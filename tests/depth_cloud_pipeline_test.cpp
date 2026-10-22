// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// DepthCloudPipeline: points straight from depth buffers and pinhole cameras,
// drawn offscreen and read back under the validation layer, with
// synchronization validation tracking the buffers the vertex stage reads.
// Each depth pixel lands where the host projection puts it, once; stride,
// gating, images read at their offsets and no further, colour by recon's rule
// through another camera, the tint where there is no colour, exact bytes on
// every target, wide points, depth against PrimitivePipeline's lines, 4K
// colour, any number of cameras in one frame in one submit or many, frames
// through a real timeline reusing blocks, teardown waiting for the
// newest frame or after one that never ran, buffers another queue family
// wrote, and every refusal recording nothing.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "color_test_support.hpp"
#include "frame_gate.hpp"
#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/camera/pinhole_camera.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/depth_cloud_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/primitive_pipeline.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;
namespace camera = volumetric_kit::gfx::camera;

using camera::PinholeCamera;
using pipelines::CloudColor;
using pipelines::DepthCloudDraw;
using pipelines::DepthCloudFrame;
using pipelines::DepthCloudPipeline;
using pipelines::DepthCloudPipelineDesc;
using pipelines::pack_rgba8;
using vg_test::open_gate;
using vg_test::OpenGateAtExit;
using vg_test::OpenGateLater;

constexpr VkFormat kUnorm = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kSrgb = VK_FORMAT_R8G8B8A8_SRGB;
constexpr VkFormat kFloat = VK_FORMAT_R32G32B32A32_SFLOAT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// The depth camera most cases use: 16 x 12, its principal point off centre.
constexpr PinholeCamera kDepthCam{14.0f, 13.0f, 7.3f, 5.6f, 16, 12};
constexpr VkExtent2D kDepthExtent{16, 12};

constexpr uint32_t kTint = pack_rgba8(255, 0, 255);

vg::RenderTargetLayout layout_for(VkFormat color) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = color;
  layout.color_count = 1;
  layout.depth_format = kDepthFormat;
  return layout;
}

VkRect2D full(VkExtent2D extent) { return {{0, 0}, extent}; }

// A rigid pose: a rotation of `degrees` about `axis`, then a translation.
glm::mat4 pose(float degrees, glm::vec3 axis, glm::vec3 translation) {
  return glm::translate(glm::mat4(1.0f), translation) *
         glm::rotate(glm::mat4(1.0f), glm::radians(degrees), axis);
}

// The depth camera's pose most cases use.
glm::mat4 depth_pose() {
  return pose(17.0f, {0.3f, 1.0f, 0.2f}, {0.4f, -0.2f, 1.5f});
}

// World to Vulkan clip space through a pinhole view `k` posed `view_to_world`,
// so that a point the camera puts at pixel (x, y) -- centres on integers --
// lands at target position (x + 0.5, y + 0.5) of a k.width x k.height
// viewport.
glm::mat4 pinhole_view_proj(const PinholeCamera& k,
                            const glm::mat4& view_to_world) {
  const double near_z = 0.05;
  const double far_z = 20.0;
  glm::dmat4 p(0.0);
  p[0][0] = 2.0 * k.fx / k.width;
  p[2][0] = 2.0 * (k.cx + 0.5) / k.width - 1.0;
  p[1][1] = 2.0 * k.fy / k.height;
  p[2][1] = 2.0 * (k.cy + 0.5) / k.height - 1.0;
  p[2][2] = far_z / (far_z - near_z);
  p[3][2] = -far_z * near_z / (far_z - near_z);
  p[2][3] = 1.0;
  return glm::mat4(p * glm::inverse(glm::dmat4(view_to_world)));
}

// Depth pixel (u, v) at `z`, in world space, in double.
glm::dvec3 unproject(const PinholeCamera& k, const glm::mat4& to_world,
                     uint32_t u, uint32_t v, double z) {
  const glm::dvec4 p((u - double{k.cx}) * z / k.fx,
                     (v - double{k.cy}) * z / k.fy, z, 1.0);
  return glm::dvec3(glm::dmat4(to_world) * p);
}

// Where `world` lands in camera `k` posed `to_world`: its pixel coordinate
// (centres on integers) and depth.
glm::dvec3 project(const PinholeCamera& k, const glm::mat4& to_world,
                   const glm::dvec3& world) {
  const glm::dvec4 c =
      glm::inverse(glm::dmat4(to_world)) * glm::dvec4(world, 1.0);
  return {k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy, c.z};
}

// What recon's rule gives a point at colour-camera coordinate `q` (depth
// q.z): the colour pixel's index, none, or ambiguous within `margin` of a
// rounding or image edge.
struct ColorPick {
  bool ambiguous = false;
  std::optional<uint32_t> index;
};

ColorPick pick_color(const PinholeCamera& k, const glm::dvec3& q,
                     double margin = 1e-3) {
  ColorPick pick;
  if (std::fabs(q.z) < margin) {
    pick.ambiguous = true;
    return pick;
  }
  if (q.z <= 0.0) {
    return pick;
  }
  const double w = k.width;
  const double h = k.height;
  if (std::fabs(q.x) < margin || std::fabs(q.y) < margin ||
      std::fabs(q.x - w) < margin || std::fabs(q.y - h) < margin) {
    pick.ambiguous = true;
    return pick;
  }
  if (!(q.x >= 0.0 && q.x < w && q.y >= 0.0 && q.y < h)) {
    return pick;
  }
  const double fx = q.x + 0.5 - std::floor(q.x + 0.5);
  const double fy = q.y + 0.5 - std::floor(q.y + 0.5);
  if (fx < margin || fx > 1.0 - margin || fy < margin || fy > 1.0 - margin) {
    pick.ambiguous = true;
    return pick;
  }
  const uint32_t x =
      std::min(static_cast<uint32_t>(std::floor(q.x + 0.5)), k.width - 1);
  const uint32_t y =
      std::min(static_cast<uint32_t>(std::floor(q.y + 0.5)), k.height - 1);
  pick.index = y * k.width + x;
  return pick;
}

// A colour image whose every word names its pixel; `seed` sets one apart.
std::vector<uint32_t> coded_colors(const PinholeCamera& k, uint32_t seed = 0) {
  std::vector<uint32_t> words(size_t{k.width} * k.height);
  for (uint32_t y = 0; y < k.height; ++y) {
    for (uint32_t x = 0; x < k.width; ++x) {
      words[size_t{y} * k.width + x] = pack_rgba8(
          static_cast<uint8_t>(10 + (x * 13 + seed * 37) % 240),
          static_cast<uint8_t>(10 + (y * 17 + seed * 53) % 240),
          static_cast<uint8_t>(10 + (x / 240 + y / 240 * 7 + seed * 60) % 240));
    }
  }
  return words;
}

// A tilted plane of depths, all positive.
std::vector<float> plane_depths(const PinholeCamera& k, float base = 1.5f) {
  std::vector<float> depths(size_t{k.width} * k.height);
  for (uint32_t v = 0; v < k.height; ++v) {
    for (uint32_t u = 0; u < k.width; ++u) {
      depths[size_t{v} * k.width + u] = base + 0.02f * u - 0.03f * v;
    }
  }
  return depths;
}

// A colour camera that sees depth pixel (u, v) at colour coordinate
// (u + dx, v + dy), from the same pose: the depth camera's intrinsics with
// its principal point moved.
PinholeCamera shifted(const PinholeCamera& k, float dx, float dy,
                      uint32_t width = 0, uint32_t height = 0) {
  PinholeCamera c = k;
  c.cx += dx;
  c.cy += dy;
  c.width = width != 0 ? width : k.width;
  c.height = height != 0 ? height : k.height;
  return c;
}

// A readback of one target.
struct Pixels {
  std::vector<uint8_t> bytes;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t texel = 4;

  const uint8_t* at(uint32_t x, uint32_t y) const {
    return &bytes[(size_t{y} * width + x) * texel];
  }
  std::array<uint8_t, 3> rgb(uint32_t x, uint32_t y) const {
    const uint8_t* p = at(x, y);
    return {p[0], p[1], p[2]};
  }
  // Whether anything was drawn at (x, y): the clear is opaque black.
  bool lit(uint32_t x, uint32_t y) const {
    const uint8_t* p = at(x, y);
    return p[0] != 0 || p[1] != 0 || p[2] != 0;
  }
  int lit_count() const {
    int count = 0;
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        count += lit(x, y) ? 1 : 0;
      }
    }
    return count;
  }
};

std::array<uint8_t, 3> rgb_of(uint32_t word) {
  return {static_cast<uint8_t>(word), static_cast<uint8_t>(word >> 8),
          static_cast<uint8_t>(word >> 16)};
}

// --- No device ---------------------------------------------------------------

TEST(DepthCloudPipelineTypesTest, IsAnAggregate) {
  static_assert(!std::is_copy_constructible_v<DepthCloudPipeline>);
  static_assert(!std::is_copy_assignable_v<DepthCloudPipeline>);
  static_assert(!std::is_move_constructible_v<DepthCloudPipeline>);
  static_assert(!std::is_move_assignable_v<DepthCloudPipeline>);
  static_assert(!std::is_default_constructible_v<DepthCloudPipeline>);
  const DepthCloudDraw draw;
  EXPECT_EQ(draw.mode, CloudColor::Solid);
  EXPECT_EQ(draw.stride, 1u);
  EXPECT_EQ(draw.color.buffer, nullptr) << "no colour unless one is given";
  EXPECT_EQ(DepthCloudPipelineDesc{}.max_draws, 16u);
}

// --- Fixtures ----------------------------------------------------------------

// Pipelines, buffers and frames on the device a fixture names: the renderer's
// shared device, or the two-family device's graphics side.
class CloudHelpers {
 public:
  virtual ~CloudHelpers() = default;

 protected:
  virtual vkc::Device& cloud_device() = 0;
  virtual vkc::Allocator& cloud_allocator() = 0;

  void make_timeline() {
    auto timeline = vkc::TimelineSemaphore::create(cloud_device(), 0);
    ASSERT_TRUE(timeline.ok()) << timeline.status().message();
    timeline_.emplace(std::move(timeline).value());
  }

  std::unique_ptr<DepthCloudPipeline> make_pipeline(VkFormat color = kUnorm,
                                                    uint32_t max_draws = 16) {
    DepthCloudPipelineDesc desc;
    desc.max_draws = max_draws;
    auto made = DepthCloudPipeline::create(cloud_device(), cloud_allocator(),
                                           layout_for(color), *timeline_, desc);
    EXPECT_TRUE(made.ok()) << made.status().message();
    return made.ok() ? std::move(made).value() : nullptr;
  }

  // `bytes` in a device-only buffer of `usage`, uploaded on the renderer's
  // queue, which has finished when this returns.
  vkc::Buffer upload(
      const void* data, size_t bytes,
      VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
    vg::BufferUploadDesc desc;
    desc.data = data;
    desc.size = bytes;
    desc.usage = usage;
    auto buffer = vg::upload_buffer(cloud_device(), cloud_allocator(), desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
  }
  template <class T>
  vkc::Buffer upload(const std::vector<T>& values) {
    return upload(values.data(), values.size() * sizeof(T));
  }

  // The next frame number.
  uint64_t next_frame() { return next_++; }

  // Records `record` inside a rendering scope over an `extent` target of
  // `color`, cleared to opaque black and depth 1, runs it, sets frame
  // `number` on the timeline from the host, and returns `record`'s status and
  // the target. An optional producer timeline is waited on by the GPU at
  // VERTEX_SHADER; there is no host wait before the render submission.
  std::pair<vkc::Status, Pixels> render_status(
      VkExtent2D extent, VkFormat color, uint64_t number,
      const std::function<vkc::Status(VkCommandBuffer)>& record,
      vkc::TimelinePoint ready = {}) {
    const vkc::Status failed = vkc::Status::invalid_argument("setup failed");
    vg::OffscreenTargetDesc desc;
    desc.extent = extent;
    desc.color_format = color;
    desc.depth_format = kDepthFormat;
    auto target = vg::OffscreenTarget::create(cloud_allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    auto pool = vkc::CommandPool::create(cloud_device().handle(),
                                         cloud_device().queue_family());
    EXPECT_TRUE(pool.ok()) << pool.status().message();
    if (!target.ok() || !pool.ok()) {
      return {failed, {}};
    }
    auto cmd = pool.value().allocate_primary();
    EXPECT_TRUE(cmd.ok()) << cmd.status().message();
    if (!cmd.ok()) {
      return {failed, {}};
    }
    EXPECT_TRUE(
        cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    const VkCommandBuffer raw = cmd.value().handle();
    target.value().prepare(raw);
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[3] = 1.0f;
    const vg::RenderTarget rt = target.value().target();
    rt.begin(raw, begin);
    const vkc::Status recorded = record(raw);
    rt.end(raw);
    target.value().record_readback(raw);
    EXPECT_TRUE(cmd.value().end().ok());
    const vkc::Status ran = [&]() -> vkc::Status {
      if (ready.semaphore == nullptr) {
        return cloud_device().submit_and_wait(raw);
      }
      VKC_ASSIGN(vkc::Fence done, vkc::Fence::create(cloud_device().handle()));
      const VkSemaphore wait = ready.semaphore->handle();
      const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
      VkTimelineSemaphoreSubmitInfo values{};
      values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
      values.waitSemaphoreValueCount = 1;
      values.pWaitSemaphoreValues = &ready.value;
      VkSubmitInfo submit{};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.pNext = &values;
      submit.waitSemaphoreCount = 1;
      submit.pWaitSemaphores = &wait;
      submit.pWaitDstStageMask = &stage;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &raw;
      VKC_VK_TRY(cloud_device().queue_submit(1, &submit, done.handle()));
      return done.wait();
    }();
    if (!ran.ok()) {
      ADD_FAILURE() << ran.message();
      return {recorded, {}};
    }
    const auto reached = timeline_->value();
    if (reached.ok() && reached.value() < number) {
      EXPECT_TRUE(timeline_->signal(number).ok());
    }
    Pixels px;
    px.width = extent.width;
    px.height = extent.height;
    px.texel = color == kFloat ? 16 : 4;
    const auto* bytes = static_cast<const uint8_t*>(target.value().pixels());
    EXPECT_NE(bytes, nullptr);
    if (bytes != nullptr) {
      px.bytes.assign(bytes,
                      bytes + size_t{px.texel} * extent.width * extent.height);
    }
    return {recorded, std::move(px)};
  }

  // render_status for `draws` in one submit of the next frame, expecting it
  // to succeed.
  Pixels render(DepthCloudPipeline& pipeline,
                const std::vector<DepthCloudDraw>& draws, VkExtent2D extent,
                VkFormat color = kUnorm, vkc::TimelinePoint ready = {}) {
    const uint64_t number = next_frame();
    auto [status, px] = render_status(
        extent, color, number,
        [&](VkCommandBuffer cmd) {
          return pipeline.submit(cmd, {extent, number, draws.data(),
                                       static_cast<uint32_t>(draws.size())});
        },
        ready);
    EXPECT_TRUE(status.ok()) << status.message();
    EXPECT_FALSE(px.bytes.empty());
    return std::move(px);
  }

  // render_status for `draws` in the next frame, one submit each, expecting
  // every submit to succeed.
  Pixels render_each(DepthCloudPipeline& pipeline,
                     const std::vector<DepthCloudDraw>& draws,
                     VkExtent2D extent) {
    const uint64_t number = next_frame();
    auto [status, px] =
        render_status(extent, kUnorm, number, [&](VkCommandBuffer cmd) {
          for (const DepthCloudDraw& draw : draws) {
            VKC_TRY(pipeline.submit(cmd, {extent, number, &draw, 1}));
          }
          return vkc::Status{};
        });
    EXPECT_TRUE(status.ok()) << status.message();
    EXPECT_FALSE(px.bytes.empty());
    return std::move(px);
  }

  // A draw of `depth` through camera `k` posed `to_world`, viewed by the same
  // camera into the whole of a k-sized target, so depth pixel (u, v) lands
  // on target pixel (u, v).
  static DepthCloudDraw self_view(const vkc::Buffer& depth,
                                  const PinholeCamera& k,
                                  const glm::mat4& to_world) {
    DepthCloudDraw draw;
    draw.depth.buffer = &depth;
    draw.depth.camera = k;
    draw.depth.to_world = to_world;
    draw.tint = kTint;
    draw.viewport = full({k.width, k.height});
    draw.view_proj = pinhole_view_proj(k, to_world);
    return draw;
  }

  std::optional<vkc::TimelineSemaphore> timeline_;
  uint64_t next_ = 1;
};

class DepthCloudTest : public vg_test::RendererDeviceTest,
                       protected CloudHelpers {
 protected:
  // The draws read storage buffers through descriptors, which
  // synchronization validation tracks only with this.
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::ShaderAccesses;
  }
  vkc::Device& cloud_device() override { return device(); }
  vkc::Allocator& cloud_allocator() override { return allocator(); }

  void SetUp() override {
    vg_test::RendererDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    make_timeline();
  }
};

// --- Creation ----------------------------------------------------------------

TEST_F(DepthCloudTest, CreatesAndRefusesWhatItCannotDraw) {
  auto pipeline = make_pipeline(kUnorm, 3);
  ASSERT_NE(pipeline, nullptr);
  EXPECT_EQ(pipeline->max_draws(), 3u);
  EXPECT_EQ(pipeline->block_count(), 1u) << "one block made up front";
  // The renderer's requirements do not enable largePoints.
  EXPECT_EQ(pipeline->max_point_size(), 1.0f);

  const auto refused = [&](const vg::RenderTargetLayout& layout,
                           const vkc::TimelineSemaphore& timeline,
                           uint32_t max_draws, const char* what) {
    DepthCloudPipelineDesc desc;
    desc.max_draws = max_draws;
    auto made = DepthCloudPipeline::create(device(), allocator(), layout,
                                           timeline, desc);
    ASSERT_FALSE(made.ok()) << what;
    EXPECT_EQ(made.status().domain(), vkc::Status::Code::InvalidArgument)
        << what;
    EXPECT_NE(made.status().message().find("DepthCloudPipeline::create"),
              std::string::npos)
        << made.status().message();
  };
  vg::RenderTargetLayout no_depth = layout_for(kUnorm);
  no_depth.depth_format = VK_FORMAT_UNDEFINED;
  refused(no_depth, *timeline_, 16, "no depth format");
  refused(layout_for(VK_FORMAT_UNDEFINED), *timeline_, 16, "no color format");
  vg::RenderTargetLayout two = layout_for(kUnorm);
  two.color_formats[1] = kUnorm;
  two.color_count = 2;
  refused(two, *timeline_, 16, "two color attachments");
  const vkc::TimelineSemaphore empty;
  refused(layout_for(kUnorm), empty, 16, "an empty timeline");
  refused(layout_for(kUnorm), *timeline_, 0, "zero max_draws");
}

// A device made for another library -- a compute one, without dynamic
// rendering -- is refused rather than drawn on.
class DepthCloudForeignDeviceTest : public vkc::test::VulkanDeviceTest {};

TEST_F(DepthCloudForeignDeviceTest, IsRefused) {
  auto timeline = vkc::TimelineSemaphore::create(device(), 0);
  ASSERT_TRUE(timeline.ok()) << timeline.status().message();
  auto made = DepthCloudPipeline::create(device(), allocator(),
                                         layout_for(kUnorm), timeline.value());
  ASSERT_FALSE(made.ok());
  EXPECT_EQ(made.status().domain(), vkc::Status::Code::Unsupported)
      << made.status().message();
}

// --- Where points land
// ---------------------------------------------------------

// Test 1: through its own camera, each depth pixel lands on its pixel, once:
// every pixel of the target holds the colour coded for it.
TEST_F(DepthCloudTest, EachPixelLandsOnItsProjectedPixelOnce) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const glm::mat4 to_world = depth_pose();
  const vkc::Buffer depth = upload(plane_depths(kDepthCam));
  // Colour coordinate (u + 0.25, v + 0.25): pixel (u, v), clear of every
  // rounding edge.
  const PinholeCamera color_cam = shifted(kDepthCam, 0.25f, 0.25f);
  const std::vector<uint32_t> words = coded_colors(color_cam);
  const vkc::Buffer color = upload(words);

  DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
  draw.mode = CloudColor::True;
  draw.color.buffer = &color;
  draw.color.camera = color_cam;
  draw.color.to_world = to_world;
  const Pixels px = render(*pipeline, {draw}, kDepthExtent);
  ASSERT_FALSE(px.bytes.empty());
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      EXPECT_EQ(px.rgb(u, v), rgb_of(words[v * kDepthCam.width + u]))
          << "pixel (" << u << ", " << v << ")";
    }
  }

  // Stride 2 draws the even pixels alone.
  draw.stride = 2;
  const Pixels even = render(*pipeline, {draw}, kDepthExtent);
  ASSERT_FALSE(even.bytes.empty());
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      if (u % 2 == 0 && v % 2 == 0) {
        EXPECT_EQ(even.rgb(u, v), rgb_of(words[v * kDepthCam.width + u]))
            << "pixel (" << u << ", " << v << ")";
      } else {
        EXPECT_FALSE(even.lit(u, v)) << "pixel (" << u << ", " << v << ")";
      }
    }
  }
  // Stride 5 on a 16 x 12 image: columns 0, 5, 10, 15 and rows 0, 5, 10.
  draw.stride = 5;
  EXPECT_EQ(render(*pipeline, {draw}, kDepthExtent).lit_count(), 4 * 3);
}

TEST_F(DepthCloudTest, CoincidentCamerasKeepEveryBorderColor) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const vkc::Buffer depth = upload(plane_depths(kDepthCam));
  const std::vector<uint32_t> words = coded_colors(kDepthCam);
  const vkc::Buffer color = upload(words);
  for (const glm::mat4& to_world : {glm::mat4(1.0f), depth_pose()}) {
    DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
    draw.mode = CloudColor::True;
    // Registered RGB-D: no principal-point offset to move the first row
    // and column away from the shader's exact image-boundary checks.
    draw.color = {&color, 0, kDepthCam, to_world, true};
    const Pixels px = render(*pipeline, {draw}, kDepthExtent);
    ASSERT_FALSE(px.bytes.empty());
    for (uint32_t v = 0; v < kDepthCam.height; ++v) {
      for (uint32_t u = 0; u < kDepthCam.width; ++u) {
        EXPECT_EQ(px.rgb(u, v), rgb_of(words[v * kDepthCam.width + u]))
            << "pixel (" << u << ", " << v << ")";
      }
    }
  }
}

// Test 1: through a view moved and with its own intrinsics and size, each
// point lands on the pixel the host projection gives it, and nothing else is
// lit.
TEST_F(DepthCloudTest, ATranslatedViewMatchesTheHostProjection) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const glm::mat4 to_world = depth_pose();
  const std::vector<float> depths = plane_depths(kDepthCam);
  const vkc::Buffer depth = upload(depths);
  const PinholeCamera color_cam = shifted(kDepthCam, 0.25f, 0.25f);
  const std::vector<uint32_t> words = coded_colors(color_cam);
  const vkc::Buffer color = upload(words);

  const PinholeCamera view{15.0f, 15.5f, 11.4f, 9.2f, 24, 20};
  const glm::mat4 view_to_world =
      to_world * pose(3.0f, {0.0f, 1.0f, 0.0f}, {0.07f, -0.05f, -0.3f});
  DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
  draw.mode = CloudColor::True;
  draw.color.buffer = &color;
  draw.color.camera = color_cam;
  draw.color.to_world = to_world;
  draw.viewport = full({view.width, view.height});
  draw.view_proj = pinhole_view_proj(view, view_to_world);
  const Pixels px = render(*pipeline, {draw}, {view.width, view.height});
  ASSERT_FALSE(px.bytes.empty());

  // Each source's target pixel, by the host; a pixel two sources share, or a
  // source within 1e-3 of a pixel edge, is not checked for its colour.
  std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> landed;
  std::vector<std::pair<uint32_t, uint32_t>> may_light;
  int checked = 0;
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      const glm::dvec3 world =
          unproject(kDepthCam, to_world, u, v, depths[v * kDepthCam.width + u]);
      const glm::dvec3 q = project(view, view_to_world, world);
      const double x = q.x + 0.5;
      const double y = q.y + 0.5;
      if (!(x >= 0.0 && x < view.width && y >= 0.0 && y < view.height)) {
        continue;
      }
      const double fx = x - std::floor(x);
      const double fy = y - std::floor(y);
      const auto pixel =
          std::make_pair(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
      if (fx < 1e-3 || fx > 1.0 - 1e-3 || fy < 1e-3 || fy > 1.0 - 1e-3) {
        for (double dx : {-1e-3, 1e-3}) {
          for (double dy : {-1e-3, 1e-3}) {
            may_light.emplace_back(static_cast<uint32_t>(x + dx),
                                   static_cast<uint32_t>(y + dy));
          }
        }
        continue;
      }
      landed[pixel].push_back(v * kDepthCam.width + u);
    }
  }
  for (const auto& [pixel, sources] : landed) {
    may_light.push_back(pixel);
    EXPECT_TRUE(px.lit(pixel.first, pixel.second))
        << "(" << pixel.first << ", " << pixel.second << ")";
    if (sources.size() == 1) {
      EXPECT_EQ(px.rgb(pixel.first, pixel.second), rgb_of(words[sources[0]]))
          << "(" << pixel.first << ", " << pixel.second << ")";
      ++checked;
    }
  }
  EXPECT_GT(checked, 150) << "the view keeps most points apart";
  for (uint32_t y = 0; y < view.height; ++y) {
    for (uint32_t x = 0; x < view.width; ++x) {
      if (px.lit(x, y)) {
        bool expected = false;
        for (const auto& p : may_light) {
          expected = expected || (p.first == x && p.second == y);
        }
        EXPECT_TRUE(expected) << "(" << x << ", " << y << ") is a stray";
      }
    }
  }
}

// Test 2: no return (0), NaN, a negative or infinite depth, and depths
// outside [min_depth, max_depth] draw nothing; the bounds themselves draw.
TEST_F(DepthCloudTest, GatesNoReturnNaNAndOutOfRangeDepths) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  std::vector<float> depths(size_t{kDepthCam.width} * kDepthCam.height, 2.0f);
  const auto at = [&](uint32_t u, uint32_t v) -> float& {
    return depths[v * kDepthCam.width + u];
  };
  at(1, 1) = 0.0f;
  at(2, 1) = std::numeric_limits<float>::quiet_NaN();
  at(3, 1) = -1.0f;
  at(4, 1) = std::numeric_limits<float>::infinity();
  at(5, 1) = 0.4f;  // below min
  at(6, 1) = 3.5f;  // above max
  at(7, 1) = 0.5f;  // min itself
  at(8, 1) = 3.0f;  // max itself
  const vkc::Buffer depth = upload(depths);
  DepthCloudDraw draw = self_view(depth, kDepthCam, depth_pose());
  draw.depth.min_depth = 0.5f;
  draw.depth.max_depth = 3.0f;

  const Pixels bounded = render(*pipeline, {draw}, kDepthExtent);
  ASSERT_FALSE(bounded.bytes.empty());
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      const bool gated = v == 1 && u >= 1 && u <= 6;
      EXPECT_EQ(bounded.lit(u, v), !gated) << "(" << u << ", " << v << ")";
    }
  }

  // Without bounds only what is not a depth -- 0, NaN, negative, infinite --
  // is dropped.
  draw.depth.min_depth = 0.0f;
  draw.depth.max_depth = 0.0f;
  const Pixels open = render(*pipeline, {draw}, kDepthExtent);
  ASSERT_FALSE(open.bytes.empty());
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      const bool gated = v == 1 && u >= 1 && u <= 4;
      EXPECT_EQ(open.lit(u, v), !gated) << "(" << u << ", " << v << ")";
    }
  }

  // Seen from 2 m behind the depth camera, with a wide view, a point at the
  // depth camera's centre (depth 0) or behind it (negative) would be in
  // view: an image of nothing but one gated value draws nothing at all.
  const PinholeCamera wide{6.0f, 6.0f, 7.5f, 5.5f, 16, 12};
  const glm::mat4 behind =
      draw.depth.to_world * pose(0.0f, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -2.0f});
  struct Gated {
    float depth;
    float min_depth;
    float max_depth;
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const Gated& g : {Gated{0.0f, 0.0f, 0.0f}, Gated{-1.0f, 0.0f, 0.0f},
                         Gated{nan, 0.0f, 0.0f}, Gated{inf, 0.0f, 0.0f},
                         Gated{-inf, 0.0f, 0.0f}, Gated{0.4f, 0.5f, 3.0f},
                         Gated{3.5f, 0.5f, 3.0f}, Gated{2.0f, 0.0f, 0.0f}}) {
    const bool shown = g.depth == 2.0f;  // the control
    SCOPED_TRACE("depth " + std::to_string(g.depth));
    const vkc::Buffer only = upload(std::vector<float>(
        size_t{kDepthCam.width} * kDepthCam.height, g.depth));
    DepthCloudDraw seen = self_view(only, kDepthCam, draw.depth.to_world);
    seen.depth.min_depth = g.min_depth;
    seen.depth.max_depth = g.max_depth;
    seen.view_proj = pinhole_view_proj(wide, behind);
    const Pixels px = render(*pipeline, {seen}, kDepthExtent);
    ASSERT_FALSE(px.bytes.empty());
    if (shown) {
      EXPECT_GT(px.lit_count(), 4) << "the control is in view";
    } else {
      EXPECT_EQ(px.lit_count(), 0);
    }
  }
}

// --- Where a draw reads
// -------------------------------------------------------

// Each image is read from its offset in a larger buffer, and no further: the
// words before and after it are valid depths and colours, any of which, read,
// would light a pixel or change its colour. The view frames the image with a
// border, so a point from past its last row or column lands in view. Holes on
// each stride's grid and depths off it pin which pixel each vertex reads, and
// a stride past the image's size draws pixel (0, 0) alone.
TEST_F(DepthCloudTest, ReadsEachImageAtItsOffsetAndNoFurther) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const uint32_t w = kDepthCam.width;
  const uint32_t h = kDepthCam.height;
  const size_t pixels = size_t{w} * h;
  // At least 64 words before each image and after it.
  const VkDeviceSize align =
      device().caps().limits().minStorageBufferOffsetAlignment;
  const VkDeviceSize offset = (256 + align - 1) / align * align;
  const size_t pad = static_cast<size_t>(offset / 4);
  const auto hole = [](uint32_t u, uint32_t v) { return (u + 2 * v) % 3 == 1; };
  std::vector<float> depth_words(pad + pixels + pad, 1.0f);
  for (uint32_t v = 0; v < h; ++v) {
    for (uint32_t u = 0; u < w; ++u) {
      depth_words[pad + size_t{v} * w + u] =
          hole(u, v) ? 0.0f : 1.5f + 0.01f * u;
    }
  }
  // Colour pixel (u, v) for depth pixel (u, v).
  const PinholeCamera color_cam = shifted(kDepthCam, 0.25f, 0.25f);
  const std::vector<uint32_t> words = coded_colors(color_cam);
  std::vector<uint32_t> color_words(pad + pixels + pad, pack_rgba8(1, 2, 3));
  std::copy(words.begin(), words.end(), color_words.begin() + pad);
  const vkc::Buffer depth = upload(depth_words);
  const vkc::Buffer color = upload(color_words);

  // Depth pixel (u, v) lands on target pixel (u + kBorder, v + kBorder).
  constexpr uint32_t kBorder = 8;
  const PinholeCamera framed =
      shifted(kDepthCam, static_cast<float>(kBorder),
              static_cast<float>(kBorder), w + 2 * kBorder, h + 2 * kBorder);
  const VkExtent2D extent{framed.width, framed.height};
  const glm::mat4 to_world = depth_pose();
  DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
  draw.depth.offset = offset;
  draw.mode = CloudColor::True;
  draw.color = {&color, offset, color_cam, to_world, false};
  draw.viewport = full(extent);
  draw.view_proj = pinhole_view_proj(framed, to_world);

  for (const uint32_t stride : {1u, 2u, 5u, 7u}) {
    SCOPED_TRACE("stride " + std::to_string(stride));
    draw.stride = stride;
    const Pixels px = render(*pipeline, {draw}, extent);
    ASSERT_FALSE(px.bytes.empty());
    int holes = 0;
    for (uint32_t y = 0; y < extent.height; ++y) {
      for (uint32_t x = 0; x < extent.width; ++x) {
        const bool inside =
            x >= kBorder && x < kBorder + w && y >= kBorder && y < kBorder + h;
        const uint32_t u = x - kBorder;
        const uint32_t v = y - kBorder;
        const bool on_grid = inside && u % stride == 0 && v % stride == 0;
        if (on_grid && !hole(u, v)) {
          EXPECT_EQ(px.rgb(x, y), rgb_of(words[size_t{v} * w + u]))
              << "(" << x << ", " << y << ")";
        } else {
          EXPECT_FALSE(px.lit(x, y)) << "(" << x << ", " << y << ")";
        }
        holes += on_grid && hole(u, v) ? 1 : 0;
      }
    }
    EXPECT_GT(holes, 0) << "the grid has holes";
  }
  for (const uint32_t stride : {16u, 1000u, 0x80000000u, 0xFFFFFFFFu}) {
    SCOPED_TRACE("stride " + std::to_string(stride));
    draw.stride = stride;
    const Pixels px = render(*pipeline, {draw}, extent);
    ASSERT_FALSE(px.bytes.empty());
    EXPECT_EQ(px.lit_count(), 1);
    EXPECT_EQ(px.rgb(kBorder, kBorder), rgb_of(words[0]));
  }
}

// --- Colour ------------------------------------------------------------------

// Test 3: through a colour camera with its own intrinsics, size and pose,
// each point takes the colour recon's rule picks, or the tint where that
// rule finds none.
TEST_F(DepthCloudTest, ColorFollowsReconsRuleThroughAnotherCamera) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const glm::mat4 to_world = depth_pose();
  const std::vector<float> depths = plane_depths(kDepthCam, 1.2f);
  const vkc::Buffer depth = upload(depths);
  const PinholeCamera color_cam{22.0f, 21.0f, 9.7f, 8.1f, 20, 15};
  const glm::mat4 color_to_world =
      to_world * pose(4.0f, {0.0f, 1.0f, 0.1f}, {0.06f, -0.02f, 0.01f});
  const std::vector<uint32_t> words = coded_colors(color_cam, 3);
  const vkc::Buffer color = upload(words);

  DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
  draw.mode = CloudColor::True;
  draw.color.buffer = &color;
  draw.color.camera = color_cam;
  draw.color.to_world = color_to_world;
  const Pixels px = render(*pipeline, {draw}, kDepthExtent);
  ASSERT_FALSE(px.bytes.empty());
  int colored = 0;
  int tinted = 0;
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      const glm::dvec3 world =
          unproject(kDepthCam, to_world, u, v, depths[v * kDepthCam.width + u]);
      const ColorPick pick =
          pick_color(color_cam, project(color_cam, color_to_world, world));
      if (pick.ambiguous) {
        continue;
      }
      const std::array<uint8_t, 3> expected =
          pick.index ? rgb_of(words[*pick.index]) : rgb_of(kTint);
      EXPECT_EQ(px.rgb(u, v), expected) << "(" << u << ", " << v << ")";
      (pick.index ? colored : tinted) += 1;
    }
  }
  EXPECT_GT(colored, 100);
  EXPECT_GT(tinted, 5) << "some points fall outside the colour image";
}

// Test 3: a point on a pixel of coverage 0, one outside the colour image and
// one behind the colour camera take the tint; the last half pixel clamps.
TEST_F(DepthCloudTest, UncoveredOutsideAndBehindTakeTheTint) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const glm::mat4 to_world = depth_pose();
  const vkc::Buffer depth = upload(plane_depths(kDepthCam));
  const uint32_t w = kDepthCam.width;

  {  // Coverage: every third pixel's word has alpha 0.
    const PinholeCamera color_cam = shifted(kDepthCam, 0.25f, 0.25f);
    std::vector<uint32_t> words = coded_colors(color_cam);
    for (size_t i = 0; i < words.size(); ++i) {
      if (i % 3 == 0) {
        words[i] &= 0x00FFFFFFu;
      }
    }
    const vkc::Buffer color = upload(words);
    DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
    draw.mode = CloudColor::True;
    draw.color = {&color, 0, color_cam, to_world, true};
    const Pixels covered = render(*pipeline, {draw}, kDepthExtent);
    draw.color.alpha_is_coverage = false;
    const Pixels ignored = render(*pipeline, {draw}, kDepthExtent);
    ASSERT_FALSE(covered.bytes.empty());
    ASSERT_FALSE(ignored.bytes.empty());
    for (uint32_t v = 0; v < kDepthCam.height; ++v) {
      for (uint32_t u = 0; u < w; ++u) {
        const uint32_t i = v * w + u;
        EXPECT_EQ(covered.rgb(u, v),
                  i % 3 == 0 ? rgb_of(kTint) : rgb_of(words[i]))
            << "(" << u << ", " << v << ")";
        EXPECT_EQ(ignored.rgb(u, v), rgb_of(words[i]))
            << "alpha ignored: (" << u << ", " << v << ")";
      }
    }
  }
  {  // Outside: colour x = u - 3.25 in a 10-wide image. Columns 0-3 fall
    // left of it and 14-15 right of it; 13 lands on 9.75, the last half
    // pixel, which clamps to column 9.
    const PinholeCamera color_cam = shifted(kDepthCam, -3.25f, 0.25f, 10, 12);
    const std::vector<uint32_t> words = coded_colors(color_cam, 1);
    const vkc::Buffer color = upload(words);
    DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
    draw.mode = CloudColor::True;
    draw.color = {&color, 0, color_cam, to_world, false};
    const Pixels px = render(*pipeline, {draw}, kDepthExtent);
    ASSERT_FALSE(px.bytes.empty());
    for (uint32_t v = 0; v < kDepthCam.height; ++v) {
      for (uint32_t u = 0; u < w; ++u) {
        std::array<uint8_t, 3> expected = rgb_of(kTint);
        if (u >= 4 && u <= 13) {
          expected = rgb_of(words[v * 10 + std::min(u - 3, 9u)]);
        }
        EXPECT_EQ(px.rgb(u, v), expected) << "(" << u << ", " << v << ")";
      }
    }
  }
  {  // Behind: the colour camera faces the other way.
    const PinholeCamera color_cam = shifted(kDepthCam, 0.25f, 0.25f);
    const vkc::Buffer color = upload(coded_colors(color_cam));
    DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
    draw.mode = CloudColor::True;
    draw.color = {&color, 0, color_cam,
                  to_world * pose(180.0f, {0.0f, 1.0f, 0.0f}, {}), false};
    const Pixels px = render(*pipeline, {draw}, kDepthExtent);
    ASSERT_FALSE(px.bytes.empty());
    for (uint32_t v = 0; v < kDepthCam.height; ++v) {
      for (uint32_t u = 0; u < w; ++u) {
        EXPECT_EQ(px.rgb(u, v), rgb_of(kTint)) << "(" << u << ", " << v << ")";
      }
    }
  }
}

// Test 3: Solid stores the tint's bytes, and TintedLuma the tint scaled by
// the colour's luma, exactly on a UNORM and an sRGB target; a float target
// holds their linear light.
TEST_F(DepthCloudTest, SolidAndTintedLumaStoreExactBytesOnEveryTarget) {
  const glm::mat4 to_world = depth_pose();
  const vkc::Buffer depth = upload(plane_depths(kDepthCam));
  const PinholeCamera color_cam = shifted(kDepthCam, 0.25f, 0.25f);
  // Every word (240, 30, 90): Y' = 0.30972; the tint (200, 120, 255) scaled
  // by it is (61.94, 37.17, 78.98), clear of every rounding edge.
  const std::vector<uint32_t> words(size_t{color_cam.width} * color_cam.height,
                                    pack_rgba8(240, 30, 90));
  const vkc::Buffer color = upload(words);
  const uint32_t tint = pack_rgba8(200, 120, 255);
  const double luma = (0.2126 * 240 + 0.7152 * 30 + 0.0722 * 90) / 255.0;

  for (const CloudColor mode : {CloudColor::Solid, CloudColor::TintedLuma}) {
    const bool solid = mode == CloudColor::Solid;
    SCOPED_TRACE(solid ? "Solid" : "TintedLuma");
    DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
    draw.mode = mode;
    draw.tint = tint;
    draw.color = {&color, 0, color_cam, to_world, true};
    // The encoded value each channel should store.
    std::array<double, 3> encoded{};
    for (int c = 0; c < 3; ++c) {
      const double t = static_cast<uint8_t>(tint >> (8 * c)) / 255.0;
      encoded[c] = solid ? t : t * luma;
    }
    for (const VkFormat format : {kUnorm, kSrgb}) {
      auto pipeline = make_pipeline(format);
      ASSERT_NE(pipeline, nullptr);
      const Pixels px = render(*pipeline, {draw}, kDepthExtent, format);
      ASSERT_FALSE(px.bytes.empty());
      for (int c = 0; c < 3; ++c) {
        const int expected = static_cast<int>(std::lround(encoded[c] * 255.0));
        EXPECT_EQ(px.at(5, 7)[c], expected)
            << (format == kUnorm ? "UNORM" : "sRGB") << " channel " << c;
        EXPECT_EQ(px.at(15, 0)[c], expected);
      }
      EXPECT_EQ(px.at(5, 7)[3], 255);
    }
    auto pipeline = make_pipeline(kFloat);
    ASSERT_NE(pipeline, nullptr);
    const Pixels px = render(*pipeline, {draw}, kDepthExtent, kFloat);
    ASSERT_FALSE(px.bytes.empty());
    float texel[4];
    std::memcpy(texel, px.at(5, 7), sizeof(texel));
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(texel[c], vg_test::srgb_to_linear(encoded[c]), 1e-4)
          << "float channel " << c;
    }
    EXPECT_EQ(texel[3], 1.0f);
  }
}

// --- Points, depth -----------------------------------------------------------

// Without largePoints, a point is one pixel whatever size it asks for.
TEST_F(DepthCloudTest, PointsAreOnePixelWithoutLargePoints) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  ASSERT_EQ(pipeline->max_point_size(), 1.0f);
  const vkc::Buffer depth = upload(plane_depths(kDepthCam));
  DepthCloudDraw draw = self_view(depth, kDepthCam, depth_pose());
  draw.stride = 5;
  draw.point_size = 9.0f;
  EXPECT_EQ(render(*pipeline, {draw}, kDepthExtent).lit_count(), 4 * 3);
}

// Test 5: a cloud and PrimitivePipeline lines in one rendering scope occlude
// each other by depth, whichever is drawn first.
TEST_F(DepthCloudTest, CloudsAndLinesOccludeEachOtherByDepth) {
  auto clouds = make_pipeline();
  ASSERT_NE(clouds, nullptr);
  auto made =
      pipelines::PrimitivePipeline::create(device(), layout_for(kUnorm));
  ASSERT_TRUE(made.ok()) << made.status().message();
  const std::unique_ptr<pipelines::PrimitivePipeline> lines =
      std::move(made).value();
  const glm::mat4 to_world = depth_pose();
  const vkc::Buffer depth = upload(
      std::vector<float>(size_t{kDepthCam.width} * kDepthCam.height, 1.0f));
  const DepthCloudDraw cloud = self_view(depth, kDepthCam, to_world);

  // Lines across row 5 at depth 0.5, in front of the cloud, and across row 8
  // at depth 2, behind it.
  const uint32_t green = pack_rgba8(0, 255, 0);
  const auto across = [&](uint32_t row, double z) {
    const auto end = [&](double u) {
      const glm::dvec4 p((u - kDepthCam.cx) * z / kDepthCam.fx,
                         (row - double{kDepthCam.cy}) * z / kDepthCam.fy, z,
                         1.0);
      return glm::vec3(glm::dmat4(to_world) * p);
    };
    return std::array<pipelines::ColorVertex, 2>{
        pipelines::ColorVertex{end(-1.0), green},
        pipelines::ColorVertex{end(16.0), green}};
  };
  std::vector<pipelines::ColorVertex> vertices;
  for (const auto& v : across(5, 0.5)) vertices.push_back(v);
  for (const auto& v : across(8, 2.0)) vertices.push_back(v);
  const vkc::Buffer line_buffer =
      upload(vertices.data(), vertices.size() * sizeof(vertices[0]),
             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  pipelines::PrimitiveDraw line;
  line.topology = pipelines::PrimitiveTopology::Lines;
  line.vertices = &line_buffer;
  line.vertex_count = 4;
  line.viewport = full(kDepthExtent);
  line.view_proj = cloud.view_proj;

  for (const bool cloud_first : {true, false}) {
    SCOPED_TRACE(cloud_first ? "cloud first" : "lines first");
    const uint64_t number = next_frame();
    auto [status, px] =
        render_status(kDepthExtent, kUnorm, number, [&](VkCommandBuffer cmd) {
          const auto draw_lines = [&] {
            return lines->submit(cmd, {kDepthExtent, &line, 1});
          };
          const auto draw_cloud = [&] {
            return clouds->submit(cmd, {kDepthExtent, number, &cloud, 1});
          };
          VKC_TRY(cloud_first ? draw_cloud() : draw_lines());
          return cloud_first ? draw_lines() : draw_cloud();
        });
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_FALSE(px.bytes.empty());
    for (uint32_t u = 1; u + 1 < kDepthCam.width; ++u) {
      EXPECT_EQ(px.rgb(u, 5), rgb_of(green)) << "the near line, u = " << u;
      EXPECT_EQ(px.rgb(u, 8), rgb_of(kTint))
          << "the far line hidden, u = " << u;
      EXPECT_EQ(px.rgb(u, 3), rgb_of(kTint)) << "the cloud, u = " << u;
    }
  }
}

// --- Any number of cameras
// -----------------------------------------------------

// N cameras of mixed sizes, each with its own depth (holes at a pattern of
// its own), pose, tint and viewport cell, in one frame: in one submit, and in
// a submit each, which share the frame's block; with blocks of two, in one
// submit that spans ceil(N / 2) blocks, and in a submit each, which fill the
// same blocks two by two.
class DepthCloudCamerasTest : public DepthCloudTest,
                              public ::testing::WithParamInterface<uint32_t> {};

TEST_P(DepthCloudCamerasTest, AnyNumberOfCamerasInOneFrame) {
  const uint32_t n = GetParam();
  const std::array<VkExtent2D, 5> sizes{
      {{16, 12}, {12, 10}, {10, 8}, {14, 12}, {8, 6}}};
  uint32_t columns = 1;
  while (columns * columns < n) {
    ++columns;
  }
  const uint32_t rows = (n + columns - 1) / columns;
  const VkExtent2D cell{16, 12};
  const VkExtent2D extent{columns * cell.width, rows * cell.height};

  std::vector<vkc::Buffer> depths;
  std::vector<DepthCloudDraw> draws;
  std::vector<PinholeCamera> cams;
  for (uint32_t i = 0; i < n; ++i) {
    const PinholeCamera k{12.0f + i,
                          11.0f + i,
                          sizes[i].width / 2.0f - 0.3f,
                          sizes[i].height / 2.0f + 0.2f,
                          sizes[i].width,
                          sizes[i].height};
    cams.push_back(k);
    std::vector<float> values(size_t{k.width} * k.height);
    for (uint32_t v = 0; v < k.height; ++v) {
      for (uint32_t u = 0; u < k.width; ++u) {
        values[size_t{v} * k.width + u] =
            (u + v) % (i + 2) == 0 ? 0.0f : 1.0f + 0.1f * i;
      }
    }
    depths.push_back(upload(values));
  }
  for (uint32_t i = 0; i < n; ++i) {
    const glm::mat4 to_world =
        pose(40.0f * i, {0.0f, 1.0f, 0.0f}, {0.5f * i, 0.0f, -0.2f * i});
    DepthCloudDraw draw = self_view(depths[i], cams[i], to_world);
    draw.tint = pack_rgba8(static_cast<uint8_t>(40 + 50 * i),
                           static_cast<uint8_t>(250 - 40 * i), 100);
    draw.viewport = {{static_cast<int32_t>((i % columns) * cell.width),
                      static_cast<int32_t>((i / columns) * cell.height)},
                     {cams[i].width, cams[i].height}};
    draws.push_back(draw);
  }
  const auto check = [&](const Pixels& px) {
    ASSERT_FALSE(px.bytes.empty());
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t x0 = (i % columns) * cell.width;
      const uint32_t y0 = (i / columns) * cell.height;
      for (uint32_t y = 0; y < cell.height; ++y) {
        for (uint32_t x = 0; x < cell.width; ++x) {
          const bool inside = x < cams[i].width && y < cams[i].height;
          const bool hole = (x + y) % (i + 2) == 0;
          if (inside && !hole) {
            EXPECT_EQ(px.rgb(x0 + x, y0 + y), rgb_of(draws[i].tint))
                << "camera " << i << " (" << x << ", " << y << ")";
          } else {
            EXPECT_FALSE(px.lit(x0 + x, y0 + y))
                << "camera " << i << " (" << x << ", " << y << ")";
          }
        }
      }
    }
  };

  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  check(render(*pipeline, draws, extent));
  check(render_each(*pipeline, draws, extent));
  EXPECT_EQ(pipeline->block_count(), 1u) << "a frame's submits share a block";

  auto pairs = make_pipeline(kUnorm, 2);
  ASSERT_NE(pairs, nullptr);
  check(render(*pairs, draws, extent));
  EXPECT_EQ(pairs->block_count(), (n + 1) / 2) << "one submit spans blocks";
  check(render_each(*pairs, draws, extent));
  EXPECT_EQ(pairs->block_count(), (n + 1) / 2)
      << "submits fill the frame's block before taking another";
}

INSTANTIATE_TEST_SUITE_P(Cameras, DepthCloudCamerasTest,
                         ::testing::Values(1u, 3u, 5u),
                         [](const ::testing::TestParamInfo<uint32_t>& info) {
                           return "N" + std::to_string(info.param);
                         });

// A depth camera with a 3840 x 2160 colour camera: each point takes its own
// colour pixel at full resolution.
TEST_F(DepthCloudTest, AColorCameraAt4K) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const PinholeCamera depth_cam{40.0f, 40.0f, 31.6f, 17.8f, 64, 36};
  // 60 colour pixels a depth pixel: depth pixel (u, v) at colour coordinate
  // (60 u + 29.75, 60 v + 29.75), so colour pixel (60 u + 30, 60 v + 30).
  PinholeCamera color_cam;
  color_cam.fx = depth_cam.fx * 60.0f;
  color_cam.fy = depth_cam.fy * 60.0f;
  color_cam.cx = depth_cam.cx * 60.0f + 29.75f;
  color_cam.cy = depth_cam.cy * 60.0f + 29.75f;
  color_cam.width = 3840;
  color_cam.height = 2160;

  const vkc::Buffer depth = upload(
      std::vector<float>(size_t{depth_cam.width} * depth_cam.height, 1.25f));
  const std::vector<uint32_t> words = coded_colors(color_cam, 1);
  const vkc::Buffer color = upload(words);
  const glm::mat4 to_world =
      pose(90.0f, {0.0f, 0.0f, 1.0f}, {1.0f, 0.5f, 0.0f});
  DepthCloudDraw draw = self_view(depth, depth_cam, to_world);
  draw.mode = CloudColor::True;
  draw.color = {&color, 0, color_cam, to_world, true};
  const Pixels px =
      render(*pipeline, {draw}, {depth_cam.width, depth_cam.height});
  ASSERT_FALSE(px.bytes.empty());
  for (uint32_t v = 0; v < depth_cam.height; ++v) {
    for (uint32_t u = 0; u < depth_cam.width; ++u) {
      const size_t index = size_t{60 * v + 30} * color_cam.width + 60 * u + 30;
      ASSERT_EQ(px.rgb(u, v), rgb_of(words[index]))
          << "(" << u << ", " << v << ")";
    }
  }
}

// --- Refusals
// ------------------------------------------------------------------

// Test 9: submit checks every draw before recording any: a malformed draw
// before or after a good one returns InvalidArgument naming it, and the
// target stays clear.
TEST_F(DepthCloudTest, RefusalsRecordNothing) {
  auto pipeline = make_pipeline(kUnorm, 4);
  ASSERT_NE(pipeline, nullptr);
  const VkPhysicalDeviceLimits& limits = device().caps().limits();
  const glm::mat4 to_world = depth_pose();
  const std::vector<float> depths = plane_depths(kDepthCam);
  const vkc::Buffer depth = upload(depths);
  const PinholeCamera color_cam = shifted(kDepthCam, 0.25f, 0.25f);
  const vkc::Buffer color = upload(coded_colors(color_cam));
  const vkc::Buffer vertex_only =
      upload(depths.data(), depths.size() * sizeof(float),
             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  const vkc::Buffer short_buffer =
      upload(depths.data(), depths.size() * sizeof(float) - 4);
  const vkc::Buffer empty;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();

  DepthCloudDraw good = self_view(depth, kDepthCam, to_world);
  good.mode = CloudColor::TintedLuma;
  good.color = {&color, 0, color_cam, to_world, true};
  struct Case {
    const char* name;
    const char* message;
    DepthCloudDraw draw;
  };
  std::vector<Case> cases;
  const auto add = [&](const char* name, const char* message, auto&& change) {
    DepthCloudDraw draw = good;
    change(draw);
    cases.push_back({name, message, draw});
  };
  add("null depth", "has no depth buffer",
      [](DepthCloudDraw& d) { d.depth.buffer = nullptr; });
  add("empty depth", "has no depth buffer",
      [&](DepthCloudDraw& d) { d.depth.buffer = &empty; });
  add("depth without storage usage", "depth buffer without STORAGE_BUFFER",
      [&](DepthCloudDraw& d) { d.depth.buffer = &vertex_only; });
  add("empty colour", "has an empty colour buffer",
      [&](DepthCloudDraw& d) { d.color.buffer = &empty; });
  add("colour without storage usage", "colour buffer without STORAGE_BUFFER",
      [&](DepthCloudDraw& d) { d.color.buffer = &vertex_only; });
  add("depth too small", "depth image beyond its buffer",
      [&](DepthCloudDraw& d) { d.depth.buffer = &short_buffer; });
  add("colour too small", "colour image beyond its buffer",
      [&](DepthCloudDraw& d) { d.color.buffer = &short_buffer; });
  add("depth offset past the end", "depth image beyond its buffer",
      [&](DepthCloudDraw& d) {
        d.depth.offset = limits.minStorageBufferOffsetAlignment;
      });
  add("misaligned depth offset", "depth offset that is not a multiple",
      [](DepthCloudDraw& d) { d.depth.offset = 1; });
  add("misaligned colour offset", "colour offset that is not a multiple",
      [](DepthCloudDraw& d) { d.color.offset = 1; });
  add("depth range above maxStorageBufferRange",
      "depth image larger than maxStorageBufferRange", [&](DepthCloudDraw& d) {
        d.depth.camera.width = 65536;
        d.depth.camera.height = limits.maxStorageBufferRange / 4 / 65536 + 1;
      });
  add("colour range above maxStorageBufferRange",
      "colour image larger than maxStorageBufferRange", [&](DepthCloudDraw& d) {
        d.color.camera.width = 65536;
        d.color.camera.height = limits.maxStorageBufferRange / 4 / 65536 + 1;
      });
  add("zero depth width", "depth camera of zero size",
      [](DepthCloudDraw& d) { d.depth.camera.width = 0; });
  add("zero colour height", "colour camera of zero size",
      [](DepthCloudDraw& d) { d.color.camera.height = 0; });
  add("zero fx", "depth camera whose focal lengths",
      [](DepthCloudDraw& d) { d.depth.camera.fx = 0.0f; });
  add("negative fy", "depth camera whose focal lengths",
      [](DepthCloudDraw& d) { d.depth.camera.fy = -13.0f; });
  add("NaN fx", "depth camera whose focal lengths",
      [&](DepthCloudDraw& d) { d.depth.camera.fx = nan; });
  add("infinite cx", "depth camera whose focal lengths",
      [&](DepthCloudDraw& d) { d.depth.camera.cx = inf; });
  add("infinite colour fy", "colour camera whose focal lengths",
      [&](DepthCloudDraw& d) { d.color.camera.fy = inf; });
  add("zero stride", "zero stride", [](DepthCloudDraw& d) { d.stride = 0; });
  add("min above max", "min_depth above max_depth", [](DepthCloudDraw& d) {
    d.depth.min_depth = 2.0f;
    d.depth.max_depth = 1.0f;
  });
  add("negative min", "depth bound that is negative or not finite",
      [](DepthCloudDraw& d) { d.depth.min_depth = -0.5f; });
  add("NaN max", "depth bound that is negative or not finite",
      [&](DepthCloudDraw& d) { d.depth.max_depth = nan; });
  add("infinite max", "depth bound that is negative or not finite",
      [&](DepthCloudDraw& d) { d.depth.max_depth = inf; });
  add("unknown mode", "unknown colour mode",
      [](DepthCloudDraw& d) { d.mode = static_cast<CloudColor>(7); });
  add("zero point size", "point size",
      [](DepthCloudDraw& d) { d.point_size = 0.0f; });
  add("negative point size", "point size",
      [](DepthCloudDraw& d) { d.point_size = -1.0f; });
  add("NaN point size", "point size",
      [&](DepthCloudDraw& d) { d.point_size = nan; });
  add("infinite point size", "point size",
      [&](DepthCloudDraw& d) { d.point_size = inf; });
  add("NaN view", "view or depth pose that is not finite",
      [&](DepthCloudDraw& d) { d.view_proj[2][1] = nan; });
  add("infinite depth pose", "view or depth pose that is not finite",
      [&](DepthCloudDraw& d) { d.depth.to_world[3][0] = inf; });
  add("singular colour pose", "colour pose that cannot be inverted",
      [](DepthCloudDraw& d) { d.color.to_world = glm::mat4(0.0f); });
  add("NaN colour pose", "colour pose that cannot be inverted",
      [&](DepthCloudDraw& d) { d.color.to_world[0][0] = nan; });
  add("viewport wider than the device's", "maxViewportDimensions",
      [&](DepthCloudDraw& d) {
        d.viewport.extent.width = limits.maxViewportDimensions[0] + 1;
      });
  add("viewport taller than the device's", "maxViewportDimensions",
      [&](DepthCloudDraw& d) {
        d.viewport.extent.height = limits.maxViewportDimensions[1] + 1;
      });
  add("viewport before the bounds range", "viewportBoundsRange",
      [](DepthCloudDraw& d) {
        d.viewport.offset.x = std::numeric_limits<int32_t>::min();
      });
  add("viewport past the bounds range", "viewportBoundsRange",
      [](DepthCloudDraw& d) {
        d.viewport.offset.y = std::numeric_limits<int32_t>::max() - 8;
      });

  const auto attempt = [&](const DepthCloudDraw* draws, uint32_t count,
                           uint64_t number) {
    return render_status(
        kDepthExtent, kUnorm, number, [&](VkCommandBuffer cmd) {
          return pipeline->submit(cmd, {kDepthExtent, number, draws, count});
        });
  };
  for (const Case& c : cases) {
    for (const bool bad_first : {false, true}) {
      const DepthCloudDraw draws[2] = {bad_first ? c.draw : good,
                                       bad_first ? good : c.draw};
      auto [status, px] = attempt(draws, 2, next_frame());
      EXPECT_EQ(status.domain(), vkc::Status::Code::InvalidArgument)
          << c.name << ", bad first: " << bad_first;
      EXPECT_NE(status.message().find(bad_first ? "draw 0 " : "draw 1 "),
                std::string::npos)
          << c.name << ": " << status.message();
      EXPECT_NE(status.message().find(c.message), std::string::npos)
          << c.name << ": " << status.message();
      EXPECT_EQ(px.lit_count(), 0)
          << c.name << ", bad first: " << bad_first << ": nothing recorded";
    }
  }

  // Frame-level refusals.
  auto [null_list, null_px] = attempt(nullptr, 1, next_frame());
  EXPECT_EQ(null_list.domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_EQ(null_px.lit_count(), 0);
  // A bad draw beyond the first block's max_draws is checked before any draw
  // is recorded too.
  std::vector<DepthCloudDraw> five(5, good);
  five[4].stride = 0;
  auto [late, late_px] = attempt(five.data(), 5, next_frame());
  EXPECT_EQ(late.domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_NE(late.message().find("draw 4 has a zero stride"), std::string::npos)
      << late.message();
  EXPECT_EQ(late_px.lit_count(), 0);

  // Not errors: no colour buffer with a colour mode (every point takes the
  // tint), a viewport outside the target or empty (skipped), and no draws.
  DepthCloudDraw no_color = good;
  no_color.color.buffer = nullptr;
  no_color.color.camera = {};
  no_color.color.to_world = glm::mat4(0.0f);  // unused without a buffer
  DepthCloudDraw outside = good;
  outside.viewport = {{16, 0}, {16, 12}};
  DepthCloudDraw empty_viewport = good;
  empty_viewport.viewport.extent.height = 0;
  const std::vector<DepthCloudDraw> fine{no_color, outside, empty_viewport};
  const Pixels px = render(*pipeline, fine, kDepthExtent);
  EXPECT_EQ(px.lit_count(), int{kDepthCam.width * kDepthCam.height});
  EXPECT_EQ(px.rgb(3, 3), rgb_of(kTint));
  auto [nothing, nothing_px] = attempt(nullptr, 0, next_frame());
  EXPECT_TRUE(nothing.ok()) << nothing.message();
  EXPECT_EQ(nothing_px.lit_count(), 0);
}

TEST_F(DepthCloudTest, RejectsImagesWhoseByteCountWouldOverflow) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const vkc::Buffer depth = upload(std::vector<float>{1.0f});
  const vkc::Buffer color = upload(std::vector<uint32_t>{kTint});
  const PinholeCamera cam{1.0f, 1.0f, 0.0f, 0.0f, 1, 1};
  vg::OffscreenTargetDesc desc;
  desc.extent = {1, 1};
  desc.color_format = kUnorm;
  desc.depth_format = kDepthFormat;
  auto target = vg::OffscreenTarget::create(allocator(), desc);
  ASSERT_TRUE(target.ok()) << target.status().message();
  auto pool =
      vkc::CommandPool::create(device().handle(), device().queue_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  for (const bool oversized_color : {false, true}) {
    SCOPED_TRACE(oversized_color ? "colour" : "depth");
    DepthCloudDraw draw = self_view(depth, cam, glm::mat4(1.0f));
    draw.color = {&color, 0, cam, glm::mat4(1.0f), false};
    draw.mode = CloudColor::True;
    auto& oversized = oversized_color ? draw.color.camera : draw.depth.camera;
    oversized.width = 0x80000000u;
    oversized.height = 0x80000000u;
    auto cmd = pool.value().allocate_primary();
    ASSERT_TRUE(cmd.ok()) << cmd.status().message();
    ASSERT_TRUE(cmd.value().begin().ok());
    target.value().prepare(cmd.value().handle());
    const vg::RenderTarget rt = target.value().target();
    rt.begin(cmd.value().handle(), {});
    const vkc::Status status = pipeline->submit(
        cmd.value().handle(), {{1, 1}, next_frame(), &draw, 1});
    EXPECT_EQ(status.domain(), vkc::Status::Code::InvalidArgument);
    EXPECT_NE(status.message().find("image larger than maxStorageBufferRange"),
              std::string::npos)
        << status.message();
    rt.end(cmd.value().handle());
    ASSERT_TRUE(cmd.value().end().ok());
    // Do not execute an invalid descriptor if this refusal regresses.
  }
}

// Test 7: frame numbers start at 1 and never go back; a refused number
// records nothing.
TEST_F(DepthCloudTest, RefusesZeroAndDecreasingFrameNumbers) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const vkc::Buffer depth = upload(plane_depths(kDepthCam));
  const DepthCloudDraw draw = self_view(depth, kDepthCam, depth_pose());
  const auto attempt = [&](uint64_t number) {
    return render_status(
        kDepthExtent, kUnorm, number, [&](VkCommandBuffer cmd) {
          return pipeline->submit(cmd, {kDepthExtent, number, &draw, 1});
        });
  };
  auto [zero, zero_px] = attempt(0);
  EXPECT_EQ(zero.domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_NE(zero.message().find("frame numbers start at 1"), std::string::npos)
      << zero.message();
  EXPECT_EQ(zero_px.lit_count(), 0);
  auto [five, five_px] = attempt(5);
  EXPECT_TRUE(five.ok()) << five.message();
  EXPECT_GT(five_px.lit_count(), 0);
  auto [four, four_px] = attempt(4);
  EXPECT_EQ(four.domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_NE(four.message().find("frame 4 is below frame 5"), std::string::npos)
      << four.message();
  EXPECT_EQ(four_px.lit_count(), 0);
  auto [again, again_px] = attempt(6);
  EXPECT_TRUE(again.ok()) << again.message();
  EXPECT_GT(again_px.lit_count(), 0);
}

// Test 7: destroying the pipeline after a frame that never reached the queue
// returns: it waits for the queues, not for a number nothing will set.
TEST_F(DepthCloudTest, TeardownAfterAFrameThatNeverRanReturns) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const vkc::Buffer depth = upload(plane_depths(kDepthCam));
  const DepthCloudDraw draw = self_view(depth, kDepthCam, depth_pose());
  vg::OffscreenTargetDesc desc;
  desc.extent = kDepthExtent;
  desc.color_format = kUnorm;
  desc.depth_format = kDepthFormat;
  auto target = vg::OffscreenTarget::create(allocator(), desc);
  ASSERT_TRUE(target.ok()) << target.status().message();
  auto pool =
      vkc::CommandPool::create(device().handle(), device().queue_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();
  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();
  target.value().prepare(raw);
  const vg::RenderTarget rt = target.value().target();
  rt.begin(raw, {});
  ASSERT_TRUE(pipeline->submit(raw, {kDepthExtent, 7, &draw, 1}).ok());
  rt.end(raw);
  ASSERT_TRUE(cmd.value().end().ok());
  // Frame 7 is never submitted, and the timeline stays below it.
  pipeline.reset();
  EXPECT_EQ(timeline_->value().value(), 0u);
}

// --- Frames through a timeline
// -------------------------------------------------

// Frames submitted as windowing::FrameLoop submits them: each sets its number
// on the timeline when it completes, recorded with core::note_timeline_signals,
// and a frame waits on the host for the frame two before it, so two are in
// flight. A gate the test sets from the host holds frames at the queue.
class DepthCloudFrameTest : public DepthCloudTest {
 protected:
  struct Frame {
    uint64_t number = 0;
    vkc::CommandBuffer cmd;
    vg::OffscreenTarget target;
  };

  void SetUp() override {
    DepthCloudTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto gate = vkc::TimelineSemaphore::create(device(), 0);
    ASSERT_TRUE(gate.ok()) << gate.status().message();
    gate_.emplace(std::move(gate).value());
    auto pool =
        vkc::CommandPool::create(device().handle(), device().queue_family());
    ASSERT_TRUE(pool.ok()) << pool.status().message();
    pool_.emplace(std::move(pool).value());
  }

  void TearDown() override {
    if (gate_) {
      open_gate(*gate_);
    }
    DepthCloudTest::TearDown();
  }

  // Records frame `number`: `draw` into a target of its own, read back.
  Frame* record(uint64_t number, DepthCloudPipeline& pipeline,
                const DepthCloudDraw& draw) {
    auto frame = std::make_unique<Frame>();
    frame->number = number;
    auto cmd = pool_->allocate_primary();
    EXPECT_TRUE(cmd.ok()) << cmd.status().message();
    vg::OffscreenTargetDesc desc;
    desc.extent = kDepthExtent;
    desc.color_format = kUnorm;
    desc.depth_format = kDepthFormat;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    if (!cmd.ok() || !target.ok()) {
      return nullptr;
    }
    frame->cmd = std::move(cmd).value();
    frame->target = std::move(target).value();
    EXPECT_TRUE(
        frame->cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    const VkCommandBuffer raw = frame->cmd.handle();
    frame->target.prepare(raw);
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[3] = 1.0f;
    const vg::RenderTarget rt = frame->target.target();
    rt.begin(raw, begin);
    const vkc::Status submitted =
        pipeline.submit(raw, {kDepthExtent, number, &draw, 1});
    EXPECT_TRUE(submitted.ok()) << submitted.message();
    rt.end(raw);
    frame->target.record_readback(raw);
    EXPECT_TRUE(frame->cmd.end().ok());
    frames_.push_back(std::move(frame));
    return frames_.back().get();
  }

  // Submits `frame`, setting its number on the timeline when it completes; a
  // `gated` one is held until the gate opens.
  void queue(const Frame& frame, bool gated) {
    const uint64_t open = 1;
    const VkSemaphore gate = gate_->handle();
    const VkPipelineStageFlags held = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkSemaphore timeline = timeline_->handle();
    const VkCommandBuffer cmd = frame.cmd.handle();
    VkTimelineSemaphoreSubmitInfo values{};
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = gated ? 1 : 0;
    values.pWaitSemaphoreValues = &open;
    values.signalSemaphoreValueCount = 1;
    values.pSignalSemaphoreValues = &frame.number;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &values;
    submit.waitSemaphoreCount = gated ? 1 : 0;
    submit.pWaitSemaphores = &gate;
    submit.pWaitDstStageMask = &held;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &timeline;
    ASSERT_EQ(device().queue_submit(1, &submit, VK_NULL_HANDLE), VK_SUCCESS);
    vkc::note_timeline_signals({{&*timeline_, frame.number}});
  }

  std::optional<vkc::TimelineSemaphore> gate_;
  std::optional<vkc::CommandPool> pool_;
  // Destroyed after the base TearDown has waited for the device.
  std::vector<std::unique_ptr<Frame>> frames_;
};

// Test 6: six frames, two in flight, alternate two depth buffers -- one with
// depth in its left half, one in its right -- and each frame draws its own;
// blocks of per-draw state come back as frames complete, so the pipeline
// holds no more than frames in flight plus one.
TEST_F(DepthCloudFrameTest, EachFrameDrawsItsOwnBufferAndBlocksAreReused) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  // The pipeline's teardown waits for the held frames.
  OpenGateAtExit open_at_exit(*gate_);
  const uint32_t w = kDepthCam.width;
  const uint32_t h = kDepthCam.height;
  std::vector<float> left(size_t{w} * h, 0.0f);
  std::vector<float> right(size_t{w} * h, 0.0f);
  for (uint32_t v = 0; v < h; ++v) {
    for (uint32_t u = 0; u < w; ++u) {
      (u < w / 2 ? left : right)[size_t{v} * w + u] = 1.5f;
    }
  }
  const vkc::Buffer left_buffer = upload(left);
  const vkc::Buffer right_buffer = upload(right);
  const glm::mat4 to_world = depth_pose();
  const DepthCloudDraw draws[2] = {
      self_view(left_buffer, kDepthCam, to_world),
      self_view(right_buffer, kDepthCam, to_world)};

  constexpr uint64_t kFrames = 6;
  constexpr uint64_t kInFlight = 2;
  std::vector<Frame*> recorded;
  for (uint64_t n = 1; n <= kFrames; ++n) {
    if (n > kInFlight) {
      // The first frames were held at the gate, so the second needed a block
      // of its own; let them run now.
      open_gate(*gate_);
      ASSERT_TRUE(timeline_->wait(n - kInFlight).ok());
    }
    Frame* frame = record(n, *pipeline, draws[n % 2]);
    ASSERT_NE(frame, nullptr);
    queue(*frame, n <= kInFlight);
    recorded.push_back(frame);
    EXPECT_LE(pipeline->block_count(), kInFlight + 1) << "frame " << n;
  }
  EXPECT_GE(pipeline->block_count(), 2u) << "frame 2 drew while 1 was held";
  ASSERT_TRUE(timeline_->wait(kFrames).ok());
  for (const Frame* frame : recorded) {
    const auto* bytes = static_cast<const uint8_t*>(frame->target.pixels());
    ASSERT_NE(bytes, nullptr);
    const bool left_half = frame->number % 2 == 0;
    for (uint32_t v = 0; v < h; ++v) {
      for (uint32_t u = 0; u < w; ++u) {
        const uint8_t* p = &bytes[(size_t{v} * w + u) * 4];
        const bool lit = p[0] != 0 || p[1] != 0 || p[2] != 0;
        EXPECT_EQ(lit, (u < w / 2) == left_half)
            << "frame " << frame->number << " (" << u << ", " << v << ")";
      }
    }
  }
}

// Destroying the pipeline waits for the newest frame -- held at the gate
// here -- before it frees the blocks that frame reads, and the frame draws
// what it was given.
TEST_F(DepthCloudFrameTest, DestructionWaitsForTheNewestFrame) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const vkc::Buffer depth = upload(plane_depths(kDepthCam));
  const DepthCloudDraw draw = self_view(depth, kDepthCam, depth_pose());
  OpenGateAtExit open_at_exit(*gate_);
  Frame* frame = record(1, *pipeline, draw);
  ASSERT_NE(frame, nullptr);
  queue(*frame, true);
  {
    OpenGateLater open_later(*gate_);
    pipeline.reset();
    EXPECT_TRUE(open_later.opened()) << "destruction waited for frame 1";
  }
  const vkc::Result<uint64_t> reached = timeline_->value();
  ASSERT_TRUE(reached.ok()) << reached.status().message();
  EXPECT_GE(reached.value(), 1u);
  const auto* bytes = static_cast<const uint8_t*>(frame->target.pixels());
  ASSERT_NE(bytes, nullptr);
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      const uint8_t* p = &bytes[(size_t{v} * kDepthCam.width + u) * 4];
      EXPECT_EQ((std::array<uint8_t, 3>{p[0], p[1], p[2]}), rgb_of(kTint))
          << "(" << u << ", " << v << ")";
    }
  }
}

// --- largePoints
// ---------------------------------------------------------------

class DepthCloudLargePointsTest : public DepthCloudTest {
 protected:
  vkc::DeviceRequirements requirements() const override {
    return requirements_;
  }

  void SetUp() override {
    // Keep the require-device policy for the renderer's baseline, but skip
    // an unavailable optional feature before requesting the shared device.
    vkc::test::VulkanTest::SetUp();
    if (base_setup_incomplete()) return;
    requirements_.features.largePoints = VK_TRUE;
    const auto wide = instance().select_physical_device(requirements_);
    if (!wide.ok() &&
        wide.status().domain() == vkc::Status::Code::Unsupported) {
      GTEST_SKIP() << "largePoints is unavailable: " << wide.status().message();
    }
    ASSERT_TRUE(wide.ok()) << wide.status().message();
    DepthCloudTest::SetUp();
  }

 private:
  vkc::DeviceRequirements requirements_ = vg::device_requirements();
};

// Test 4: a point of size 3 covers 3 x 3 pixels around its own; a size below
// a pixel is drawn at one. The view puts the point 0.4 px right of its
// pixel's centre, where a quarter-pixel point would cover no centre.
TEST_F(DepthCloudLargePointsTest, SizeThreeCoversThreeByThreeAndClamps) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  EXPECT_EQ(pipeline->max_point_size(),
            device().caps().limits().pointSizeRange[1]);
  if (pipeline->max_point_size() < 3.0f) {
    GTEST_SKIP() << "the device's points stop at " << pipeline->max_point_size()
                 << " px";
  }
  std::vector<float> depths(size_t{kDepthCam.width} * kDepthCam.height, 0.0f);
  depths[5 * kDepthCam.width + 7] = 1.5f;
  const vkc::Buffer depth = upload(depths);
  DepthCloudDraw draw = self_view(depth, kDepthCam, depth_pose());
  draw.view_proj =
      pinhole_view_proj(shifted(kDepthCam, 0.4f, 0.0f), draw.depth.to_world);
  draw.point_size = 3.0f;
  const Pixels big = render(*pipeline, {draw}, kDepthExtent);
  ASSERT_FALSE(big.bytes.empty());
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      const bool inside = u >= 6 && u <= 8 && v >= 4 && v <= 6;
      EXPECT_EQ(big.lit(u, v), inside) << "(" << u << ", " << v << ")";
    }
  }
  draw.point_size = 0.25f;
  const Pixels small = render(*pipeline, {draw}, kDepthExtent);
  EXPECT_EQ(small.lit_count(), 1);
  EXPECT_TRUE(small.lit(7, 5));
}

// --- Another queue family
// ------------------------------------------------------

// Test 8: depth and colour the compute library's queue family wrote, in
// CONCURRENT buffers, are drawn on the renderer's family in place, exactly,
// with a producer-signalled semaphore wait at VERTEX_SHADER. No host fence
// wait retires synchronization validation's producer history before the draw.
class DepthCloudCrossFamilyTest : public vkc::test::SharedDeviceTest,
                                  protected CloudHelpers {
 protected:
  vkc::DeviceRequirements graphics_requirements() const override {
    return vg::device_requirements();
  }
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::ShaderAccesses;
  }
  vkc::Device& cloud_device() override { return two_family().graphics(); }
  vkc::Allocator& cloud_allocator() override {
    return two_family().graphics_allocator();
  }

  void SetUp() override {
    vkc::test::SharedDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    make_timeline();
  }

  // A storage buffer allocated by the compute library, CONCURRENT across
  // both families. The test submits the writes asynchronously.
  vkc::Buffer compute_buffer(VkDeviceSize bytes) {
    vkc::test::TwoFamilyDevice& two = two_family();
    vkc::BufferDesc desc;
    desc.size = bytes;
    desc.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.queue_families = two.families().data();
    desc.queue_family_count = vkc::test::TwoFamilyDevice::kFamilyCount;
    auto buffer = two.compute_allocator().create_buffer(desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    if (!buffer.ok()) {
      return vkc::Buffer{};
    }
    EXPECT_EQ(buffer.value().sharing_mode(), VK_SHARING_MODE_CONCURRENT);
    return std::move(buffer).value();
  }
};

TEST_F(DepthCloudCrossFamilyTest,
       SemaphoreMakesComputeWritesVisibleToVertices) {
  ASSERT_TRUE(two_family().crosses_families());
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const glm::mat4 to_world = depth_pose();
  std::vector<float> depths = plane_depths(kDepthCam);
  depths[3 * kDepthCam.width + 4] = 0.0f;  // a hole
  const PinholeCamera color_cam = shifted(kDepthCam, 0.25f, 0.25f);
  const std::vector<uint32_t> words = coded_colors(color_cam, 2);
  const vkc::Buffer depth = compute_buffer(depths.size() * sizeof(float));
  const vkc::Buffer color = compute_buffer(words.size() * sizeof(uint32_t));
  ASSERT_TRUE(depth.valid());
  ASSERT_TRUE(color.valid());
  auto ready = vkc::TimelineSemaphore::create(two_family().compute(), 0);
  ASSERT_TRUE(ready.ok()) << ready.status().message();
  auto producer = two_family().compute().submit_pending(
      [&](VkCommandBuffer cmd) {
        vkCmdUpdateBuffer(cmd, depth.handle(), 0, depth.size(), depths.data());
        vkCmdUpdateBuffer(cmd, color.handle(), 0, color.size(), words.data());
      },
      {}, {{&ready.value(), 1}});
  ASSERT_TRUE(producer.ok()) << producer.status().message();

  DepthCloudDraw draw = self_view(depth, kDepthCam, to_world);
  draw.mode = CloudColor::True;
  draw.color = {&color, 0, color_cam, to_world, true};
  const Pixels px =
      render(*pipeline, {draw}, kDepthExtent, kUnorm, {&ready.value(), 1});
  ASSERT_TRUE(producer.value().wait().ok());
  ASSERT_FALSE(px.bytes.empty());
  for (uint32_t v = 0; v < kDepthCam.height; ++v) {
    for (uint32_t u = 0; u < kDepthCam.width; ++u) {
      if (u == 4 && v == 3) {
        EXPECT_FALSE(px.lit(u, v)) << "the hole";
      } else {
        EXPECT_EQ(px.rgb(u, v), rgb_of(words[v * kDepthCam.width + u]))
            << "(" << u << ", " << v << ")";
      }
    }
  }
}

}  // namespace

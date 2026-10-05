// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImagePipeline and ImageTexture: creation and update validation, move
// semantics, and end-to-end offscreen draws read back -- texel-exact
// magnification, mip chains that average in linear light, each mapping (color,
// grey, ramp, NV12 with its siting), buffer and image planes, clipping, and an
// update / draw / update / draw sequence in one command buffer. Runs under the
// validation layer with synchronization validation. Skips when the runner
// exposes no Vulkan device.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
#include "volumetric_kit/gfx/core/mip_chain.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/image_pipeline.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

constexpr VkFormat kUnorm = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kSrgb = VK_FORMAT_R8G8B8A8_SRGB;

vg::RenderTargetLayout color_layout(VkFormat format) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = format;
  layout.color_count = 1;
  return layout;
}

// A readback: RGBA bytes, row-major, of a `width`-wide target.
struct Pixels {
  std::vector<uint8_t> bytes;
  uint32_t width = 0;

  const uint8_t* at(uint32_t x, uint32_t y) const {
    return &bytes[(static_cast<size_t>(y) * width + x) * 4];
  }
};

// The expected byte for a channel, within `tolerance`.
void expect_rgb(const uint8_t* px, int r, int g, int b, int tolerance,
                const char* what) {
  EXPECT_NEAR(px[0], r, tolerance) << what << " (red)";
  EXPECT_NEAR(px[1], g, tolerance) << what << " (green)";
  EXPECT_NEAR(px[2], b, tolerance) << what << " (blue)";
}

// --- No device needed --------------------------------------------------------

TEST(ImagePipelineTest, DefaultConstructedIsEmpty) {
  const pipelines::ImagePipeline pipeline;
  EXPECT_FALSE(pipeline.valid());
  const pipelines::ImageTexture texture;
  EXPECT_FALSE(texture.valid());
  EXPECT_FALSE(texture.has_picture());
  EXPECT_EQ(texture.extent().width, 0u);
  EXPECT_EQ(texture.mip_levels(), 0u);
}

TEST(MipChainTest, CountsLevelsDownToOneByOne) {
  EXPECT_EQ(vg::mip_level_count({1, 1}), 1u);
  EXPECT_EQ(vg::mip_level_count({2, 1}), 2u);
  EXPECT_EQ(vg::mip_level_count({300, 200}), 9u);
  EXPECT_EQ(vg::mip_level_count({3840, 2160}), 12u);
  const VkExtent2D e = vg::mip_level_extent({3840, 2160}, 5);
  EXPECT_EQ(e.width, 120u);
  EXPECT_EQ(e.height, 67u);  // 2160 >> 5, floored
  const VkExtent2D last = vg::mip_level_extent({3840, 2160}, 11);
  EXPECT_EQ(last.width, 1u);
  EXPECT_EQ(last.height, 1u);
}

// --- On a device -------------------------------------------------------------

class ImagePipelineDeviceTest : public VulkanDeviceTest {
 protected:
  bool wants_validation() const override { return true; }
  // The update's barriers -- plane copies, the convert pass, the chain, the
  // draws -- are what these tests exercise, so missing ones must be reported.
  bool wants_sync_validation() const override { return true; }

  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

  pipelines::ImagePipeline make_pipeline(VkFormat target = kUnorm) {
    auto pipeline =
        pipelines::ImagePipeline::create(*device_, color_layout(target));
    EXPECT_TRUE(pipeline.ok()) << pipeline.status().message();
    return pipeline.ok() ? std::move(pipeline).value()
                         : pipelines::ImagePipeline{};
  }

  pipelines::ImageTexture make_texture(const pipelines::ImagePipeline& pipeline,
                                       VkExtent2D extent,
                                       pipelines::ImageMapping mapping,
                                       VkFormat format) {
    pipelines::ImageTextureDesc desc;
    desc.extent = extent;
    desc.mapping = mapping;
    desc.format = format;
    auto texture = pipelines::ImageTexture::create(pipeline, *allocator_, desc);
    EXPECT_TRUE(texture.ok()) << texture.status().message();
    return texture.ok() ? std::move(texture).value()
                        : pipelines::ImageTexture{};
  }

  // A device-only buffer holding `bytes`, readable by a transfer.
  vkc::Buffer source_buffer(const std::vector<uint8_t>& bytes) {
    vg::BufferUploadDesc desc;
    desc.data = bytes.data();
    desc.size = bytes.size();
    desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto buffer = vg::upload_buffer(*device_, *allocator_, desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
  }

  // A device image holding `bytes`, left SHADER_READ_ONLY_OPTIMAL by the
  // upload; ready_for_copy moves it to TRANSFER_SRC_OPTIMAL in a command
  // buffer, as a producer would leave a picture for a copy to read.
  vkc::Image source_image(VkFormat format, VkExtent2D extent,
                          const std::vector<uint8_t>& bytes) {
    vg::ImageUploadDesc desc;
    desc.extent = extent;
    desc.format = format;
    desc.pixels = bytes.data();
    desc.size = bytes.size();
    auto image = vg::upload_texture(*device_, *allocator_, desc);
    EXPECT_TRUE(image.ok()) << image.status().message();
    if (!image.ok()) {
      return vkc::Image{};
    }
    image.value().set_layout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return std::move(image).value();
  }

  static void ready_for_copy(VkCommandBuffer cmd, vkc::Image& image) {
    vg::ImageBarrierDesc b;
    b.image = image.handle();
    b.src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    b.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    b.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
    b.old_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vg::cmd_image_barrier(cmd, b);
    image.set_layout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  }

  // One step of a recorded command buffer: work before the render (updates),
  // then the draws into a fresh `extent` target of `format`, read back.
  struct Pass {
    std::function<void(VkCommandBuffer)> before;
    std::vector<pipelines::ImageDraw> draws;
  };

  // Records every pass into one command buffer -- each into its own target --
  // submits it, and returns each target's pixels.
  std::vector<Pixels> render_passes(const pipelines::ImagePipeline& pipeline,
                                    const std::vector<Pass>& passes,
                                    VkExtent2D extent,
                                    VkFormat format = kUnorm) {
    std::vector<vg::OffscreenTarget> targets;
    for (size_t i = 0; i < passes.size(); ++i) {
      vg::OffscreenTargetDesc desc;
      desc.extent = extent;
      desc.color_format = format;
      auto target = vg::OffscreenTarget::create(*allocator_, desc);
      EXPECT_TRUE(target.ok()) << target.status().message();
      if (!target.ok()) {
        return {};
      }
      targets.push_back(std::move(target).value());
    }

    auto pool = vkc::CommandPool::create(device(), device_->queue_family());
    EXPECT_TRUE(pool.ok()) << pool.status().message();
    if (!pool.ok()) {
      return {};
    }
    auto cmd = pool.value().allocate_primary();
    EXPECT_TRUE(cmd.ok()) << cmd.status().message();
    if (!cmd.ok()) {
      return {};
    }
    EXPECT_TRUE(
        cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    const VkCommandBuffer raw = cmd.value().handle();

    for (size_t i = 0; i < passes.size(); ++i) {
      if (passes[i].before) {
        passes[i].before(raw);
      }
      targets[i].prepare(raw);
      vg::RenderTargetBeginInfo begin;
      begin.clear_color.float32[2] = 1.0f;  // opaque blue: "nothing drawn"
      begin.clear_color.float32[3] = 1.0f;
      const vg::RenderTarget rt = targets[i].target();
      rt.begin(raw, begin);
      pipeline.submit(raw, {extent, passes[i].draws.data(),
                            static_cast<uint32_t>(passes[i].draws.size())});
      rt.end(raw);
      targets[i].record_readback(raw);
    }
    EXPECT_TRUE(cmd.value().end().ok());
    submit_and_wait(raw);
    if (HasFatalFailure()) {
      return {};
    }

    std::vector<Pixels> out;
    for (const vg::OffscreenTarget& target : targets) {
      const auto* px = static_cast<const uint8_t*>(target.pixels());
      out.push_back(
          {std::vector<uint8_t>(
               px, px + static_cast<size_t>(extent.width) * extent.height * 4),
           extent.width});
    }
    return out;
  }

  // The single-pass case: update `texture` from `update`, then draw `draws`.
  Pixels update_and_draw(
      const pipelines::ImagePipeline& pipeline,
      pipelines::ImageTexture& texture, const pipelines::ImageUpdate& update,
      const std::vector<pipelines::ImageDraw>& draws, VkExtent2D extent,
      VkFormat format = kUnorm,
      const std::function<void(VkCommandBuffer)>& prepare = nullptr) {
    Pass pass;
    pass.before = [&](VkCommandBuffer cmd) {
      if (prepare) {
        prepare(cmd);
      }
      const vkc::Status s = pipeline.record_update(cmd, texture, update);
      EXPECT_TRUE(s.ok()) << s.message();
    };
    pass.draws = draws;
    std::vector<Pixels> out = render_passes(pipeline, {pass}, extent, format);
    return out.empty() ? Pixels{} : std::move(out.front());
  }

  static pipelines::ImageDraw draw_of(const pipelines::ImageTexture& texture,
                                      VkExtent2D target, float scale,
                                      glm::vec2 origin = {0.0f, 0.0f}) {
    pipelines::ImageDraw draw;
    draw.texture = &texture;
    draw.viewport = {{0, 0}, target};
    draw.origin = origin;
    draw.scale = scale;
    return draw;
  }

  std::optional<vkc::Allocator> allocator_;
};

TEST_F(ImagePipelineDeviceTest, CreateRejectsALayoutWithoutColor) {
  auto pipeline =
      pipelines::ImagePipeline::create(*device_, vg::RenderTargetLayout{});
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(ImagePipelineDeviceTest, CreateRejectsAMultisampledLayout) {
  vg::RenderTargetLayout layout = color_layout(kUnorm);
  layout.samples = VK_SAMPLE_COUNT_4_BIT;
  auto pipeline = pipelines::ImagePipeline::create(*device_, layout);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(ImagePipelineDeviceTest, TextureHasAFullSrgbChainAndNoPictureYet) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  ASSERT_TRUE(pipeline.valid());
  const pipelines::ImageTexture texture = make_texture(
      pipeline, {300, 200}, pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
  ASSERT_TRUE(texture.valid());
  EXPECT_EQ(texture.mip_levels(), 9u);
  EXPECT_EQ(texture.display().format(), VK_FORMAT_R8G8B8A8_SRGB);
  EXPECT_EQ(texture.extent().width, 300u);
  EXPECT_FALSE(texture.has_picture());
}

TEST_F(ImagePipelineDeviceTest, TextureCreateRejectsBadDescriptions) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  ASSERT_TRUE(pipeline.valid());
  auto create = [&](VkExtent2D extent, pipelines::ImageMapping mapping,
                    VkFormat format) {
    pipelines::ImageTextureDesc desc;
    desc.extent = extent;
    desc.mapping = mapping;
    desc.format = format;
    return pipelines::ImageTexture::create(pipeline, *allocator_, desc);
  };
  using M = pipelines::ImageMapping;
  const auto zero = create({0, 8}, M::Color, kUnorm);
  EXPECT_EQ(zero.status().domain(), vkc::Status::Code::InvalidArgument);
  const auto huge = create({1u << 30, 8}, M::Color, kUnorm);
  EXPECT_EQ(huge.status().domain(), vkc::Status::Code::InvalidArgument);
  const auto grey_rgba = create({8, 8}, M::Grey, kUnorm);
  EXPECT_EQ(grey_rgba.status().domain(), vkc::Status::Code::InvalidArgument);
  const auto color_r8 = create({8, 8}, M::Color, VK_FORMAT_R8_UNORM);
  EXPECT_EQ(color_r8.status().domain(), vkc::Status::Code::InvalidArgument);
  const auto ramp_rg = create({8, 8}, M::Ramp, VK_FORMAT_R8G8_UNORM);
  EXPECT_EQ(ramp_rg.status().domain(), vkc::Status::Code::InvalidArgument);

  pipelines::ImageTextureDesc desc;
  desc.extent = {8, 8};
  const pipelines::ImagePipeline empty;
  const auto no_pipeline =
      pipelines::ImageTexture::create(empty, *allocator_, desc);
  EXPECT_EQ(no_pipeline.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(ImagePipelineDeviceTest, UpdateRejectsBadPlanesAndParameters) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  ASSERT_TRUE(pipeline.valid());
  pipelines::ImageTexture grey = make_texture(
      pipeline, {4, 4}, pipelines::ImageMapping::Grey, VK_FORMAT_R16_UNORM);
  pipelines::ImageTexture srgb =
      make_texture(pipeline, {4, 4}, pipelines::ImageMapping::Color, kSrgb);
  pipelines::ImageTexture ramp = make_texture(
      pipeline, {4, 4}, pipelines::ImageMapping::Ramp, VK_FORMAT_R8_UNORM);
  pipelines::ImageTexture nv12 = make_texture(
      pipeline, {4, 4}, pipelines::ImageMapping::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(grey.valid() && srgb.valid() && ramp.valid() && nv12.valid());

  const vkc::Buffer r16 = source_buffer(std::vector<uint8_t>(4 * 4 * 2, 0));
  const vkc::Buffer rgba = source_buffer(std::vector<uint8_t>(4 * 4 * 4, 0));
  const vkc::Buffer r8 = source_buffer(std::vector<uint8_t>(4 * 4, 0));
  const vkc::Image wrong_format =
      source_image(VK_FORMAT_R8_UNORM, {4, 4}, std::vector<uint8_t>(16, 0));

  auto pool = vkc::CommandPool::create(device(), device_->queue_family());
  ASSERT_TRUE(pool.ok());
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok());
  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();

  auto rejected = [&](pipelines::ImageTexture& texture,
                      const pipelines::ImageUpdate& update, const char* what) {
    const vkc::Status s = pipeline.record_update(raw, texture, update);
    EXPECT_EQ(s.domain(), vkc::Status::Code::InvalidArgument) << what;
    EXPECT_FALSE(texture.has_picture()) << what;
  };

  pipelines::ImageUpdate none;
  rejected(grey, none, "no plane");

  pipelines::ImageUpdate both;
  both.planes[0].buffer = &r16;
  both.planes[0].image = &wrong_format;
  rejected(grey, both, "an image and a buffer");

  pipelines::ImageUpdate extra;
  extra.planes[0].buffer = &r16;
  extra.planes[1].buffer = &r16;
  rejected(grey, extra, "plane 1 on a grey source");

  pipelines::ImageUpdate small;
  small.planes[0].buffer = &r8;  // 16 bytes; R16 needs 32
  rejected(grey, small, "a buffer too small");

  pipelines::ImageUpdate misaligned;
  misaligned.planes[0].buffer = &r16;
  misaligned.planes[0].offset = 1;
  rejected(grey, misaligned, "an offset off the texel size");

  pipelines::ImageUpdate short_rows;
  short_rows.planes[0].buffer = &r16;
  short_rows.planes[0].row_length = 3;
  rejected(grey, short_rows, "rows shorter than the picture");

  pipelines::ImageUpdate wrong_image;
  wrong_image.planes[0].image = &wrong_format;  // R8, still SHADER_READ
  rejected(grey, wrong_image, "an image of the wrong format and layout");

  pipelines::ImageUpdate linear_srgb;
  linear_srgb.planes[0].buffer = &rgba;
  linear_srgb.encoding = pipelines::ImageEncoding::Linear;
  rejected(srgb, linear_srgb, "linear encoding for an _SRGB source");

  pipelines::ImageUpdate flat_ramp;
  flat_ramp.planes[0].buffer = &r8;
  flat_ramp.ramp_min = 10.0f;
  flat_ramp.ramp_max = 10.0f;
  rejected(ramp, flat_ramp, "equal ramp bounds");

  const vkc::Buffer chroma = source_buffer(std::vector<uint8_t>(2 * 2 * 2, 0));
  pipelines::ImageUpdate bad_matrix;
  bad_matrix.planes[0].buffer = &r8;
  bad_matrix.planes[1].buffer = &chroma;
  bad_matrix.kr = 0.6f;
  bad_matrix.kb = 0.5f;
  rejected(nv12, bad_matrix, "kr + kb >= 1");

  pipelines::ImageUpdate one_plane;
  one_plane.planes[0].buffer = &r8;
  rejected(nv12, one_plane, "NV12 without its chroma plane");

  pipelines::ImageTexture empty;
  pipelines::ImageUpdate fine;
  fine.planes[0].buffer = &r16;
  EXPECT_EQ(pipeline.record_update(raw, empty, fine).domain(),
            vkc::Status::Code::InvalidArgument);
  EXPECT_TRUE(cmd.value().end().ok());
}

// A Color source drawn one texel per pixel stores its own bytes, on a _UNORM
// target and an _SRGB one alike -- the decode, the _SRGB display and the
// draw's encode round-trip -- and a B8G8R8A8 source keeps its channel order.
TEST_F(ImagePipelineDeviceTest, ColorDrawsItsOwnBytesAtOneToOne) {
  const std::vector<uint8_t> rgba = {
      200, 30, 10,  255, 15, 180, 60, 255,  // row 0
      90,  90, 220, 255, 5,  5,   5,  255,  // row 1
  };
  const vkc::Buffer buffer = source_buffer(rgba);
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &buffer;

  for (VkFormat target : {kUnorm, kSrgb}) {
    const pipelines::ImagePipeline pipeline = make_pipeline(target);
    pipelines::ImageTexture texture =
        make_texture(pipeline, {2, 2}, pipelines::ImageMapping::Color, kUnorm);
    ASSERT_TRUE(texture.valid());
    const Pixels px =
        update_and_draw(pipeline, texture, update,
                        {draw_of(texture, {2, 2}, 1.0f)}, {2, 2}, target);
    ASSERT_EQ(px.bytes.size(), 16u);
    expect_rgb(px.at(0, 0), 200, 30, 10, 1, "texel (0, 0)");
    expect_rgb(px.at(1, 0), 15, 180, 60, 1, "texel (1, 0)");
    expect_rgb(px.at(0, 1), 90, 90, 220, 1, "texel (0, 1)");
    expect_rgb(px.at(1, 1), 5, 5, 5, 1, "texel (1, 1)");
    EXPECT_TRUE(texture.has_picture());
  }

  // The same bytes read as B, G, R, A.
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture bgra =
      make_texture(pipeline, {2, 2}, pipelines::ImageMapping::Color,
                   VK_FORMAT_B8G8R8A8_UNORM);
  const Pixels px = update_and_draw(pipeline, bgra, update,
                                    {draw_of(bgra, {2, 2}, 1.0f)}, {2, 2});
  ASSERT_EQ(px.bytes.size(), 16u);
  expect_rgb(px.at(0, 0), 10, 30, 200, 1, "texel (0, 0) as BGRA");
}

// Magnified nearest, each texel is a crisp 8 x 8 block -- its first and last
// pixel exactly the texel; magnified linear, a block's edge blends with the
// next block.
TEST_F(ImagePipelineDeviceTest, NearestMagnificationIsTexelExact) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture = make_texture(
      pipeline, {4, 1}, pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
  const std::vector<uint8_t> grey = {0, 255, 60, 180};
  const vkc::Buffer buffer = source_buffer(grey);
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &buffer;

  const Pixels nearest = update_and_draw(
      pipeline, texture, update, {draw_of(texture, {32, 8}, 8.0f)}, {32, 8});
  ASSERT_EQ(nearest.bytes.size(), 32u * 8 * 4);
  for (uint32_t t = 0; t < 4; ++t) {
    for (uint32_t x : {8 * t, 8 * t + 7}) {
      EXPECT_NEAR(nearest.at(x, 4)[0], grey[t], 1)
          << "pixel " << x << " belongs to texel " << t;
    }
  }

  pipelines::ImageDraw smooth = draw_of(texture, {32, 8}, 8.0f);
  smooth.magnify = pipelines::ImageFilter::Linear;
  const Pixels linear =
      update_and_draw(pipeline, texture, update, {smooth}, {32, 8});
  ASSERT_EQ(linear.bytes.size(), 32u * 8 * 4);
  // Pixel 7 sits just short of the texel 0 / texel 1 boundary: bilinear
  // filtering gives it part of each.
  EXPECT_GT(linear.at(7, 4)[0], 20);
  EXPECT_LT(linear.at(7, 4)[0], 235);
}

// The point of the _SRGB display image: a one-texel black-and-white
// checkerboard shrunk 2x or 4x is the linear-light average, 50% grey, which
// sRGB-encodes to 188 -- not the 128 that averaging the encoded bytes gives.
TEST_F(ImagePipelineDeviceTest, MipsAverageInLinearLight) {
  constexpr uint32_t kChecker = 64;
  std::vector<uint8_t> checker(kChecker * kChecker);
  for (uint32_t y = 0; y < kChecker; ++y) {
    for (uint32_t x = 0; x < kChecker; ++x) {
      checker[y * kChecker + x] = ((x + y) % 2 == 0) ? 0 : 255;
    }
  }
  const vkc::Buffer buffer = source_buffer(checker);
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &buffer;

  for (VkFormat target : {kUnorm, kSrgb}) {
    const pipelines::ImagePipeline pipeline = make_pipeline(target);
    pipelines::ImageTexture texture =
        make_texture(pipeline, {kChecker, kChecker},
                     pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
    ASSERT_EQ(texture.mip_levels(), 7u);
    for (float scale : {0.5f, 0.25f}) {
      const uint32_t side = static_cast<uint32_t>(kChecker * scale);
      const Pixels px = update_and_draw(pipeline, texture, update,
                                        {draw_of(texture, {side, side}, scale)},
                                        {side, side}, target);
      ASSERT_EQ(px.bytes.size(), static_cast<size_t>(side) * side * 4);
      for (uint32_t y = 0; y < side; y += side / 4) {
        for (uint32_t x = 0; x < side; x += side / 4) {
          EXPECT_NEAR(px.at(x, y)[0], 188, 3)
              << "scale " << scale << " pixel (" << x << ", " << y << ")";
        }
      }
    }
  }
}

// An odd-sized image builds every level of its chain from real texels: a
// uniform color drawn at a tenth of its size is still that color.
TEST_F(ImagePipelineDeviceTest, OddSizedChainsHoldTheImage) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture =
      make_texture(pipeline, {37, 23}, pipelines::ImageMapping::Color, kUnorm);
  std::vector<uint8_t> rgba(37 * 23 * 4);
  for (size_t i = 0; i < rgba.size(); i += 4) {
    rgba[i] = 40;
    rgba[i + 1] = 160;
    rgba[i + 2] = 90;
    rgba[i + 3] = 255;
  }
  const vkc::Buffer buffer = source_buffer(rgba);
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &buffer;
  const Pixels px = update_and_draw(pipeline, texture, update,
                                    {draw_of(texture, {4, 3}, 0.1f)}, {4, 3});
  ASSERT_EQ(px.bytes.size(), 4u * 3 * 4);
  expect_rgb(px.at(1, 1), 40, 160, 90, 2, "a pixel of the shrunken image");
}

// Grey replicates its channel; its encoding decides whether 128 is shown as
// itself (sRGB) or as linear 0.5, which encodes to 188.
TEST_F(ImagePipelineDeviceTest, GreyHonorsItsEncoding) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture = make_texture(
      pipeline, {2, 2}, pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
  const vkc::Buffer buffer = source_buffer(std::vector<uint8_t>(4, 128));
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &buffer;

  const Pixels srgb = update_and_draw(pipeline, texture, update,
                                      {draw_of(texture, {2, 2}, 1.0f)}, {2, 2});
  ASSERT_EQ(srgb.bytes.size(), 16u);
  expect_rgb(srgb.at(0, 0), 128, 128, 128, 1, "sRGB-encoded 128");

  update.encoding = pipelines::ImageEncoding::Linear;
  const Pixels linear = update_and_draw(
      pipeline, texture, update, {draw_of(texture, {2, 2}, 1.0f)}, {2, 2});
  ASSERT_EQ(linear.bytes.size(), 16u);
  expect_rgb(linear.at(0, 0), 188, 188, 188, 2, "linear 128 / 255");
}

// A ramp maps its bounds to its end colors and its middle to the middle
// stop, and shows a stored 0 as black.
TEST_F(ImagePipelineDeviceTest, RampMapsItsBoundsAndShowsZeroAsEmpty) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture = make_texture(
      pipeline, {4, 1}, pipelines::ImageMapping::Ramp, VK_FORMAT_R16_UNORM);
  const std::vector<uint16_t> depth = {0, 1000, 3000, 5000};
  std::vector<uint8_t> bytes(depth.size() * 2);
  std::memcpy(bytes.data(), depth.data(), bytes.size());
  const vkc::Buffer buffer = source_buffer(bytes);
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &buffer;
  update.ramp_min = 1000.0f;
  update.ramp_max = 5000.0f;

  const Pixels px = update_and_draw(pipeline, texture, update,
                                    {draw_of(texture, {32, 8}, 8.0f)}, {32, 8});
  ASSERT_EQ(px.bytes.size(), 32u * 8 * 4);
  expect_rgb(px.at(4, 4), 0, 0, 0, 0, "a stored 0");
  expect_rgb(px.at(12, 4), 48, 18, 59, 2, "ramp_min: the first stop");
  expect_rgb(px.at(20, 4), 26, 224, 115, 2, "the middle: the third stop");
  expect_rgb(px.at(28, 4), 163, 10, 3, 2, "ramp_max: the last stop");
}

// NV12 from two image planes, as a decoder leaves them: BT.601 full-range
// red, and limited-range mid grey.
TEST_F(ImagePipelineDeviceTest, Nv12ConvertsWithItsMatrixAndRange) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture = make_texture(
      pipeline, {4, 4}, pipelines::ImageMapping::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(texture.valid());

  // Red: Y = 0.299 * 255, Cb = 128 - 0.299 / 1.772 * 255, Cr clamps at 255.
  vkc::Image luma =
      source_image(VK_FORMAT_R8_UNORM, {4, 4}, std::vector<uint8_t>(16, 76));
  vkc::Image chroma = source_image(VK_FORMAT_R8G8_UNORM, {2, 2},
                                   {85, 255, 85, 255, 85, 255, 85, 255});
  pipelines::ImageUpdate update;
  update.planes[0].image = &luma;
  update.planes[1].image = &chroma;
  const auto ready = [&](VkCommandBuffer cmd) {
    ready_for_copy(cmd, luma);
    ready_for_copy(cmd, chroma);
  };
  const Pixels red =
      update_and_draw(pipeline, texture, update,
                      {draw_of(texture, {4, 4}, 1.0f)}, {4, 4}, kUnorm, ready);
  ASSERT_EQ(red.bytes.size(), 64u);
  expect_rgb(red.at(1, 1), 254, 0, 0, 2, "BT.601 full-range red");

  // Mid grey: limited-range Y 126 is (126 - 16) / 219 = 0.502.
  vkc::Image grey_luma =
      source_image(VK_FORMAT_R8_UNORM, {4, 4}, std::vector<uint8_t>(16, 126));
  vkc::Image grey_chroma =
      source_image(VK_FORMAT_R8G8_UNORM, {2, 2}, std::vector<uint8_t>(8, 128));
  update.planes[0].image = &grey_luma;
  update.planes[1].image = &grey_chroma;
  update.full_range = false;
  const auto ready_grey = [&](VkCommandBuffer cmd) {
    ready_for_copy(cmd, grey_luma);
    ready_for_copy(cmd, grey_chroma);
  };
  const Pixels grey = update_and_draw(pipeline, texture, update,
                                      {draw_of(texture, {4, 4}, 1.0f)}, {4, 4},
                                      kUnorm, ready_grey);
  ASSERT_EQ(grey.bytes.size(), 64u);
  expect_rgb(grey.at(2, 2), 128, 128, 128, 1, "limited-range mid grey");
}

// Chroma siting: a 4 x 1 picture whose two chroma samples differ only in Cr,
// read through the red channel (R' = Y' + 1.402 Cr'). Centered chroma sits
// between luma pairs, so pixel 1 reads 3/4 of sample 0; left-sited chroma
// sits on luma 0 and 2, so pixel 1 reads half of each and pixel 2 sample 1
// alone.
TEST_F(ImagePipelineDeviceTest, Nv12HonorsItsChromaSiting) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture = make_texture(
      pipeline, {4, 1}, pipelines::ImageMapping::Nv12, VK_FORMAT_UNDEFINED);
  const vkc::Buffer luma = source_buffer(std::vector<uint8_t>(4, 128));
  const vkc::Buffer chroma = source_buffer({128, 200, 128, 56});
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &luma;
  update.planes[1].buffer = &chroma;

  // R' for Cr = 200 is 229, for 56 is 27, and for their mixes in between.
  update.chroma_siting = pipelines::ChromaSiting::Center;
  const Pixels center = update_and_draw(
      pipeline, texture, update, {draw_of(texture, {4, 1}, 1.0f)}, {4, 1});
  ASSERT_EQ(center.bytes.size(), 16u);
  EXPECT_NEAR(center.at(0, 0)[0], 229, 3);
  EXPECT_NEAR(center.at(1, 0)[0], 178, 3);
  EXPECT_NEAR(center.at(2, 0)[0], 78, 3);
  EXPECT_NEAR(center.at(3, 0)[0], 27, 3);

  update.chroma_siting = pipelines::ChromaSiting::Left;
  const Pixels left = update_and_draw(pipeline, texture, update,
                                      {draw_of(texture, {4, 1}, 1.0f)}, {4, 1});
  ASSERT_EQ(left.bytes.size(), 16u);
  EXPECT_NEAR(left.at(0, 0)[0], 229, 3);
  EXPECT_NEAR(left.at(1, 0)[0], 128, 3);
  EXPECT_NEAR(left.at(2, 0)[0], 27, 3);
  EXPECT_NEAR(left.at(3, 0)[0], 27, 3);
}

// A buffer plane with padded rows and an offset reads only the picture.
TEST_F(ImagePipelineDeviceTest, BufferPlanesHonorOffsetAndRowLength) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture = make_texture(
      pipeline, {2, 2}, pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
  // Four bytes of header, then rows of 3 texels of which the picture uses 2.
  const std::vector<uint8_t> bytes = {9, 9, 9, 9, 20, 40, 9, 60, 80, 9};
  const vkc::Buffer buffer = source_buffer(bytes);
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &buffer;
  update.planes[0].offset = 4;
  update.planes[0].row_length = 3;
  const Pixels px = update_and_draw(pipeline, texture, update,
                                    {draw_of(texture, {2, 2}, 1.0f)}, {2, 2});
  ASSERT_EQ(px.bytes.size(), 16u);
  EXPECT_NEAR(px.at(0, 0)[0], 20, 1);
  EXPECT_NEAR(px.at(1, 0)[0], 40, 1);
  EXPECT_NEAR(px.at(0, 1)[0], 60, 1);
  EXPECT_NEAR(px.at(1, 1)[0], 80, 1);
}

// Draws are clipped to their viewport; a texture never updated, and a
// non-positive scale, draw nothing.
TEST_F(ImagePipelineDeviceTest, SubmitClipsToTheViewportAndSkipsEmptyDraws) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture = make_texture(
      pipeline, {2, 2}, pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
  pipelines::ImageTexture never_updated = make_texture(
      pipeline, {2, 2}, pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
  const vkc::Buffer buffer = source_buffer(std::vector<uint8_t>(4, 255));
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &buffer;

  // The image covers the whole 8 x 8 target, but its viewport is the left
  // half.
  pipelines::ImageDraw half = draw_of(texture, {8, 8}, 4.0f);
  half.viewport = {{0, 0}, {4, 8}};
  pipelines::ImageDraw nothing = draw_of(never_updated, {8, 8}, 4.0f);
  pipelines::ImageDraw zero_scale = draw_of(texture, {8, 8}, 0.0f);
  const Pixels px = update_and_draw(pipeline, texture, update,
                                    {nothing, half, zero_scale}, {8, 8});
  ASSERT_EQ(px.bytes.size(), 8u * 8 * 4);
  expect_rgb(px.at(1, 4), 255, 255, 255, 0, "inside the viewport");
  expect_rgb(px.at(6, 4), 0, 0, 255, 0, "outside it: the clear color");
}

// A live source: update, draw, update again, draw again -- one texture, one
// command buffer. Each pass draws the picture 1:1 (level 0) and shrunk 4x
// (level 2) into the corner, so the second update overwrites levels the first
// pass's draws are still reading. Each draw shows the picture recorded before
// it, and the synchronization layer sees every hazard between them covered.
TEST_F(ImagePipelineDeviceTest, UpdatesAndDrawsInterleaveInOneSubmission) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture = make_texture(
      pipeline, {16, 16}, pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
  const vkc::Buffer dark = source_buffer(std::vector<uint8_t>(256, 30));
  const vkc::Buffer light = source_buffer(std::vector<uint8_t>(256, 220));
  pipelines::ImageUpdate first;
  first.planes[0].buffer = &dark;
  pipelines::ImageUpdate second;
  second.planes[0].buffer = &light;

  const std::vector<pipelines::ImageDraw> draws = {
      draw_of(texture, {16, 16}, 1.0f), draw_of(texture, {16, 16}, 0.25f)};
  auto update_with = [&](const pipelines::ImageUpdate& update) {
    return [&, update](VkCommandBuffer cmd) {
      const vkc::Status s = pipeline.record_update(cmd, texture, update);
      EXPECT_TRUE(s.ok()) << s.message();
    };
  };
  const std::vector<Pixels> px = render_passes(
      pipeline, {{update_with(first), draws}, {update_with(second), draws}},
      {16, 16});
  ASSERT_EQ(px.size(), 2u);
  EXPECT_NEAR(px[0].at(10, 10)[0], 30, 1) << "pass 1, level 0";
  EXPECT_NEAR(px[0].at(2, 2)[0], 30, 1) << "pass 1, level 2";
  EXPECT_NEAR(px[1].at(10, 10)[0], 220, 1) << "pass 2, level 0";
  EXPECT_NEAR(px[1].at(2, 2)[0], 220, 1) << "pass 2, level 2";
}

TEST_F(ImagePipelineDeviceTest, TextureMoveLeavesTheSourceEmpty) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture a =
      make_texture(pipeline, {8, 8}, pipelines::ImageMapping::Color, kUnorm);
  ASSERT_TRUE(a.valid());
  const VkImage image = a.display().handle();

  pipelines::ImageTexture b(std::move(a));
  EXPECT_FALSE(a.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(a.extent().width, 0u);
  EXPECT_FALSE(a.has_picture());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.display().handle(), image);

  // Over a live texture: it is released, then the source adopted.
  pipelines::ImageTexture c = make_texture(
      pipeline, {4, 4}, pipelines::ImageMapping::Grey, VK_FORMAT_R8_UNORM);
  ASSERT_TRUE(c.valid());
  c = std::move(b);
  EXPECT_FALSE(b.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(c.display().handle(), image);
  EXPECT_EQ(c.extent().width, 8u);

  // Launder through a pointer so -Wself-move does not fire under -Werror.
  pipelines::ImageTexture* alias = &c;
  c = std::move(*alias);
  EXPECT_TRUE(c.valid());
  EXPECT_EQ(c.display().handle(), image);
}

TEST_F(ImagePipelineDeviceTest, PipelineMoveLeavesTheSourceEmpty) {
  pipelines::ImagePipeline a = make_pipeline();
  ASSERT_TRUE(a.valid());
  pipelines::ImagePipeline b(std::move(a));
  EXPECT_FALSE(a.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(b.valid());

  // A texture cannot be made for the moved-from pipeline.
  pipelines::ImageTextureDesc desc;
  desc.extent = {4, 4};
  EXPECT_EQ(
      pipelines::ImageTexture::create(a, *allocator_, desc).status().domain(),
      vkc::Status::Code::InvalidArgument);

  pipelines::ImagePipeline c = make_pipeline();
  c = std::move(b);
  EXPECT_FALSE(b.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(c.valid());

  pipelines::ImagePipeline* alias = &c;
  c = std::move(*alias);
  EXPECT_TRUE(c.valid());
}

}  // namespace

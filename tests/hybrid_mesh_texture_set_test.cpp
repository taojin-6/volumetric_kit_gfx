// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HybridMeshTextureSet: a HybridMeshPipeline's set 0 over an image the test
// owns. Each draw is an unlit quad over the whole target whose uv0 is its own
// position in [0, 1]^2, so on a target the image's size each pixel samples the
// centre of the image pixel it lies over, and the readback is the image. The
// images are sRGB: the shader decodes what it samples and, on the UNORM
// target, encodes what it writes. Runs under the validation layer with
// synchronization validation, tracking what shaders read through descriptors.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <glm/mat4x4.hpp>

#include "color_test_support.hpp"
#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_set.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;
namespace assets = volumetric_kit::gfx::assets;

using Options = pipelines::HybridMeshTextureSet::Options;
using Rgba = std::array<uint8_t, 4>;

constexpr VkFormat kImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
constexpr VkFormat kTargetFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// Two 2x2 images, row-major, every component distinct from the other image's.
const std::vector<Rgba> kFirst = {
    Rgba{200, 40, 90, 255}, Rgba{30, 180, 120, 255}, Rgba{90, 110, 230, 255},
    Rgba{250, 220, 10, 255}};
const std::vector<Rgba> kSecond = {
    Rgba{60, 140, 20, 255}, Rgba{170, 70, 200, 255}, Rgba{10, 250, 160, 255},
    Rgba{130, 30, 60, 255}};

vg::RenderTargetLayout target_layout() {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = kTargetFormat;
  layout.color_count = 1;
  layout.depth_format = kDepthFormat;
  return layout;
}

// A quad over the whole target, uv0 its own position in [0, 1]^2.
assets::Mesh make_quad() {
  auto vert = [](float x, float y) {
    assets::Vertex v;
    v.position = {x, y, 0.5f};
    v.normal = {0.0f, 0.0f, 1.0f};
    v.uv0 = {(x + 1.0f) * 0.5f, (y + 1.0f) * 0.5f};
    v.color = {1.0f, 1.0f, 1.0f, 1.0f};
    return v;
  };
  assets::Mesh mesh;
  mesh.vertices = {vert(-1.0f, -1.0f), vert(1.0f, -1.0f), vert(1.0f, 1.0f),
                   vert(-1.0f, 1.0f)};
  mesh.indices = {0, 2, 1, 0, 3, 2};
  return mesh;
}

// The byte a UNORM target stores for linear light `linear`.
int encoded_byte(double linear) {
  return static_cast<int>(std::lround(255.0 * vg_test::linear_to_srgb(linear)));
}

// Expects every component of `drawn` within `tolerance` of `expected`.
void expect_near(const std::vector<Rgba>& drawn,
                 const std::vector<Rgba>& expected, int tolerance) {
  ASSERT_EQ(drawn.size(), expected.size());
  for (size_t i = 0; i < drawn.size(); ++i) {
    for (size_t c = 0; c < 4; ++c) {
      EXPECT_LE(std::abs(int{drawn[i][c]} - int{expected[i][c]}), tolerance)
          << "pixel " << i << " component " << c << ": drew "
          << int{drawn[i][c]} << ", expected " << int{expected[i][c]};
    }
  }
}

class HybridMeshTextureSetTest : public vg_test::RendererDeviceTest {
 protected:
  // The draws sample the image through a descriptor, which synchronization
  // validation tracks only with this.
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::ShaderAccesses;
  }

  void SetUp() override {
    vg_test::RendererDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto pipeline = pipelines::HybridMeshPipeline::create(device(), allocator(),
                                                          target_layout());
    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    pipeline_.emplace(std::move(pipeline).value());
    auto quad = pipelines::upload_mesh(device(), allocator(), make_quad());
    ASSERT_TRUE(quad.ok()) << quad.status().message();
    quad_.emplace(std::move(quad).value());
  }

  // `pixels`, a 2x2 sRGB image, in SHADER_READ_ONLY_OPTIMAL.
  vkc::Result<vkc::Image> upload(const std::vector<Rgba>& pixels) {
    vg::ImageUploadDesc desc;
    desc.extent = {2, 2};
    desc.format = kImageFormat;
    desc.pixels = pixels.data();
    desc.size = pixels.size() * sizeof(Rgba);
    return vg::upload_texture(device(), allocator(), desc);
  }

  vkc::Result<std::unique_ptr<pipelines::HybridMeshTextureSet>> make_set(
      VkImageView view, const Options& options = {}) {
    return pipelines::HybridMeshTextureSet::create(device(), *pipeline_, view,
                                                   options);
  }

  // Draws the quad with `set` as set 0 into a fresh `side` x `side` target
  // and returns its pixels, row-major; empty on a failure, which has been
  // reported.
  std::vector<Rgba> draw(VkDescriptorSet set, uint32_t side) {
    vg::OffscreenTargetDesc desc;
    desc.extent = {side, side};
    desc.color_format = kTargetFormat;
    desc.depth_format = kDepthFormat;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    auto pool =
        vkc::CommandPool::create(device().handle(), device().queue_family());
    EXPECT_TRUE(pool.ok()) << pool.status().message();
    if (!target.ok() || !pool.ok()) {
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

    target.value().prepare(raw);
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[3] = 1.0f;
    const vg::RenderTarget rt = target.value().target();
    rt.begin(raw, begin);
    const pipelines::HybridMeshDraw quad{&*quad_};
    // In declaration order: the extent, the identity view-projection (the
    // quad is in clip space), the default light, unlit flags (the albedo
    // itself), the set under test as set 0, and the draw list.
    const pipelines::HybridMeshFrame frame{
        {side, side},
        glm::mat4(1.0f),
        pipelines::HybridMeshFrame{}.light_dir,
        0u,
        set,
        &quad,
        1};
    pipeline_->submit(raw, frame);
    rt.end(raw);
    target.value().record_readback(raw);
    EXPECT_TRUE(cmd.value().end().ok());
    const vkc::Status submitted = device().submit_and_wait(raw);
    if (!submitted.ok()) {
      ADD_FAILURE() << submitted.message();
      return {};
    }

    const auto* px = static_cast<const Rgba*>(target.value().pixels());
    EXPECT_NE(px, nullptr);
    if (px == nullptr) {
      return {};
    }
    return std::vector<Rgba>(px, px + static_cast<size_t>(side) * side);
  }

  std::optional<pipelines::HybridMeshPipeline> pipeline_;
  std::optional<pipelines::GpuMesh> quad_;
};

TEST_F(HybridMeshTextureSetTest, DrawReadsBackTheImage) {
  auto image = upload(kFirst);
  ASSERT_TRUE(image.ok()) << image.status().message();
  auto texture = make_set(image.value().view());
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  ASSERT_NE(texture.value()->set(), VK_NULL_HANDLE);

  expect_near(draw(texture.value()->set(), 2), kFirst, 1);
}

TEST_F(HybridMeshTextureSetTest, RebindDrawsTheSecondImage) {
  auto first = upload(kFirst);
  ASSERT_TRUE(first.ok()) << first.status().message();
  auto second = upload(kSecond);
  ASSERT_TRUE(second.ok()) << second.status().message();
  auto texture = make_set(first.value().view());
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  pipelines::HybridMeshTextureSet& bound = *texture.value();
  const VkDescriptorSet before = bound.set();

  expect_near(draw(bound.set(), 2), kFirst, 1);
  // The draw has completed, so no frame that bound the set is pending.
  bound.rebind(second.value().view());
  EXPECT_EQ(bound.set(), before) << "rebind rewrites the same set";
  expect_near(draw(bound.set(), 2), kSecond, 1);
}

// On a 4x4 target each image pixel spans two target pixels, whose centres
// sample a quarter of the way from the image pixel's centre toward each edge.
// A black left column and a white right column then read, along a row:
// LINEAR, CLAMP_TO_EDGE: black, a quarter white, three quarters, white;
// NEAREST: black, black, white, white; LINEAR, REPEAT: the edges blend with
// the opposite column. A quarter and three quarters are linear light.
TEST_F(HybridMeshTextureSetTest, OptionsChooseTheFilterAndAddressing) {
  const Rgba black{0, 0, 0, 255};
  const Rgba white{255, 255, 255, 255};
  auto image = upload({black, white, black, white});
  ASSERT_TRUE(image.ok()) << image.status().message();

  const auto grey = [](int byte) {
    const auto b = static_cast<uint8_t>(byte);
    return Rgba{b, b, b, 255};
  };
  const Rgba quarter = grey(encoded_byte(0.25));
  const Rgba three_quarters = grey(encoded_byte(0.75));
  const auto rows = [](const Rgba& a, const Rgba& b, const Rgba& c,
                       const Rgba& d) {
    std::vector<Rgba> all;
    for (int y = 0; y < 4; ++y) {
      all.insert(all.end(), {a, b, c, d});
    }
    return all;
  };

  struct Case {
    const char* name;
    Options options;
    std::vector<Rgba> expected;
  };
  Options nearest;
  nearest.filter = VK_FILTER_NEAREST;
  Options repeat;
  repeat.address = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  const Case cases[] = {
      {"defaults", Options{}, rows(black, quarter, three_quarters, white)},
      {"nearest", nearest, rows(black, black, white, white)},
      {"repeat", repeat,
       rows(quarter, quarter, three_quarters, three_quarters)},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    auto texture = make_set(image.value().view(), c.options);
    ASSERT_TRUE(texture.ok()) << texture.status().message();
    // A filtered blend may round either way.
    expect_near(draw(texture.value()->set(), 4), c.expected, 2);
  }
}

// A 4x4 image on a 2x2 target is minified to mip level 1. Level 0 is black and
// levels 1 and 2 white, so the default max_lod of 0 draws black and the whole
// chain white.
TEST_F(HybridMeshTextureSetTest, MaxLodBoundsTheLevelsSampled) {
  const Rgba black{0, 0, 0, 255};
  const Rgba white{255, 255, 255, 255};
  std::vector<Rgba> levels(4 * 4, black);
  levels.insert(levels.end(), 2 * 2 + 1, white);
  vg::ImageUploadDesc desc;
  desc.extent = {4, 4};
  desc.format = kImageFormat;
  desc.pixels = levels.data();
  desc.size = levels.size() * sizeof(Rgba);
  desc.mip_levels = 3;
  auto image = vg::upload_texture(device(), allocator(), desc);
  ASSERT_TRUE(image.ok()) << image.status().message();

  Options whole_chain;
  whole_chain.max_lod = VK_LOD_CLAMP_NONE;
  const struct {
    const char* name;
    Options options;
    Rgba expected;
  } cases[] = {{"defaults", Options{}, black},
               {"whole chain", whole_chain, white}};
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    auto texture = make_set(image.value().view(), c.options);
    ASSERT_TRUE(texture.ok()) << texture.status().message();
    expect_near(draw(texture.value()->set(), 2),
                std::vector<Rgba>(4, c.expected), 2);
  }
}

TEST_F(HybridMeshTextureSetTest, CreateRefusesWhatItCannotBind) {
  auto image = upload(kFirst);
  ASSERT_TRUE(image.ok()) << image.status().message();
  const VkImageView view = image.value().view();

  const pipelines::HybridMeshPipeline empty;
  EXPECT_EQ(pipelines::HybridMeshTextureSet::create(device(), empty, view)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
  EXPECT_EQ(make_set(VK_NULL_HANDLE).status().domain(),
            vkc::Status::Code::InvalidArgument);

  Options cubic;
  cubic.filter = VK_FILTER_CUBIC_EXT;
  Options mirror_clamp;
  mirror_clamp.address = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
  Options negative_lod;
  negative_lod.max_lod = -1.0f;
  Options nan_lod;
  nan_lod.max_lod = std::numeric_limits<float>::quiet_NaN();
  for (const Options& options : {cubic, mirror_clamp, negative_lod, nan_lod}) {
    EXPECT_EQ(make_set(view, options).status().domain(),
              vkc::Status::Code::InvalidArgument);
  }

  Options whole_chain;
  whole_chain.max_lod = VK_LOD_CLAMP_NONE;
  EXPECT_TRUE(make_set(view, whole_chain).ok());
}

}  // namespace

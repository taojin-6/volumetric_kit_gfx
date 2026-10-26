// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HybridMeshTextureSet: a HybridMeshPipeline's set 0 over a picture the test
// owns -- an RGBA image, or a Y'CbCr 4:2:0 picture's two planes. Each draw is
// an unlit quad over the whole target whose uv0 is its own position in
// [0, 1]^2, so on a target the picture's size each pixel samples the centre
// of the picture texel it lies over, and the readback is the picture. The RGBA
// images are sRGB: the shader decodes what it samples and, on the UNORM
// target, encodes what it writes. A two-plane picture is checked texel for
// texel against a double-precision reference, across matrices, ranges,
// sitings and odd and padded planes, and against ImagePipeline's NV12
// convert. Runs under the validation layer with synchronization validation,
// tracking what shaders read through descriptors.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/mat4x4.hpp>

#include "color_test_support.hpp"
#include "gfx_test_support.hpp"
#include "image_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_set.hpp"
#include "volumetric_kit/gfx/pipelines/image_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/ycbcr.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;
namespace assets = volumetric_kit::gfx::assets;

using Options = pipelines::HybridMeshTextureSet::Options;
using Rgba = std::array<uint8_t, 4>;

constexpr VkFormat kImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
constexpr VkFormat kTargetFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kFloatFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// Two 2x2 images, row-major, every component distinct from the other image's.
const std::vector<Rgba> kFirst = {
    Rgba{200, 40, 90, 255}, Rgba{30, 180, 120, 255}, Rgba{90, 110, 230, 255},
    Rgba{250, 220, 10, 255}};
const std::vector<Rgba> kSecond = {
    Rgba{60, 140, 20, 255}, Rgba{170, 70, 200, 255}, Rgba{10, 250, 160, 255},
    Rgba{130, 30, 60, 255}};

vg::RenderTargetLayout target_layout(VkFormat color = kTargetFormat) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = color;
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

// --- Two-plane pictures and their reference ----------------------------------

// A Y'CbCr 4:2:0 picture's planes as bytes, the picture at their corner.
struct PlaneBytes {
  VkExtent2D extent{};          // the picture
  VkExtent2D luma_size{};       // the luma plane, at least `extent`
  VkExtent2D chroma_size{};     // the chroma plane, at least half, rounded up
  std::vector<uint8_t> luma;    // row-major
  std::vector<uint8_t> chroma;  // row-major, Cb then Cr
};

VkExtent2D half_up(VkExtent2D e) {
  return {(e.width + 1) / 2, (e.height + 1) / 2};
}

// A picture whose luma ramps through every code, row by row, and whose
// chroma is a checker of 3x3-sample cells: the 0/255 extremes either way
// round, and two gradients. Texels beyond the picture, in a padded plane,
// alternate between the extremes, so a read of them shows.
PlaneBytes make_planes(VkExtent2D extent, VkExtent2D luma_size,
                       VkExtent2D chroma_size) {
  PlaneBytes p{extent, luma_size, chroma_size, {}, {}};
  p.luma.resize(static_cast<size_t>(luma_size.width) * luma_size.height);
  for (uint32_t y = 0; y < luma_size.height; ++y) {
    for (uint32_t x = 0; x < luma_size.width; ++x) {
      const bool inside = x < extent.width && y < extent.height;
      p.luma[y * luma_size.width + x] = static_cast<uint8_t>(
          inside ? (y * extent.width + x) % 256 : ((x + y) % 2) * 255);
    }
  }
  const VkExtent2D chroma_extent = half_up(extent);
  p.chroma.resize(static_cast<size_t>(chroma_size.width) * chroma_size.height *
                  2);
  for (uint32_t j = 0; j < chroma_size.height; ++j) {
    for (uint32_t i = 0; i < chroma_size.width; ++i) {
      uint32_t cb = 0;
      uint32_t cr = 0;
      if (i >= chroma_extent.width || j >= chroma_extent.height) {
        cb = cr = ((i + j) % 2) * 255;
      } else {
        switch ((i / 3 + j / 3) % 4) {
          case 0:
            cb = 0;
            cr = 255;
            break;
          case 1:
            cb = 255;
            cr = 0;
            break;
          case 2:
            cb = (i * 37) % 256;
            cr = (j * 53) % 256;
            break;
          default:
            cb = 255 - (j * 29) % 256;
            cr = (i * 17 + j * 5) % 256;
            break;
        }
      }
      const size_t at = (static_cast<size_t>(j) * chroma_size.width + i) * 2;
      p.chroma[at] = static_cast<uint8_t>(cb);
      p.chroma[at + 1] = static_cast<uint8_t>(cr);
    }
  }
  return p;
}

// A flat picture -- luma `y`, chroma (`cb`, `cr`) -- in make_planes's planes,
// whose padding alternates between the extremes.
PlaneBytes make_flat_planes(VkExtent2D extent, VkExtent2D luma_size,
                            VkExtent2D chroma_size, uint8_t y, uint8_t cb,
                            uint8_t cr) {
  PlaneBytes p = make_planes(extent, luma_size, chroma_size);
  for (uint32_t row = 0; row < extent.height; ++row) {
    std::fill_n(p.luma.begin() + size_t{row} * luma_size.width, extent.width,
                y);
  }
  const VkExtent2D chroma_extent = half_up(extent);
  for (uint32_t j = 0; j < chroma_extent.height; ++j) {
    for (uint32_t i = 0; i < chroma_extent.width; ++i) {
      const size_t at = (size_t{j} * chroma_size.width + i) * 2;
      p.chroma[at] = cb;
      p.chroma[at + 1] = cr;
    }
  }
  return p;
}

// R'G'B' of the picture's texel (x, y), in double precision: the chroma
// bilinear at the declared siting within the picture's chroma texel centres,
// moved by `shift` codes (Cb, Cr), the range expanded, and the matrix
// R' = Y' + 2 (1 - Kr) Cr', B' = Y' + 2 (1 - Kb) Cb',
// G' = (Y' - Kr R' - Kb B') / Kg, each clamped to [0, 1].
std::array<double, 3> reference(const PlaneBytes& p,
                                const pipelines::YcbcrCoding& coding,
                                uint32_t x, uint32_t y,
                                std::array<double, 2> shift = {0.0, 0.0}) {
  const double luma = p.luma[y * p.luma_size.width + x] / 255.0;
  const VkExtent2D ce = half_up(p.extent);
  const bool left = coding.chroma_siting == pipelines::ChromaSiting::Left;
  const double cx =
      std::clamp((x + 0.5) * 0.5 + (left ? 0.25 : 0.0), 0.5, ce.width - 0.5);
  const double cy = std::clamp((y + 0.5) * 0.5, 0.5, ce.height - 0.5);
  const double fx = cx - 0.5;
  const double fy = cy - 0.5;
  const auto x0 = static_cast<uint32_t>(std::floor(fx));
  const auto y0 = static_cast<uint32_t>(std::floor(fy));
  const uint32_t x1 = std::min(x0 + 1, ce.width - 1);
  const uint32_t y1 = std::min(y0 + 1, ce.height - 1);
  const double ax = fx - x0;
  const double ay = fy - y0;
  const auto at = [&](uint32_t i, uint32_t j, int c) {
    return p.chroma[(static_cast<size_t>(j) * p.chroma_size.width + i) * 2 +
                    c] /
           255.0;
  };
  double cbcr[2];
  for (int c = 0; c < 2; ++c) {
    cbcr[c] = (1 - ay) * ((1 - ax) * at(x0, y0, c) + ax * at(x1, y0, c)) +
              ay * ((1 - ax) * at(x0, y1, c) + ax * at(x1, y1, c)) +
              shift[c] / 255.0;
  }

  double Y = luma;
  double cb = cbcr[0] - 128.0 / 255.0;
  double cr = cbcr[1] - 128.0 / 255.0;
  if (!coding.full_range) {
    Y = (luma * 255.0 - 16.0) / 219.0;
    cb = (cbcr[0] * 255.0 - 128.0) / 224.0;
    cr = (cbcr[1] * 255.0 - 128.0) / 224.0;
  }
  const double kr = coding.kr;
  const double kb = coding.kb;
  const double r = Y + 2.0 * (1.0 - kr) * cr;
  const double b = Y + 2.0 * (1.0 - kb) * cb;
  const double g = (Y - kr * r - kb * b) / (1.0 - kr - kb);
  return {std::clamp(r, 0.0, 1.0), std::clamp(g, 0.0, 1.0),
          std::clamp(b, 0.0, 1.0)};
}

pipelines::YcbcrCoding coding_of(double kr, double kb, bool full_range,
                                 pipelines::ChromaSiting siting) {
  pipelines::YcbcrCoding c;
  c.kr = static_cast<float>(kr);
  c.kb = static_cast<float>(kb);
  c.full_range = full_range;
  c.chroma_siting = siting;
  return c;
}

std::string name_of(const pipelines::YcbcrCoding& c) {
  return std::string(c.kr < 0.25f ? "BT.709" : "BT.601") +
         (c.full_range ? " full" : " limited") +
         (c.chroma_siting == pipelines::ChromaSiting::Left ? " left"
                                                           : " center");
}

// Every matrix, range and siting the tests cover.
std::vector<pipelines::YcbcrCoding> all_codings() {
  std::vector<pipelines::YcbcrCoding> out;
  for (const auto& m : {std::array<double, 2>{0.2126, 0.0722},
                        std::array<double, 2>{0.299, 0.114}}) {
    for (bool full : {false, true}) {
      for (auto siting :
           {pipelines::ChromaSiting::Center, pipelines::ChromaSiting::Left}) {
        out.push_back(coding_of(m[0], m[1], full, siting));
      }
    }
  }
  return out;
}

// The picture sizes, with their planes: exact, even, odd, and a picture in
// padded planes.
struct Size {
  const char* name;
  VkExtent2D extent;
  VkExtent2D luma_size;
  VkExtent2D chroma_size;
};
const Size kSizes[] = {
    {"64x36", {64, 36}, {64, 36}, {32, 18}},
    {"130x66", {130, 66}, {130, 66}, {65, 33}},
    {"odd 129x65", {129, 65}, {129, 65}, {65, 33}},
    {"120x56 in 128x64", {120, 56}, {128, 64}, {64, 32}},
};

class HybridMeshTextureSetTest : public vg_test::RendererDeviceTest,
                                 protected vg_test::ImageHelpers {
 protected:
  // The draws sample the image through a descriptor, which synchronization
  // validation tracks only with this.
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::ShaderAccesses;
  }
  vkc::Device& image_device() override { return device(); }
  vkc::Allocator& image_allocator() override { return allocator(); }

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
    return pipelines::HybridMeshTextureSet::create(device(), allocator(),
                                                   *pipeline_, view, options);
  }

  // The planes of `bytes`, uploaded, in SHADER_READ_ONLY_OPTIMAL.
  struct Planes {
    vkc::Image luma;
    vkc::Image chroma;
  };
  Planes upload_planes(const PlaneBytes& bytes) {
    return {
        source_image(VK_FORMAT_R8_UNORM, bytes.luma_size, bytes.luma),
        source_image(VK_FORMAT_R8G8_UNORM, bytes.chroma_size, bytes.chroma)};
  }

  static pipelines::YcbcrPicture picture_of(
      const Planes& planes, VkExtent2D extent,
      const pipelines::YcbcrCoding& coding = {}) {
    pipelines::YcbcrPicture picture;
    picture.luma = &planes.luma;
    picture.chroma = &planes.chroma;
    picture.extent = extent;
    picture.coding = coding;
    return picture;
  }

  // Draws the quad through `pipeline`, with `set` as set 0, into a fresh
  // `extent` target of `format`, and returns its texels' bytes, row-major;
  // empty on a failure, which has been reported.
  std::vector<uint8_t> draw_bytes(const pipelines::HybridMeshPipeline& pipeline,
                                  VkDescriptorSet set, VkExtent2D extent,
                                  VkFormat format) {
    vg::OffscreenTargetDesc desc;
    desc.extent = extent;
    desc.color_format = format;
    desc.depth_format = kDepthFormat;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    if (!target.ok()) {
      return {};
    }
    const vkc::Status submitted = submit([&](VkCommandBuffer raw) {
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
          extent,
          glm::mat4(1.0f),
          pipelines::HybridMeshFrame{}.light_dir,
          0u,
          set,
          &quad,
          1};
      pipeline.submit(raw, frame);
      rt.end(raw);
      target.value().record_readback(raw);
    });
    if (!submitted.ok()) {
      ADD_FAILURE() << submitted.message();
      return {};
    }
    const auto* px = static_cast<const uint8_t*>(target.value().pixels());
    EXPECT_NE(px, nullptr);
    if (px == nullptr) {
      return {};
    }
    const size_t bytes = static_cast<size_t>(extent.width) * extent.height *
                         (format == kFloatFormat ? 16 : 4);
    return std::vector<uint8_t>(px, px + bytes);
  }

  // The UNORM case, as RGBA texels.
  std::vector<Rgba> draw(VkDescriptorSet set, VkExtent2D extent) {
    const std::vector<uint8_t> bytes =
        draw_bytes(*pipeline_, set, extent, kTargetFormat);
    std::vector<Rgba> texels(bytes.size() / 4);
    std::memcpy(texels.data(), bytes.data(), bytes.size());
    return texels;
  }
  std::vector<Rgba> draw(VkDescriptorSet set, uint32_t side) {
    return draw(set, {side, side});
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
  EXPECT_EQ(pipelines::HybridMeshTextureSet::create(device(), allocator(),
                                                    empty, view)
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

// Every texel centre of a two-plane picture, drawn into a float target, is
// within one 8-bit code of the double-precision reference: for BT.709 and
// BT.601, limited and full range, centred and left-sited chroma, on even, odd
// and padded planes. One set per size is rebound to each coding.
//
// A GPU filters 8-bit texels to more than 8 bits. Lavapipe, a CPU device,
// returns a filtered 8-bit texel on the 8-bit grid, so its chroma blend may
// be a code off before the matrix scales it (by up to 2.1, limited range's B');
// there the reference's chroma may be anywhere within a code of the blend.
TEST_F(HybridMeshTextureSetTest, TwoPlanePictureMatchesTheReference) {
  auto float_pipeline = pipelines::HybridMeshPipeline::create(
      device(), allocator(), target_layout(kFloatFormat));
  ASSERT_TRUE(float_pipeline.ok()) << float_pipeline.status().message();
  const double chroma_slack =
      device().caps().properties().deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU
          ? 1.0
          : 0.0;

  int worst = 0;             // in rounded codes
  double worst_error = 0.0;  // before rounding, in codes, from the exact blend
  for (const Size& size : kSizes) {
    SCOPED_TRACE(size.name);
    const PlaneBytes bytes =
        make_planes(size.extent, size.luma_size, size.chroma_size);
    const Planes planes = upload_planes(bytes);
    ASSERT_TRUE(planes.luma.valid() && planes.chroma.valid());
    auto texture = pipelines::HybridMeshTextureSet::create(
        device(), allocator(), float_pipeline.value(),
        picture_of(planes, size.extent));
    ASSERT_TRUE(texture.ok()) << texture.status().message();

    for (const pipelines::YcbcrCoding& coding : all_codings()) {
      SCOPED_TRACE(name_of(coding));
      // The last draw has completed, so no frame that bound the set is
      // pending.
      const vkc::Status rebound =
          texture.value()->rebind(picture_of(planes, size.extent, coding));
      ASSERT_TRUE(rebound.ok()) << rebound.message();
      const std::vector<uint8_t> drawn =
          draw_bytes(float_pipeline.value(), texture.value()->set(),
                     size.extent, kFloatFormat);
      ASSERT_EQ(drawn.size(),
                size_t{size.extent.width} * size.extent.height * 16);

      int max_codes = 0;
      int reported = 0;
      for (uint32_t y = 0; y < size.extent.height; ++y) {
        for (uint32_t x = 0; x < size.extent.width; ++x) {
          float rgba[4] = {};
          std::memcpy(rgba, &drawn[(size_t{y} * size.extent.width + x) * 16],
                      sizeof(rgba));
          const std::array<double, 3> want = reference(bytes, coding, x, y);
          // The reference's range over the chroma slack: each channel is
          // monotonic in Cb and Cr, so the corners bound it.
          std::array<double, 3> low = want;
          std::array<double, 3> high = want;
          for (double cb : {-chroma_slack, chroma_slack}) {
            for (double cr : {-chroma_slack, chroma_slack}) {
              const std::array<double, 3> corner =
                  reference(bytes, coding, x, y, {cb, cr});
              for (int c = 0; c < 3; ++c) {
                low[c] = std::min(low[c], corner[c]);
                high[c] = std::max(high[c], corner[c]);
              }
            }
          }
          for (int c = 0; c < 3; ++c) {
            // The target holds linear light; the reference is R'G'B'.
            const double got = vg_test::linear_to_srgb(
                std::clamp(static_cast<double>(rgba[c]), 0.0, 1.0));
            const long drawn_code = std::lround(got * 255.0);
            const long low_code = std::lround(low[c] * 255.0);
            const long high_code = std::lround(high[c] * 255.0);
            const int codes = static_cast<int>(
                std::max({0L, low_code - drawn_code, drawn_code - high_code}));
            if (codes > 1 && reported++ < 5) {
              ADD_FAILURE() << "texel (" << x << ", " << y << ") channel " << c
                            << ": drew " << got * 255.0 << ", expected "
                            << want[c] * 255.0;
            }
            max_codes = std::max(max_codes, codes);
            worst_error =
                std::max(worst_error, std::abs(got - want[c]) * 255.0);
          }
        }
      }
      worst = std::max(worst, max_codes);
      ASSERT_LE(max_codes, 1);
    }
  }
  RecordProperty("max_codes", worst);
  RecordProperty("max_error_codes", std::to_string(worst_error));
}

// At texel centres the mesh draws what ImagePipeline's NV12 convert shows:
// the same planes, read 1:1, agree within one code on every size.
TEST_F(HybridMeshTextureSetTest, TwoPlaneAgreesWithImagePipeline) {
  const pipelines::ImagePipeline images = make_pipeline();
  ASSERT_TRUE(images.valid());
  const pipelines::YcbcrCoding codings[] = {
      coding_of(0.299, 0.114, true, pipelines::ChromaSiting::Center),
      coding_of(0.2126, 0.0722, false, pipelines::ChromaSiting::Left)};

  for (const Size& size : kSizes) {
    SCOPED_TRACE(size.name);
    for (const pipelines::YcbcrCoding& coding : codings) {
      SCOPED_TRACE(name_of(coding));
      const PlaneBytes bytes =
          make_planes(size.extent, size.luma_size, size.chroma_size);
      Planes planes = upload_planes(bytes);
      ASSERT_TRUE(planes.luma.valid() && planes.chroma.valid());

      // The mesh first, while the planes are SHADER_READ_ONLY_OPTIMAL.
      auto texture = pipelines::HybridMeshTextureSet::create(
          device(), allocator(), *pipeline_,
          picture_of(planes, size.extent, coding));
      ASSERT_TRUE(texture.ok()) << texture.status().message();
      const std::vector<Rgba> mesh = draw(texture.value()->set(), size.extent);
      ASSERT_EQ(mesh.size(), size_t{size.extent.width} * size.extent.height);

      // Then ImagePipeline, which copies the planes' corner.
      pipelines::ImageTexture image =
          make_texture(images, size.extent, pipelines::ImageMapping::Nv12,
                       VK_FORMAT_UNDEFINED);
      ASSERT_TRUE(image.valid());
      pipelines::ImageUpdate update;
      static_cast<pipelines::YcbcrCoding&>(update) = coding;
      update.planes[0].image = &planes.luma;
      update.planes[1].image = &planes.chroma;
      const vg_test::Pixels shown = update_and_draw(
          images, image, update, {draw_of(image, size.extent, 1.0f)},
          size.extent, vg_test::kUnorm, [&](VkCommandBuffer cmd) {
            ready_for_copy(cmd, planes.luma);
            ready_for_copy(cmd, planes.chroma);
          });
      ASSERT_EQ(shown.bytes.size(), mesh.size() * 4);

      int max_codes = 0;
      for (uint32_t y = 0; y < size.extent.height; ++y) {
        for (uint32_t x = 0; x < size.extent.width; ++x) {
          const Rgba& m = mesh[size_t{y} * size.extent.width + x];
          for (int c = 0; c < 3; ++c) {
            max_codes = std::max(max_codes,
                                 std::abs(int{m[c]} - int{shown.at(x, y)[c]}));
          }
        }
      }
      EXPECT_LE(max_codes, 1);
    }
  }
}

// Magnified, fragments sample between texel centres out to the picture's
// edge. A flat picture in padded planes then draws its one colour everywhere:
// each plane's reads are held to the picture's last texel centres, so no
// filter reaches the padding, whose extremes would show along the right and
// bottom edges. Left-sited chroma reaches past an odd picture's last chroma
// centre too.
TEST_F(HybridMeshTextureSetTest, MagnifiedPictureNeverReadsThePadding) {
  auto float_pipeline = pipelines::HybridMeshPipeline::create(
      device(), allocator(), target_layout(kFloatFormat));
  ASSERT_TRUE(float_pipeline.ok()) << float_pipeline.status().message();
  const Size sizes[] = {
      {"120x56 in 128x64", {120, 56}, {128, 64}, {64, 32}},
      {"odd 33x17 in 40x20", {33, 17}, {40, 20}, {20, 10}},
  };
  const pipelines::YcbcrCoding codings[] = {
      coding_of(0.2126, 0.0722, false, pipelines::ChromaSiting::Left),
      coding_of(0.299, 0.114, true, pipelines::ChromaSiting::Center)};

  for (const Size& size : sizes) {
    SCOPED_TRACE(size.name);
    const PlaneBytes bytes = make_flat_planes(size.extent, size.luma_size,
                                              size.chroma_size, 150, 90, 170);
    const Planes planes = upload_planes(bytes);
    ASSERT_TRUE(planes.luma.valid() && planes.chroma.valid());
    auto texture = pipelines::HybridMeshTextureSet::create(
        device(), allocator(), float_pipeline.value(),
        picture_of(planes, size.extent));
    ASSERT_TRUE(texture.ok()) << texture.status().message();
    // About five target pixels to two texels, so the last column and row
    // sample beyond the picture's last texel centres.
    const VkExtent2D target{size.extent.width * 5 / 2,
                            size.extent.height * 5 / 2};

    for (const pipelines::YcbcrCoding& coding : codings) {
      SCOPED_TRACE(name_of(coding));
      const vkc::Status rebound =
          texture.value()->rebind(picture_of(planes, size.extent, coding));
      ASSERT_TRUE(rebound.ok()) << rebound.message();
      const std::vector<uint8_t> drawn = draw_bytes(
          float_pipeline.value(), texture.value()->set(), target, kFloatFormat);
      ASSERT_EQ(drawn.size(), size_t{target.width} * target.height * 16);
      const std::array<double, 3> want = reference(bytes, coding, 0, 0);

      int max_codes = 0;
      int reported = 0;
      for (uint32_t y = 0; y < target.height; ++y) {
        for (uint32_t x = 0; x < target.width; ++x) {
          float rgba[4] = {};
          std::memcpy(rgba, &drawn[(size_t{y} * target.width + x) * 16],
                      sizeof(rgba));
          for (int c = 0; c < 3; ++c) {
            const double got = vg_test::linear_to_srgb(
                std::clamp(static_cast<double>(rgba[c]), 0.0, 1.0));
            const int codes =
                static_cast<int>(std::lround(std::abs(got - want[c]) * 255.0));
            if (codes > 1 && reported++ < 5) {
              ADD_FAILURE() << "target (" << x << ", " << y << ") channel " << c
                            << ": drew " << got * 255.0 << ", expected "
                            << want[c] * 255.0;
            }
            max_codes = std::max(max_codes, codes);
          }
        }
      }
      EXPECT_LE(max_codes, 1);
    }
  }
}

// A picture's coding is a uniform in host-mapped device memory. A device
// without that memory still makes and draws RGBA sets, which bind the
// pipeline's device-only uniform, and refuses a picture as Unsupported.
TEST_F(HybridMeshTextureSetTest, APictureAloneNeedsHostMappedMemory) {
  auto image = upload(kFirst);
  ASSERT_TRUE(image.ok()) << image.status().message();
  PlaneBytes red{{2, 2}, {2, 2}, {1, 1}, {76, 76, 76, 76}, {85, 255}};
  const Planes planes = upload_planes(red);
  ASSERT_TRUE(planes.luma.valid() && planes.chroma.valid());
  const pipelines::YcbcrPicture picture = picture_of(planes, {2, 2});
  const bool mapped = device().caps().device_mapped_memory();

  auto rgba = make_set(image.value().view());
  ASSERT_TRUE(rgba.ok()) << rgba.status().message();
  const vkc::Status rebound = rgba.value()->rebind(picture);
  EXPECT_EQ(rebound.ok(), mapped);
  auto made = pipelines::HybridMeshTextureSet::create(device(), allocator(),
                                                      *pipeline_, picture);
  EXPECT_EQ(made.ok(), mapped);
  if (!mapped) {
    EXPECT_EQ(rebound.domain(), vkc::Status::Code::Unsupported);
    EXPECT_EQ(made.status().domain(), vkc::Status::Code::Unsupported);
    expect_near(draw(rgba.value()->set(), 2), kFirst, 1);
  }
}

// One set, created over either form, switches to the other and back on
// rebind, keeping its handle: the RGBA image, then BT.601 full-range red from
// a 2x2 picture (as image_pipeline_test's NV12 red), then the image again.
TEST_F(HybridMeshTextureSetTest, RebindSwitchesTheFormBothWays) {
  auto image = upload(kFirst);
  ASSERT_TRUE(image.ok()) << image.status().message();
  PlaneBytes red{{2, 2}, {2, 2}, {1, 1}, {76, 76, 76, 76}, {85, 255}};
  const Planes planes = upload_planes(red);
  ASSERT_TRUE(planes.luma.valid() && planes.chroma.valid());
  const pipelines::YcbcrPicture picture = picture_of(planes, {2, 2});
  const std::vector<Rgba> kRed(4, Rgba{254, 0, 0, 255});

  auto from_rgba = make_set(image.value().view());
  ASSERT_TRUE(from_rgba.ok()) << from_rgba.status().message();
  pipelines::HybridMeshTextureSet& a = *from_rgba.value();
  const VkDescriptorSet a_set = a.set();
  expect_near(draw(a.set(), 2), kFirst, 1);
  ASSERT_TRUE(a.rebind(picture).ok());
  expect_near(draw(a.set(), 2), kRed, 2);
  a.rebind(image.value().view());
  expect_near(draw(a.set(), 2), kFirst, 1);
  EXPECT_EQ(a.set(), a_set);

  auto from_picture = pipelines::HybridMeshTextureSet::create(
      device(), allocator(), *pipeline_, picture);
  ASSERT_TRUE(from_picture.ok()) << from_picture.status().message();
  pipelines::HybridMeshTextureSet& b = *from_picture.value();
  const VkDescriptorSet b_set = b.set();
  expect_near(draw(b.set(), 2), kRed, 2);
  b.rebind(image.value().view());
  expect_near(draw(b.set(), 2), kFirst, 1);
  ASSERT_TRUE(b.rebind(picture).ok());
  expect_near(draw(b.set(), 2), kRed, 2);
  EXPECT_EQ(b.set(), b_set);
}

// Planes left in GENERAL, as a camera picture's imported planes are, are
// sampled in it.
TEST_F(HybridMeshTextureSetTest, PlanesInGeneralAreSampledThere) {
  PlaneBytes red{{2, 2}, {2, 2}, {1, 1}, {76, 76, 76, 76}, {85, 255}};
  Planes planes = upload_planes(red);
  ASSERT_TRUE(planes.luma.valid() && planes.chroma.valid());
  const vkc::Status moved = submit([&](VkCommandBuffer cmd) {
    for (vkc::Image* plane : {&planes.luma, &planes.chroma}) {
      vg::ImageBarrierDesc b;
      b.image = plane->handle();
      b.src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      b.dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      b.dst_access = VK_ACCESS_SHADER_READ_BIT;
      b.old_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      b.new_layout = VK_IMAGE_LAYOUT_GENERAL;
      vg::cmd_image_barrier(cmd, b);
    }
  });
  ASSERT_TRUE(moved.ok()) << moved.message();
  planes.luma.set_layout(VK_IMAGE_LAYOUT_GENERAL);
  planes.chroma.set_layout(VK_IMAGE_LAYOUT_GENERAL);

  auto texture = pipelines::HybridMeshTextureSet::create(
      device(), allocator(), *pipeline_, picture_of(planes, {2, 2}));
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  expect_near(draw(texture.value()->set(), 2),
              std::vector<Rgba>(4, Rgba{254, 0, 0, 255}), 2);
}

TEST_F(HybridMeshTextureSetTest, TwoPlaneCreateAndRebindRefuseBadPictures) {
  const PlaneBytes bytes = make_planes({64, 36}, {64, 36}, {32, 18});
  const Planes planes = upload_planes(bytes);
  ASSERT_TRUE(planes.luma.valid() && planes.chroma.valid());
  const pipelines::YcbcrPicture good = picture_of(planes, {64, 36});

  // An image the set cannot sample: no SAMPLED usage, no view, or nothing
  // written yet (UNDEFINED).
  const auto make_image = [&](VkImageUsageFlags usage, bool with_view) {
    vkc::ImageDesc desc;
    desc.extent = {64, 36};
    desc.format = VK_FORMAT_R8_UNORM;
    desc.usage = usage;
    desc.with_view = with_view;
    auto image = allocator().create_image(desc);
    EXPECT_TRUE(image.ok()) << image.status().message();
    return image.ok() ? std::move(image).value() : vkc::Image{};
  };
  const vkc::Image unsampled =
      make_image(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true);
  const vkc::Image viewless = make_image(VK_IMAGE_USAGE_SAMPLED_BIT, false);
  const vkc::Image unwritten = make_image(VK_IMAGE_USAGE_SAMPLED_BIT, true);
  // A chroma plane a texel short of half the picture.
  const vkc::Image narrow_chroma = source_image(
      VK_FORMAT_R8G8_UNORM, {31, 18}, std::vector<uint8_t>(31 * 18 * 2, 128));
  // A luma plane for a picture 65 wide or 37 high, whose chroma -- half of
  // it, rounded up -- is a column or a row more than the good one's 32x18.
  const vkc::Image odd_luma = source_image(VK_FORMAT_R8_UNORM, {65, 37},
                                           std::vector<uint8_t>(65 * 37, 128));

  struct Case {
    const char* name;
    pipelines::YcbcrPicture picture;
  };
  std::vector<Case> cases;
  const auto with = [&](const char* name, auto change) {
    pipelines::YcbcrPicture p = good;
    change(p);
    cases.push_back({name, p});
  };
  with("no luma", [](auto& p) { p.luma = nullptr; });
  with("no chroma", [](auto& p) { p.chroma = nullptr; });
  with("luma not R8", [&](auto& p) { p.luma = &planes.chroma; });
  with("chroma not R8G8", [&](auto& p) { p.chroma = &planes.luma; });
  with("unsampled luma", [&](auto& p) { p.luma = &unsampled; });
  with("viewless luma", [&](auto& p) { p.luma = &viewless; });
  with("unwritten luma", [&](auto& p) { p.luma = &unwritten; });
  with("zero extent", [](auto& p) { p.extent = {0, 36}; });
  with("luma too small", [](auto& p) { p.extent = {65, 36}; });
  with("chroma too small", [&](auto& p) { p.chroma = &narrow_chroma; });
  with("chroma half an odd width, rounded down", [&](auto& p) {
    p.luma = &odd_luma;
    p.extent = {65, 36};
  });
  with("chroma half an odd height, rounded down", [&](auto& p) {
    p.luma = &odd_luma;
    p.extent = {64, 37};
  });
  with("kr of 0", [](auto& p) { p.coding.kr = 0.0f; });
  with("weights summing to 1", [](auto& p) {
    p.coding.kr = 0.5f;
    p.coding.kb = 0.5f;
  });
  with("NaN weight",
       [](auto& p) { p.coding.kb = std::numeric_limits<float>::quiet_NaN(); });
  with("unknown siting", [](auto& p) {
    p.coding.chroma_siting = static_cast<pipelines::ChromaSiting>(7);
  });

  auto texture = pipelines::HybridMeshTextureSet::create(device(), allocator(),
                                                         *pipeline_, good);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  const std::vector<Rgba> before =
      draw(texture.value()->set(), VkExtent2D{64, 36});
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    EXPECT_EQ(pipelines::HybridMeshTextureSet::create(device(), allocator(),
                                                      *pipeline_, c.picture)
                  .status()
                  .domain(),
              vkc::Status::Code::InvalidArgument);
    EXPECT_EQ(texture.value()->rebind(c.picture).domain(),
              vkc::Status::Code::InvalidArgument);
  }
  // A refused rebind leaves the set as it was.
  expect_near(draw(texture.value()->set(), VkExtent2D{64, 36}), before, 0);

  // A picture is sampled CLAMP_TO_EDGE alone: refused at create, and by
  // rebind on an RGBA set made with another address mode, which keeps
  // drawing its image.
  Options repeat;
  repeat.address = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  EXPECT_EQ(pipelines::HybridMeshTextureSet::create(device(), allocator(),
                                                    *pipeline_, good, repeat)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
  auto image = upload(kFirst);
  ASSERT_TRUE(image.ok()) << image.status().message();
  auto rgba = make_set(image.value().view(), repeat);
  ASSERT_TRUE(rgba.ok()) << rgba.status().message();
  EXPECT_EQ(rgba.value()->rebind(good).domain(),
            vkc::Status::Code::InvalidArgument);
  expect_near(draw(rgba.value()->set(), 2), kFirst, 1);

  const pipelines::HybridMeshPipeline empty;
  EXPECT_EQ(pipelines::HybridMeshTextureSet::create(device(), allocator(),
                                                    empty, good)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
}

}  // namespace

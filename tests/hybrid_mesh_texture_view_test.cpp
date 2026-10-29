// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HybridMeshTextureView and PicturePyramid: a mesh's set-0 texture drawn flat.
// The reference for each picture pixel is the mesh's own draw of it -- an
// unlit quad whose uv0 is its position, on a target the picture's size --
// read back as linear light from a float target, so the view is checked
// against what the mesh shows. Magnified, each target pixel is the picture
// pixel under it; minified, it is checked against the exact box mean, in
// linear light, of the picture pixels under it. RGBA (sRGB) images and
// Y'CbCr 4:2:0 pictures, one or two. Runs under the validation layer with
// synchronization validation, tracking what shaders read through
// descriptors. And HybridMeshFrame::viewport: a mesh drawn into part of the
// target.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "color_test_support.hpp"
#include "gfx_test_support.hpp"
#include "image_test_support.hpp"
#include "texture_view_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_set.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_view.hpp"
#include "volumetric_kit/gfx/pipelines/ycbcr.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;
namespace assets = volumetric_kit::gfx::assets;
using vg_test::bt709_full;
using vg_test::make_card;
using vg_test::Pixels;
using vg_test::PlaneBytes;
using vg_test::planes_of;
using vg_test::Rgb;

constexpr VkFormat kTarget = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kFloat = VK_FORMAT_R32G32B32A32_SFLOAT;
constexpr VkFormat kDepth = VK_FORMAT_D32_SFLOAT;
// The clear: opaque blue, which no test picture holds.
constexpr std::array<uint8_t, 3> kClear{0, 0, 255};

vg::RenderTargetLayout layout_of(VkFormat color) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = color;
  layout.color_count = 1;
  layout.depth_format = kDepth;
  return layout;
}

// A quad over the whole target, uv0 its own position in [0, 1]^2, plus `dv`
// on uv0.y.
assets::Mesh make_quad(float dv = 0.0f) {
  assets::Mesh mesh;
  for (const auto& xy :
       {std::array<float, 2>{-1.0f, -1.0f}, std::array<float, 2>{1.0f, -1.0f},
        std::array<float, 2>{1.0f, 1.0f}, std::array<float, 2>{-1.0f, 1.0f}}) {
    assets::Vertex v;
    v.position = {xy[0], xy[1], 0.5f};
    v.normal = {0.0f, 0.0f, 1.0f};
    v.uv0 = {(xy[0] + 1.0f) * 0.5f, (xy[1] + 1.0f) * 0.5f + dv};
    v.color = {1.0f, 1.0f, 1.0f, 1.0f};
    mesh.vertices.push_back(v);
  }
  mesh.indices = {0, 2, 1, 0, 3, 2};
  return mesh;
}

// Linear light, RGB, row-major: what a reference holds.
struct Linear {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<std::array<double, 3>> pixels;
  const std::array<double, 3>& at(uint32_t x, uint32_t y) const {
    return pixels[static_cast<size_t>(y) * width + x];
  }
};

// The mean of `lin` over [x0, x1) x [y0, y1), in its pixels, each pixel
// weighted by its area inside.
std::array<double, 3> box_mean(const Linear& lin, double x0, double y0,
                               double x1, double y1) {
  std::array<double, 3> sum{0.0, 0.0, 0.0};
  double area = 0.0;
  for (auto y = static_cast<uint32_t>(std::floor(y0));
       y < static_cast<uint32_t>(std::ceil(y1)); ++y) {
    const double wy =
        std::min(y1, y + 1.0) - std::max(y0, static_cast<double>(y));
    for (auto x = static_cast<uint32_t>(std::floor(x0));
         x < static_cast<uint32_t>(std::ceil(x1)); ++x) {
      const double w =
          wy * (std::min(x1, x + 1.0) - std::max(x0, static_cast<double>(x)));
      for (int c = 0; c < 3; ++c) {
        sum[c] += w * lin.at(x, y)[c];
      }
      area += w;
    }
  }
  for (double& s : sum) {
    s /= area;
  }
  return sum;
}

int encoded(double linear) {
  return static_cast<int>(std::lround(
      255.0 * vg_test::linear_to_srgb(std::clamp(linear, 0.0, 1.0))));
}

// How far a drawn picture is from its reference: the largest and the mean
// difference over every component compared, in codes.
struct Error {
  int max = 0;
  double mean = 0.0;
  size_t compared = 0;
};

class TextureViewTest : public vg_test::RendererDeviceTest,
                        protected vg_test::ImageHelpers {
 protected:
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
                                                          layout_of(kTarget));
    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    pipeline_.emplace(std::move(pipeline).value());
    auto linear = pipelines::HybridMeshPipeline::create(device(), allocator(),
                                                        layout_of(kFloat));
    ASSERT_TRUE(linear.ok()) << linear.status().message();
    linear_.emplace(std::move(linear).value());
    auto view = pipelines::HybridMeshTextureView::create(device(), *pipeline_,
                                                         layout_of(kTarget));
    ASSERT_TRUE(view.ok()) << view.status().message();
    view_ = std::move(view).value();
    for (int p = 0; p < 2; ++p) {
      auto quad = pipelines::upload_mesh(device(), allocator(),
                                         make_quad(static_cast<float>(p)));
      ASSERT_TRUE(quad.ok()) << quad.status().message();
      quads_[p].emplace(std::move(quad).value());
    }
  }

  // `rgb` as an sRGB image, its set over it.
  struct RgbaTexture {
    vkc::Image image;
    std::unique_ptr<pipelines::HybridMeshTextureSet> set;
  };
  RgbaTexture rgba_texture(const Rgb& rgb) {
    std::vector<uint8_t> bytes;
    for (const auto& p : rgb.pixels) {
      bytes.insert(bytes.end(), {p[0], p[1], p[2], 255});
    }
    RgbaTexture t;
    t.image = source_image(vg_test::kSrgb, {rgb.width, rgb.height}, bytes);
    auto set = pipelines::HybridMeshTextureSet::create(
        device(), allocator(), *pipeline_, t.image.view());
    EXPECT_TRUE(set.ok()) << set.status().message();
    if (set.ok()) {
      t.set = std::move(set).value();
    }
    return t;
  }

  // `rgbs` as Y'CbCr pictures, each in planes padded by `pad` texels, their
  // set over them.
  struct PictureTexture {
    std::vector<vkc::Image> planes;
    std::unique_ptr<pipelines::HybridMeshTextureSet> set;
  };
  PictureTexture picture_texture(const std::vector<const Rgb*>& rgbs,
                                 uint32_t pad = 0) {
    PictureTexture t;
    t.planes.reserve(2 * rgbs.size());
    std::vector<pipelines::YcbcrPicture> pictures;
    for (const Rgb* rgb : rgbs) {
      const PlaneBytes bytes =
          planes_of(*rgb, {rgb->width + pad, rgb->height + pad});
      t.planes.push_back(
          source_image(VK_FORMAT_R8_UNORM, bytes.luma_size, bytes.luma));
      t.planes.push_back(
          source_image(VK_FORMAT_R8G8_UNORM, bytes.chroma_size, bytes.chroma));
      pipelines::YcbcrPicture picture;
      picture.luma = &t.planes[t.planes.size() - 2];
      picture.chroma = &t.planes.back();
      picture.extent = {rgb->width, rgb->height};
      picture.coding = bt709_full();
      pictures.push_back(picture);
    }
    auto set = pipelines::HybridMeshTextureSet::create(
        device(), allocator(), *pipeline_, pictures.data(),
        static_cast<uint32_t>(pictures.size()));
    EXPECT_TRUE(set.ok()) << set.status().message();
    if (set.ok()) {
      t.set = std::move(set).value();
    }
    return t;
  }

  std::unique_ptr<pipelines::PicturePyramid> make_pyramid(VkExtent2D size,
                                                          uint32_t count) {
    auto pyramid =
        pipelines::PicturePyramid::create(*view_, allocator(), size, count);
    EXPECT_TRUE(pyramid.ok()) << pyramid.status().message();
    return pyramid.ok() ? std::move(pyramid).value() : nullptr;
  }

  // Records `before` (outside the rendering scope), then `inside` into a
  // fresh `extent` target of `format`, cleared to kClear, and reads it back.
  Pixels render(VkExtent2D extent, VkFormat format,
                const std::function<void(VkCommandBuffer)>& before,
                const std::function<void(VkCommandBuffer)>& inside) {
    vg::OffscreenTargetDesc desc;
    desc.extent = extent;
    desc.color_format = format;
    desc.depth_format = kDepth;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    if (!target.ok()) {
      return {};
    }
    const vkc::Status submitted = submit([&](VkCommandBuffer cmd) {
      if (before) {
        before(cmd);
      }
      target.value().prepare(cmd);
      vg::RenderTargetBeginInfo begin;
      begin.clear_color.float32[0] = kClear[0] / 255.0f;
      begin.clear_color.float32[1] = kClear[1] / 255.0f;
      begin.clear_color.float32[2] = kClear[2] / 255.0f;
      begin.clear_color.float32[3] = 1.0f;
      const vg::RenderTarget rt = target.value().target();
      rt.begin(cmd, begin);
      inside(cmd);
      rt.end(cmd);
      target.value().record_readback(cmd);
    });
    EXPECT_TRUE(submitted.ok()) << submitted.message();
    if (!submitted.ok()) {
      return {};
    }
    const uint32_t texel = format == kFloat ? 16 : 4;
    const auto* px = static_cast<const uint8_t*>(target.value().pixels());
    return {std::vector<uint8_t>(px, px + static_cast<size_t>(extent.width) *
                                              extent.height * texel),
            extent.width, texel};
  }

  // The mesh's draw of picture `p` of `set`, a quad on a target the
  // picture's `size`, so each target pixel is the picture pixel it covers.
  void draw_mesh(VkCommandBuffer cmd, const pipelines::HybridMeshPipeline& with,
                 VkDescriptorSet set, VkExtent2D size, uint32_t p,
                 VkRect2D viewport = {}) {
    const pipelines::HybridMeshDraw quad{&*quads_[p]};
    pipelines::HybridMeshFrame frame;
    frame.extent = size;
    frame.flags = 0;  // unlit: the albedo itself
    frame.atlas = set;
    frame.draws = &quad;
    frame.draw_count = 1;
    frame.viewport = viewport;
    with.submit(cmd, frame);
  }

  // Picture `p` of `set` as the mesh shows it, in linear light.
  Linear reference(VkDescriptorSet set, VkExtent2D size, uint32_t p = 0) {
    const Pixels px = render(size, kFloat, nullptr, [&](VkCommandBuffer cmd) {
      draw_mesh(cmd, *linear_, set, size, p);
    });
    Linear lin{size.width, size.height, {}};
    for (uint32_t y = 0; y < size.height && !px.bytes.empty(); ++y) {
      for (uint32_t x = 0; x < size.width; ++x) {
        float rgba[4];
        std::memcpy(rgba, px.at(x, y), sizeof(rgba));
        lin.pixels.push_back({rgba[0], rgba[1], rgba[2]});
      }
    }
    return lin;
  }

  // The view's draw of `draw` (its viewport the whole target, if empty) into
  // an `extent` UNORM target, after `pyramid`'s build from the draw's set.
  Pixels draw_view(VkExtent2D extent, pipelines::TextureViewDraw draw,
                   pipelines::PicturePyramid* pyramid = nullptr) {
    if (draw.viewport.extent.width == 0) {
      draw.viewport = {{0, 0}, extent};
    }
    draw.pyramid = pyramid;
    return render(
        extent, kTarget,
        [&](VkCommandBuffer cmd) {
          if (pyramid != nullptr) {
            pyramid->record(cmd, draw.set);
          }
        },
        [&](VkCommandBuffer cmd) { view_->submit(cmd, extent, &draw, 1); });
  }

  static pipelines::TextureViewDraw draw_of(VkDescriptorSet set, glm::vec2 size,
                                            float scale,
                                            glm::vec2 origin = {0.0f, 0.0f},
                                            uint32_t picture = 0) {
    pipelines::TextureViewDraw draw;
    draw.set = set;
    draw.uv = {0.0f, static_cast<float>(picture), 1.0f,
               static_cast<float>(picture) + 1.0f};
    draw.size = size;
    draw.origin = origin;
    draw.scale = scale;
    return draw;
  }

  // The view's draw at `scale` from `origin` against the box mean of `lin`
  // under each pixel whose footprint lies a pixel or more inside the picture.
  static Error minified_error(const Pixels& drawn, const Linear& lin,
                              float scale, glm::vec2 origin) {
    Error e;
    double sum = 0.0;
    const double f = 1.0 / scale;
    const uint32_t width = drawn.width;
    const auto height =
        static_cast<uint32_t>(drawn.bytes.size() / (4 * size_t{width}));
    for (uint32_t ty = 0; ty < height; ++ty) {
      for (uint32_t tx = 0; tx < width; ++tx) {
        const double x0 = (tx - origin.x) * f;
        const double y0 = (ty - origin.y) * f;
        if (x0 < 1.0 || y0 < 1.0 || x0 + f > lin.width - 1.0 ||
            y0 + f > lin.height - 1.0) {
          continue;
        }
        const std::array<double, 3> mean =
            box_mean(lin, x0, y0, x0 + f, y0 + f);
        for (int c = 0; c < 3; ++c) {
          const int d = std::abs(int{drawn.at(tx, ty)[c]} - encoded(mean[c]));
          e.max = std::max(e.max, d);
          sum += d;
          ++e.compared;
        }
      }
    }
    e.mean = e.compared > 0 ? sum / e.compared : 0.0;
    return e;
  }

  std::optional<pipelines::HybridMeshPipeline> pipeline_;
  std::optional<pipelines::HybridMeshPipeline> linear_;
  std::unique_ptr<pipelines::HybridMeshTextureView> view_;
  std::optional<pipelines::GpuMesh> quads_[2];
};

// --- Magnified: a picture pixel is a sharp square ---------------------------

// At 1:1 the view draws exactly what the mesh draws over the same target, and
// at 4:1 each 4 x 4 square is one picture pixel: RGBA and Y'CbCr alike.
TEST_F(TextureViewTest, MagnifiedEachPictureIsSharpSquares) {
  constexpr VkExtent2D kSize{48, 32};
  const Rgb card = make_card(kSize.width, kSize.height);
  RgbaTexture rgba = rgba_texture(card);
  PictureTexture picture = picture_texture({&card});
  ASSERT_TRUE(rgba.set && picture.set);
  for (const VkDescriptorSet set : {rgba.set->set(), picture.set->set()}) {
    SCOPED_TRACE(set == rgba.set->set() ? "RGBA" : "Y'CbCr");
    const Pixels mesh = render(kSize, kTarget, nullptr, [&](VkCommandBuffer c) {
      draw_mesh(c, *pipeline_, set, kSize, 0);
    });
    const glm::vec2 size{kSize.width, kSize.height};
    const Pixels one = draw_view(kSize, draw_of(set, size, 1.0f));
    ASSERT_EQ(one.bytes.size(), mesh.bytes.size());
    int worst = 0;
    for (size_t i = 0; i < one.bytes.size(); ++i) {
      worst = std::max(worst, std::abs(int{one.bytes[i]} - mesh.bytes[i]));
    }
    EXPECT_LE(worst, 1) << "1:1 against the mesh";

    // 4:1 from a fractional origin: the squares move with it, whole.
    const glm::vec2 origin{0.25f, 0.0f};
    const Pixels four = draw_view({4 * kSize.width, 4 * kSize.height},
                                  draw_of(set, size, 4.0f, origin));
    int off = 0;
    for (uint32_t y = 0; y < 4 * kSize.height; ++y) {
      for (uint32_t x = 1; x < 4 * kSize.width; ++x) {
        const auto px = static_cast<uint32_t>((x + 0.5f - origin.x) / 4.0f);
        const auto py = static_cast<uint32_t>((y + 0.5f) / 4.0f);
        for (int c = 0; c < 3; ++c) {
          off = std::max(
              off, std::abs(int{four.at(x, y)[c]} - int{one.at(px, py)[c]}));
        }
      }
    }
    EXPECT_LE(off, 1) << "4:1 against 1:1";
  }
}

// --- Minified: the mean of the footprint ------------------------------------

// Each pixel against the exact box mean, in linear light, of the picture
// pixels under it. On the picture itself (to 1/4) the view's mean is that
// mean, at any origin; so it is on the pyramid where the footprint's edges
// fall between its pixels, at 1/8 and 1/16 from a whole origin, to the
// pyramid's 8-bit rounding. Elsewhere a pyramid pixel the footprint's edge
// cuts is taken as a whole, which shows only at sharp edges: within 1.6
// codes on average, and 32 at the most.
TEST_F(TextureViewTest, MinifiedEachPixelIsTheMeanOfItsFootprint) {
  constexpr VkExtent2D kSize{512, 256};
  const Rgb card = make_card(kSize.width, kSize.height);
  RgbaTexture rgba = rgba_texture(card);
  PictureTexture picture = picture_texture({&card});
  ASSERT_TRUE(rgba.set && picture.set);
  for (const VkDescriptorSet set : {rgba.set->set(), picture.set->set()}) {
    SCOPED_TRACE(set == rgba.set->set() ? "RGBA" : "Y'CbCr");
    const Linear lin = reference(set, kSize);
    ASSERT_EQ(lin.pixels.size(), size_t{kSize.width} * kSize.height);
    std::unique_ptr<pipelines::PicturePyramid> pyramid = make_pyramid(kSize, 1);
    ASSERT_NE(pyramid, nullptr);
    struct Case {
      float zoom;
      glm::vec2 origin;
      int max;
      double mean;
    };
    const glm::vec2 whole{0.0f, 0.0f};
    const glm::vec2 off{0.3f, 0.6f};
    const Case cases[] = {
        {1.0f / 2, whole, 1, 0.1},  {1.0f / 4, whole, 1, 0.1},
        {1.0f / 8, whole, 1, 0.25}, {1.0f / 16, whole, 1, 0.25},
        {1.0f / 2, off, 2, 0.1},    {1.0f / 3, off, 2, 0.1},
        {1.0f / 5, off, 32, 1.6},   {1.0f / 8, off, 32, 1.6},
        {1.0f / 12, off, 32, 1.6},
    };
    for (const Case& c : cases) {
      SCOPED_TRACE(::testing::Message()
                   << "zoom 1/" << 1.0f / c.zoom << ", origin " << c.origin.x
                   << ", " << c.origin.y);
      const VkExtent2D extent{
          static_cast<uint32_t>(std::ceil(kSize.width * c.zoom + c.origin.x)),
          static_cast<uint32_t>(std::ceil(kSize.height * c.zoom + c.origin.y))};
      const Pixels drawn = draw_view(
          extent, draw_of(set, {kSize.width, kSize.height}, c.zoom, c.origin),
          pyramid.get());
      const Error e = minified_error(drawn, lin, c.zoom, c.origin);
      EXPECT_GT(e.compared, 0u);
      EXPECT_LE(e.max, c.max);
      EXPECT_LE(e.mean, c.mean);
    }
  }
}

// --- Parts, clipping, pictures ----------------------------------------------

// A uv part draws that part alone, at its own place; pixels outside the
// draw's viewport keep the clear.
TEST_F(TextureViewTest, APartDrawsThatPartAloneInsideItsViewport) {
  constexpr VkExtent2D kSize{64, 32};
  const Rgb card = make_card(kSize.width, kSize.height);
  PictureTexture picture = picture_texture({&card});
  ASSERT_TRUE(picture.set);
  const VkDescriptorSet set = picture.set->set();
  const Pixels whole =
      draw_view(kSize, draw_of(set, {kSize.width, kSize.height}, 1.0f));

  // The middle half of each axis, at (8, 4).
  pipelines::TextureViewDraw part = draw_of(set, {32.0f, 16.0f}, 1.0f, {8, 4});
  part.uv = {0.25f, 0.25f, 0.75f, 0.75f};
  part.viewport = {{0, 0}, kSize};
  const Pixels drawn = draw_view(kSize, part);
  for (uint32_t y = 0; y < kSize.height; ++y) {
    for (uint32_t x = 0; x < kSize.width; ++x) {
      const bool inside = x >= 8 && x < 40 && y >= 4 && y < 20;
      for (int c = 0; c < 3; ++c) {
        const int want =
            inside ? whole.at(x - 8 + 16, y - 4 + 8)[c] : kClear[c];
        ASSERT_NEAR(drawn.at(x, y)[c], want, 1) << x << ", " << y;
      }
    }
  }

  // The whole picture, clipped to a viewport of (10, 6) to (30, 20).
  pipelines::TextureViewDraw clipped =
      draw_of(set, {kSize.width, kSize.height}, 1.0f);
  clipped.viewport = {{10, 6}, {20, 14}};
  const Pixels kept = draw_view(kSize, clipped);
  for (uint32_t y = 0; y < kSize.height; ++y) {
    for (uint32_t x = 0; x < kSize.width; ++x) {
      const bool inside = x >= 10 && x < 30 && y >= 6 && y < 20;
      for (int c = 0; c < 3; ++c) {
        ASSERT_NEAR(kept.at(x, y)[c], inside ? whole.at(x, y)[c] : kClear[c], 1)
            << x << ", " << y;
      }
    }
  }
}

// Over two pictures, uv v in [1, 2) draws picture 1, at 1:1 and minified
// through its own pyramid, as a set over picture 1 alone draws it.
TEST_F(TextureViewTest, TwoPicturesEachDrawsItsOwn) {
  constexpr VkExtent2D kSize{128, 64};
  const Rgb first = make_card(kSize.width, kSize.height, 0);
  const Rgb second = make_card(kSize.width, kSize.height, 5);
  PictureTexture both = picture_texture({&first, &second});
  PictureTexture alone = picture_texture({&second});
  ASSERT_TRUE(both.set && alone.set);
  std::unique_ptr<pipelines::PicturePyramid> two = make_pyramid(kSize, 2);
  std::unique_ptr<pipelines::PicturePyramid> one = make_pyramid(kSize, 1);
  ASSERT_TRUE(two && one);
  const glm::vec2 size{kSize.width, kSize.height};
  for (const float zoom : {1.0f, 1.0f / 8}) {
    SCOPED_TRACE(::testing::Message() << "zoom " << zoom);
    const VkExtent2D extent{static_cast<uint32_t>(kSize.width * zoom),
                            static_cast<uint32_t>(kSize.height * zoom)};
    const Pixels drawn = draw_view(
        extent, draw_of(both.set->set(), size, zoom, {0, 0}, 1), two.get());
    const Pixels expected =
        draw_view(extent, draw_of(alone.set->set(), size, zoom), one.get());
    ASSERT_EQ(drawn.bytes.size(), expected.bytes.size());
    for (size_t i = 0; i < drawn.bytes.size(); ++i) {
      ASSERT_NEAR(drawn.bytes[i], expected.bytes[i], 1) << "byte " << i;
    }
  }
}

// A picture in padded planes, whose padding alternates between the extremes,
// draws minified as the same picture in exact planes: neither the taps nor
// the pyramid read past the picture.
TEST_F(TextureViewTest, NothingPastThePictureIsRead) {
  constexpr VkExtent2D kSize{96, 48};
  const Rgb card = make_card(kSize.width, kSize.height);
  PictureTexture exact = picture_texture({&card});
  PictureTexture padded = picture_texture({&card}, 16);
  ASSERT_TRUE(exact.set && padded.set);
  const glm::vec2 size{kSize.width, kSize.height};
  for (const float zoom : {1.0f / 3, 1.0f / 8, 1.0f / 16}) {
    SCOPED_TRACE(::testing::Message() << "zoom 1/" << 1.0f / zoom);
    const VkExtent2D extent{
        static_cast<uint32_t>(std::ceil(kSize.width * zoom)),
        static_cast<uint32_t>(std::ceil(kSize.height * zoom))};
    std::unique_ptr<pipelines::PicturePyramid> a = make_pyramid(kSize, 1);
    std::unique_ptr<pipelines::PicturePyramid> b = make_pyramid(kSize, 1);
    ASSERT_TRUE(a && b);
    const Pixels want =
        draw_view(extent, draw_of(exact.set->set(), size, zoom), a.get());
    const Pixels got =
        draw_view(extent, draw_of(padded.set->set(), size, zoom), b.get());
    ASSERT_EQ(want.bytes.size(), got.bytes.size());
    for (size_t i = 0; i < want.bytes.size(); ++i) {
      ASSERT_NEAR(got.bytes[i], want.bytes[i], 1) << "byte " << i;
    }
  }
}

// A draw with no set, or a scale or size that is not positive and finite,
// draws nothing; so does a pyramid discarded before any draw reads it, whose
// draw reads the picture alone.
TEST_F(TextureViewTest, DegenerateDrawsDrawNothing) {
  constexpr VkExtent2D kSize{32, 16};
  const Rgb card = make_card(kSize.width, kSize.height);
  PictureTexture picture = picture_texture({&card});
  ASSERT_TRUE(picture.set);
  const glm::vec2 size{kSize.width, kSize.height};
  std::vector<pipelines::TextureViewDraw> draws;
  draws.push_back(draw_of(VK_NULL_HANDLE, size, 1.0f));
  for (const float bad : {0.0f, -1.0f, NAN, INFINITY}) {
    draws.push_back(draw_of(picture.set->set(), size, bad));
    draws.push_back(draw_of(picture.set->set(), {bad, size.y}, 1.0f));
  }
  for (pipelines::TextureViewDraw& d : draws) {
    d.viewport = {{0, 0}, kSize};
  }
  const Pixels drawn =
      render(kSize, kTarget, nullptr, [&](VkCommandBuffer cmd) {
        view_->submit(cmd, kSize, draws.data(),
                      static_cast<uint32_t>(draws.size()));
        view_->submit(cmd, kSize, nullptr, 3);
      });
  for (uint32_t y = 0; y < kSize.height; ++y) {
    for (uint32_t x = 0; x < kSize.width; ++x) {
      for (int c = 0; c < 3; ++c) {
        ASSERT_EQ(drawn.at(x, y)[c], kClear[c]);
      }
    }
  }

  std::unique_ptr<pipelines::PicturePyramid> pyramid = make_pyramid(kSize, 1);
  ASSERT_NE(pyramid, nullptr);
  EXPECT_FALSE(pyramid->built());
  pipelines::TextureViewDraw small = draw_of(picture.set->set(), size, 0.25f);
  small.viewport = {{0, 0}, {8, 4}};
  small.pyramid = pyramid.get();
  const Pixels unbuilt =
      render({8, 4}, kTarget, nullptr,
             [&](VkCommandBuffer c) { view_->submit(c, {8, 4}, &small, 1); });
  EXPECT_NE(unbuilt.at(4, 2)[2], kClear[2]);
}

TEST_F(TextureViewTest, CreateRefusesWhatItCannotDraw) {
  pipelines::HybridMeshPipeline empty;
  EXPECT_EQ(pipelines::HybridMeshTextureView::create(device(), empty,
                                                     layout_of(kTarget))
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
  vg::RenderTargetLayout two = layout_of(kTarget);
  two.color_formats[1] = kTarget;
  two.color_count = 2;
  EXPECT_EQ(pipelines::HybridMeshTextureView::create(device(), *pipeline_, two)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
  vg::RenderTargetLayout none = layout_of(kTarget);
  none.color_count = 0;
  EXPECT_EQ(pipelines::HybridMeshTextureView::create(device(), *pipeline_, none)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
  for (const auto& [size, count] :
       {std::pair<VkExtent2D, uint32_t>{{0, 8}, 1},
        {{8, 0}, 1},
        {{8, 8}, 0},
        {{8, 8}, pipelines::HybridMeshTextureSet::kMaxPictures + 1}}) {
    EXPECT_EQ(
        pipelines::PicturePyramid::create(*view_, allocator(), size, count)
            .status()
            .domain(),
        vkc::Status::Code::InvalidArgument);
  }
  auto pyramid =
      pipelines::PicturePyramid::create(*view_, allocator(), {640, 360}, 2);
  ASSERT_TRUE(pyramid.ok()) << pyramid.status().message();
  EXPECT_EQ(pyramid.value()->size().width, 640u);
  EXPECT_EQ(pyramid.value()->count(), 2u);
  // Levels 1 (320 x 180) down to 1 x 1.
  EXPECT_EQ(pyramid.value()->levels(), 9u);
}

// texture_view_draw places the part through the view: its origin, scale and
// rounded viewport.
TEST_F(TextureViewTest, TextureViewDrawFollowsTheView) {
  vg::camera::ImageView2D image({200.0f, 100.0f}, {10.4f, 20.6f},
                                {400.0f, 300.0f});
  image.zoom_about({110.0f, 120.0f}, 3.0f);
  const VkDescriptorSet set = reinterpret_cast<VkDescriptorSet>(uintptr_t{8});
  const pipelines::TextureViewDraw draw =
      pipelines::texture_view_draw(set, nullptr, image, {0.0f, 50.0f},
                                   {200.0f, 50.0f}, {0.0f, 1.0f, 1.0f, 2.0f});
  EXPECT_EQ(draw.set, set);
  EXPECT_EQ(draw.pyramid, nullptr);
  EXPECT_FLOAT_EQ(draw.scale, image.scale());
  EXPECT_FLOAT_EQ(draw.origin.x, image.image_to_target({0.0f, 50.0f}).x);
  EXPECT_FLOAT_EQ(draw.origin.y, image.image_to_target({0.0f, 50.0f}).y);
  EXPECT_EQ(draw.size, glm::vec2(200.0f, 50.0f));
  EXPECT_EQ(draw.uv, glm::vec4(0.0f, 1.0f, 1.0f, 2.0f));
  EXPECT_EQ(draw.viewport.offset.x, 10);
  EXPECT_EQ(draw.viewport.offset.y, 21);
  EXPECT_EQ(draw.viewport.extent.width, 400u);
  EXPECT_EQ(draw.viewport.extent.height, 300u);
}

// --- HybridMeshFrame::viewport ----------------------------------------------

// A mesh drawn with a viewport fills that rectangle alone -- the quad over
// its whole NDC square lands exactly there -- and leaves the rest of the
// target clear; one wholly outside the target draws nothing.
TEST_F(TextureViewTest, AMeshWithAViewportLeavesTheRestClear) {
  constexpr VkExtent2D kSize{48, 32};
  constexpr VkExtent2D kTargetSize{80, 40};
  const Rgb card = make_card(kSize.width, kSize.height);
  PictureTexture picture = picture_texture({&card});
  ASSERT_TRUE(picture.set);
  const VkDescriptorSet set = picture.set->set();
  const Pixels whole = render(kSize, kTarget, nullptr, [&](VkCommandBuffer c) {
    draw_mesh(c, *pipeline_, set, kSize, 0);
  });
  const VkRect2D at{{20, 5}, kSize};
  const Pixels placed =
      render(kTargetSize, kTarget, nullptr, [&](VkCommandBuffer c) {
        draw_mesh(c, *pipeline_, set, kTargetSize, 0, at);
      });
  for (uint32_t y = 0; y < kTargetSize.height; ++y) {
    for (uint32_t x = 0; x < kTargetSize.width; ++x) {
      const bool inside = x >= 20 && x < 68 && y >= 5 && y < 37;
      for (int c = 0; c < 3; ++c) {
        ASSERT_NEAR(placed.at(x, y)[c],
                    inside ? whole.at(x - 20, y - 5)[c] : kClear[c], 1)
            << x << ", " << y;
      }
    }
  }
  const Pixels outside =
      render(kTargetSize, kTarget, nullptr, [&](VkCommandBuffer c) {
        draw_mesh(c, *pipeline_, set, kTargetSize, 0, {{100, 0}, kSize});
      });
  for (size_t i = 0; i < outside.bytes.size(); i += 4) {
    ASSERT_EQ(outside.bytes[i + 2], kClear[2]);
  }
}

}  // namespace

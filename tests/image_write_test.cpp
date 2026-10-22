// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImagePipeline's region writes: several sources written into one picture
// and converted once. Each tile equals its own texture; edge chroma takes
// exactly its documented share; the gutter is black; a later write keeps the
// other tiles; mips keep tiles apart across a gutter; draws between a write
// and its convert show the previous picture; a producer may refill a source
// right after the write; discard() recovers from a dropped command buffer;
// sources written on another queue family are read in place; any number of
// regions of mixed sizes, up to a 2 x 2 4K mosaic; abutting regions,
// odd NV12 edges among them, make the picture one update would; every
// refusal records nothing; and image_draw bounds what any view gives it. Runs
// under the validation layer with synchronization validation; skips without a
// device, and the cross-family case without two queue families.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <glm/vec2.hpp>

#include "gfx_test_support.hpp"
#include "image_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/core/mip_chain.hpp"
#include "volumetric_kit/gfx/pipelines/image_pipeline.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

using vg_test::Pixels;
using M = pipelines::ImageMapping;

constexpr VkFormat kLuma = VK_FORMAT_R8_UNORM;
constexpr VkFormat kChroma = VK_FORMAT_R8G8_UNORM;

// A host NV12 picture: luma rows, then interleaved Cb, Cr rows at half size.
struct Nv12 {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> luma;
  std::vector<uint8_t> chroma;

  VkExtent2D extent() const { return {width, height}; }
  VkExtent2D chroma_extent() const {
    return {(width + 1) / 2, (height + 1) / 2};
  }
};

// A picture whose every texel differs from its neighbours, `seed` apart from
// another's; `neutral` keeps its chroma at 128.
Nv12 pattern(uint32_t width, uint32_t height, uint32_t seed,
             bool neutral = false) {
  Nv12 p;
  p.width = width;
  p.height = height;
  p.luma.resize(static_cast<size_t>(width) * height);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      p.luma[static_cast<size_t>(y) * width + x] =
          static_cast<uint8_t>(16 + (x * 7 + y * 3 + seed * 50) % 220);
    }
  }
  const VkExtent2D c = p.chroma_extent();
  p.chroma.resize(static_cast<size_t>(c.width) * c.height * 2);
  for (uint32_t y = 0; y < c.height; ++y) {
    for (uint32_t x = 0; x < c.width; ++x) {
      const size_t i = (static_cast<size_t>(y) * c.width + x) * 2;
      p.chroma[i] =
          neutral ? 128 : static_cast<uint8_t>(64 + (x * 5 + seed * 20) % 128);
      p.chroma[i + 1] =
          neutral ? 128 : static_cast<uint8_t>(64 + (y * 9 + seed * 30) % 128);
    }
  }
  return p;
}

// A picture of one color.
Nv12 flat(uint32_t width, uint32_t height, uint8_t y, uint8_t cb, uint8_t cr) {
  Nv12 p;
  p.width = width;
  p.height = height;
  p.luma.assign(static_cast<size_t>(width) * height, y);
  const VkExtent2D c = p.chroma_extent();
  p.chroma.resize(static_cast<size_t>(c.width) * c.height * 2);
  for (size_t i = 0; i < p.chroma.size(); i += 2) {
    p.chroma[i] = cb;
    p.chroma[i + 1] = cr;
  }
  return p;
}

// The `size` texels of `p` at `at`, an even offset: their luma, and the
// chroma texels that cover them.
Nv12 crop(const Nv12& p, VkOffset2D at, VkExtent2D size) {
  const uint32_t x0 = static_cast<uint32_t>(at.x);
  const uint32_t y0 = static_cast<uint32_t>(at.y);
  Nv12 c;
  c.width = size.width;
  c.height = size.height;
  for (uint32_t y = 0; y < size.height; ++y) {
    const uint8_t* row = &p.luma[static_cast<size_t>(y0 + y) * p.width + x0];
    c.luma.insert(c.luma.end(), row, row + size.width);
  }
  const uint32_t pitch = p.chroma_extent().width;
  const VkExtent2D ce = c.chroma_extent();
  for (uint32_t y = 0; y < ce.height; ++y) {
    const uint8_t* row =
        &p.chroma[(static_cast<size_t>(y0 / 2 + y) * pitch + x0 / 2) * 2];
    c.chroma.insert(c.chroma.end(), row, row + ce.width * 2);
  }
  return c;
}

// `rows` of `width` texels of `texel` bytes, laid out from byte `offset`
// with `row_length` texels a row, padding filled with 0xEE.
std::vector<uint8_t> padded(const std::vector<uint8_t>& rows, uint32_t width,
                            uint32_t texel, uint32_t row_length,
                            size_t offset) {
  const size_t height = rows.size() / (static_cast<size_t>(width) * texel);
  std::vector<uint8_t> out(offset + height * row_length * texel, 0xEE);
  for (size_t y = 0; y < height; ++y) {
    std::memcpy(&out[offset + y * row_length * texel], &rows[y * width * texel],
                static_cast<size_t>(width) * texel);
  }
  return out;
}

// Whether `a`'s region at `(ax, ay)` equals `b`'s at `(bx, by)` over
// `w x h`, every channel within `tolerance`; reports the first difference.
::testing::AssertionResult same_region(const Pixels& a, uint32_t ax,
                                       uint32_t ay, const Pixels& b,
                                       uint32_t bx, uint32_t by, uint32_t w,
                                       uint32_t h, int tolerance = 1) {
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const uint8_t* pa = a.at(ax + x, ay + y);
      const uint8_t* pb = b.at(bx + x, by + y);
      for (uint32_t c = 0; c < 4; ++c) {
        if (std::abs(int{pa[c]} - int{pb[c]}) > tolerance) {
          return ::testing::AssertionFailure()
                 << "texel (" << ax + x << ", " << ay + y << ") channel " << c
                 << " is " << int{pa[c]} << ", not " << int{pb[c]};
        }
      }
    }
  }
  return ::testing::AssertionSuccess();
}

// Whether every texel of `px` in `w x h` at `(x0, y0)` is `rgb`, within
// `tolerance`.
::testing::AssertionResult all_rgb(const Pixels& px, uint32_t x0, uint32_t y0,
                                   uint32_t w, uint32_t h, int r, int g, int b,
                                   int tolerance = 0) {
  for (uint32_t y = y0; y < y0 + h; ++y) {
    for (uint32_t x = x0; x < x0 + w; ++x) {
      const uint8_t* p = px.at(x, y);
      if (std::abs(p[0] - r) > tolerance || std::abs(p[1] - g) > tolerance ||
          std::abs(p[2] - b) > tolerance) {
        return ::testing::AssertionFailure()
               << "texel (" << x << ", " << y << ") is " << int{p[0]} << ", "
               << int{p[1]} << ", " << int{p[2]} << ", not " << r << ", " << g
               << ", " << b;
      }
    }
  }
  return ::testing::AssertionSuccess();
}

// A grid of `count` cells, `columns` = ceil(sqrt(count)) wide, each `cell`
// texels plus `gutter`, unaligned: the picture's extent and each cell's
// top-left.
struct Grid {
  VkExtent2D extent{};
  std::vector<VkOffset2D> origins;
};

Grid grid(uint32_t count, VkExtent2D cell, uint32_t gutter) {
  uint32_t columns = 1;
  while (columns * columns < count) {
    ++columns;
  }
  const uint32_t rows = (count + columns - 1) / columns;
  Grid g;
  g.extent = {columns * cell.width + (columns - 1) * gutter,
              rows * cell.height + (rows - 1) * gutter};
  for (uint32_t i = 0; i < count; ++i) {
    g.origins.push_back(
        {static_cast<int32_t>((i % columns) * (cell.width + gutter)),
         static_cast<int32_t>((i / columns) * (cell.height + gutter))});
  }
  return g;
}

// The plan's mosaic: a 2 x 2 of 64 x 48 tiles with a 16-texel gutter.
constexpr VkExtent2D kTile{64, 48};
constexpr uint32_t kGutter = 16;

class ImageWriteTest : public vg_test::RendererDeviceTest,
                       protected vg_test::ImageHelpers {
 protected:
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::ShaderAccesses;
  }
  vkc::Device& image_device() override { return device(); }
  vkc::Allocator& image_allocator() override { return allocator(); }

  // A picture's planes as device buffers, tightly packed.
  struct BufferPlanes {
    vkc::Buffer luma;
    vkc::Buffer chroma;
  };
  BufferPlanes buffers_of(const Nv12& p) {
    return {source_buffer(p.luma), source_buffer(p.chroma)};
  }

  // A picture's planes as device images, left SHADER_READ until a command
  // buffer readies them (ready).
  struct ImagePlanes {
    vkc::Image luma;
    vkc::Image chroma;
    void ready(VkCommandBuffer cmd) {
      ready_for_copy(cmd, luma);
      ready_for_copy(cmd, chroma);
    }
  };
  ImagePlanes images_of(const Nv12& p) {
    return {source_image(kLuma, p.extent(), p.luma),
            source_image(kChroma, p.chroma_extent(), p.chroma)};
  }

  static pipelines::ImageWrite write_of(const vkc::Buffer& luma,
                                        const vkc::Buffer& chroma,
                                        VkOffset2D at, VkExtent2D extent) {
    pipelines::ImageWrite w;
    w.planes[0].buffer = &luma;
    w.planes[1].buffer = &chroma;
    w.offset = at;
    w.extent = extent;
    return w;
  }
  static pipelines::ImageWrite write_of(const BufferPlanes& planes,
                                        VkOffset2D at, VkExtent2D extent) {
    return write_of(planes.luma, planes.chroma, at, extent);
  }
  static pipelines::ImageWrite write_of(const ImagePlanes& planes,
                                        VkOffset2D at, VkExtent2D extent) {
    pipelines::ImageWrite w;
    w.planes[0].image = &planes.luma;
    w.planes[1].image = &planes.chroma;
    w.offset = at;
    w.extent = extent;
    return w;
  }

  // Writes `writes` and converts, expecting both to be recorded.
  void write_and_convert(VkCommandBuffer cmd, const pipelines::ImagePipeline& p,
                         pipelines::ImageTexture& texture,
                         const std::vector<pipelines::ImageWrite>& writes,
                         const pipelines::ImageConversion& conversion = {}) {
    const vkc::Status w = p.record_write(cmd, texture, writes.data(),
                                         static_cast<uint32_t>(writes.size()));
    EXPECT_TRUE(w.ok()) << w.message();
    const vkc::Status c = p.record_convert(cmd, texture, conversion);
    EXPECT_TRUE(c.ok()) << c.message();
  }
};

// Test 1: four tiles -- two from images (one larger than its tile, as a
// decoder's padded surface is), one from tight buffers, one from buffers with
// an offset and padded rows -- in one write and one convert: each tile's
// interior equals its own texture updated from the same source.
TEST_F(ImageWriteTest, EachTileEqualsItsOwnUpdate) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  ASSERT_TRUE(pipeline.valid());
  const Grid g = grid(4, kTile, kGutter);
  ASSERT_EQ(g.extent.width, 144u);
  ASSERT_EQ(g.extent.height, 112u);
  pipelines::ImageTexture mosaic =
      make_texture(pipeline, g.extent, M::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(mosaic.valid());

  std::array<Nv12, 4> pictures;
  for (uint32_t i = 0; i < 4; ++i) {
    pictures[i] = pattern(kTile.width, kTile.height, i);
  }
  ImagePlanes image0 = images_of(pictures[0]);
  // A padded surface: 70 x 50, of which the tile is the top-left 64 x 48.
  Nv12 big = pattern(70, 50, 1);
  for (uint32_t y = 0; y < kTile.height; ++y) {
    std::memcpy(&big.luma[y * 70], &pictures[1].luma[y * kTile.width],
                kTile.width);
  }
  for (uint32_t y = 0; y < kTile.height / 2; ++y) {
    std::memcpy(&big.chroma[y * 35 * 2],
                &pictures[1].chroma[y * (kTile.width / 2) * 2], kTile.width);
  }
  ImagePlanes image1 = images_of(big);
  const BufferPlanes tight = buffers_of(pictures[2]);
  const vkc::Buffer padded_luma =
      source_buffer(padded(pictures[3].luma, kTile.width, 1, 70, 8));
  const vkc::Buffer padded_chroma =
      source_buffer(padded(pictures[3].chroma, kTile.width / 2, 2, 40, 4));
  pipelines::ImageWrite strided =
      write_of(padded_luma, padded_chroma, g.origins[3], kTile);
  strided.planes[0].offset = 8;
  strided.planes[0].row_length = 70;
  strided.planes[1].offset = 4;
  strided.planes[1].row_length = 40;
  const std::vector<pipelines::ImageWrite> writes = {
      write_of(image0, g.origins[0], kTile),
      write_of(image1, g.origins[1], kTile),
      write_of(tight, g.origins[2], kTile), strided};

  std::array<pipelines::ImageTexture, 4> own;
  for (pipelines::ImageTexture& t : own) {
    t = make_texture(pipeline, kTile, M::Nv12, VK_FORMAT_UNDEFINED);
    ASSERT_TRUE(t.valid());
  }
  const Pixels whole = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
    image0.ready(cmd);
    image1.ready(cmd);
    write_and_convert(cmd, pipeline, mosaic, writes);
    for (uint32_t i = 0; i < 4; ++i) {
      pipelines::ImageUpdate update;
      update.planes = writes[i].planes;
      const vkc::Status s = pipeline.record_update(cmd, own[i], update);
      EXPECT_TRUE(s.ok()) << s.message();
    }
  });
  ASSERT_EQ(whole.bytes.size(), 144u * 112 * 4);
  for (uint32_t i = 0; i < 4; ++i) {
    SCOPED_TRACE("tile " + std::to_string(i));
    const Pixels alone = read_level(own[i], 0);
    ASSERT_FALSE(alone.bytes.empty());
    // The outermost ring takes its neighbours' chroma (EdgeChroma...).
    const uint32_t ox = static_cast<uint32_t>(g.origins[i].x);
    const uint32_t oy = static_cast<uint32_t>(g.origins[i].y);
    EXPECT_TRUE(same_region(whole, ox + 1, oy + 1, alone, 1, 1, kTile.width - 2,
                            kTile.height - 2));
  }
}

// Test 2: a tile of one saturated color in a neutral gutter. Its interior is
// exact; each outermost texel takes the documented share of the gutter's
// neutral chroma -- 1/4 in the top and bottom rows, 1/4 in the side columns
// when centered and 0 left / 1/2 right when left-sited, both at a corner --
// and none where the tile meets the picture's edge; and the gutter texel
// beside each edge takes the mirrored share of the tile's.
TEST_F(ImageWriteTest, EdgeChromaTakesItsDocumentedShare) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  // Tile A on the picture's left edge, tile B with gutter on every side.
  pipelines::ImageTexture texture =
      make_texture(pipeline, {48, 24}, M::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(texture.valid());
  const BufferPlanes red = buffers_of(flat(16, 8, 128, 128, 200));
  const std::vector<pipelines::ImageWrite> writes = {
      write_of(red, {0, 8}, {16, 8}), write_of(red, {24, 8}, {16, 8})};

  // R' = Y' + 1.402 Cr': 229 for this tile, 128 for the neutral gutter, and
  // linear in Cr between them, so a share s of the gutter's gives
  // 229 - 101 s.
  for (pipelines::ChromaSiting siting :
       {pipelines::ChromaSiting::Center, pipelines::ChromaSiting::Left}) {
    const bool left = siting == pipelines::ChromaSiting::Left;
    SCOPED_TRACE(left ? "left-sited" : "centered");
    pipelines::ImageConversion conversion;
    conversion.chroma_siting = siting;
    const Pixels px = read_level(texture, 0, [&](VkCommandBuffer cmd) {
      write_and_convert(cmd, pipeline, texture, writes, conversion);
    });
    ASSERT_FALSE(px.bytes.empty());
    const int own = px.at(28, 12)[0];
    EXPECT_NEAR(own, 229, 2) << "an interior texel";
    const auto expect_share = [&](uint32_t x, uint32_t y, double share,
                                  const char* what) {
      EXPECT_NEAR(px.at(x, y)[0], own - share * (own - 128), 1.5)
          << what << " at (" << x << ", " << y << ")";
    };
    // Tile B's interior is exact, every texel of it.
    EXPECT_TRUE(
        all_rgb(px, 25, 9, 14, 6, own, px.at(28, 12)[1], px.at(28, 12)[2], 1));
    const double side = left ? 0.0 : 0.25;  // left column
    const double right = left ? 0.5 : 0.25;
    expect_share(24, 12, side, "B's left column");
    expect_share(39, 12, right, "B's right column");
    expect_share(30, 8, 0.25, "B's top row");
    expect_share(30, 15, 0.25, "B's bottom row");
    expect_share(24, 8, 1.0 - (1.0 - side) * 0.75, "B's top-left corner");
    expect_share(39, 15, 1.0 - (1.0 - right) * 0.75, "B's bottom-right corner");
    // Tile A's left column is on the picture's edge: it takes none.
    expect_share(0, 12, 0.0, "A's left column");
    expect_share(15, 12, right, "A's right column");
    // The gutter's luma is 0, so R' is the tile's share alone: 101 s.
    const auto expect_tint = [&](uint32_t x, uint32_t y, double share,
                                 const char* what) {
      EXPECT_NEAR(px.at(x, y)[0], share * (own - 128), 1.5)
          << what << " at (" << x << ", " << y << ")";
    };
    // Mirrored: the texel left of B takes what B's right column does.
    expect_tint(23, 12, right, "the gutter left of B");
    expect_tint(40, 12, side, "the gutter right of B");
    expect_tint(30, 7, 0.25, "the gutter above B");
    expect_tint(30, 16, 0.25, "the gutter below B");
  }
}

// Test 3: texels no write covers are black -- for NV12 at both ranges, but
// for the texel beside a region, and for a grey picture -- after a fresh
// texture's first write and after a discard, when the planes still hold an
// earlier picture.
TEST_F(ImageWriteTest, GutterAndUnwrittenTexelsAreBlack) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  const Grid g = grid(4, kTile, kGutter);
  pipelines::ImageTexture mosaic =
      make_texture(pipeline, g.extent, M::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(mosaic.valid());
  const BufferPlanes white = buffers_of(flat(144, 112, 255, 128, 128));
  const BufferPlanes tile = buffers_of(pattern(kTile.width, kTile.height, 0));
  // Tiles 0 and 3 only: tiles 1 and 2 are unwritten.
  const std::vector<pipelines::ImageWrite> two = {
      write_of(tile, g.origins[0], kTile), write_of(tile, g.origins[3], kTile)};

  for (bool full_range : {true, false}) {
    SCOPED_TRACE(full_range ? "full range" : "limited range");
    pipelines::ImageConversion conversion;
    conversion.full_range = full_range;
    for (bool after_discard : {false, true}) {
      SCOPED_TRACE(after_discard ? "after a discard" : "first write");
      if (after_discard) {
        // A white picture that ran, then forgotten.
        const Pixels earlier = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
          write_and_convert(cmd, pipeline, mosaic,
                            {write_of(white, {0, 0}, g.extent)}, conversion);
        });
        ASSERT_FALSE(earlier.bytes.empty());
        ASSERT_TRUE(all_rgb(earlier, 0, 0, 144, 112, 255, 255, 255, 1));
        mosaic.discard();
      }
      const Pixels px = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
        write_and_convert(cmd, pipeline, mosaic, two, conversion);
      });
      ASSERT_FALSE(px.bytes.empty());
      // The gutter texels beside a tile take a share of its chroma (a tint
      // at zero luma, as the header says); the rest of the gutter is black.
      EXPECT_TRUE(all_rgb(px, 65, 0, 14, 112, 0, 0, 0)) << "the gutter column";
      EXPECT_TRUE(all_rgb(px, 0, 49, 144, 14, 0, 0, 0)) << "the gutter row";
      EXPECT_TRUE(all_rgb(px, 80, 0, 64, 48, 0, 0, 0)) << "tile 1";
      EXPECT_TRUE(all_rgb(px, 0, 64, 64, 48, 0, 0, 0)) << "tile 2";
      EXPECT_GT(px.at(10, 10)[0] + px.at(10, 10)[1] + px.at(10, 10)[2], 0)
          << "tile 0 is written";
      mosaic.discard();
    }
  }

  // A grey picture is cleared to 0 too.
  pipelines::ImageTexture grey =
      make_texture(pipeline, {8, 8}, M::Grey, VK_FORMAT_R8_UNORM);
  const vkc::Buffer bright = source_buffer(std::vector<uint8_t>(16, 200));
  pipelines::ImageWrite middle;
  middle.planes[0].buffer = &bright;
  middle.offset = {2, 2};
  middle.extent = {4, 4};
  const Pixels px = read_level(grey, 0, [&](VkCommandBuffer cmd) {
    write_and_convert(cmd, pipeline, grey, {middle});
  });
  ASSERT_FALSE(px.bytes.empty());
  EXPECT_TRUE(all_rgb(px, 2, 2, 4, 4, 200, 200, 200, 1));
  EXPECT_TRUE(all_rgb(px, 0, 0, 8, 2, 0, 0, 0));
  EXPECT_TRUE(all_rgb(px, 0, 6, 8, 2, 0, 0, 0));
  EXPECT_TRUE(all_rgb(px, 0, 2, 2, 4, 0, 0, 0));
  EXPECT_TRUE(all_rgb(px, 6, 2, 2, 4, 0, 0, 0));
}

// Test 4: a later write of tile 1 alone keeps tiles 0, 2 and 3 and the gutter,
// whether it follows a convert or another write not yet converted.
TEST_F(ImageWriteTest, ALaterWriteOfOneTileKeepsTheOthers) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  const Grid g = grid(4, kTile, kGutter);
  pipelines::ImageTexture mosaic =
      make_texture(pipeline, g.extent, M::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(mosaic.valid());
  std::vector<BufferPlanes> first;
  std::vector<pipelines::ImageWrite> all;
  first.reserve(4);
  for (uint32_t i = 0; i < 4; ++i) {
    first.push_back(buffers_of(pattern(kTile.width, kTile.height, i)));
  }
  for (uint32_t i = 0; i < 4; ++i) {
    all.push_back(write_of(first[i], g.origins[i], kTile));
  }
  const BufferPlanes second = buffers_of(pattern(kTile.width, kTile.height, 7));
  const BufferPlanes third = buffers_of(pattern(kTile.width, kTile.height, 9));

  const Pixels before = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
    write_and_convert(cmd, pipeline, mosaic, all);
  });
  ASSERT_FALSE(before.bytes.empty());
  const Pixels after = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
    write_and_convert(cmd, pipeline, mosaic,
                      {write_of(second, g.origins[1], kTile)});
  });
  ASSERT_FALSE(after.bytes.empty());
  // Tile 1 changed; the rest, gutter included, is the same picture, but for
  // the edge chroma beside tile 1, which its new chroma changes.
  EXPECT_FALSE(same_region(after, 81, 1, before, 81, 1, 62, 46));
  EXPECT_TRUE(same_region(after, 0, 0, before, 0, 0, 63, 112, 0));
  EXPECT_TRUE(same_region(after, 0, 49, before, 0, 49, 144, 63, 0));

  // Two writes before one convert: tile 2 in one, tile 1 in the next.
  const Pixels twice = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
    const pipelines::ImageWrite a = write_of(third, g.origins[2], kTile);
    const pipelines::ImageWrite b = write_of(first[1], g.origins[1], kTile);
    ASSERT_TRUE(pipeline.record_write(cmd, mosaic, &a, 1).ok());
    ASSERT_TRUE(pipeline.record_write(cmd, mosaic, &b, 1).ok());
    ASSERT_TRUE(pipeline.record_convert(cmd, mosaic, {}).ok());
  });
  ASSERT_FALSE(twice.bytes.empty());
  EXPECT_TRUE(same_region(twice, 0, 0, before, 0, 0, 144, 47, 0))
      << "tile 1 is back";
  EXPECT_FALSE(same_region(twice, 1, 65, before, 1, 65, 62, 46))
      << "tile 2 is new";
}

// Test 5: with a 16-texel gutter and 16-aligned tiles in a picture whose
// extent is a multiple of 16, levels up to 4 keep tiles and gutter apart:
// white tiles stay white to their edges and the level-4 gutter stays black;
// and each tile's levels 1 to 4 equal those of its own texture.
TEST_F(ImageWriteTest, MipsKeepTilesApartAcrossTheGutter) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  const Grid g = grid(4, kTile, kGutter);
  pipelines::ImageTexture mosaic =
      make_texture(pipeline, g.extent, M::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(mosaic.valid());
  ASSERT_GE(mosaic.mip_levels(), 5u);

  const BufferPlanes white = buffers_of(flat(64, 48, 255, 128, 128));
  std::vector<pipelines::ImageWrite> whites;
  for (const VkOffset2D& at : g.origins) {
    whites.push_back(write_of(white, at, kTile));
  }
  for (uint32_t level = 0; level <= 4; ++level) {
    SCOPED_TRACE("level " + std::to_string(level));
    const Pixels px = read_level(mosaic, level, [&](VkCommandBuffer cmd) {
      if (level == 0) {
        write_and_convert(cmd, pipeline, mosaic, whites);
      }
    });
    ASSERT_FALSE(px.bytes.empty());
    const uint32_t w = kTile.width >> level;
    const uint32_t h = kTile.height >> level;
    for (const VkOffset2D& at : g.origins) {
      EXPECT_TRUE(all_rgb(px, static_cast<uint32_t>(at.x) >> level,
                          static_cast<uint32_t>(at.y) >> level, w, h, 255, 255,
                          255, 1));
    }
    if (level == 4) {
      EXPECT_TRUE(all_rgb(px, 4, 0, 1, 7, 0, 0, 0)) << "the gutter column";
      EXPECT_TRUE(all_rgb(px, 0, 3, 9, 1, 0, 0, 0)) << "the gutter row";
    }
  }

  // Patterned luma on neutral chroma, so level 0 has no edge chroma to tell
  // the tiles from their own textures.
  std::vector<BufferPlanes> sources;
  sources.reserve(4);
  std::vector<pipelines::ImageWrite> tiles;
  std::array<pipelines::ImageTexture, 4> own;
  for (uint32_t i = 0; i < 4; ++i) {
    sources.push_back(buffers_of(pattern(kTile.width, kTile.height, i, true)));
    tiles.push_back(write_of(sources[i], g.origins[i], kTile));
    own[i] = make_texture(pipeline, kTile, M::Nv12, VK_FORMAT_UNDEFINED);
    ASSERT_TRUE(own[i].valid());
  }
  const Pixels unused = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
    write_and_convert(cmd, pipeline, mosaic, tiles);
    for (uint32_t i = 0; i < 4; ++i) {
      pipelines::ImageUpdate update;
      update.planes = tiles[i].planes;
      ASSERT_TRUE(pipeline.record_update(cmd, own[i], update).ok());
    }
  });
  ASSERT_FALSE(unused.bytes.empty());
  for (uint32_t level = 1; level <= 4; ++level) {
    const Pixels px = read_level(mosaic, level);
    for (uint32_t i = 0; i < 4; ++i) {
      SCOPED_TRACE("level " + std::to_string(level) + ", tile " +
                   std::to_string(i));
      const Pixels alone = read_level(own[i], level);
      ASSERT_FALSE(alone.bytes.empty());
      EXPECT_TRUE(
          same_region(px, static_cast<uint32_t>(g.origins[i].x) >> level,
                      static_cast<uint32_t>(g.origins[i].y) >> level, alone, 0,
                      0, kTile.width >> level, kTile.height >> level));
    }
  }
}

// Test 6: a write touches only the planes, so a draw between it and its
// convert shows the previous picture, and one after the convert the new one.
TEST_F(ImageWriteTest, ADrawBetweenWriteAndConvertShowsThePreviousPicture) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture =
      make_texture(pipeline, {4, 4}, M::Grey, VK_FORMAT_R8_UNORM);
  const vkc::Buffer dark = source_buffer(std::vector<uint8_t>(16, 30));
  const vkc::Buffer light = source_buffer(std::vector<uint8_t>(16, 220));
  pipelines::ImageWrite first;
  first.planes[0].buffer = &dark;
  first.extent = {4, 4};
  pipelines::ImageWrite second = first;
  second.planes[0].buffer = &light;

  const std::vector<pipelines::ImageDraw> draws = {
      draw_of(texture, {4, 4}, 1.0f)};
  Pass a;
  a.before = [&](VkCommandBuffer cmd) {
    write_and_convert(cmd, pipeline, texture, {first});
  };
  a.draws = draws;
  Pass b;
  b.before = [&](VkCommandBuffer cmd) {
    ASSERT_TRUE(pipeline.record_write(cmd, texture, &second, 1).ok());
  };
  b.draws = draws;
  Pass c;
  c.before = [&](VkCommandBuffer cmd) {
    ASSERT_TRUE(pipeline.record_convert(cmd, texture, {}).ok());
  };
  c.draws = draws;
  const std::vector<Pixels> px = render_passes(pipeline, {a, b, c}, {4, 4});
  ASSERT_EQ(px.size(), 3u);
  vg_test::expect_rgb(px[0].at(1, 1), 30, 30, 30, 1, "the first picture");
  vg_test::expect_rgb(px[1].at(1, 1), 30, 30, 30, 1, "written, not converted");
  vg_test::expect_rgb(px[2].at(1, 1), 220, 220, 220, 1, "converted");
}

// Test 7: a producer on the same queue writes a tile's source just before the
// write and the next picture just after it, with no barrier of its own: the
// tile is the first write's, and synchronization validation sees no hazard.
TEST_F(ImageWriteTest, AProducerRefillAfterTheWriteDoesNotChangeTheTile) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture =
      make_texture(pipeline, {8, 4}, M::Grey, VK_FORMAT_R8_UNORM);
  const vkc::Buffer source = source_buffer(
      std::vector<uint8_t>(16, 0),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  const auto producer = vg_test::make_word_writer(device(), source);
  ASSERT_NE(producer, nullptr);
  pipelines::ImageWrite right;
  right.planes[0].buffer = &source;
  right.offset = {4, 0};
  right.extent = {4, 4};

  const Pixels px = read_level(texture, 0, [&](VkCommandBuffer cmd) {
    producer->record(cmd, 0x80808080u, 4);
    ASSERT_TRUE(pipeline.record_write(cmd, texture, &right, 1).ok());
    producer->record(cmd, 0xFFFFFFFFu, 4);
    ASSERT_TRUE(pipeline.record_convert(cmd, texture, {}).ok());
  });
  ASSERT_FALSE(px.bytes.empty());
  EXPECT_TRUE(all_rgb(px, 4, 0, 4, 4, 128, 128, 128, 1)) << "the first write";
  EXPECT_TRUE(all_rgb(px, 0, 0, 4, 4, 0, 0, 0)) << "unwritten";
}

// Test 8: a command buffer that wrote and converted the texture is dropped.
// After discard() the texture shows nothing, a convert is refused until a
// write, and the next write clears from scratch: validation finds every
// layout as recorded, and the picture is the new region on black.
TEST_F(ImageWriteTest, DiscardAfterADroppedCommandBufferStartsOver) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture texture =
      make_texture(pipeline, {8, 8}, M::Grey, VK_FORMAT_R8_UNORM);
  const vkc::Buffer full = source_buffer(std::vector<uint8_t>(64, 90));
  const vkc::Buffer region = source_buffer(std::vector<uint8_t>(16, 200));
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &full;

  {
    auto pool =
        vkc::CommandPool::create(device().handle(), device().queue_family());
    ASSERT_TRUE(pool.ok());
    auto dropped = pool.value().allocate_primary();
    ASSERT_TRUE(dropped.ok());
    ASSERT_TRUE(dropped.value().begin().ok());
    ASSERT_TRUE(
        pipeline.record_update(dropped.value().handle(), texture, update).ok());
    ASSERT_TRUE(dropped.value().end().ok());
    EXPECT_TRUE(texture.has_picture());
  }  // never submitted
  texture.discard();
  EXPECT_FALSE(texture.has_picture());
  EXPECT_EQ(texture.display().layout(), VK_IMAGE_LAYOUT_UNDEFINED);

  pipelines::ImageWrite middle;
  middle.planes[0].buffer = &region;
  middle.offset = {2, 2};
  middle.extent = {4, 4};
  const std::vector<pipelines::ImageDraw> draws = {
      draw_of(texture, {8, 8}, 1.0f)};
  Pass written;
  written.before = [&](VkCommandBuffer cmd) {
    EXPECT_EQ(pipeline.record_convert(cmd, texture, {}).domain(),
              vkc::Status::Code::InvalidArgument)
        << "a convert with nothing written";
    ASSERT_TRUE(pipeline.record_write(cmd, texture, &middle, 1).ok());
  };
  written.draws = draws;
  Pass converted;
  converted.before = [&](VkCommandBuffer cmd) {
    ASSERT_TRUE(pipeline.record_convert(cmd, texture, {}).ok());
  };
  converted.draws = draws;
  const std::vector<Pixels> px =
      render_passes(pipeline, {written, converted}, {8, 8});
  ASSERT_EQ(px.size(), 2u);
  EXPECT_TRUE(all_rgb(px[0], 0, 0, 8, 8, 0, 0, 255))
      << "no picture until the convert: the clear color";
  EXPECT_TRUE(all_rgb(px[1], 2, 2, 4, 4, 200, 200, 200, 1));
  EXPECT_TRUE(all_rgb(px[1], 0, 0, 8, 2, 0, 0, 0)) << "cleared, not 90";
  EXPECT_TRUE(texture.has_picture());
}

// Any number of regions of mixed sizes: 1, 3 and 5 sources in cells sized to
// the largest, each region where it was written and the rest black.
TEST_F(ImageWriteTest, AnyNumberOfRegionsOfMixedSizes) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  constexpr VkExtent2D kCell{40, 32};
  for (uint32_t n : {1u, 3u, 5u}) {
    SCOPED_TRACE(std::to_string(n) + " regions");
    const Grid g = grid(n, kCell, kGutter);
    pipelines::ImageTexture texture =
        make_texture(pipeline, g.extent, M::Nv12, VK_FORMAT_UNDEFINED);
    ASSERT_TRUE(texture.valid());
    std::vector<BufferPlanes> sources;
    sources.reserve(n);
    std::vector<pipelines::ImageWrite> writes;
    std::vector<VkExtent2D> sizes;
    for (uint32_t i = 0; i < n; ++i) {
      const VkExtent2D size{kCell.width - 8 * (i % 2),
                            kCell.height - 4 * (i % 3)};
      sizes.push_back(size);
      sources.push_back(
          buffers_of(flat(size.width, size.height,
                          static_cast<uint8_t>(60 + 30 * i), 128, 128)));
      writes.push_back(write_of(sources[i], g.origins[i], size));
    }
    const Pixels px = read_level(texture, 0, [&](VkCommandBuffer cmd) {
      write_and_convert(cmd, pipeline, texture, writes);
    });
    ASSERT_FALSE(px.bytes.empty());
    std::vector<uint8_t> covered(static_cast<size_t>(g.extent.width) *
                                 g.extent.height);
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t ox = static_cast<uint32_t>(g.origins[i].x);
      const uint32_t oy = static_cast<uint32_t>(g.origins[i].y);
      const int y = 60 + 30 * static_cast<int>(i);
      EXPECT_TRUE(
          all_rgb(px, ox, oy, sizes[i].width, sizes[i].height, y, y, y, 1))
          << "region " << i;
      for (uint32_t r = 0; r < sizes[i].height; ++r) {
        std::fill_n(&covered[(oy + r) * g.extent.width + ox], sizes[i].width,
                    1);
      }
    }
    for (uint32_t y = 0; y < g.extent.height; ++y) {
      for (uint32_t x = 0; x < g.extent.width; ++x) {
        if (covered[y * g.extent.width + x] == 0) {
          ASSERT_TRUE(all_rgb(px, x, y, 1, 1, 0, 0, 0));
        }
      }
    }
  }
}

// The rig mosaic at full size: four 4K pictures in a 7696 x 4336 picture with
// a 16-texel gutter, its extent a multiple of 16. Level 4 (481 x 271) holds
// each tile's color inside and black between. Each 4K picture is flat, filled
// on the GPU ahead of the write, so none is made on the host.
TEST_F(ImageWriteTest, ATwoByTwoMosaicOf4KPictures) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  constexpr VkExtent2D k4K{3840, 2160};
  const Grid g = grid(4, k4K, kGutter);
  ASSERT_EQ(g.extent.width, 7696u);
  ASSERT_EQ(g.extent.height, 4336u);
  const VkExtent2D max = pipeline.max_texture_extent();
  if (max.width < g.extent.width || max.height < g.extent.height) {
    GTEST_SKIP() << "the device's largest picture is " << max.width << " x "
                 << max.height;
  }
  pipelines::ImageTexture mosaic =
      make_texture(pipeline, g.extent, M::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(mosaic.valid());

  const std::array<std::array<uint8_t, 3>, 4> colors = {
      {{90, 88, 168}, {150, 168, 88}, {60, 168, 168}, {200, 88, 88}}};
  const auto fillable = [&](VkDeviceSize bytes) {
    vkc::BufferDesc desc;
    desc.size = bytes;
    desc.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto buffer = image_allocator().create_buffer(desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
  };
  const VkDeviceSize luma_bytes = VkDeviceSize{k4K.width} * k4K.height;
  std::vector<BufferPlanes> sources;
  sources.reserve(4);
  std::vector<pipelines::ImageWrite> writes;
  // Each color's display bytes, from a small texture of its own.
  std::array<pipelines::ImageTexture, 4> swatch;
  std::vector<BufferPlanes> swatches;
  swatches.reserve(4);
  for (uint32_t i = 0; i < 4; ++i) {
    const auto& c = colors[i];
    sources.push_back({fillable(luma_bytes), fillable(luma_bytes / 2)});
    writes.push_back(write_of(sources[i], g.origins[i], k4K));
    swatches.push_back(buffers_of(flat(16, 16, c[0], c[1], c[2])));
    swatch[i] = make_texture(pipeline, {16, 16}, M::Nv12, VK_FORMAT_UNDEFINED);
  }
  const Pixels level4 = read_level(mosaic, 4, [&](VkCommandBuffer cmd) {
    for (uint32_t i = 0; i < 4; ++i) {
      // A word of luma bytes, and of Cb, Cr pairs.
      const auto& c = colors[i];
      vkCmdFillBuffer(cmd, sources[i].luma.handle(), 0, VK_WHOLE_SIZE,
                      c[0] * 0x01010101u);
      vkCmdFillBuffer(cmd, sources[i].chroma.handle(), 0, VK_WHOLE_SIZE,
                      c[1] * 0x00010001u + c[2] * 0x01000100u);
    }
    write_and_convert(cmd, pipeline, mosaic, writes);
    for (uint32_t i = 0; i < 4; ++i) {
      write_and_convert(cmd, pipeline, swatch[i],
                        {write_of(swatches[i], {0, 0}, {16, 16})});
    }
  });
  ASSERT_EQ(level4.width, 481u);
  // Tile i covers (origin / 16) + 240 x 135 at level 4; its outermost ring
  // carries the edge chroma.
  for (uint32_t i = 0; i < 4; ++i) {
    SCOPED_TRACE("tile " + std::to_string(i));
    const Pixels own = read_level(swatch[i], 0);
    ASSERT_FALSE(own.bytes.empty());
    const uint8_t* c = own.at(8, 8);
    EXPECT_TRUE(all_rgb(level4, static_cast<uint32_t>(g.origins[i].x) / 16 + 1,
                        static_cast<uint32_t>(g.origins[i].y) / 16 + 1, 238,
                        133, c[0], c[1], c[2], 1));
  }
  // Black, but for the tint the gutter texels beside each tile take from its
  // chroma: 2 of each 16 at a quarter share, under 2/255 here.
  EXPECT_TRUE(all_rgb(level4, 240, 0, 1, 271, 0, 0, 0, 2))
      << "the gutter column";
  EXPECT_TRUE(all_rgb(level4, 0, 135, 481, 1, 0, 0, 0, 2)) << "the gutter row";
}

// Test 11: record_update is a whole-picture write and its convert: the same
// picture, level for level.
TEST_F(ImageWriteTest, UpdateEqualsWriteThenConvert) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  pipelines::ImageTexture updated =
      make_texture(pipeline, {64, 48}, M::Nv12, VK_FORMAT_UNDEFINED);
  pipelines::ImageTexture written =
      make_texture(pipeline, {64, 48}, M::Nv12, VK_FORMAT_UNDEFINED);
  const BufferPlanes source = buffers_of(pattern(64, 48, 3));
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &source.luma;
  update.planes[1].buffer = &source.chroma;
  update.kr = 0.2126f;  // BT.709, limited, left-sited
  update.kb = 0.0722f;
  update.full_range = false;
  update.chroma_siting = pipelines::ChromaSiting::Left;

  const Pixels a = read_level(updated, 0, [&](VkCommandBuffer cmd) {
    ASSERT_TRUE(pipeline.record_update(cmd, updated, update).ok());
    write_and_convert(cmd, pipeline, written,
                      {write_of(source, {0, 0}, {64, 48})}, update);
  });
  ASSERT_FALSE(a.bytes.empty());
  for (uint32_t level = 0; level < updated.mip_levels(); ++level) {
    SCOPED_TRACE("level " + std::to_string(level));
    const Pixels u = level == 0 ? a : read_level(updated, level);
    const Pixels w = read_level(written, level);
    ASSERT_EQ(u.bytes.size(), w.bytes.size());
    EXPECT_EQ(u.bytes, w.bytes);
  }
}

// Regions that abut, with no gutter, and together cover the picture --
// NV12 regions whose odd extents reach the right and bottom edges among them
// -- make the picture one update of it would, level for level: on the first
// write, which clears nothing, and on a second over the converted picture.
TEST_F(ImageWriteTest, AbuttingRegionsMakeThePictureOneUpdateWould) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  struct Region {
    VkOffset2D at;
    VkExtent2D size;
  };
  struct Case {
    const char* what;
    VkExtent2D extent;
    std::vector<Region> regions;
  };
  const std::vector<Case> cases = {
      {"a 2 x 2 of 64 x 48 tiles",
       {128, 96},
       {{{0, 0}, {64, 48}},
        {{64, 0}, {64, 48}},
        {{0, 48}, {64, 48}},
        {{64, 48}, {64, 48}}}},
      {"7 x 5 as 4 x 5 and 3 x 5",
       {7, 5},
       {{{0, 0}, {4, 5}}, {{4, 0}, {3, 5}}}},
      {"7 x 5 as 7 x 2 and 7 x 3",
       {7, 5},
       {{{0, 0}, {7, 2}}, {{0, 2}, {7, 3}}}},
      {"9 x 7 as a 2 x 2",
       {9, 7},
       {{{0, 0}, {4, 4}},
        {{4, 0}, {5, 4}},
        {{0, 4}, {4, 3}},
        {{4, 4}, {5, 3}}}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.what);
    pipelines::ImageTexture written =
        make_texture(pipeline, c.extent, M::Nv12, VK_FORMAT_UNDEFINED);
    pipelines::ImageTexture updated =
        make_texture(pipeline, c.extent, M::Nv12, VK_FORMAT_UNDEFINED);
    ASSERT_TRUE(written.valid());
    ASSERT_TRUE(updated.valid());
    for (uint32_t pass = 0; pass < 2; ++pass) {
      SCOPED_TRACE(pass == 0 ? "the first write" : "a second write");
      const Nv12 whole = pattern(c.extent.width, c.extent.height, 3 + pass);
      const BufferPlanes all = buffers_of(whole);
      std::vector<BufferPlanes> parts;
      parts.reserve(c.regions.size());
      std::vector<pipelines::ImageWrite> writes;
      for (const Region& r : c.regions) {
        parts.push_back(buffers_of(crop(whole, r.at, r.size)));
        writes.push_back(write_of(parts.back(), r.at, r.size));
      }
      pipelines::ImageUpdate update;
      update.planes[0].buffer = &all.luma;
      update.planes[1].buffer = &all.chroma;
      const Pixels level0 = read_level(written, 0, [&](VkCommandBuffer cmd) {
        write_and_convert(cmd, pipeline, written, writes);
        const vkc::Status s = pipeline.record_update(cmd, updated, update);
        EXPECT_TRUE(s.ok()) << s.message();
      });
      ASSERT_FALSE(level0.bytes.empty());
      for (uint32_t level = 0; level < written.mip_levels(); ++level) {
        SCOPED_TRACE("level " + std::to_string(level));
        const Pixels w = level == 0 ? level0 : read_level(written, level);
        const Pixels u = read_level(updated, level);
        ASSERT_EQ(w.bytes.size(), u.bytes.size());
        EXPECT_EQ(w.bytes, u.bytes);
      }
    }
  }
}

// Test 10: every refusal names its reason, records nothing and leaves the
// texture as it was: its picture, and the layouts a later write and convert
// start from.
TEST_F(ImageWriteTest, RefusalsRecordNothingAndKeepThePicture) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  const Grid g = grid(4, kTile, kGutter);
  pipelines::ImageTexture mosaic =
      make_texture(pipeline, g.extent, M::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(mosaic.valid());
  const BufferPlanes tile = buffers_of(pattern(kTile.width, kTile.height, 2));
  std::vector<pipelines::ImageWrite> tiles;
  for (const VkOffset2D& at : g.origins) {
    tiles.push_back(write_of(tile, at, kTile));
  }
  const Pixels before = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
    write_and_convert(cmd, pipeline, mosaic, tiles);
  });
  ASSERT_FALSE(before.bytes.empty());

  pipelines::ImageTexture fresh =
      make_texture(pipeline, {8, 8}, M::Grey, VK_FORMAT_R8_UNORM);
  pipelines::ImageTexture srgb =
      make_texture(pipeline, {8, 8}, M::Color, vg_test::kSrgb);
  pipelines::ImageTexture ramp =
      make_texture(pipeline, {8, 8}, M::Ramp, VK_FORMAT_R8_UNORM);
  pipelines::ImageTexture odd =
      make_texture(pipeline, {5, 3}, M::Nv12, VK_FORMAT_UNDEFINED);
  const vkc::Buffer rgba = source_buffer(std::vector<uint8_t>(8 * 8 * 4, 9));
  const vkc::Buffer r8 = source_buffer(std::vector<uint8_t>(8 * 8, 9));
  const BufferPlanes small = buffers_of(pattern(32, 24, 0));
  const BufferPlanes five = buffers_of(pattern(5, 3, 0));
  ImagePlanes small_images = images_of(pattern(32, 24, 0));

  const vkc::Status recorded = submit([&](VkCommandBuffer cmd) {
    small_images.ready(cmd);
    const auto refused = [&](std::vector<pipelines::ImageWrite> writes,
                             const char* what, const char* says) {
      const vkc::Status s = pipeline.record_write(
          cmd, mosaic, writes.data(), static_cast<uint32_t>(writes.size()));
      EXPECT_EQ(s.domain(), vkc::Status::Code::InvalidArgument) << what;
      EXPECT_NE(s.message().find(says), std::string::npos)
          << what << ": " << s.message();
      EXPECT_TRUE(mosaic.has_picture()) << what;
    };
    refused({write_of(tile, {100, 0}, kTile)}, "a region past the right edge",
            "outside");
    refused({write_of(tile, {80, 0}, {65, 48})},
            "one texel past the right edge", "outside");
    refused({write_of(tile, {0, 64}, {64, 49})},
            "one texel past the bottom edge", "outside");
    refused({write_of(tile, {-2, 0}, kTile)}, "a negative offset", "outside");
    refused({write_of(tile, {0, 0}, {0, 48})}, "a zero width", "zero size");
    refused({write_of(tile, {1, 0}, kTile)}, "an odd NV12 offset", "even");
    refused({write_of(tile, {0, 0}, {63, 48})},
            "an odd extent short of the edge", "even");
    refused({write_of(small, {0, 0}, kTile)}, "buffers smaller than the region",
            "past the end");
    refused({write_of(small_images, {0, 0}, kTile)},
            "images smaller than the region", "smaller than its region");
    pipelines::ImageWrite luma_only = write_of(tile, {0, 0}, kTile);
    luma_only.planes[1] = {};
    refused({luma_only}, "NV12 without its chroma", "plane 1");
    refused({write_of(tile, {0, 0}, kTile), write_of(tile, {32, 16}, kTile)},
            "overlapping writes", "overlap");
    std::vector<pipelines::ImageWrite> bad_last = tiles;
    bad_last[3].offset.x += 1;
    refused(bad_last, "a bad write at index 3 of 4", "write 3");
    EXPECT_EQ(pipeline.record_write(cmd, mosaic, tiles.data(), 0).domain(),
              vkc::Status::Code::InvalidArgument)
        << "no writes";
    EXPECT_EQ(pipeline.record_write(cmd, mosaic, nullptr, 1).domain(),
              vkc::Status::Code::InvalidArgument)
        << "a null span";

    // A grey picture takes one plane.
    pipelines::ImageWrite extra;
    extra.planes[0].buffer = &r8;
    extra.planes[1].buffer = &r8;
    extra.extent = {8, 8};
    EXPECT_EQ(pipeline.record_write(cmd, fresh, &extra, 1).domain(),
              vkc::Status::Code::InvalidArgument)
        << "plane 1 on a grey picture";
    EXPECT_EQ(pipeline.record_convert(cmd, fresh, {}).domain(),
              vkc::Status::Code::InvalidArgument)
        << "a convert before any write";
    EXPECT_FALSE(fresh.has_picture());

    // An odd NV12 extent is fine where it reaches the picture's edge.
    EXPECT_TRUE(pipeline
                    .record_write(cmd, odd,
                                  std::vector<pipelines::ImageWrite>{
                                      write_of(five, {2, 0}, {3, 3})}
                                      .data(),
                                  1)
                    .ok());
    EXPECT_EQ(pipeline
                  .record_write(cmd, odd,
                                std::vector<pipelines::ImageWrite>{
                                    write_of(five, {0, 0}, {3, 3})}
                                    .data(),
                                1)
                  .domain(),
              vkc::Status::Code::InvalidArgument);

    // Conversions, each on a texture with a picture written.
    pipelines::ImageWrite colour;
    colour.planes[0].buffer = &rgba;
    colour.extent = {8, 8};
    ASSERT_TRUE(pipeline.record_write(cmd, srgb, &colour, 1).ok());
    pipelines::ImageConversion linear;
    linear.encoding = pipelines::ImageEncoding::Linear;
    EXPECT_EQ(pipeline.record_convert(cmd, srgb, linear).domain(),
              vkc::Status::Code::InvalidArgument)
        << "Linear on an _SRGB source";
    pipelines::ImageWrite grey;
    grey.planes[0].buffer = &r8;
    grey.extent = {8, 8};
    ASSERT_TRUE(pipeline.record_write(cmd, ramp, &grey, 1).ok());
    pipelines::ImageConversion flat_ramp;
    flat_ramp.ramp_min = flat_ramp.ramp_max = 10.0f;
    EXPECT_EQ(pipeline.record_convert(cmd, ramp, flat_ramp).domain(),
              vkc::Status::Code::InvalidArgument)
        << "equal ramp bounds";
    pipelines::ImageConversion nan_ramp;
    nan_ramp.ramp_max = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(pipeline.record_convert(cmd, ramp, nan_ramp).domain(),
              vkc::Status::Code::InvalidArgument)
        << "a NaN ramp bound";
    for (const std::array<float, 2>& kr_kb :
         {std::array<float, 2>{0.6f, 0.5f}, std::array<float, 2>{0.0f, 0.1f},
          std::array<float, 2>{0.3f, -0.1f}}) {
      pipelines::ImageConversion matrix;
      matrix.kr = kr_kb[0];
      matrix.kb = kr_kb[1];
      EXPECT_EQ(pipeline.record_convert(cmd, mosaic, matrix).domain(),
                vkc::Status::Code::InvalidArgument)
          << "kr " << kr_kb[0] << ", kb " << kr_kb[1];
    }
    // record_update checks its convert before recording its write.
    pipelines::ImageUpdate bad_update;
    bad_update.planes = tiles[0].planes;
    bad_update.kr = 0.9f;
    pipelines::ImageTexture own =
        make_texture(pipeline, kTile, M::Nv12, VK_FORMAT_UNDEFINED);
    EXPECT_EQ(pipeline.record_update(cmd, own, bad_update).domain(),
              vkc::Status::Code::InvalidArgument);
    EXPECT_FALSE(own.has_picture());
    EXPECT_EQ(pipeline.record_convert(cmd, own, {}).domain(),
              vkc::Status::Code::InvalidArgument)
        << "the refused update wrote nothing";
  });
  ASSERT_TRUE(recorded.ok()) << recorded.message();

  // The picture is unchanged, and a write of one tile then starts from the
  // layouts the texture recorded: validation would report one it did not.
  EXPECT_EQ(read_level(mosaic, 0).bytes, before.bytes);
  const Pixels again = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
    write_and_convert(cmd, pipeline, mosaic, {tiles[1]});
  });
  EXPECT_EQ(again.bytes, before.bytes);
}

// The device's largest picture bounds a texture, and the refusal says what it
// is, so a caller can page its regions over several textures.
TEST_F(ImageWriteTest, ATextureBeyondTheLargestPictureNamesTheLimit) {
  const pipelines::ImagePipeline pipeline = make_pipeline();
  const VkExtent2D max = pipeline.max_texture_extent();
  const VkPhysicalDeviceLimits& limits = device().caps().limits();
  EXPECT_EQ(max.width,
            std::min({limits.maxImageDimension2D, limits.maxFramebufferWidth,
                      limits.maxViewportDimensions[0]}));
  EXPECT_EQ(max.height,
            std::min({limits.maxImageDimension2D, limits.maxFramebufferHeight,
                      limits.maxViewportDimensions[1]}));
  pipelines::ImageTextureDesc desc;
  desc.extent = {max.width + 1, 16};
  desc.mapping = M::Nv12;
  const auto made =
      pipelines::ImageTexture::create(pipeline, allocator(), desc);
  ASSERT_EQ(made.status().domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_NE(made.status().message().find(std::to_string(max.width) + " x " +
                                         std::to_string(max.height)),
            std::string::npos)
      << made.status().message();
  EXPECT_EQ(pipelines::ImagePipeline{}.max_texture_extent().width, 0u);
}

// Test 9: sources written on the compute library's queue family -- images
// and buffers, CONCURRENT across both -- are written into one picture and
// converted on the renderer's, in place.
class ImageCrossFamilyTest : public vkc::test::SharedDeviceTest,
                             protected vg_test::ImageHelpers {
 protected:
  vkc::DeviceRequirements graphics_requirements() const override {
    return vg::device_requirements();
  }
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::ShaderAccesses;
  }
  vkc::Device& image_device() override { return two_family().graphics(); }
  vkc::Allocator& image_allocator() override {
    return two_family().graphics_allocator();
  }

  // A plane the compute family wrote, CONCURRENT across both families.
  vkc::Image compute_image(VkFormat format, VkExtent2D extent,
                           const std::vector<uint8_t>& texels) {
    vkc::test::TwoFamilyDevice& two = two_family();
    vkc::ImageDesc desc;
    desc.extent = extent;
    desc.format = format;
    desc.usage =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    desc.with_view = false;  // transfer only, as a decoder's picture can be
    desc.queue_families = two.families().data();
    desc.queue_family_count = vkc::test::TwoFamilyDevice::kFamilyCount;
    auto image = two.compute_allocator().create_image(desc);
    EXPECT_TRUE(image.ok()) << image.status().message();
    if (!image.ok()) {
      return vkc::Image{};
    }
    const vkc::Status s =
        two.write_image(image.value(), texels.data(), texels.size(),
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    EXPECT_TRUE(s.ok()) << s.message();
    return std::move(image).value();
  }
  vkc::Buffer compute_buffer(const std::vector<uint8_t>& bytes) {
    vkc::test::TwoFamilyDevice& two = two_family();
    vkc::BufferDesc desc;
    desc.size = bytes.size();
    desc.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.queue_families = two.families().data();
    desc.queue_family_count = vkc::test::TwoFamilyDevice::kFamilyCount;
    auto buffer = two.compute_allocator().create_buffer(desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    if (!buffer.ok()) {
      return vkc::Buffer{};
    }
    const vkc::Status s =
        two.write_buffer(buffer.value(), 0, bytes.data(), bytes.size());
    EXPECT_TRUE(s.ok()) << s.message();
    return std::move(buffer).value();
  }
};

TEST_F(ImageCrossFamilyTest, WritesSourcesTheComputeFamilyFilledInPlace) {
  ASSERT_TRUE(two_family().crosses_families());
  const pipelines::ImagePipeline pipeline = make_pipeline();
  ASSERT_TRUE(pipeline.valid());
  const Grid g = grid(2, kTile, kGutter);
  pipelines::ImageTexture mosaic =
      make_texture(pipeline, g.extent, M::Nv12, VK_FORMAT_UNDEFINED);
  ASSERT_TRUE(mosaic.valid());

  const Nv12 a = pattern(kTile.width, kTile.height, 0);
  const Nv12 b = pattern(kTile.width, kTile.height, 1);
  const vkc::Image a_luma = compute_image(kLuma, a.extent(), a.luma);
  const vkc::Image a_chroma =
      compute_image(kChroma, a.chroma_extent(), a.chroma);
  const vkc::Buffer b_luma = compute_buffer(b.luma);
  const vkc::Buffer b_chroma = compute_buffer(b.chroma);
  std::vector<pipelines::ImageWrite> writes(2);
  writes[0].planes[0].image = &a_luma;
  writes[0].planes[1].image = &a_chroma;
  writes[0].offset = g.origins[0];
  writes[0].extent = kTile;
  writes[1].planes[0].buffer = &b_luma;
  writes[1].planes[1].buffer = &b_chroma;
  writes[1].offset = g.origins[1];
  writes[1].extent = kTile;

  // The same pictures from the renderer's own buffers, for reference.
  const vkc::Buffer a_luma_here = source_buffer(a.luma);
  const vkc::Buffer a_chroma_here = source_buffer(a.chroma);
  const vkc::Buffer b_luma_here = source_buffer(b.luma);
  const vkc::Buffer b_chroma_here = source_buffer(b.chroma);
  std::array<pipelines::ImageTexture, 2> own = {
      make_texture(pipeline, kTile, M::Nv12, VK_FORMAT_UNDEFINED),
      make_texture(pipeline, kTile, M::Nv12, VK_FORMAT_UNDEFINED)};
  const Pixels px = read_level(mosaic, 0, [&](VkCommandBuffer cmd) {
    ASSERT_TRUE(pipeline
                    .record_write(cmd, mosaic, writes.data(),
                                  static_cast<uint32_t>(writes.size()))
                    .ok());
    ASSERT_TRUE(pipeline.record_convert(cmd, mosaic, {}).ok());
    pipelines::ImageUpdate update;
    update.planes[0].buffer = &a_luma_here;
    update.planes[1].buffer = &a_chroma_here;
    ASSERT_TRUE(pipeline.record_update(cmd, own[0], update).ok());
    update.planes[0].buffer = &b_luma_here;
    update.planes[1].buffer = &b_chroma_here;
    ASSERT_TRUE(pipeline.record_update(cmd, own[1], update).ok());
  });
  ASSERT_FALSE(px.bytes.empty());
  for (uint32_t i = 0; i < 2; ++i) {
    SCOPED_TRACE("tile " + std::to_string(i));
    const Pixels alone = read_level(own[i], 0);
    ASSERT_FALSE(alone.bytes.empty());
    EXPECT_TRUE(same_region(px, static_cast<uint32_t>(g.origins[i].x) + 1, 1,
                            alone, 1, 1, kTile.width - 2, kTile.height - 2));
  }
}

// --- No device needed --------------------------------------------------------

// Test 12: image_draw takes the view's placement, and bounds the viewport
// whatever the view holds: huge, NaN and negative rectangles give a bounded
// or empty one, never an out-of-range conversion.
TEST(ImageDrawTest, TakesTheViewAndBoundsItsViewport) {
  const pipelines::ImageTexture texture;
  using vg::camera::ImageView2D;

  const ImageView2D view({64.0f, 48.0f}, {10.4f, 20.6f}, {100.2f, 50.5f});
  const pipelines::ImageDraw draw =
      pipelines::image_draw(texture, view, pipelines::ImageFilter::Linear);
  EXPECT_EQ(draw.texture, &texture);
  EXPECT_EQ(draw.viewport.offset.x, 10);
  EXPECT_EQ(draw.viewport.offset.y, 21);
  EXPECT_EQ(draw.viewport.extent.width, 101u);  // round(110.6) - 10
  EXPECT_EQ(draw.viewport.extent.height, 50u);  // round(71.1) - 21
  EXPECT_EQ(draw.origin, view.origin());
  EXPECT_EQ(draw.scale, view.scale());
  EXPECT_EQ(draw.magnify, pipelines::ImageFilter::Linear);
  EXPECT_EQ(pipelines::image_draw(texture, view).magnify,
            pipelines::ImageFilter::Nearest);

  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
  struct Case {
    glm::vec2 origin;
    glm::vec2 size;
    bool empty;
  };
  const Case cases[] = {
      {{-3e9f, -3e9f}, {100.0f, 100.0f}, true},  // far off to the left
      {{3e9f, 3e9f}, {100.0f, 100.0f}, true},    // far off to the right
      {{0.0f, 0.0f}, {3e9f, 3e9f}, false},       // past Vulkan's range
      {{-3e9f, -3e9f}, {6e9f, 6e9f}, false},     // across it
      {{kNan, kNan}, {100.0f, 100.0f}, true},
      {{5.0f, 5.0f}, {kNan, kNan}, true},
      {{5.0f, 5.0f}, {-40.0f, -40.0f}, true},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "origin " << c.origin.x << ", size " << c.size.x);
    ImageView2D v({64.0f, 48.0f}, {0.0f, 0.0f}, {1.0f, 1.0f});
    v.set_viewport(c.origin, c.size);
    const pipelines::ImageDraw d = pipelines::image_draw(texture, v);
    EXPECT_GE(d.viewport.offset.x, 0);
    EXPECT_GE(d.viewport.offset.y, 0);
    EXPECT_LE(int64_t{d.viewport.offset.x} + d.viewport.extent.width, kMax);
    EXPECT_LE(int64_t{d.viewport.offset.y} + d.viewport.extent.height, kMax);
    if (c.empty) {
      EXPECT_EQ(d.viewport.extent.width, 0u);
      EXPECT_EQ(d.viewport.extent.height, 0u);
    } else {
      EXPECT_GT(d.viewport.extent.width, 0u);
      EXPECT_GT(d.viewport.extent.height, 0u);
    }
  }
}

}  // namespace

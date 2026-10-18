// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// PrimitivePipeline: colored points and lines drawn offscreen and read back,
// under the validation layer: where a point lands, which pixels a line
// covers, how a viewport places and clips a draw, depth between primitives
// and draws, what each target stores of an sRGB color, and the draws submit
// refuses before recording anything.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "color_test_support.hpp"
#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/primitive_pipeline.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;
namespace camera = volumetric_kit::gfx::camera;

using pipelines::ColorVertex;
using pipelines::pack_rgba8;
using pipelines::PrimitiveDraw;
using pipelines::PrimitivePipeline;
using pipelines::PrimitiveTopology;

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t kSize = 64;
constexpr VkRect2D kFullTarget{{0, 0}, {kSize, kSize}};

constexpr uint32_t kRed = pack_rgba8(255, 0, 0);
constexpr uint32_t kGreen = pack_rgba8(0, 255, 0);
constexpr uint32_t kBlue = pack_rgba8(0, 0, 255);

vg::RenderTargetLayout layout_for(VkFormat color = kColorFormat) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = color;
  layout.color_count = 1;
  layout.depth_format = kDepthFormat;
  return layout;
}

// A view-projection that puts world (x, y) at target pixel (x, y), so pixel
// (i, j)'s center is world (i + 0.5, j + 0.5), and keeps z as depth.
glm::mat4 pixel_space() {
  glm::mat4 m(1.0f);
  m[0][0] = 2.0f / kSize;
  m[1][1] = 2.0f / kSize;
  m[3][0] = -1.0f;
  m[3][1] = -1.0f;
  return m;
}

// The first `count` vertices of `buffer` drawn as `topology`, by default into
// the whole target in pixel space.
PrimitiveDraw draw_of(const vkc::Buffer& buffer, uint32_t count,
                      PrimitiveTopology topology,
                      const glm::mat4& view_proj = pixel_space(),
                      VkRect2D viewport = kFullTarget) {
  PrimitiveDraw draw;
  draw.topology = topology;
  draw.vertices = &buffer;
  draw.vertex_count = count;
  draw.viewport = viewport;
  draw.view_proj = view_proj;
  return draw;
}

// An RGBA8 readback of a kSize x kSize target.
struct Pixels {
  std::vector<uint8_t> bytes;

  std::array<uint8_t, 4> at(uint32_t x, uint32_t y) const {
    const uint8_t* p = &bytes[(static_cast<size_t>(y) * kSize + x) * 4];
    return {p[0], p[1], p[2], p[3]};
  }
  // Whether anything was drawn at (x, y): the clear is opaque black.
  bool lit(uint32_t x, uint32_t y) const {
    const std::array<uint8_t, 4> p = at(x, y);
    return p[0] != 0 || p[1] != 0 || p[2] != 0;
  }
  int lit_count() const {
    int count = 0;
    for (uint32_t y = 0; y < kSize; ++y) {
      for (uint32_t x = 0; x < kSize; ++x) {
        count += lit(x, y) ? 1 : 0;
      }
    }
    return count;
  }
};

// The bytes of a packed color, as an RGBA8 target stores it opaque.
std::array<uint8_t, 4> opaque(uint32_t rgba) {
  return {static_cast<uint8_t>(rgba), static_cast<uint8_t>(rgba >> 8),
          static_cast<uint8_t>(rgba >> 16), 255};
}

// --- No device ---------------------------------------------------------------

TEST(ColorVertexTest, IsSixteenBytesWithRedFirstInMemory) {
  static_assert(sizeof(ColorVertex) == 16);
  const ColorVertex v{{1.0f, 2.0f, 3.0f}, pack_rgba8(10, 20, 30, 40)};
  uint8_t bytes[sizeof(ColorVertex)];
  std::memcpy(bytes, &v, sizeof(v));
  float position[3];
  std::memcpy(position, bytes, sizeof(position));
  EXPECT_EQ(position[0], 1.0f);
  EXPECT_EQ(position[2], 3.0f);
  EXPECT_EQ(bytes[12], 10) << "R8G8B8A8 reads red from the first byte";
  EXPECT_EQ(bytes[13], 20);
  EXPECT_EQ(bytes[14], 30);
  EXPECT_EQ(bytes[15], 40);
  EXPECT_EQ(pack_rgba8(1, 2, 3), 0xFF030201u) << "alpha defaults to opaque";
}

TEST(PrimitivePipelineTest, IsAnAggregate) {
  static_assert(!std::is_copy_constructible_v<PrimitivePipeline>);
  static_assert(!std::is_copy_assignable_v<PrimitivePipeline>);
  static_assert(!std::is_move_constructible_v<PrimitivePipeline>);
  static_assert(!std::is_move_assignable_v<PrimitivePipeline>);
  static_assert(!std::is_default_constructible_v<PrimitivePipeline>);
  SUCCEED();
}

// --- Creation ----------------------------------------------------------------

class PrimitivePipelineDeviceTest : public vg_test::RendererDeviceTest {};

TEST_F(PrimitivePipelineDeviceTest, CreatesForAColorAndDepthLayout) {
  auto pipeline = PrimitivePipeline::create(device(), layout_for());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
  ASSERT_NE(pipeline.value(), nullptr);
  // The renderer's requirements do not enable largePoints.
  EXPECT_EQ(pipeline.value()->max_point_size(), 1.0f);
}

TEST_F(PrimitivePipelineDeviceTest, RefusesALayoutWithoutDepth) {
  vg::RenderTargetLayout layout = layout_for();
  layout.depth_format = VK_FORMAT_UNDEFINED;
  auto pipeline = PrimitivePipeline::create(device(), layout);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(PrimitivePipelineDeviceTest, RefusesALayoutWithoutExactlyOneColor) {
  for (uint32_t count : {0u, 2u}) {
    vg::RenderTargetLayout layout = layout_for();
    layout.color_formats[1] = kColorFormat;
    layout.color_count = count;
    auto pipeline = PrimitivePipeline::create(device(), layout);
    ASSERT_FALSE(pipeline.ok()) << count << " color attachments";
    EXPECT_EQ(pipeline.status().domain(), vkc::Status::Code::InvalidArgument);
  }
  // One attachment of no format, refused before any Vulkan object is made.
  auto pipeline =
      PrimitivePipeline::create(device(), layout_for(VK_FORMAT_UNDEFINED));
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_NE(pipeline.status().message().find("PrimitivePipeline::create"),
            std::string::npos)
      << pipeline.status().message();
}

// A device made for another library -- here a compute one, without dynamic
// rendering -- is refused rather than drawn on.
using PrimitiveForeignDeviceTest = vkc::test::VulkanDeviceTest;

TEST_F(PrimitiveForeignDeviceTest, IsRefused) {
  auto pipeline = PrimitivePipeline::create(device(), layout_for());
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vkc::Status::Code::Unsupported)
      << pipeline.status().message();
}

// --- Drawing: validation-with-teeth ------------------------------------------

class PrimitiveRenderTest : public vg_test::RendererDeviceTest {
 protected:
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::On;
  }

  std::unique_ptr<PrimitivePipeline> make_pipeline(
      VkFormat color = kColorFormat) {
    auto pipeline = PrimitivePipeline::create(device(), layout_for(color));
    EXPECT_TRUE(pipeline.ok()) << pipeline.status().message();
    return pipeline.ok() ? std::move(pipeline).value() : nullptr;
  }

  // `vertices` in a device-only vertex buffer.
  vkc::Buffer upload(const std::vector<ColorVertex>& vertices) {
    vg::BufferUploadDesc desc;
    desc.data = vertices.data();
    desc.size = vertices.size() * sizeof(ColorVertex);
    desc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    auto buffer = vg::upload_buffer(device(), allocator(), desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
  }

  // Records `draws` into a kSize x kSize target of `color` cleared to opaque
  // black, runs it, and returns submit's status and the texels' bytes (4 or
  // 16 per texel).
  // TODO: fold into one shared offscreen render-and-readback test helper,
  // with the copies in the other pipeline and target tests.
  std::pair<vkc::Status, std::vector<uint8_t>> render_bytes(
      const PrimitivePipeline& pipeline, const PrimitiveDraw* draws,
      uint32_t draw_count, VkFormat color = kColorFormat) {
    vg::OffscreenTargetDesc desc;
    desc.extent = {kSize, kSize};
    desc.color_format = color;
    desc.depth_format = kDepthFormat;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    auto pool =
        vkc::CommandPool::create(device().handle(), device().queue_family());
    EXPECT_TRUE(pool.ok()) << pool.status().message();
    if (!target.ok() || !pool.ok()) {
      return {vkc::Status::invalid_argument("setup failed"), {}};
    }
    auto cmd = pool.value().allocate_primary();
    EXPECT_TRUE(cmd.ok()) << cmd.status().message();
    if (!cmd.ok()) {
      return {vkc::Status::invalid_argument("setup failed"), {}};
    }
    EXPECT_TRUE(
        cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    const VkCommandBuffer raw = cmd.value().handle();
    target.value().prepare(raw);
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[3] = 1.0f;  // opaque black, depth 1
    const vg::RenderTarget rt = target.value().target();
    rt.begin(raw, begin);
    const vkc::Status submitted =
        pipeline.submit(raw, {{kSize, kSize}, draws, draw_count});
    rt.end(raw);
    target.value().record_readback(raw);
    EXPECT_TRUE(cmd.value().end().ok());
    const vkc::Status ran = device().submit_and_wait(raw);
    if (!ran.ok()) {
      ADD_FAILURE() << ran.message();
      return {submitted, {}};
    }
    const size_t texel = color == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : 4;
    const auto* px = static_cast<const uint8_t*>(target.value().pixels());
    EXPECT_NE(px, nullptr);
    if (px == nullptr) {
      return {submitted, {}};
    }
    return {submitted,
            std::vector<uint8_t>(px, px + texel * size_t{kSize} * kSize)};
  }

  // render_bytes into an RGBA8 target, expecting submit to succeed.
  Pixels render(const PrimitivePipeline& pipeline,
                const std::vector<PrimitiveDraw>& draws,
                VkFormat color = kColorFormat) {
    auto [status, bytes] = render_bytes(
        pipeline, draws.data(), static_cast<uint32_t>(draws.size()), color);
    EXPECT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(bytes.size(), size_t{kSize} * kSize * 4);
    return Pixels{std::move(bytes)};
  }
};

// Points drawn through a perspective camera land on the pixel the camera
// projects them to: each is placed by unprojecting a pixel's center, and
// lights that pixel alone, in its color.
TEST_F(PrimitiveRenderTest, PointLandsOnItsProjectedPixel) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const glm::mat4 view_proj =
      camera::Camera::look_at_perspective(
          {0.4f, -0.3f, 3.0f}, {0.0f, 0.2f, 0.0f}, {0.0f, 1.0f, 0.0f},
          glm::radians(50.0f), 1.0f, 0.1f, 10.0f)
          .view_proj();
  const glm::mat4 inverse = glm::inverse(view_proj);

  const std::array<std::array<uint32_t, 2>, 5> pixels{
      {{5, 7}, {40, 12}, {31, 50}, {63, 62}, {0, 33}}};
  std::vector<ColorVertex> points;
  for (size_t i = 0; i < pixels.size(); ++i) {
    const float x = static_cast<float>(pixels[i][0]) + 0.5f;
    const float y = static_cast<float>(pixels[i][1]) + 0.5f;
    const glm::vec4 ndc{x / kSize * 2.0f - 1.0f, y / kSize * 2.0f - 1.0f, 0.5f,
                        1.0f};
    const glm::vec4 world = inverse * ndc;
    points.push_back({glm::vec3(world) / world.w,
                      pack_rgba8(static_cast<uint8_t>(50 * i + 40), 200,
                                 static_cast<uint8_t>(255 - 50 * i))});
  }
  const vkc::Buffer buffer = upload(points);
  const Pixels drawn =
      render(*pipeline, {draw_of(buffer, static_cast<uint32_t>(points.size()),
                                 PrimitiveTopology::Points, view_proj)});

  for (size_t i = 0; i < pixels.size(); ++i) {
    EXPECT_EQ(drawn.at(pixels[i][0], pixels[i][1]), opaque(points[i].rgba))
        << "point " << i;
  }
  EXPECT_EQ(drawn.lit_count(), static_cast<int>(pixels.size()))
      << "each point lights its pixel and no other";
}

// Without largePoints, a point is one pixel whatever size it asks for.
TEST_F(PrimitiveRenderTest, PointsAreOnePixelWithoutLargePoints) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  ASSERT_EQ(pipeline->max_point_size(), 1.0f);
  const vkc::Buffer buffer = upload({{{20.5f, 20.5f, 0.5f}, kRed}});
  PrimitiveDraw draw = draw_of(buffer, 1, PrimitiveTopology::Points);
  draw.point_size = 9.0f;
  const Pixels drawn = render(*pipeline, {draw});
  EXPECT_TRUE(drawn.lit(20, 20));
  EXPECT_EQ(drawn.lit_count(), 1);
}

// A line lights the pixels between its ends and no others: one row or
// column for an axis-aligned line, and a one- or two-pixel band hugging a
// slanted one (strict lines may cover two pixels in a column).
TEST_F(PrimitiveRenderTest, LinesCoverThePixelsBetweenTheirEnds) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);

  {  // Horizontal, along row 20 from pixel 5 to pixel 50.
    const vkc::Buffer buffer =
        upload({{{5.5f, 20.5f, 0.5f}, kRed}, {{50.5f, 20.5f, 0.5f}, kRed}});
    const Pixels drawn =
        render(*pipeline, {draw_of(buffer, 2, PrimitiveTopology::Lines)});
    for (uint32_t x = 7; x <= 48; ++x) {
      EXPECT_TRUE(drawn.lit(x, 20)) << "x = " << x;
    }
    for (uint32_t y = 0; y < kSize; ++y) {
      for (uint32_t x = 0; x < kSize; ++x) {
        if (y != 20 || x < 4 || x > 51) {
          EXPECT_FALSE(drawn.lit(x, y)) << "(" << x << ", " << y << ")";
        }
      }
    }
  }
  {  // Vertical, along column 40 from pixel 5 to pixel 58.
    const vkc::Buffer buffer =
        upload({{{40.5f, 5.5f, 0.5f}, kRed}, {{40.5f, 58.5f, 0.5f}, kRed}});
    const Pixels drawn =
        render(*pipeline, {draw_of(buffer, 2, PrimitiveTopology::Lines)});
    for (uint32_t y = 7; y <= 56; ++y) {
      EXPECT_TRUE(drawn.lit(40, y)) << "y = " << y;
    }
    for (uint32_t y = 0; y < kSize; ++y) {
      for (uint32_t x = 0; x < kSize; ++x) {
        if (x != 40 || y < 4 || y > 59) {
          EXPECT_FALSE(drawn.lit(x, y)) << "(" << x << ", " << y << ")";
        }
      }
    }
  }
  {  // Slanted, slope 2/3, from (6.5, 10.5) to (54.5, 42.5).
    const vkc::Buffer buffer =
        upload({{{6.5f, 10.5f, 0.5f}, kRed}, {{54.5f, 42.5f, 0.5f}, kRed}});
    const Pixels drawn =
        render(*pipeline, {draw_of(buffer, 2, PrimitiveTopology::Lines)});
    const auto line_y = [](float x) {
      return 10.5f + (x - 6.5f) * 2.0f / 3.0f;
    };
    for (uint32_t x = 0; x < kSize; ++x) {
      int in_column = 0;
      for (uint32_t y = 0; y < kSize; ++y) {
        if (!drawn.lit(x, y)) {
          continue;
        }
        ++in_column;
        const float cx = static_cast<float>(x) + 0.5f;
        EXPECT_LT(std::fabs(static_cast<float>(y) + 0.5f - line_y(cx)), 1.0f)
            << "(" << x << ", " << y << ") is off the line";
        EXPECT_TRUE(cx > 5.0f && cx < 56.0f)
            << "(" << x << ", " << y << ") is past an end";
      }
      if (x >= 9 && x <= 51) {
        EXPECT_GE(in_column, 1) << "column " << x << " is not covered";
      }
      EXPECT_LE(in_column, 2) << "column " << x << " is too thick";
    }
  }
}

// A draw's viewport is where normalized device coordinates [-1, 1] land,
// and nothing of the draw reaches past it or the target -- also when the
// viewport hangs off the target's edge, where the scissor must be clipped
// for Vulkan to accept it.
TEST_F(PrimitiveRenderTest, ViewportPlacesAndClipsADraw) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  // NDC of pixel `p`'s center in a viewport starting at `offset`, `size` long.
  const auto ndc = [](float p, float offset, float size) {
    return (p + 0.5f - offset) / size * 2.0f - 1.0f;
  };

  {
    const VkRect2D rect{{16, 8}, {32, 24}};
    const std::vector<ColorVertex> vertices{
        // A point at pixel (20, 10).
        {{ndc(20, 16, 32), ndc(10, 8, 24), 0.5f}, kBlue},
        // Lines across far more than the viewport, along row 20 and column 30.
        {{-3.0f, ndc(20, 8, 24), 0.5f}, kRed},
        {{3.0f, ndc(20, 8, 24), 0.5f}, kRed},
        {{ndc(30, 16, 32), -3.0f, 0.5f}, kGreen},
        {{ndc(30, 16, 32), 3.0f, 0.5f}, kGreen},
    };
    const vkc::Buffer buffer = upload(vertices);
    PrimitiveDraw point =
        draw_of(buffer, 1, PrimitiveTopology::Points, glm::mat4(1.0f), rect);
    PrimitiveDraw lines =
        draw_of(buffer, 4, PrimitiveTopology::Lines, glm::mat4(1.0f), rect);
    lines.first_vertex = 1;
    const Pixels drawn = render(*pipeline, {point, lines});

    EXPECT_EQ(drawn.at(20, 10), opaque(kBlue));
    for (uint32_t x = 17; x <= 46; ++x) {
      if (x != 30) {
        EXPECT_EQ(drawn.at(x, 20), opaque(kRed)) << "x = " << x;
      }
    }
    for (uint32_t y = 9; y <= 30; ++y) {
      if (y != 20) {
        EXPECT_EQ(drawn.at(30, y), opaque(kGreen)) << "y = " << y;
      }
    }
    for (uint32_t y = 0; y < kSize; ++y) {
      for (uint32_t x = 0; x < kSize; ++x) {
        const bool inside = x >= 16 && x < 48 && y >= 8 && y < 32;
        if (!inside) {
          EXPECT_FALSE(drawn.lit(x, y)) << "(" << x << ", " << y << ")";
        }
      }
    }
  }
  {
    // Hanging off the target's left and bottom edges: x in [-16, 32), y in
    // [40, 88). A line across row 50 and a point at row 70, below the target.
    const VkRect2D rect{{-16, 40}, {48, 48}};
    const std::vector<ColorVertex> vertices{
        {{-1.0f, ndc(50, 40, 48), 0.5f}, kRed},
        {{1.0f, ndc(50, 40, 48), 0.5f}, kRed},
        {{ndc(10, -16, 48), ndc(70, 40, 48), 0.5f}, kBlue},
    };
    const vkc::Buffer buffer = upload(vertices);
    PrimitiveDraw lines =
        draw_of(buffer, 2, PrimitiveTopology::Lines, glm::mat4(1.0f), rect);
    PrimitiveDraw point =
        draw_of(buffer, 1, PrimitiveTopology::Points, glm::mat4(1.0f), rect);
    point.first_vertex = 2;
    const Pixels drawn = render(*pipeline, {lines, point});
    for (uint32_t x = 0; x <= 30; ++x) {
      EXPECT_TRUE(drawn.lit(x, 50)) << "x = " << x;
    }
    for (uint32_t y = 0; y < kSize; ++y) {
      for (uint32_t x = 0; x < kSize; ++x) {
        if (y != 50 || x >= 32) {
          EXPECT_FALSE(drawn.lit(x, y)) << "(" << x << ", " << y << ")";
        }
      }
    }
  }
  {
    // Wholly outside the target, or empty: skipped.
    const vkc::Buffer buffer = upload({{{0.0f, 0.0f, 0.5f}, kRed}});
    PrimitiveDraw outside = draw_of(buffer, 1, PrimitiveTopology::Points,
                                    glm::mat4(1.0f), {{64, 0}, {16, 16}});
    PrimitiveDraw empty = draw_of(buffer, 1, PrimitiveTopology::Points,
                                  glm::mat4(1.0f), {{8, 8}, {0, 16}});
    EXPECT_EQ(render(*pipeline, {outside, empty}).lit_count(), 0);
  }
}

// The nearer of two primitives on one pixel shows, whichever is drawn first,
// within a draw and across draws of either topology.
TEST_F(PrimitiveRenderTest, NearerPrimitivesOccludeFartherOnesInEitherOrder) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const ColorVertex near_point{{20.5f, 20.5f, 0.25f}, kRed};
  const ColorVertex far_point{{20.5f, 20.5f, 0.75f}, kGreen};

  for (bool near_first : {true, false}) {
    const vkc::Buffer one =
        upload(near_first ? std::vector{near_point, far_point}
                          : std::vector{far_point, near_point});
    EXPECT_EQ(render(*pipeline, {draw_of(one, 2, PrimitiveTopology::Points)})
                  .at(20, 20),
              opaque(kRed))
        << "one draw, near first: " << near_first;

    const vkc::Buffer near = upload({near_point});
    const vkc::Buffer far = upload({far_point});
    const PrimitiveDraw near_draw = draw_of(near, 1, PrimitiveTopology::Points);
    const PrimitiveDraw far_draw = draw_of(far, 1, PrimitiveTopology::Points);
    EXPECT_EQ(render(*pipeline, near_first ? std::vector{near_draw, far_draw}
                                           : std::vector{far_draw, near_draw})
                  .at(20, 20),
              opaque(kRed))
        << "two draws, near first: " << near_first;
  }

  // A near point drawn before a far line across it, and a near line drawn
  // before a far point on it.
  const vkc::Buffer point = upload({{{30.5f, 40.5f, 0.25f}, kRed}});
  const vkc::Buffer line =
      upload({{{4.5f, 40.5f, 0.75f}, kGreen}, {{60.5f, 40.5f, 0.75f}, kGreen}});
  const vkc::Buffer near_line =
      upload({{{4.5f, 50.5f, 0.25f}, kBlue}, {{60.5f, 50.5f, 0.25f}, kBlue}});
  const vkc::Buffer far_point_on_line = upload({{{25.5f, 50.5f, 0.75f}, kRed}});
  const Pixels drawn = render(
      *pipeline, {draw_of(point, 1, PrimitiveTopology::Points),
                  draw_of(line, 2, PrimitiveTopology::Lines),
                  draw_of(near_line, 2, PrimitiveTopology::Lines),
                  draw_of(far_point_on_line, 1, PrimitiveTopology::Points)});
  EXPECT_EQ(drawn.at(30, 40), opaque(kRed)) << "the near point stays";
  EXPECT_EQ(drawn.at(31, 40), opaque(kGreen)) << "the line beside it";
  EXPECT_EQ(drawn.at(25, 50), opaque(kBlue)) << "the near line stays";
}

// A UNORM and an sRGB target store a color's own bytes, a float target its
// linear light; and a line's color interpolates in the encoding, so a gradient
// stores alike on UNORM and sRGB targets.
TEST_F(PrimitiveRenderTest, ColorsKeepTheirEncodingOnEveryTarget) {
  constexpr uint8_t kR = 200, kG = 100, kB = 30;
  const std::vector<ColorVertex> vertices{
      {{20.5f, 20.5f, 0.5f}, pack_rgba8(kR, kG, kB, 0)},  // alpha ignored
      {{0.5f, 40.5f, 0.5f}, pack_rgba8(0, 0, 0)},
      {{63.5f, 40.5f, 0.5f}, pack_rgba8(255, 255, 255)},
  };
  const vkc::Buffer buffer = upload(vertices);
  PrimitiveDraw lines = draw_of(buffer, 2, PrimitiveTopology::Lines);
  lines.first_vertex = 1;
  const std::vector<PrimitiveDraw> draws{
      draw_of(buffer, 1, PrimitiveTopology::Points), lines};

  auto unorm_pipeline = make_pipeline(VK_FORMAT_R8G8B8A8_UNORM);
  auto srgb_pipeline = make_pipeline(VK_FORMAT_R8G8B8A8_SRGB);
  auto float_pipeline = make_pipeline(VK_FORMAT_R32G32B32A32_SFLOAT);
  ASSERT_NE(unorm_pipeline, nullptr);
  ASSERT_NE(srgb_pipeline, nullptr);
  ASSERT_NE(float_pipeline, nullptr);

  const Pixels unorm = render(*unorm_pipeline, draws, VK_FORMAT_R8G8B8A8_UNORM);
  EXPECT_EQ(unorm.at(20, 20), (std::array<uint8_t, 4>{kR, kG, kB, 255}));

  const Pixels srgb = render(*srgb_pipeline, draws, VK_FORMAT_R8G8B8A8_SRGB);
  const std::array<uint8_t, 4> stored = srgb.at(20, 20);
  EXPECT_NEAR(stored[0], kR, 1);
  EXPECT_NEAR(stored[1], kG, 1);
  EXPECT_NEAR(stored[2], kB, 1);
  EXPECT_EQ(stored[3], 255);
  for (uint32_t x = 4; x < 60; ++x) {
    EXPECT_NEAR(srgb.at(x, 40)[0], unorm.at(x, 40)[0], 1) << "x = " << x;
  }
  EXPECT_NEAR(unorm.at(32, 40)[0], 128, 6) << "halfway in the encoding";

  auto [status, bytes] = render_bytes(*float_pipeline, draws.data(),
                                      static_cast<uint32_t>(draws.size()),
                                      VK_FORMAT_R32G32B32A32_SFLOAT);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_EQ(bytes.size(), size_t{kSize} * kSize * 16);
  float texel[4];
  std::memcpy(texel, &bytes[(size_t{20} * kSize + 20) * 16], sizeof(texel));
  EXPECT_NEAR(texel[0], vg_test::srgb_to_linear(kR / 255.0), 2e-3);
  EXPECT_NEAR(texel[1], vg_test::srgb_to_linear(kG / 255.0), 2e-3);
  EXPECT_NEAR(texel[2], vg_test::srgb_to_linear(kB / 255.0), 2e-3);
  EXPECT_EQ(texel[3], 1.0f);
}

// submit checks every draw before recording any: a malformed draw before or
// after a good one returns InvalidArgument naming it, and the target stays
// clear.
TEST_F(PrimitiveRenderTest, RefusesMalformedDrawsAndRecordsNothing) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  const vkc::Buffer buffer = upload({{{10.5f, 10.5f, 0.5f}, kRed},
                                     {{20.5f, 20.5f, 0.5f}, kRed},
                                     {{30.5f, 30.5f, 0.5f}, kRed}});
  vkc::BufferDesc storage_desc;
  storage_desc.size = 3 * sizeof(ColorVertex);
  storage_desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  auto storage = allocator().create_buffer(storage_desc);
  ASSERT_TRUE(storage.ok()) << storage.status().message();
  const vkc::Buffer empty;
  const VkPhysicalDeviceLimits& limits = device().caps().limits();

  const PrimitiveDraw good = draw_of(buffer, 3, PrimitiveTopology::Points);
  std::vector<std::pair<const char*, PrimitiveDraw>> cases;
  const auto add = [&](const char* name, auto&& change) {
    PrimitiveDraw draw = good;
    change(draw);
    cases.emplace_back(name, draw);
  };
  add("null buffer", [](PrimitiveDraw& d) { d.vertices = nullptr; });
  add("empty buffer", [&](PrimitiveDraw& d) { d.vertices = &empty; });
  add("no VERTEX_BUFFER usage",
      [&](PrimitiveDraw& d) { d.vertices = &storage.value(); });
  add("one past the end", [](PrimitiveDraw& d) { d.first_vertex = 1; });
  add("past the end", [](PrimitiveDraw& d) { d.first_vertex = 2; });
  add("wraps uint32", [](PrimitiveDraw& d) {
    d.first_vertex = std::numeric_limits<uint32_t>::max();
  });
  add("unknown topology",
      [](PrimitiveDraw& d) { d.topology = static_cast<PrimitiveTopology>(7); });
  add("odd line count",
      [](PrimitiveDraw& d) { d.topology = PrimitiveTopology::Lines; });
  add("zero point size", [](PrimitiveDraw& d) { d.point_size = 0.0f; });
  add("negative point size", [](PrimitiveDraw& d) { d.point_size = -2.0f; });
  add("NaN point size", [](PrimitiveDraw& d) {
    d.point_size = std::numeric_limits<float>::quiet_NaN();
  });
  add("infinite point size", [](PrimitiveDraw& d) {
    d.point_size = std::numeric_limits<float>::infinity();
  });
  add("viewport wider than the device's", [&](PrimitiveDraw& d) {
    d.viewport.extent.width = limits.maxViewportDimensions[0] + 1;
  });
  add("viewport taller than the device's", [&](PrimitiveDraw& d) {
    d.viewport.extent.height = limits.maxViewportDimensions[1] + 1;
  });
  add("viewport x before the bounds range", [](PrimitiveDraw& d) {
    d.viewport.offset.x = std::numeric_limits<int32_t>::min();
  });
  add("viewport y before the bounds range", [](PrimitiveDraw& d) {
    d.viewport.offset.y = std::numeric_limits<int32_t>::min();
  });
  add("viewport x past the bounds range", [](PrimitiveDraw& d) {
    d.viewport.offset.x = std::numeric_limits<int32_t>::max() - 8;
  });
  add("viewport y past the bounds range", [](PrimitiveDraw& d) {
    d.viewport.offset.y = std::numeric_limits<int32_t>::max() - 8;
  });

  for (const auto& [name, bad] : cases) {
    for (const bool bad_first : {false, true}) {
      const PrimitiveDraw draws[2] = {bad_first ? bad : good,
                                      bad_first ? good : bad};
      auto [status, bytes] = render_bytes(*pipeline, draws, 2);
      EXPECT_EQ(status.domain(), vkc::Status::Code::InvalidArgument)
          << name << ", bad first: " << bad_first;
      EXPECT_NE(status.message().find(bad_first ? "draw 0 " : "draw 1 "),
                std::string::npos)
          << status.message();
      ASSERT_EQ(bytes.size(), size_t{kSize} * kSize * 4) << name;
      EXPECT_EQ(Pixels{std::move(bytes)}.lit_count(), 0)
          << name << ", bad first: " << bad_first << ": nothing is recorded";
    }
  }

  auto [null_list, bytes] = render_bytes(*pipeline, nullptr, 1);
  EXPECT_EQ(null_list.domain(), vkc::Status::Code::InvalidArgument);
  ASSERT_EQ(bytes.size(), size_t{kSize} * kSize * 4);
  EXPECT_EQ(Pixels{std::move(bytes)}.lit_count(), 0);

  // Not errors: a range that ends at the buffer's end, a line's point size,
  // which is ignored, and a draw of no vertices, which needs no buffer.
  PrimitiveDraw last_two = good;
  last_two.first_vertex = 1;
  last_two.vertex_count = 2;
  PrimitiveDraw line = draw_of(buffer, 2, PrimitiveTopology::Lines);
  line.point_size = std::numeric_limits<float>::quiet_NaN();
  PrimitiveDraw nothing = good;
  nothing.vertices = nullptr;
  nothing.vertex_count = 0;
  const Pixels drawn = render(*pipeline, {last_two, line, nothing});
  EXPECT_TRUE(drawn.lit(30, 30)) << "the buffer's last vertex is drawn";
  EXPECT_GT(drawn.lit_count(), 2);
}

// --- largePoints -------------------------------------------------------------

class PrimitiveLargePointsTest : public PrimitiveRenderTest {
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
    PrimitiveRenderTest::SetUp();
  }

 private:
  vkc::DeviceRequirements requirements_ = vg::device_requirements();
};

// With largePoints, a point is a square of its size, centered on its pixel;
// a size below 1 still draws a pixel.
TEST_F(PrimitiveLargePointsTest, PointsCoverTheirSize) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  EXPECT_EQ(pipeline->max_point_size(),
            device().caps().limits().pointSizeRange[1]);
  if (pipeline->max_point_size() < 5.0f) {
    GTEST_SKIP() << "the device's points stop at " << pipeline->max_point_size()
                 << " px";
  }

  const vkc::Buffer big = upload({{{20.5f, 20.5f, 0.5f}, kRed}});
  PrimitiveDraw draw = draw_of(big, 1, PrimitiveTopology::Points);
  draw.point_size = 5.0f;
  const Pixels drawn = render(*pipeline, {draw});
  for (uint32_t y = 18; y <= 22; ++y) {
    for (uint32_t x = 18; x <= 22; ++x) {
      EXPECT_TRUE(drawn.lit(x, y)) << "(" << x << ", " << y << ")";
    }
  }
  EXPECT_EQ(drawn.lit_count(), 25);

  // A quarter-pixel point off the pixel's center covers no pixel center;
  // drawn at 1 px it covers one.
  const vkc::Buffer tiny = upload({{{20.9f, 20.5f, 0.5f}, kRed}});
  PrimitiveDraw small = draw_of(tiny, 1, PrimitiveTopology::Points);
  small.point_size = 0.25f;
  EXPECT_EQ(render(*pipeline, {small}).lit_count(), 1);
}

// A wide point stops at its viewport's edge, which only the scissor enforces:
// Vulkan clips a point by its center alone. Centered on the viewport's left
// column and on its top row, each 5 px point keeps 3 x 5 pixels inside.
TEST_F(PrimitiveLargePointsTest, WidePointsStopAtTheirViewport) {
  auto pipeline = make_pipeline();
  ASSERT_NE(pipeline, nullptr);
  if (pipeline->max_point_size() < 5.0f) {
    GTEST_SKIP() << "the device's points stop at " << pipeline->max_point_size()
                 << " px";
  }
  // The viewport spans pixels [16, 48) on both axes; NDC of pixel p's center.
  const VkRect2D rect{{16, 16}, {32, 32}};
  const auto ndc = [](float p) {
    return (p + 0.5f - 16.0f) / 32.0f * 2.0f - 1.0f;
  };
  const vkc::Buffer buffer = upload(
      {{{ndc(16), ndc(30), 0.5f}, kRed}, {{ndc(40), ndc(16), 0.5f}, kGreen}});
  PrimitiveDraw draw =
      draw_of(buffer, 2, PrimitiveTopology::Points, glm::mat4(1.0f), rect);
  draw.point_size = 5.0f;
  const Pixels drawn = render(*pipeline, {draw});
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const bool left = x >= 16 && x <= 18 && y >= 28 && y <= 32;
      const bool top = x >= 38 && x <= 42 && y >= 16 && y <= 18;
      EXPECT_EQ(drawn.lit(x, y), left || top) << "(" << x << ", " << y << ")";
    }
  }
}

}  // namespace

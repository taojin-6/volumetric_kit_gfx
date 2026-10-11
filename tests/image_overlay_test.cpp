// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImageOverlay on a CPU-only ImGui frame (no device): the image-to-display
// mapping, the viewport clip, and the geometry each mark adds to a draw list.

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <glm/common.hpp>
#include <glm/vec2.hpp>

#include "glm_test_util.hpp"
#include "imgui.h"
#include "ui_test_support.hpp"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/ui/image_overlay.hpp"

namespace ui = volumetric_kit::gfx::ui;
namespace camera = volumetric_kit::gfx::camera;

namespace {

using vg_test::expect_near;

constexpr ImU32 kRed = IM_COL32(255, 0, 0, 255);
constexpr ImU32 kGreen = IM_COL32(0, 255, 0, 255);
constexpr ImU32 kBlue = IM_COL32(0, 0, 255, 255);
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// A 640 x 480 image fitted in a 1280 x 960 viewport at (40, 20): 2 target
// pixels per texel, image (0, 0) at target (40, 20).
camera::ImageView2D make_view() {
  return camera::ImageView2D({640.0f, 480.0f}, {40.0f, 20.0f},
                             {1280.0f, 960.0f});
}

// The vertices added from index `first` on that carry color `col`.
struct Added {
  glm::vec2 min{std::numeric_limits<float>::max()};
  glm::vec2 max{std::numeric_limits<float>::lowest()};
  glm::vec2 sum{0.0f, 0.0f};
  int count = 0;
};

Added added(const ImDrawList& list, int first, ImU32 col) {
  Added a;
  for (int i = first; i < list.VtxBuffer.Size; ++i) {
    const ImDrawVert& v = list.VtxBuffer[i];
    if (v.col != col) continue;
    const glm::vec2 p(v.pos.x, v.pos.y);
    a.min = glm::min(a.min, p);
    a.max = glm::max(a.max, p);
    a.sum += p;
    ++a.count;
  }
  return a;
}

// A headless 800 x 600 ImGui frame. Anti-aliasing is off, so every vertex of a
// mark carries its color and sits on its outline.
class ImageOverlayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ImGui::GetStyle().AntiAliasedFill = false;
    ImGui::GetStyle().AntiAliasedLines = false;
  }

  void TearDown() override {
    if (in_frame_) ImGui::EndFrame();
  }

  // The scale is set after NewFrame, which would otherwise bake the font at
  // that density on this backend-less atlas; an overlay reads it when made.
  void begin_frame(glm::vec2 framebuffer_scale) {
    ImGui::NewFrame();
    in_frame_ = true;
    set_framebuffer_scale(framebuffer_scale);
  }

  static void set_framebuffer_scale(glm::vec2 scale) {
    ImGui::GetIO().DisplayFramebufferScale = ImVec2(scale.x, scale.y);
  }

  static ImDrawList& background() { return *ImGui::GetBackgroundDrawList(); }

 private:
  vg_test::HeadlessImGui imgui_{ImVec2(800.0f, 600.0f)};
  bool in_frame_ = false;
};

// OpenCV's (0, 0), the first texel's center, lands at origin + 0.5 * scale
// target pixels, divided by the framebuffer scale; Vulkan's is the corner.
TEST_F(ImageOverlayTest, MapsTexelCentersToDisplayCoordinates) {
  begin_frame({2.0f, 2.0f});
  const camera::ImageView2D view = make_view();
  const ui::ImageOverlay integer(view, ui::PixelCenters::Integer);
  const ui::ImageOverlay half(view, ui::PixelCenters::Half);
  expect_near(integer.to_display({0.0f, 0.0f}),
              (view.origin() + 0.5f * view.scale()) / 2.0f);
  expect_near(integer.to_display({0.0f, 0.0f}), {20.5f, 10.5f});
  expect_near(half.to_display({0.0f, 0.0f}), view.origin() / 2.0f);

  // The dot is drawn there, its radius in display pixels.
  const int first = background().VtxBuffer.Size;
  integer.point({0.0f, 0.0f}, kRed, 3.0f);
  const Added dot = added(background(), first, kRed);
  ASSERT_GT(dot.count, 0);
  expect_near(dot.sum / static_cast<float>(dot.count), {20.5f, 10.5f}, 1e-3f);
  expect_near(dot.max - dot.min, {6.0f, 6.0f}, 0.1f);
}

// The mapping follows a panned, zoomed view and divides per axis; a
// framebuffer scale ImGui was not given (non-positive or non-finite) counts as
// 1 rather than sending marks to infinity.
TEST_F(ImageOverlayTest, FollowsTheViewAndFramebufferScale) {
  begin_frame({2.0f, 1.5f});
  camera::ImageView2D view = make_view();
  view.zoom_about({300.0f, 200.0f}, 3.7f);
  view.pan({-55.0f, 31.0f});
  const ui::ImageOverlay overlay(view, ui::PixelCenters::Integer);
  for (const glm::vec2 p : {glm::vec2{0.0f, 0.0f}, glm::vec2{17.25f, 3.5f},
                            glm::vec2{639.0f, 479.0f}}) {
    expect_near(overlay.to_display(p),
                view.image_to_target(p + 0.5f) / glm::vec2(2.0f, 1.5f), 1e-3f);
  }

  for (const glm::vec2 unusable :
       {glm::vec2{0.0f, kNaN}, glm::vec2{kInf, -1.0f}}) {
    set_framebuffer_scale(unusable);
    expect_near(ui::ImageOverlay(view, ui::PixelCenters::Half)
                    .to_display({10.0f, 20.0f}),
                view.image_to_target({10.0f, 20.0f}), 1e-3f);
  }
}

// Every mark's draw command is clipped to the viewport in display coordinates
// -- (40, 20) + (1280, 960) target pixels, halved: x 20 to 660, y 10 to 490 --
// within the list's own clip, which is restored after each mark.
TEST_F(ImageOverlayTest, MarksAreClippedToTheViewport) {
  begin_frame({2.0f, 2.0f});
  ImDrawList& list = background();
  list.PushClipRect(ImVec2(0.0f, 50.0f), ImVec2(700.0f, 300.0f));
  const ImVec2 outer_min = list.GetClipRectMin();
  const ImVec2 outer_max = list.GetClipRectMax();
  const ui::ImageOverlay overlay(make_view(), ui::PixelCenters::Integer);
  const glm::vec2 line[] = {{0.0f, 100.0f}, {100.0f, 150.0f}};
  const ImU32 fills[] = {kGreen};
  overlay.point({10.0f, 100.0f}, kRed);
  overlay.polyline(line, 2, kRed);
  overlay.label({10.0f, 100.0f}, "id", kRed);
  overlay.grid(1, 1, fills, 1, kBlue);

  int drawn = 0;
  for (const ImDrawCmd& cmd : list.CmdBuffer) {
    if (cmd.ElemCount == 0) continue;
    ++drawn;
    EXPECT_FLOAT_EQ(cmd.ClipRect.x, 20.0f);
    EXPECT_FLOAT_EQ(cmd.ClipRect.y, 50.0f);
    EXPECT_FLOAT_EQ(cmd.ClipRect.z, 660.0f);
    EXPECT_FLOAT_EQ(cmd.ClipRect.w, 300.0f);
  }
  EXPECT_GT(drawn, 0);
  EXPECT_EQ(list.GetClipRectMin().x, outer_min.x);
  EXPECT_EQ(list.GetClipRectMin().y, outer_min.y);
  EXPECT_EQ(list.GetClipRectMax().x, outer_max.x);
  EXPECT_EQ(list.GetClipRectMax().y, outer_max.y);
  list.PopClipRect();
}

// A label wholly outside the viewport adds nothing, not even its background
// box, however far off (ImGui's text layout truncates a position to an int);
// one in it adds glyphs from its anchor on, over a box that pads them.
TEST_F(ImageOverlayTest, LabelsOnlyInTheViewport) {
  begin_frame({1.0f, 1.0f});
  ImDrawList& list = background();
  // 320 x 240 texels at (100, 100), scale 1: the viewport ends at (420, 340).
  const ui::ImageOverlay overlay(
      camera::ImageView2D({320.0f, 240.0f}, {100.0f, 100.0f}, {320.0f, 240.0f}),
      ui::PixelCenters::Half);

  int first = list.VtxBuffer.Size;
  overlay.label({330.0f, 10.0f}, "x", kRed, kBlue);    // right of it
  overlay.label({-1e30f, 10.0f}, "far", kRed, kBlue);  // far off to the left
  overlay.label({10.0f, 10.0f}, "x", kRed, kBlue, {-200.0f, 0.0f});  // offset
  EXPECT_EQ(list.VtxBuffer.Size, first);

  overlay.label({10.0f, 10.0f}, "x", kRed);
  const Added glyphs = added(list, first, kRed);
  EXPECT_GT(glyphs.count, 0);
  EXPECT_GE(glyphs.min.x, 110.0f);
  EXPECT_GE(glyphs.min.y, 110.0f);

  first = list.VtxBuffer.Size;
  overlay.label({10.0f, 10.0f}, "Wg", kRed, kBlue, {4.0f, 4.0f});
  const Added box = added(list, first, kBlue);
  const Added text = added(list, first, kRed);
  ASSERT_EQ(box.count, 4);
  ASSERT_GT(text.count, 0);
  expect_near(box.min, {112.0f, 112.0f});  // anchor + offset - 2 px padding
  EXPECT_LE(box.min.x, text.min.x);
  EXPECT_LE(box.min.y, text.min.y);
  EXPECT_GE(box.max.x, text.max.x);
  EXPECT_GE(box.max.y, text.max.y);             // the descender too
  EXPECT_EQ(list.VtxBuffer[first].col, kBlue);  // drawn first, under the text
}

// A fractional anchor is truncated to a display pixel, where ImGui draws the
// text, so the box pads it evenly -- unless the list keeps text unsnapped.
TEST_F(ImageOverlayTest, LabelsSnapAsImGuiTextDoes) {
  begin_frame({1.0f, 1.0f});
  ImDrawList& list = background();
  const ui::ImageOverlay overlay(
      camera::ImageView2D({320.0f, 240.0f}, {100.0f, 100.0f}, {320.0f, 240.0f}),
      ui::PixelCenters::Half);

  int first = list.VtxBuffer.Size;
  overlay.label({10.9f, 10.25f}, "x", kRed, kBlue);
  expect_near(added(list, first, kBlue).min, {108.0f, 108.0f}, 1e-3f);

  list.Flags |= ImDrawListFlags_TextNoPixelSnap;
  first = list.VtxBuffer.Size;
  overlay.label({10.9f, 10.25f}, "x", kRed, kBlue);
  expect_near(added(list, first, kBlue).min, {108.9f, 108.25f}, 1e-3f);
}

TEST_F(ImageOverlayTest, PolylineStrokesThroughItsPoints) {
  begin_frame({1.0f, 1.0f});
  ImDrawList& list = background();
  const ui::ImageOverlay overlay(
      camera::ImageView2D({320.0f, 240.0f}, {0.0f, 0.0f}, {320.0f, 240.0f}),
      ui::PixelCenters::Integer);
  const glm::vec2 points[] = {{10.0f, 10.0f}, {100.0f, 10.0f}, {100.0f, 60.0f}};

  int first = list.VtxBuffer.Size;
  overlay.polyline(points, 3, kRed, 2.0f);
  const Added open = added(list, first, kRed);
  ASSERT_GT(open.count, 0);
  // Each leg is a quad half the thickness either side of its points (+ 0.5
  // for integer centers), with no end caps.
  expect_near(open.min, {10.5f, 9.5f}, 1e-2f);
  expect_near(open.max, {101.5f, 60.5f}, 1e-2f);

  first = list.VtxBuffer.Size;
  overlay.polyline(points, 3, kRed, 2.0f, true);
  EXPECT_GT(added(list, first, kRed).count, open.count);  // the closing leg
}

struct StrokeStyle {
  const char* name;
  ImDrawListFlags flags;
  float thickness;
};

using LargePolylineParam = std::tuple<StrokeStyle, bool, bool>;
class LargePolylineTest
    : public ImageOverlayTest,
      public ::testing::WithParamInterface<LargePolylineParam> {};

// Resolve indices as the renderer does, including each command's vertex
// offset. Every vertex of the mark must be reachable, and its indices must
// never refer to a different mark, even if those indices are in bounds.
void expect_referenced_vertices(const ImDrawList& list, int first_index,
                                int last_index, int first_vertex,
                                int last_vertex) {
  ASSERT_GT(last_index, first_index);
  ASSERT_GT(last_vertex, first_vertex);
  std::vector<bool> referenced(last_vertex - first_vertex, false);
  for (const ImDrawCmd& cmd : list.CmdBuffer) {
    const unsigned int begin =
        std::max(cmd.IdxOffset, static_cast<unsigned int>(first_index));
    const unsigned int end = std::min(cmd.IdxOffset + cmd.ElemCount,
                                      static_cast<unsigned int>(last_index));
    for (unsigned int i = begin; i < end; ++i) {
      const size_t vertex =
          static_cast<size_t>(cmd.VtxOffset) + list.IdxBuffer[i];
      ASSERT_GE(vertex, static_cast<size_t>(first_vertex));
      ASSERT_LT(vertex, static_cast<size_t>(last_vertex));
      referenced[vertex - first_vertex] = true;
    }
  }
  EXPECT_EQ(std::count(referenced.begin(), referenced.end(), true),
            last_vertex - first_vertex);
}

// A single stroke exceeds 64K vertices in every tessellation mode. Vertex
// offsets alone cannot split that allocation; 16-bit indices wrap inside it
// and, on the AA paths, also corrupt the point drawn afterward.
TEST_P(LargePolylineTest, ReferencesEntirePathAndFollowingPoint) {
  const auto& [style, closed, vertex_offsets] = GetParam();
  if (vertex_offsets) {
    ImGui::GetIO().BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
  }
  begin_frame({1.0f, 1.0f});
  ImDrawList& list = background();
  list.Flags &= ~(ImDrawListFlags_AntiAliasedLines |
                  ImDrawListFlags_AntiAliasedLinesUseTex);
  list.Flags |= style.flags;
  const ui::ImageOverlay overlay(make_view(), ui::PixelCenters::Integer);
  // Include existing geometry so the stroke need not start at vertex zero.
  overlay.point({5.0f, 5.0f}, kBlue);

  std::vector<glm::vec2> points(33000);
  for (size_t i = 0; i < points.size(); ++i) {
    points[i] = {static_cast<float>(i % 600),
                 20.0f + static_cast<float>(i / 600)};
  }
  const int path_vertex = list.VtxBuffer.Size;
  const int path_index = list.IdxBuffer.Size;
  overlay.polyline(points.data(), points.size(), kRed, style.thickness, closed);
  const int point_vertex = list.VtxBuffer.Size;
  const int point_index = list.IdxBuffer.Size;
  ASSERT_GT(point_vertex - path_vertex, 65536);
  expect_referenced_vertices(list, path_index, point_index, path_vertex,
                             point_vertex);

  overlay.point({200.0f, 200.0f}, kGreen);
  expect_referenced_vertices(list, point_index, list.IdxBuffer.Size,
                             point_vertex, list.VtxBuffer.Size);
}

INSTANTIATE_TEST_SUITE_P(
    StrokeModes, LargePolylineTest,
    ::testing::Combine(
        ::testing::Values(
            StrokeStyle{"NoAA", ImDrawListFlags_None, 2.0f},
            StrokeStyle{"TexturedAA",
                        ImDrawListFlags_AntiAliasedLines |
                            ImDrawListFlags_AntiAliasedLinesUseTex,
                        2.0f},
            StrokeStyle{"ThinAA", ImDrawListFlags_AntiAliasedLines, 1.0f},
            StrokeStyle{"ThickAA", ImDrawListFlags_AntiAliasedLines, 2.5f}),
        ::testing::Bool(), ::testing::Bool()),
    ([](const ::testing::TestParamInfo<LargePolylineParam>& info) {
      const auto& [style, closed, vertex_offsets] = info.param;
      return std::string(style.name) + (closed ? "Closed" : "Open") +
             (vertex_offsets ? "WithOffsets" : "WithoutOffsets");
    }));

// The cells span the image's edges whatever the pixel-center convention, each
// filled with its own color in row-major order (a transparent one, or one past
// the fills given, left empty), with borders centered on the edges across the
// whole image.
TEST_F(ImageOverlayTest, GridSpansTheImageEdges) {
  begin_frame({2.0f, 2.0f});
  ImDrawList& list = background();
  const camera::ImageView2D view = make_view();
  const ui::ImageOverlay overlay(view, ui::PixelCenters::Integer);
  const auto display = [&](glm::vec2 p) {
    return view.image_to_target(p) / 2.0f;
  };

  // 3 x 2 cells, given 3 fills: (column 2, row 0) is green, (1, 1) unfilled.
  const ImU32 fills[] = {0, 0, kGreen, 0, kRed, 0};
  const int first = list.VtxBuffer.Size;
  overlay.grid(3, 2, fills, 3, kBlue, 1.0f);

  const Added cell = added(list, first, kGreen);
  ASSERT_EQ(cell.count, 4);
  expect_near(cell.min, display({640.0f * 2.0f / 3.0f, 0.0f}), 1e-3f);
  expect_near(cell.max, display({640.0f, 240.0f}));
  EXPECT_EQ(added(list, first, kRed).count, 0);

  const Added borders = added(list, first, kBlue);
  EXPECT_EQ(borders.count, 7 * 4);  // 4 + 3 lines, a quad each
  expect_near(borders.min, display({0.0f, 0.0f}) - 0.5f, 1e-2f);
  expect_near(borders.max, display({640.0f, 480.0f}) + 0.5f, 1e-2f);
}

// A non-finite coordinate on either axis, a NaN radius, too few points, null
// or empty text, a non-positive grid size, or no visible fills or borders add
// nothing; a polyline cut short drops the path it began.
TEST_F(ImageOverlayTest, DegenerateMarksAddNothing) {
  begin_frame({1.0f, 1.0f});
  ImDrawList& list = background();
  const ui::ImageOverlay overlay(make_view(), ui::PixelCenters::Integer);
  const glm::vec2 bad_x[] = {{0.0f, 0.0f}, {kNaN, 1.0f}, {5.0f, 5.0f}};
  const glm::vec2 bad_y[] = {{0.0f, 0.0f}, {1.0f, kInf}, {5.0f, 5.0f}};
  const ImU32 fills[] = {kGreen};

  const int first = list.VtxBuffer.Size;
  overlay.point({kNaN, 0.0f}, kRed);
  overlay.point({0.0f, kNaN}, kRed);
  overlay.point({0.0f, 0.0f}, kRed, kNaN);
  overlay.polyline(bad_x, 3, kRed);
  overlay.polyline(bad_y, 3, kRed);
  overlay.polyline(bad_x, 1, kRed);
  overlay.polyline(nullptr, 3, kRed);
  overlay.label({0.0f, kNaN}, "x", kRed, kBlue);
  overlay.label({0.0f, kInf}, "x", kRed, kBlue);
  overlay.label({0.0f, 0.0f}, nullptr, kRed, kBlue);
  overlay.label({0.0f, 0.0f}, "", kRed, kBlue);
  overlay.grid(0, 1, fills, 1, kBlue);
  overlay.grid(1, -1, fills, 1, kBlue);
  overlay.grid(1, 1, fills, 0, 0);    // no fills given, transparent borders
  overlay.grid(1, 1, nullptr, 1, 0);  // null fills
  EXPECT_EQ(list.VtxBuffer.Size, first);

  // A polyline after a cut-short one strokes only its own leg.
  overlay.polyline(bad_y, 3, kRed);
  const glm::vec2 leg[] = {{10.0f, 10.0f}, {20.0f, 10.0f}};
  overlay.polyline(leg, 2, kGreen);
  EXPECT_EQ(added(list, first, kGreen).count, 4);
}

// A radius or thickness is clamped to 10000 display pixels, so ImGui never
// converts one past INT_MAX to an int (undefined; UBSan flags it). Extents are
// checked to 0.1%: on x86 ImGui normalizes strokes with an approximate rsqrt.
TEST_F(ImageOverlayTest, SizesAreClamped) {
  ImGui::GetStyle().AntiAliasedLines = true;  // the path taking (int)thickness
  begin_frame({1.0f, 1.0f});
  ImDrawList& list = background();
  const ui::ImageOverlay overlay(
      camera::ImageView2D({320.0f, 240.0f}, {0.0f, 0.0f}, {320.0f, 240.0f}),
      ui::PixelCenters::Half);

  int first = list.VtxBuffer.Size;
  overlay.point({100.0f, 100.0f}, kRed, 3e9f);
  const Added dot = added(list, first, kRed);
  ASSERT_GT(dot.count, 0);
  expect_near(dot.max - dot.min, {2e4f, 2e4f}, 20.0f);

  first = list.VtxBuffer.Size;
  const glm::vec2 leg[] = {{0.0f, 100.0f}, {100.0f, 100.0f}};
  overlay.polyline(leg, 2, kGreen, 3e9f);
  const Added line = added(list, first, kGreen);
  ASSERT_GT(line.count, 0);
  EXPECT_NEAR(line.max.y - line.min.y, 1e4f, 10.0f);

  first = list.VtxBuffer.Size;
  overlay.grid(1, 1, nullptr, 0, kBlue, 3e9f);
  const Added borders = added(list, first, kBlue);
  ASSERT_GT(borders.count, 0);
  EXPECT_NEAR(borders.max.x - borders.min.x, 320.0f + 1e4f, 10.0f);
}

// A given draw list is drawn into; the background list is the default.
TEST_F(ImageOverlayTest, DrawsIntoTheGivenList) {
  begin_frame({1.0f, 1.0f});
  ImDrawList& back = background();
  ImDrawList& front = *ImGui::GetForegroundDrawList();
  const int back_first = back.VtxBuffer.Size;
  const int front_first = front.VtxBuffer.Size;

  ui::ImageOverlay(make_view(), ui::PixelCenters::Half, &front)
      .point({1.0f, 1.0f}, kRed);
  EXPECT_EQ(back.VtxBuffer.Size, back_first);
  EXPECT_GT(front.VtxBuffer.Size, front_first);

  ui::ImageOverlay(make_view(), ui::PixelCenters::Half)
      .point({1.0f, 1.0f}, kRed);
  EXPECT_GT(back.VtxBuffer.Size, back_first);
}

}  // namespace

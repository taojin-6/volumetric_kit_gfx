// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImageOverlay on a CPU-only ImGui frame (no device): the image-to-display
// mapping, the viewport clip, and the geometry each mark adds to a draw list.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <ostream>
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
// the fills given, left empty), with borders across the whole image: the outer
// ones just inside its edges.
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
  expect_near(borders.min, display({0.0f, 0.0f}), 1e-2f);
  expect_near(borders.max, display({640.0f, 480.0f}), 1e-2f);
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

  // The outer borders, 1e4 wide inside the image's edges: from 320 - 1e4 to
  // 1e4.
  first = list.VtxBuffer.Size;
  overlay.grid(1, 1, nullptr, 0, kBlue, 3e9f);
  const Added borders = added(list, first, kBlue);
  ASSERT_GT(borders.count, 0);
  EXPECT_NEAR(borders.max.x - borders.min.x, 2e4f - 320.0f, 20.0f);
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

// A mark given in a region's coordinates lands at the region's origin plus its
// point (+ 0.5 for integer centers), however the view pans and zooms; with no
// region, the coordinates are the whole image's.
TEST_F(ImageOverlayTest, MarksLandInTheirRegion) {
  const glm::vec2 fb(2.0f, 1.5f);
  begin_frame(fb);
  camera::ImageView2D view = make_view();
  view.zoom_about({900.0f, 700.0f}, 2.5f);
  view.pan({-40.0f, 25.0f});
  const ui::OverlayRegion region{{336.0f, 256.0f}, {304.0f, 224.0f}};
  const ui::ImageOverlay integer(view, ui::PixelCenters::Integer, region);
  const ui::ImageOverlay half(view, ui::PixelCenters::Half, region);
  const ui::ImageOverlay whole(view, ui::PixelCenters::Integer);
  for (const glm::vec2 p : {glm::vec2{0.0f, 0.0f}, glm::vec2{17.25f, 3.5f},
                            glm::vec2{303.0f, 223.0f}}) {
    expect_near(integer.to_display(p),
                view.image_to_target(region.origin + p + 0.5f) / fb, 1e-3f);
    expect_near(half.to_display(p),
                view.image_to_target(region.origin + p) / fb, 1e-3f);
    expect_near(whole.to_display(region.origin + p), integer.to_display(p),
                1e-3f);
  }

  // The dot is drawn there.
  const int first = background().VtxBuffer.Size;
  integer.point({10.0f, 20.0f}, kRed, 3.0f);
  const Added dot = added(background(), first, kRed);
  ASSERT_GT(dot.count, 0);
  expect_near(
      dot.sum / static_cast<float>(dot.count),
      view.image_to_target(region.origin + glm::vec2(10.5f, 20.5f)) / fb,
      1e-2f);
}

// edge_to_display takes the region's points in Vulkan's coordinates, texel
// edges on integers: half a texel before OpenCV's centers, at Vulkan's.
TEST_F(ImageOverlayTest, EdgesSitHalfATexelBeforeIntegerCenters) {
  const glm::vec2 fb(2.0f, 1.5f);
  begin_frame(fb);
  camera::ImageView2D view = make_view();
  view.zoom_about({300.0f, 200.0f}, 3.7f);
  view.pan({-55.0f, 31.0f});
  const ui::OverlayRegion region{{100.0f, 50.0f}, {200.0f, 150.0f}};
  const ui::ImageOverlay integer(view, ui::PixelCenters::Integer, region);
  const ui::ImageOverlay half(view, ui::PixelCenters::Half, region);
  for (const glm::vec2 p : {glm::vec2{0.0f, 0.0f}, glm::vec2{17.25f, 3.5f},
                            glm::vec2{199.0f, 149.0f}}) {
    expect_near(integer.edge_to_display(p), integer.to_display(p - 0.5f),
                1e-3f);
    expect_near(half.edge_to_display(p), half.to_display(p), 1e-3f);
  }
  // The region's corner is its origin, whatever the convention.
  expect_near(integer.edge_to_display({0.0f, 0.0f}),
              view.image_to_target(region.origin) / fb, 1e-3f);
  expect_near(half.edge_to_display({0.0f, 0.0f}),
              integer.edge_to_display({0.0f, 0.0f}), 1e-3f);
}

// Marks are clipped to the region within the viewport, clip_min to clip_max
// in display coordinates, which every mark's draw command takes: a region in
// view, and the same one panned so the viewport's edge cuts it. Marks past
// the region's edges are cut there.
TEST_F(ImageOverlayTest, MarksAreClippedToTheRegionInTheViewport) {
  begin_frame({2.0f, 2.0f});
  ImDrawList& list = background();
  // 2 target pixels per texel from (40, 20), halved: display = (20, 10) +
  // texel; the viewport is x 20 to 660, y 10 to 490.
  const ui::OverlayRegion region{{100.0f, 60.0f}, {200.0f, 150.0f}};
  struct Case {
    glm::vec2 pan;
    glm::vec2 min;
    glm::vec2 max;
  };
  const Case cases[] = {
      {{0.0f, 0.0f}, {120.0f, 70.0f}, {320.0f, 220.0f}},
      // 150 texels left: the region starts off the viewport's left edge.
      {{-300.0f, 0.0f}, {20.0f, 70.0f}, {170.0f, 220.0f}},
  };
  const glm::vec2 line[] = {{-20.0f, 10.0f}, {250.0f, 10.0f}};
  for (const Case& c : cases) {
    camera::ImageView2D view = make_view();
    view.pan(c.pan);
    const ui::ImageOverlay overlay(view, ui::PixelCenters::Integer, region);
    expect_near(overlay.clip_min(), c.min, 1e-3f);
    expect_near(overlay.clip_max(), c.max, 1e-3f);

    const auto first = static_cast<unsigned int>(list.IdxBuffer.Size);
    overlay.point({-10.0f, -10.0f}, kRed);  // outside the region
    overlay.point({50.0f, 50.0f}, kRed);
    overlay.polyline(line, 2, kRed);                     // past both sides
    overlay.label({190.0f, 140.0f}, "id", kRed, kBlue);  // past the corner
    overlay.grid(2, 2, nullptr, 0, kBlue);
    overlay.caption("cam", kRed, kBlue);
    int drawn = 0;
    for (const ImDrawCmd& cmd : list.CmdBuffer) {
      if (cmd.ElemCount == 0 || cmd.IdxOffset < first) continue;
      ++drawn;
      EXPECT_FLOAT_EQ(cmd.ClipRect.x, c.min.x);
      EXPECT_FLOAT_EQ(cmd.ClipRect.y, c.min.y);
      EXPECT_FLOAT_EQ(cmd.ClipRect.z, c.max.x);
      EXPECT_FLOAT_EQ(cmd.ClipRect.w, c.max.y);
    }
    EXPECT_GT(drawn, 0);
  }
}

// The clip is on whole target pixels, those whose centers lie in the region,
// so ImGui's Vulkan backend -- which multiplies it by the framebuffer scale
// and truncates its corner and size -- scissors exactly those, at any scale.
TEST_F(ImageOverlayTest, ClipScissorsTheRegionsPixelsAtAnyScale) {
  begin_frame({1.0f, 1.0f});
  // 1 target pixel per texel.
  const camera::ImageView2D view({4096.0f, 4096.0f}, {0.0f, 0.0f},
                                 {4096.0f, 4096.0f});
  for (const float scale :
       {1.0f, 1.25f, 1.5f, 1.75f, 2.0f, 1.1f, 4.0f / 3.0f}) {
    set_framebuffer_scale({scale, scale});
    int wrong = 0;
    for (int k = 0; k < 2000; ++k) {
      const float step = static_cast<float>(k);
      const ui::OverlayRegion region{{0.37f * step, 1.13f * step},
                                     {17.0f + 0.29f * (step / 7.0f), 23.5f}};
      const ui::ImageOverlay overlay(view, ui::PixelCenters::Half, region);
      for (int axis = 0; axis < 2; ++axis) {
        const float lo = std::ceil(region.origin[axis] - 0.5f);
        const float hi =
            std::ceil(region.origin[axis] + region.size[axis] - 0.5f);
        EXPECT_NEAR(overlay.clip_min()[axis], lo / scale, 1e-3f);
        // As the backend projects it, a statement at a time.
        const float start = overlay.clip_min()[axis] * scale;
        const float end = overlay.clip_max()[axis] * scale;
        const auto offset = static_cast<int32_t>(start);
        const auto extent = static_cast<uint32_t>(end - start);
        if (offset != static_cast<int32_t>(lo) ||
            static_cast<int64_t>(offset) + extent != static_cast<int64_t>(hi)) {
          ++wrong;
        }
      }
    }
    EXPECT_EQ(wrong, 0) << "scale " << scale;
  }
}

// A region out of view, with no width, a negative height or a non-finite
// component, one whose mapping overflows, or a view with an empty viewport,
// leaves a finite, empty clip rectangle, and no mark adds anything. A
// non-finite region still maps through as given.
TEST_F(ImageOverlayTest, AnEmptyClipAddsNothing) {
  begin_frame({1.0f, 1.0f});
  ImDrawList& list = background();
  const camera::ImageView2D view = make_view();  // fills its viewport
  const camera::ImageView2D shut({640.0f, 480.0f}, {40.0f, 20.0f},
                                 {0.0f, 0.0f});
  camera::ImageView2D zoomed = make_view();  // 64 target pixels per texel
  zoomed.zoom_about({0.0f, 0.0f}, 1e9f);
  const ui::ImageOverlay overlays[] = {
      {view, ui::PixelCenters::Integer, {{700.0f, 0.0f}, {100.0f, 100.0f}}},
      {view, ui::PixelCenters::Integer, {{10.0f, 10.0f}, {0.0f, 50.0f}}},
      {view, ui::PixelCenters::Integer, {{10.0f, 10.0f}, {50.0f, -5.0f}}},
      {view, ui::PixelCenters::Integer, {{kNaN, 10.0f}, {50.0f, 50.0f}}},
      {view, ui::PixelCenters::Integer, {{10.0f, 10.0f}, {kInf, 50.0f}}},
      {view, ui::PixelCenters::Integer, {{3e38f, 0.0f}, {100.0f, 100.0f}}},
      {zoomed, ui::PixelCenters::Integer, {{1e37f, 1e37f}, {10.0f, 10.0f}}},
      {shut, ui::PixelCenters::Integer},
  };
  const glm::vec2 nan_mapped = overlays[3].to_display({1.0f, 1.0f});
  EXPECT_TRUE(std::isnan(nan_mapped.x));
  const glm::vec2 leg[] = {{10.0f, 10.0f}, {20.0f, 10.0f}};
  const ImU32 fills[] = {kGreen};

  const int first = list.VtxBuffer.Size;
  for (const ui::ImageOverlay& overlay : overlays) {
    const glm::vec2 min = overlay.clip_min();
    const glm::vec2 max = overlay.clip_max();
    EXPECT_TRUE(std::isfinite(min.x) && std::isfinite(min.y) &&
                std::isfinite(max.x) && std::isfinite(max.y));
    EXPECT_GE(max.x, min.x);
    EXPECT_GE(max.y, min.y);
    EXPECT_TRUE(max.x == min.x || max.y == min.y);
    overlay.point({1.0f, 1.0f}, kRed);
    overlay.polyline(leg, 2, kRed);
    overlay.label({1.0f, 1.0f}, "x", kRed, kBlue);
    overlay.caption("x", kRed, kBlue);
    overlay.grid(1, 1, fills, 1, kBlue);
  }
  EXPECT_EQ(list.VtxBuffer.Size, first);

  // Nor does text reaching past an empty clip into the list's own clip.
  list.PushClipRect(ImVec2(100.0f, 0.0f), ImVec2(800.0f, 600.0f));
  const ui::ImageOverlay beside(
      camera::ImageView2D({640.0f, 480.0f}, {0.0f, 0.0f}, {640.0f, 480.0f}),
      ui::PixelCenters::Half, {{50.0f, 0.0f}, {0.0f, 100.0f}});
  beside.label({0.0f, 10.0f}, "long id", kRed, kBlue, {40.0f, 0.0f});
  beside.caption("a caption", kRed, kBlue);
  list.PopClipRect();
  EXPECT_EQ(list.VtxBuffer.Size, first);
}

// The cells span the region's edges whatever the pixel-center convention:
// cell (2, 0) of 3 x 2 over a 300 x 200 region covers its texels
// [200, 300) x [0, 100). The borders span the region's pixels, its clip: the
// inner ones centered on the cell edges, the outer ones, whole, along its
// outermost pixels -- at a fractional origin too, and when thick.
TEST_F(ImageOverlayTest, GridSpansTheRegion) {
  const glm::vec2 fb(2.0f, 2.0f);
  begin_frame(fb);
  ImDrawList& list = background();
  const camera::ImageView2D view = make_view();
  struct Case {
    glm::vec2 origin;
    float thickness;
  };
  const Case cases[] = {
      {{64.0f, 48.0f}, 1.0f},
      // Target (168.6, 117.4): the region's pixels start at (169, 117).
      {{64.3f, 48.7f}, 1.0f},
      {{64.3f, 48.7f}, 3.0f},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "origin " << c.origin.x << ", " << c.origin.y
                 << ", thickness " << c.thickness);
    const ui::OverlayRegion region{c.origin, {300.0f, 200.0f}};
    const ui::ImageOverlay overlay(view, ui::PixelCenters::Integer, region);
    const auto display = [&](glm::vec2 p) {
      return view.image_to_target(region.origin + p) / fb;
    };
    const glm::vec2 lo =
        glm::ceil(view.image_to_target(region.origin) - 0.5f) / fb;
    const glm::vec2 hi =
        glm::ceil(view.image_to_target(region.origin + region.size) - 0.5f) /
        fb;
    expect_near(overlay.clip_min(), lo, 1e-3f);
    expect_near(overlay.clip_max(), hi, 1e-3f);

    const ImU32 fills[] = {0, 0, kGreen};
    const int first = list.VtxBuffer.Size;
    overlay.grid(3, 2, fills, 3, kBlue, c.thickness);

    const Added cell = added(list, first, kGreen);
    ASSERT_EQ(cell.count, 4);
    expect_near(cell.min, display({200.0f, 0.0f}), 1e-3f);
    expect_near(cell.max, display({300.0f, 100.0f}), 1e-3f);
    const Added borders = added(list, first, kBlue);
    EXPECT_EQ(borders.count, 7 * 4);  // 4 + 3 lines, a quad each
    expect_near(borders.min, lo, 1e-3f);
    expect_near(borders.max, hi, 1e-3f);
    // Each vertex is on a border's side: an outer one's, from the clip's
    // edge in, or an inner one's, half the width either side of its edge.
    const float w = c.thickness;
    const auto on = [&](float v, std::initializer_list<float> sides) {
      return std::any_of(sides.begin(), sides.end(), [&](float side) {
        return std::abs(v - side) < 1e-3f;
      });
    };
    const float x1 = display({100.0f, 0.0f}).x;
    const float x2 = display({200.0f, 0.0f}).x;
    const float y1 = display({0.0f, 100.0f}).y;
    for (int k = first; k < list.VtxBuffer.Size; ++k) {
      const ImDrawVert& v = list.VtxBuffer[k];
      if (v.col != kBlue) continue;
      EXPECT_TRUE(on(v.pos.x, {lo.x, lo.x + w, hi.x - w, hi.x, x1 - w / 2,
                               x1 + w / 2, x2 - w / 2, x2 + w / 2}))
          << v.pos.x;
      EXPECT_TRUE(
          on(v.pos.y, {lo.y, lo.y + w, hi.y - w, hi.y, y1 - w / 2, y1 + w / 2}))
          << v.pos.y;
    }
  }
}

// A caption's text starts 4 display pixels inside the clip rectangle's
// top-left corner, snapped as ImGui snaps text, so its box keeps inside: at
// the region's corner while it is in view, at the viewport's once the view
// pans past it. Text too large for the region starts there too, clipped.
TEST_F(ImageOverlayTest, CaptionIsPinnedInsideTheClip) {
  const glm::vec2 fb(1.5f, 1.5f);
  begin_frame(fb);
  ImDrawList& list = background();
  struct Case {
    ui::OverlayRegion region;
    glm::vec2 pan;
    ImDrawListFlags flags;
    bool fits;
  };
  const Case cases[] = {
      // The corner at display (67.33, 20.67): the box at (69, 22).
      {{{101.0f, 31.0f}, {200.0f, 120.0f}}, {0.0f, 0.0f}, 0, true},
      {{{101.0f, 31.0f}, {200.0f, 120.0f}},
       {0.0f, 0.0f},
       ImDrawListFlags_TextNoPixelSnap,
       true},
      // Panned past it: the viewport's corner, (0, 0).
      {{{101.0f, 31.0f}, {200.0f, 120.0f}}, {-150.0f, -50.0f}, 0, true},
      {{{101.0f, 31.0f}, {10.0f, 8.0f}}, {0.0f, 0.0f}, 0, false},
  };
  const ImDrawListFlags flags = list.Flags;
  for (const Case& c : cases) {
    list.Flags = flags | c.flags;
    camera::ImageView2D view({640.0f, 480.0f}, {0.0f, 0.0f}, {640.0f, 480.0f});
    view.pan(c.pan);
    const ui::ImageOverlay overlay(view, ui::PixelCenters::Integer, c.region);
    const glm::vec2 corner =
        glm::max(view.image_to_target(c.region.origin), glm::vec2(0.0f)) / fb;
    expect_near(overlay.clip_min(), corner, 1e-3f);
    const glm::vec2 text_at =
        c.flags != 0 ? corner + 4.0f : glm::trunc(corner + 4.0f);

    const int first = list.VtxBuffer.Size;
    const auto first_index = static_cast<unsigned int>(list.IdxBuffer.Size);
    overlay.caption("camera 1", kRed, kBlue);
    const Added box = added(list, first, kBlue);
    const Added text = added(list, first, kRed);
    ASSERT_EQ(box.count, 4);
    ASSERT_GT(text.count, 0);
    expect_near(box.min, text_at - 2.0f, 1e-3f);
    EXPECT_GE(box.min.x, overlay.clip_min().x);
    EXPECT_GE(box.min.y, overlay.clip_min().y);
    EXPECT_EQ(
        box.max.x <= overlay.clip_max().x && box.max.y <= overlay.clip_max().y,
        c.fits);
    EXPECT_GE(text.min.x, box.min.x);
    EXPECT_GE(text.min.y, box.min.y);
    EXPECT_LE(text.max.x, box.max.x);
    EXPECT_LE(text.max.y, box.max.y);
    for (const ImDrawCmd& cmd : list.CmdBuffer) {
      if (cmd.ElemCount == 0 || cmd.IdxOffset < first_index) continue;
      EXPECT_FLOAT_EQ(cmd.ClipRect.x, overlay.clip_min().x);
      EXPECT_FLOAT_EQ(cmd.ClipRect.y, overlay.clip_min().y);
      EXPECT_FLOAT_EQ(cmd.ClipRect.z, overlay.clip_max().x);
      EXPECT_FLOAT_EQ(cmd.ClipRect.w, overlay.clip_max().y);
    }
  }
  list.Flags = flags;
}

// A mosaic: `count` regions of `sizes` in turn, row-major, `columns` cells
// across, each cell the size of the largest region and `gutter` texels from
// the next.
struct Mosaic {
  const char* name;
  int count;
  int columns;
  std::vector<glm::vec2> sizes;
  float gutter;
};

void PrintTo(const Mosaic& mosaic, std::ostream* os) { *os << mosaic.name; }

std::vector<ui::OverlayRegion> layout(const Mosaic& mosaic, glm::vec2& extent) {
  glm::vec2 cell(0.0f);
  for (const glm::vec2 size : mosaic.sizes) cell = glm::max(cell, size);
  const glm::vec2 pitch = cell + mosaic.gutter;
  std::vector<ui::OverlayRegion> regions;
  for (int i = 0; i < mosaic.count; ++i) {
    const glm::vec2 at(static_cast<float>(i % mosaic.columns),
                       static_cast<float>(i / mosaic.columns));
    regions.push_back(
        {pitch * at,
         mosaic.sizes[static_cast<size_t>(i) % mosaic.sizes.size()]});
  }
  const int rows = (mosaic.count + mosaic.columns - 1) / mosaic.columns;
  extent = pitch * glm::vec2(static_cast<float>(mosaic.columns),
                             static_cast<float>(rows)) -
           mosaic.gutter;
  return regions;
}

class MosaicOverlayTest : public ImageOverlayTest,
                          public ::testing::WithParamInterface<Mosaic> {};

// However many cameras, of whatever sizes: each camera's marks, given in its
// own pixels, land on its region and are clipped to the region within the
// viewport, and its caption keeps inside that, at every pan and zoom. The
// moves give, for every count, regions that fit their caption, regions the
// viewport's corner cuts, and regions out of view.
TEST_P(MosaicOverlayTest, EachRegionKeepsItsMarks) {
  const glm::vec2 fb(2.0f, 2.0f);
  const glm::vec2 viewport(1600.0f, 1200.0f);  // the whole 800 x 600 display
  begin_frame(fb);
  ImDrawList& list = background();
  glm::vec2 extent(0.0f);
  const std::vector<ui::OverlayRegion> regions = layout(GetParam(), extent);
  struct Move {
    glm::vec2 about;
    float zoom;
    glm::vec2 pan;
  };
  // Zooms of x0.5 to x40, which the zoom limit (64 target pixels per texel)
  // reduces for a small mosaic; the last move pans it all out of view.
  const Move moves[] = {
      {{800.0f, 600.0f}, 1.0f, {0.0f, 0.0f}},
      {{300.0f, 900.0f}, 2.5f, {-210.3f, 95.7f}},
      {{1500.0f, 100.0f}, 9.0f, {640.5f, -333.25f}},
      {{800.0f, 600.0f}, 0.5f, {37.0f, 12.0f}},
      {{1100.0f, 700.0f}, 40.0f, {0.0f, 0.0f}},
      {{800.0f, 600.0f}, 1.0f, {2.0f * viewport.x, 0.0f}},
  };
  const char* text = "camera 4: good";
  const ImVec2 text_size = ImGui::CalcTextSize(text);
  int fitted = 0;
  int pinned = 0;
  int out_of_view = 0;
  for (const Move& move : moves) {
    camera::ImageView2D view(extent, {0.0f, 0.0f}, viewport);
    view.zoom_about(move.about, move.zoom);
    view.pan(move.pan);
    for (size_t i = 0; i < regions.size(); ++i) {
      SCOPED_TRACE(::testing::Message() << "zoom " << move.zoom << ", pan "
                                        << move.pan.x << ", region " << i);
      const ui::OverlayRegion& region = regions[i];
      const ui::ImageOverlay overlay(view, ui::PixelCenters::Integer, region);
      const glm::vec2 last = region.size - 1.0f;
      expect_near(overlay.to_display({0.0f, 0.0f}),
                  view.image_to_target(region.origin + 0.5f) / fb, 1e-2f);
      expect_near(overlay.to_display(last),
                  view.image_to_target(region.origin + last + 0.5f) / fb,
                  1e-2f);
      // The target pixels whose centers lie in the region and the viewport.
      const glm::vec2 corner =
          glm::ceil(view.image_to_target(region.origin) - 0.5f);
      const glm::vec2 lo = glm::max(corner, glm::vec2(0.0f)) / fb;
      const glm::vec2 hi = glm::max(
          lo,
          glm::ceil(glm::min(view.image_to_target(region.origin + region.size),
                             viewport) -
                    0.5f) /
              fb);
      expect_near(overlay.clip_min(), lo, 1e-2f);
      expect_near(overlay.clip_max(), hi, 1e-2f);

      // Every mark is clipped to it, or adds nothing when it is empty.
      const bool empty = !(hi.x > lo.x && hi.y > lo.y);
      if (empty) {
        ++out_of_view;
      } else if (lo != corner / fb) {
        ++pinned;
      }
      const glm::vec2 diagonal[] = {{0.0f, 0.0f}, last};
      int first = list.VtxBuffer.Size;
      const auto first_index = static_cast<unsigned int>(list.IdxBuffer.Size);
      overlay.point(region.size * 0.5f, kRed);
      overlay.polyline(diagonal, 2, kRed);
      overlay.grid(2, 2, nullptr, 0, kGreen);
      // A command whose indices reach past first_index drew some (ImGui
      // merges one into an earlier command of the same clip).
      int drawn = 0;
      for (const ImDrawCmd& cmd : list.CmdBuffer) {
        if (cmd.ElemCount == 0 ||
            cmd.IdxOffset + cmd.ElemCount <= first_index) {
          continue;
        }
        ++drawn;
        EXPECT_FLOAT_EQ(cmd.ClipRect.x, overlay.clip_min().x);
        EXPECT_FLOAT_EQ(cmd.ClipRect.y, overlay.clip_min().y);
        EXPECT_FLOAT_EQ(cmd.ClipRect.z, overlay.clip_max().x);
        EXPECT_FLOAT_EQ(cmd.ClipRect.w, overlay.clip_max().y);
      }
      if (empty) {
        EXPECT_EQ(list.VtxBuffer.Size, first);
      } else {
        EXPECT_GT(drawn, 0);
      }

      // A clip too narrow to reach the caption's box -- or empty -- shows
      // none of it.
      const glm::vec2 box_min = glm::trunc(lo + 4.0f) - 2.0f;
      first = list.VtxBuffer.Size;
      overlay.caption(text, kRed, kBlue);
      if (!(hi.x > box_min.x && hi.y > box_min.y)) {
        EXPECT_EQ(list.VtxBuffer.Size, first);
        continue;
      }
      const Added box = added(list, first, kBlue);
      ASSERT_EQ(box.count, 4);
      expect_near(box.min, box_min, 1e-3f);
      EXPECT_GE(box.min.x, lo.x);
      EXPECT_GE(box.min.y, lo.y);
      if (hi.x - lo.x >= text_size.x + 6.0f &&
          hi.y - lo.y >= text_size.y + 6.0f) {
        ++fitted;
        const Added glyphs = added(list, first, kRed);
        EXPECT_LE(box.max.x, hi.x);
        EXPECT_LE(box.max.y, hi.y);
        EXPECT_LE(glyphs.max.x, box.max.x);
        EXPECT_LE(glyphs.max.y, box.max.y);
      }
    }
  }
  EXPECT_GT(fitted, 0);
  EXPECT_GT(pinned, 0);
  EXPECT_GT(out_of_view, 0);
}

INSTANTIATE_TEST_SUITE_P(
    Cameras, MosaicOverlayTest,
    ::testing::Values(
        Mosaic{"One", 1, 1, {{640.0f, 576.0f}}, 16.0f},
        Mosaic{"Three", 3, 2, {{1280.0f, 720.0f}}, 16.0f},
        Mosaic{"FiveOfMixedSizes",
               5,
               3,
               {{1280.0f, 720.0f}, {640.0f, 576.0f}},
               16.0f},
        // 7696 x 4336, origins (0, 0), (3856, 0), (0, 2176), (3856, 2176).
        Mosaic{"TwoByTwo4K", 4, 2, {{3840.0f, 2160.0f}}, 16.0f}),
    [](const ::testing::TestParamInfo<Mosaic>& info) {
      return std::string(info.param.name);
    });

}  // namespace

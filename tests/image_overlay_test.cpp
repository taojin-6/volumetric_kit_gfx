// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImageOverlay on a CPU-only ImGui frame (no device): the image-to-display
// mapping, the viewport clip, and the geometry each mark adds to a draw list.

#include <gtest/gtest.h>

#include <limits>

#include <glm/common.hpp>
#include <glm/vec2.hpp>

#include "imgui.h"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/ui/image_overlay.hpp"

namespace ui = volumetric_kit::gfx::ui;
namespace camera = volumetric_kit::gfx::camera;

namespace {

constexpr ImU32 kRed = IM_COL32(255, 0, 0, 255);
constexpr ImU32 kGreen = IM_COL32(0, 255, 0, 255);
constexpr ImU32 kBlue = IM_COL32(0, 0, 255, 255);
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// A 640 x 480 image fitted in a 1280 x 960 viewport at (40, 20): 2 target
// pixels per texel, image (0, 0) at target (40, 20).
camera::ImageView2D make_view() {
  return camera::ImageView2D({640.0f, 480.0f}, {40.0f, 20.0f},
                             {1280.0f, 960.0f});
}

// The vertices added from index `first` on that carry colour `col`.
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

void expect_near(glm::vec2 actual, glm::vec2 expected, float eps = 1e-4f) {
  EXPECT_NEAR(actual.x, expected.x, eps);
  EXPECT_NEAR(actual.y, expected.y, eps);
}

// A headless 800 x 600 ImGui frame with no renderer backend: the font atlas
// is built on the CPU so NewFrame's atlas assert holds. Anti-aliasing is off,
// so every vertex of a mark carries its colour and sits on its outline.
class ImageOverlayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = ImGui::CreateContext();
    ImGui::SetCurrentContext(context_);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // hermetic: no imgui.ini in the CWD
    io.DisplaySize = ImVec2(800.0f, 600.0f);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    ImGui::GetStyle().AntiAliasedFill = false;
    ImGui::GetStyle().AntiAliasedLines = false;
  }

  void TearDown() override {
    if (in_frame_) ImGui::EndFrame();
    ImGui::DestroyContext(context_);
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
  ImGuiContext* context_ = nullptr;
  bool in_frame_ = false;
};

// OpenCV's (0, 0), the first texel's centre, lands at origin + 0.5 * scale
// target pixels, divided by the framebuffer scale; Vulkan's is the corner.
TEST_F(ImageOverlayTest, MapsTexelCentresToDisplayCoordinates) {
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

  set_framebuffer_scale({0.0f, kNaN});
  expect_near(
      ui::ImageOverlay(view, ui::PixelCenters::Half).to_display({10.0f, 20.0f}),
      view.image_to_target({10.0f, 20.0f}), 1e-3f);
}

// Every mark's draw command is clipped to the viewport in display coordinates
// -- (40, 20) + (1280, 960) target pixels, halved, cut to the 800 x 600
// display -- and the list's own clip is restored after each mark.
TEST_F(ImageOverlayTest, MarksAreClippedToTheViewport) {
  begin_frame({2.0f, 2.0f});
  ImDrawList& list = background();
  const ImVec4 outer = list._CmdHeader.ClipRect;
  const ui::ImageOverlay overlay(make_view(), ui::PixelCenters::Integer);
  const glm::vec2 line[] = {{0.0f, 0.0f}, {100.0f, 50.0f}};
  const ImU32 fills[] = {kGreen};
  overlay.point({10.0f, 10.0f}, kRed);
  overlay.polyline(line, 2, kRed);
  overlay.label({10.0f, 10.0f}, "id", kRed);
  overlay.grid(1, 1, fills, kBlue);

  int drawn = 0;
  for (const ImDrawCmd& cmd : list.CmdBuffer) {
    if (cmd.ElemCount == 0) continue;
    ++drawn;
    EXPECT_FLOAT_EQ(cmd.ClipRect.x, 20.0f);
    EXPECT_FLOAT_EQ(cmd.ClipRect.y, 10.0f);
    EXPECT_FLOAT_EQ(cmd.ClipRect.z, 660.0f);
    EXPECT_FLOAT_EQ(cmd.ClipRect.w, 490.0f);
  }
  EXPECT_GT(drawn, 0);
  const ImVec4 after = list._CmdHeader.ClipRect;
  EXPECT_EQ(after.x, outer.x);
  EXPECT_EQ(after.y, outer.y);
  EXPECT_EQ(after.z, outer.z);
  EXPECT_EQ(after.w, outer.w);
}

// A label wholly outside the viewport adds nothing, not even its background
// box, however far off (ImGui's text layout truncates a position to an int);
// one in it adds glyphs from its anchor on, over its box.
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
  overlay.label({10.0f, 10.0f}, "x", kRed, kBlue, {4.0f, 4.0f});
  const Added box = added(list, first, kBlue);
  ASSERT_EQ(box.count, 4);
  expect_near(box.min, {112.0f, 112.0f});  // anchor + offset - 2 px padding
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
  // for integer centres), with no end caps.
  expect_near(open.min, {10.5f, 9.5f}, 1e-2f);
  expect_near(open.max, {101.5f, 60.5f}, 1e-2f);

  first = list.VtxBuffer.Size;
  overlay.polyline(points, 3, kRed, 2.0f, true);
  EXPECT_GT(added(list, first, kRed).count, open.count);  // the closing leg
}

// The cells span the image's edges whatever the pixel-centre convention, each
// filled with its own colour (a transparent one left empty), with borders
// centred on the edges across the whole image.
TEST_F(ImageOverlayTest, GridSpansTheImageEdges) {
  begin_frame({2.0f, 2.0f});
  ImDrawList& list = background();
  const camera::ImageView2D view = make_view();
  const ui::ImageOverlay overlay(view, ui::PixelCenters::Integer);
  const auto display = [&](glm::vec2 p) {
    return view.image_to_target(p) / 2.0f;
  };

  const ImU32 fills[] = {0, kGreen, 0, 0};  // row-major: (column 1, row 0)
  const int first = list.VtxBuffer.Size;
  overlay.grid(2, 2, fills, kBlue, 1.0f);

  const Added cell = added(list, first, kGreen);
  ASSERT_EQ(cell.count, 4);
  expect_near(cell.min, display({320.0f, 0.0f}));
  expect_near(cell.max, display({640.0f, 240.0f}));

  const Added borders = added(list, first, kBlue);
  EXPECT_EQ(borders.count, 6 * 4);  // 3 + 3 lines, a quad each
  expect_near(borders.min, display({0.0f, 0.0f}) - 0.5f, 1e-2f);
  expect_near(borders.max, display({640.0f, 480.0f}) + 0.5f, 1e-2f);
}

TEST_F(ImageOverlayTest, DegenerateMarksAddNothing) {
  begin_frame({1.0f, 1.0f});
  ImDrawList& list = background();
  const ui::ImageOverlay overlay(make_view(), ui::PixelCenters::Integer);
  const float inf = std::numeric_limits<float>::infinity();
  const glm::vec2 points[] = {{0.0f, 0.0f}, {kNaN, 1.0f}, {5.0f, 5.0f}};
  const ImU32 fills[] = {kGreen};

  const int first = list.VtxBuffer.Size;
  overlay.point({kNaN, 0.0f}, kRed);
  overlay.polyline(points, 3, kRed);
  EXPECT_EQ(list._Path.Size, 0);  // the half-built path was dropped
  overlay.polyline(points, 1, kRed);
  overlay.polyline(nullptr, 3, kRed);
  overlay.label({0.0f, inf}, "x", kRed, kBlue);
  overlay.label({0.0f, 0.0f}, nullptr, kRed, kBlue);
  overlay.grid(0, 1, fills, kBlue);
  overlay.grid(1, -1, fills, kBlue);
  overlay.grid(1, 1, nullptr, 0);  // no fills, transparent borders
  EXPECT_EQ(list.VtxBuffer.Size, first);
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

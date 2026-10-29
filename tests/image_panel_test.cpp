// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImagePanel on CPU-only ImGui frames: the layout on the screens it is meant
// for, with their safe areas (through ImGuiTouch, as the iOS shell feeds
// them); the shares, the least size and the pictures' arrangement; the view
// across resizes; and the chrome's header buttons and handle, driven with the
// mouse at framebuffer scale 1 and 2.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/vec2.hpp>

#include "imgui.h"
#include "imgui_internal.h"  // FindWindowByName, for the chrome's rects
#include "ui_test_support.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/ui/image_panel.hpp"
#include "volumetric_kit/gfx/ui/imgui_touch.hpp"

namespace {

namespace ui = volumetric_kit::gfx::ui;

// A screen: its size in points, pixels per point, safe area and the share a
// shell gives the panel there.
struct Screen {
  const char* name;
  glm::vec2 points;
  float scale;
  ui::EdgeInsets safe;
  float share;
  VkExtent2D pixels() const {
    return {static_cast<uint32_t>(points.x * scale),
            static_cast<uint32_t>(points.y * scale)};
  }
};

// The screens of the panel's design: two desktops, and an iPhone and an iPad
// each way up (top, left, bottom, right insets in points).
const Screen kMacBook{"MacBook Pro 16", {1728, 1117}, 2.0f, {}, 0.4f};
const Screen k4K{"Linux 4K", {3840, 2160}, 1.0f, {}, 0.4f};
const Screen kPhonePortrait{
    "iPhone portrait", {440, 956}, 3.0f, {62, 0, 34, 0}, 0.45f};
const Screen kPhoneLandscape{
    "iPhone landscape", {956, 440}, 3.0f, {0, 62, 21, 62}, 0.45f};
const Screen kPadLandscape{
    "iPad landscape", {1210, 834}, 2.0f, {24, 0, 20, 0}, 0.4f};
const Screen kPadPortrait{
    "iPad portrait", {834, 1210}, 2.0f, {24, 0, 20, 0}, 0.4f};

// One tall picture, and two wide ones (as one picture of them, on a phone).
const glm::vec2 kTall{1024.0f, 2048.0f};
const glm::vec2 kWide{4096.0f, 2176.0f};

ui::ImagePanelConfig open_at(float share) {
  ui::ImagePanelConfig config;
  config.state.open = true;
  config.state.right_share = share;
  config.state.bottom_share = share;
  return config;
}

float right_of(const VkRect2D& r) {
  return static_cast<float>(r.offset.x) + static_cast<float>(r.extent.width);
}
float bottom_of(const VkRect2D& r) {
  return static_cast<float>(r.offset.y) + static_cast<float>(r.extent.height);
}

class ImagePanelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto touch = ui::ImGuiTouch::create(ImGui::GetCurrentContext());
    ASSERT_TRUE(touch.ok()) << touch.status().message();
    touch_ = std::move(touch).value();
  }

  // One frame of `panel` on `screen`: its layout, then its chrome when
  // `chrome` names a title.
  ui::PanelLayout frame(ui::ImagePanel& panel, const Screen& screen,
                        const char* chrome = nullptr) {
    seconds_ += 1.0 / 60.0;
    touch_->new_frame({screen.pixels(), screen.scale, screen.safe, seconds_});
    ImGui::NewFrame();
    const ui::PanelLayout layout = panel.layout(screen.pixels());
    if (chrome != nullptr) {
      panel.draw_chrome(chrome);
    }
    ImGui::Render();
    return layout;
  }

  // The pointer to `at` (points), then pressed and released, a frame each.
  void click(ui::ImagePanel& panel, const Screen& screen, ImVec2 at) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(at.x, at.y);
    frame(panel, screen, kTitle);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    frame(panel, screen, kTitle);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    frame(panel, screen, kTitle);
  }

  // A drag from `from` to `to` (points), released there.
  void drag(ui::ImagePanel& panel, const Screen& screen, ImVec2 from,
            ImVec2 to) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(from.x, from.y);
    frame(panel, screen, kTitle);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    frame(panel, screen, kTitle);
    io.AddMousePosEvent(0.5f * (from.x + to.x), 0.5f * (from.y + to.y));
    frame(panel, screen, kTitle);
    io.AddMousePosEvent(to.x, to.y);
    frame(panel, screen, kTitle);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    frame(panel, screen, kTitle);
  }

  static ImRect window_rect(const std::string& name) {
    const ImGuiWindow* window = ImGui::FindWindowByName(name.c_str());
    EXPECT_NE(window, nullptr) << name;
    return window != nullptr ? window->Rect() : ImRect();
  }

  // The header's buttons' centres, right-aligned as the panel lays them out:
  // Fit, 1:1, then the close button.
  struct Buttons {
    ImVec2 fit;
    ImVec2 one;
    ImVec2 close;
  };
  static Buttons buttons() {
    const ImRect header = window_rect(std::string(kTitle) + "##header");
    const ImGuiStyle& style = ImGui::GetStyle();
    const float frame = ImGui::GetFrameHeight();
    const float y = header.GetCenter().y;
    const float close_right = header.Max.x - style.WindowPadding.x;
    const float one_right = close_right - frame - style.ItemSpacing.x;
    const float one_width =
        ImGui::CalcTextSize("1:1").x + 2.0f * style.FramePadding.x;
    const float fit_right = one_right - one_width - style.ItemSpacing.x;
    const float fit_width =
        ImGui::CalcTextSize("Fit").x + 2.0f * style.FramePadding.x;
    return {ImVec2(fit_right - 0.5f * fit_width, y),
            ImVec2(one_right - 0.5f * one_width, y),
            ImVec2(close_right - 0.5f * frame, y)};
  }

  static constexpr const char* kTitle = "Pictures";
  vg_test::HeadlessImGui imgui_{ImVec2(64.0f, 64.0f)};
  std::unique_ptr<ui::ImGuiTouch> touch_;
  double seconds_ = 0.0;
};

// --- Layout ------------------------------------------------------------------

// Closed, the scene is the whole drawable and nothing is over the pictures.
TEST_F(ImagePanelTest, ClosedTheSceneIsTheWholeDrawable) {
  ui::ImagePanel panel;
  panel.set_pictures(kTall, 1);
  const ui::PanelLayout layout = frame(panel, kMacBook);
  EXPECT_EQ(layout.scene.offset.x, 0);
  EXPECT_EQ(layout.scene.offset.y, 0);
  EXPECT_EQ(layout.scene.extent.width, 3456u);
  EXPECT_EQ(layout.scene.extent.height, 2234u);
  EXPECT_EQ(layout.pictures.extent.width, 0u);
  EXPECT_FALSE(panel.over_pictures({3000.0f, 1000.0f}));
  EXPECT_FALSE(panel.open());
}

// The side on the design's screens: where the pictures show larger by more
// than 25 %, else at the end of the long axis. A tall picture docks right
// even on a portrait iPad (it shows 50 % larger there), but at the bottom on
// a portrait iPhone (within 25 %: the long axis). Two wide pictures dock at
// the bottom of the portrait screens, and right of the landscape ones even
// where the bottom shows them up to 14 % larger.
TEST_F(ImagePanelTest, TheSideOnTheDesignsScreens) {
  struct Case {
    const Screen* screen;
    glm::vec2 size;
    uint32_t count;
    ui::DockSide side;
  };
  const auto R = ui::DockSide::Right;
  const auto B = ui::DockSide::Bottom;
  const Case cases[] = {
      {&kMacBook, kTall, 1, R},
      {&kMacBook, kWide, 2, R},
      {&k4K, kTall, 1, R},
      {&k4K, kWide, 2, R},
      {&kPhonePortrait, kTall, 1, B},
      {&kPhonePortrait, kWide, 1, B},
      {&kPhoneLandscape, kTall, 1, R},
      {&kPhoneLandscape, kWide, 1, R},
      {&kPadLandscape, kTall, 1, R},
      {&kPadLandscape, kWide, 2, R},
      {&kPadPortrait, kTall, 1, R},
      {&kPadPortrait, kWide, 2, B},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << c.screen->name << ", " << c.count << " of " << c.size.x
                 << " x " << c.size.y);
    ui::ImagePanel panel(open_at(c.screen->share));
    panel.set_pictures(c.size, c.count);
    const ui::PanelLayout layout = frame(panel, *c.screen);
    EXPECT_EQ(layout.side, c.side);

    // The scene and the panel split the drawable along the dock axis, and
    // the pictures lie in the safe area, beyond the scene.
    const VkExtent2D px = c.screen->pixels();
    const float s = c.screen->scale;
    const ui::EdgeInsets& safe = c.screen->safe;
    if (layout.side == R) {
      EXPECT_EQ(layout.scene.extent.height, px.height);
      EXPECT_GT(static_cast<float>(layout.pictures.offset.x),
                right_of(layout.scene));
      EXPECT_FLOAT_EQ(right_of(layout.pictures), px.width - safe.right * s);
    } else {
      EXPECT_EQ(layout.scene.extent.width, px.width);
      EXPECT_GT(static_cast<float>(layout.pictures.offset.y),
                bottom_of(layout.scene));
      EXPECT_FLOAT_EQ(static_cast<float>(layout.pictures.offset.x),
                      safe.left * s);
    }
    EXPECT_GE(static_cast<float>(layout.pictures.offset.y), safe.top * s);
    EXPECT_LE(bottom_of(layout.pictures), px.height - safe.bottom * s);
    EXPECT_GE(static_cast<float>(layout.pictures.offset.x), safe.left * s);
  }
}

// The share is held to 25-60 % of the safe length, the panel is at least its
// least size across, and at most 60 %.
TEST_F(ImagePanelTest, TheShareIsHeldAndTheLeastSizeKept) {
  const Screen wide{"wide", {2000, 1000}, 1.0f, {}, 0.4f};
  const auto inner_edge = [&](float share, const Screen& screen) {
    ui::ImagePanel panel(open_at(share));
    const ui::PanelLayout layout = frame(panel, screen);
    EXPECT_EQ(layout.side, ui::DockSide::Right);
    return screen.pixels().width - right_of(layout.scene);
  };
  EXPECT_FLOAT_EQ(inner_edge(0.4f, wide), 800.0f);
  EXPECT_FLOAT_EQ(inner_edge(0.9f, wide), 1200.0f);
  EXPECT_FLOAT_EQ(inner_edge(0.1f, wide), 500.0f);
  const Screen small{"small", {600, 400}, 2.0f, {}, 0.4f};
  EXPECT_FLOAT_EQ(inner_edge(0.25f, small), 480.0f);  // 240 points
  const Screen narrow{"narrow", {350, 200}, 1.0f, {}, 0.4f};
  EXPECT_FLOAT_EQ(inner_edge(0.25f, narrow), 210.0f);  // 60 %, not 240
}

// Several pictures run along the panel, kGap apart, as the view's one image;
// one picture's origin is the image's.
TEST_F(ImagePanelTest, PicturesRunAlongThePanel) {
  ui::ImagePanel right(open_at(0.4f));
  right.set_pictures({300.0f, 200.0f}, 2);
  ASSERT_EQ(frame(right, kMacBook).side, ui::DockSide::Right);
  EXPECT_EQ(right.picture_origin(0), glm::vec2(0.0f, 0.0f));
  EXPECT_EQ(right.picture_origin(1), glm::vec2(0.0f, 216.0f));
  EXPECT_EQ(right.view().image_size(), glm::vec2(300.0f, 416.0f));

  ui::ImagePanel bottom(open_at(0.4f));
  bottom.set_pictures(kWide, 2);
  ASSERT_EQ(frame(bottom, kPadPortrait).side, ui::DockSide::Bottom);
  EXPECT_EQ(bottom.picture_origin(1), glm::vec2(4096.0f + 16.0f, 0.0f));
  EXPECT_EQ(bottom.view().image_size(),
            glm::vec2(2.0f * 4096.0f + 16.0f, 2176.0f));
}

// The view keeps its zoom and centre across a resize and a rotation, which
// moves one picture to the other side, and fits again on new pictures; two
// pictures fit again when a rotation rearranges them.
TEST_F(ImagePanelTest, ZoomAndCentreSurviveResizesNotNewPictures) {
  ui::ImagePanel panel(open_at(0.4f));
  panel.set_pictures(kTall, 1);
  frame(panel, kPadLandscape);
  panel.view().zoom_about({2200.0f, 900.0f}, 3.0f);
  const float zoom = panel.view().zoom();
  const glm::vec2 centre = panel.view().center();
  ASSERT_GT(zoom, 2.9f);

  const Screen bigger{"bigger", {1300, 900}, 2.0f, {24, 0, 20, 0}, 0.4f};
  frame(panel, bigger);
  EXPECT_FLOAT_EQ(panel.view().zoom(), zoom);
  EXPECT_EQ(panel.view().center(), centre);
  frame(panel, kPhonePortrait);
  EXPECT_EQ(panel.state().open, true);
  EXPECT_FLOAT_EQ(panel.view().zoom(), zoom);
  EXPECT_EQ(panel.view().center(), centre);

  panel.set_pictures(kTall, 1);  // the same: nothing changes
  frame(panel, kPhonePortrait);
  EXPECT_FLOAT_EQ(panel.view().zoom(), zoom);
  panel.set_pictures({1000.0f, 2048.0f}, 1);
  frame(panel, kPhonePortrait);
  EXPECT_FLOAT_EQ(panel.view().zoom(), 1.0f);

  panel.set_pictures(kWide, 2);
  frame(panel, kPadLandscape);
  panel.view().zoom_about({2200.0f, 900.0f}, 2.0f);
  frame(panel, kPadPortrait);  // right to bottom: side by side now
  EXPECT_FLOAT_EQ(panel.view().zoom(), 1.0f);
}

// Over the pictures' viewport only, open; a double tap goes from fit to one
// picture pixel a drawable pixel about the point, and back.
TEST_F(ImagePanelTest, OverPicturesAndDoubleTap) {
  ui::ImagePanel panel(open_at(0.4f));
  panel.set_pictures(kTall, 1);
  const ui::PanelLayout layout = frame(panel, kMacBook);
  const glm::vec2 inside{static_cast<float>(layout.pictures.offset.x) + 10.0f,
                         static_cast<float>(layout.pictures.offset.y) + 10.0f};
  EXPECT_TRUE(panel.over_pictures(inside));
  EXPECT_FALSE(panel.over_pictures({10.0f, 10.0f}));
  EXPECT_FALSE(panel.over_pictures(
      {right_of(layout.pictures), bottom_of(layout.pictures) - 1.0f}));

  const glm::vec2 at{static_cast<float>(layout.pictures.offset.x) + 300.0f,
                     900.0f};
  const glm::vec2 under = panel.view().target_to_image(at);
  panel.double_tap(at);
  EXPECT_NEAR(panel.view().scale(), 1.0f, 1e-5f);
  const glm::vec2 still = panel.view().target_to_image(at);
  EXPECT_NEAR(still.x, under.x, 1e-2f);
  EXPECT_NEAR(still.y, under.y, 1e-2f);
  panel.double_tap(at);
  EXPECT_FLOAT_EQ(panel.view().zoom(), 1.0f);

  panel.set_open(false);
  EXPECT_FALSE(panel.over_pictures(inside));
}

// --- Chrome ------------------------------------------------------------------

// At framebuffer scale 1 and 2: Fit fits, 1:1 shows a picture pixel a
// drawable pixel, a drag on the handle sets the share under the pointer, a
// release below 25 % closes the panel and keeps the share it had, and close
// closes it.
TEST_F(ImagePanelTest, TheHeaderAndTheHandleChangeTheState) {
  for (const float scale : {1.0f, 2.0f}) {
    SCOPED_TRACE(::testing::Message() << "framebuffer scale " << scale);
    const Screen screen{"desktop", {1200, 800}, scale, {}, 0.4f};
    ui::ImagePanel panel(open_at(0.4f));
    panel.set_pictures({400.0f, 800.0f}, 1);
    frame(panel, screen, kTitle);
    frame(panel, screen, kTitle);

    panel.view().zoom_about({1000.0f * scale, 300.0f * scale}, 2.5f);
    click(panel, screen, buttons().fit);
    EXPECT_FLOAT_EQ(panel.view().zoom(), 1.0f);
    click(panel, screen, buttons().one);
    EXPECT_NEAR(panel.view().scale(), 1.0f, 1e-5f);

    // The handle, from the panel's inner edge to half the width.
    const ImVec2 handle =
        window_rect(std::string(kTitle) + "##handle").GetCenter();
    EXPECT_NEAR(handle.x, 1200.0f * 0.6f + 3.0f, 1.0f);
    drag(panel, screen, handle, ImVec2(600.0f, handle.y));
    EXPECT_NEAR(panel.state().right_share, 0.5f, 1e-3f);
    EXPECT_TRUE(panel.open());
    const ImVec2 moved =
        window_rect(std::string(kTitle) + "##handle").GetCenter();
    drag(panel, screen, moved, ImVec2(1150.0f, moved.y));  // 4 %
    EXPECT_FALSE(panel.open());
    EXPECT_NEAR(panel.state().right_share, 0.5f, 1e-3f);

    panel.set_open(true);
    frame(panel, screen, kTitle);
    frame(panel, screen, kTitle);
    click(panel, screen, buttons().close);
    EXPECT_FALSE(panel.open());
  }
}

// With detents, a release snaps to the nearest; between drags the share
// follows the pointer, held to 25-60 %.
TEST_F(ImagePanelTest, AReleaseSnapsToTheNearestDetent) {
  ui::ImagePanelConfig config = open_at(0.45f);
  config.detents = {0.45f, 0.6f};
  ui::ImagePanel panel(config);
  panel.set_pictures(kTall, 1);
  frame(panel, kPhonePortrait, kTitle);
  ASSERT_EQ(frame(panel, kPhonePortrait, kTitle).side, ui::DockSide::Bottom);
  // The safe height is 860 points, from 62 down: 55 % is 473 up from 922.
  const ImVec2 handle =
      window_rect(std::string(kTitle) + "##handle").GetCenter();
  drag(panel, kPhonePortrait, handle, ImVec2(handle.x, 922.0f - 473.0f));
  EXPECT_FLOAT_EQ(panel.state().bottom_share, 0.6f);
  frame(panel, kPhonePortrait, kTitle);  // the handle at the snapped share
  const ImVec2 moved =
      window_rect(std::string(kTitle) + "##handle").GetCenter();
  drag(panel, kPhonePortrait, moved, ImVec2(moved.x, 922.0f - 344.0f));  // 40 %
  EXPECT_FLOAT_EQ(panel.state().bottom_share, 0.45f);
}

}  // namespace

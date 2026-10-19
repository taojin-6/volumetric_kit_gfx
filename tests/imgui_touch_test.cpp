// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImGuiTouch on a CPU-only ImGui context: the display and clock it feeds, a
// tap as ImGui takes it across frames, the one-finger pointer, a panel's
// capture, the safe area as the main viewport's work inset, and the binding.
// No device is needed.

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <utility>

#include "imgui.h"
#include "ui_test_support.hpp"
#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/gfx/ui/imgui_touch.hpp"

namespace ui = volumetric_kit::gfx::ui;
namespace vkc = volumetric_kit::core;

namespace {

// A 400 x 300-point display at 2 pixels per point.
ui::DisplayFrame display(double seconds) {
  ui::DisplayFrame frame;
  frame.pixels = {800, 600};
  frame.pixels_per_point = 2.0f;
  frame.seconds = seconds;
  return frame;
}

// A CPU-only context, current, with an ImGuiTouch bound to it.
class ImGuiTouchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto touch = ui::ImGuiTouch::create(ImGui::GetCurrentContext());
    ASSERT_TRUE(touch.ok()) << touch.status().message();
    touch_ = std::move(touch).value();
  }

  // One whole frame, 1/60 s after the last: the touch half's new_frame, then
  // ImGui's, with `build` between NewFrame and Render.
  template <class Build>
  void frame(Build&& build) {
    seconds_ += 1.0 / 60.0;
    touch_->new_frame(display(seconds_));
    ImGui::NewFrame();
    build();
    ImGui::Render();
  }
  void frame() {
    frame([] {});
  }

  vg_test::HeadlessImGui imgui_{ImVec2(64.0f, 64.0f)};
  std::unique_ptr<ui::ImGuiTouch> touch_;  // destroyed before the context
  double seconds_ = 0.0;
};

TEST_F(ImGuiTouchTest, NewFrameSetsTheDisplayInPoints) {
  ui::DisplayFrame frame = display(0.0);
  frame.pixels = {1170, 2532};
  frame.pixels_per_point = 3.0f;
  touch_->new_frame(frame);
  const ImGuiIO& io = ImGui::GetIO();
  EXPECT_FLOAT_EQ(io.DisplaySize.x, 390.0f);
  EXPECT_FLOAT_EQ(io.DisplaySize.y, 844.0f);
  EXPECT_FLOAT_EQ(io.DisplayFramebufferScale.x, 3.0f);
  EXPECT_FLOAT_EQ(io.DisplayFramebufferScale.y, 3.0f);
}

TEST_F(ImGuiTouchTest, DeltaTimeIsASixtiethThenTheClockStep) {
  ImGuiIO& io = ImGui::GetIO();
  io.DeltaTime = 0.5f;  // the helper's own, which the first frame replaces
  touch_->new_frame(display(100.0));
  EXPECT_FLOAT_EQ(io.DeltaTime, 1.0f / 60.0f);
  touch_->new_frame(display(100.25));
  EXPECT_FLOAT_EQ(io.DeltaTime, 0.25f);
  // A clock that goes back, or stalls, still steps forward, a little.
  touch_->new_frame(display(99.0));
  EXPECT_GT(io.DeltaTime, 0.0f);
  EXPECT_LT(io.DeltaTime, 0.001f);
  touch_->new_frame(display(99.0));
  EXPECT_GT(io.DeltaTime, 0.0f);
  ImGui::NewFrame();
  ImGui::Render();
}

// ImGui takes a tap's events a kind per frame: the position, with the
// touch-screen source so the finger has no hover before its press; the press;
// the release; and the position cleared.
TEST_F(ImGuiTouchTest, ATapArrivesAcrossFrames) {
  touch_->touch_began(1, {50.0f, 60.0f});
  touch_->touch_ended(1);
  const ImGuiIO& io = ImGui::GetIO();
  frame();
  EXPECT_EQ(io.MouseSource, ImGuiMouseSource_TouchScreen);
  EXPECT_EQ(io.MousePos.x, 50.0f);
  EXPECT_EQ(io.MousePos.y, 60.0f);
  EXPECT_FALSE(io.MouseDown[0]);
  frame();
  EXPECT_TRUE(io.MouseDown[0]);
  EXPECT_TRUE(io.MouseClicked[0]);
  frame();
  EXPECT_FALSE(io.MouseDown[0]);
  EXPECT_TRUE(ImGui::IsMousePosValid());
  frame();
  EXPECT_FALSE(ImGui::IsMousePosValid());
}

TEST_F(ImGuiTouchTest, OnlyTheFirstFingerDownIsThePointer) {
  const ImGuiIO& io = ImGui::GetIO();
  touch_->touch_began(1, {10.0f, 20.0f});
  touch_->touch_began(2, {200.0f, 100.0f});  // a pinch's second finger
  touch_->touch_moved(2, {210.0f, 110.0f});
  touch_->touch_ended(2);
  frame();
  frame();
  EXPECT_EQ(io.MousePos.x, 10.0f);
  EXPECT_EQ(io.MousePos.y, 20.0f);
  EXPECT_TRUE(io.MouseDown[0]);
  touch_->touch_moved(1, {30.0f, 40.0f});
  frame();
  EXPECT_EQ(io.MousePos.x, 30.0f);
  EXPECT_EQ(io.MousePos.y, 40.0f);
  EXPECT_TRUE(io.MouseDown[0]);

  // A cancel, as when a recognizer takes the finger, releases as a lift does.
  // A finger still down from before stays the app's.
  touch_->touch_ended(1);
  touch_->touch_moved(2, {220.0f, 120.0f});
  frame();
  frame();
  EXPECT_FALSE(io.MouseDown[0]);
  EXPECT_FALSE(ImGui::IsMousePosValid());

  // The next finger down is the pointer.
  touch_->touch_began(3, {50.0f, 60.0f});
  frame();
  frame();
  EXPECT_EQ(io.MousePos.x, 50.0f);
  EXPECT_EQ(io.MousePos.y, 60.0f);
  EXPECT_TRUE(io.MouseDown[0]);
}

// A finger that lands on a panel is ImGui's from the frame that takes its
// position, through its press; one that lands elsewhere is the app's.
TEST_F(ImGuiTouchTest, WantCaptureMouseIsTrueOnlyOverAPanel) {
  const auto panel = [] {
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(100.0f, 100.0f));
    ImGui::Begin("panel");
    ImGui::End();
  };
  frame(panel);
  frame(panel);
  const ImGuiIO& io = ImGui::GetIO();
  touch_->touch_began(1, {50.0f, 50.0f});
  frame(panel);
  EXPECT_TRUE(io.WantCaptureMouse);
  frame(panel);
  EXPECT_TRUE(io.WantCaptureMouse);
  touch_->touch_ended(1);
  frame(panel);
  frame(panel);

  touch_->touch_began(2, {300.0f, 200.0f});
  frame(panel);
  EXPECT_FALSE(io.WantCaptureMouse);
  frame(panel);
  EXPECT_FALSE(io.WantCaptureMouse);
}

// The safe area is the main viewport's work inset for the coming frame,
// applied once however many new_frames precede it, and replaced by the next
// (a rotation).
TEST_F(ImGuiTouchTest, TheSafeAreaIsTheWorkInsetOnce) {
  ui::DisplayFrame frame = display(1.0);
  frame.safe_area = {20.0f, 10.0f, 30.0f, 5.0f};  // top, left, bottom, right
  const auto expect_work_rect = [](ImVec2 pos, ImVec2 size) {
    const ImGuiViewport& viewport = *ImGui::GetMainViewport();
    EXPECT_EQ(viewport.WorkPos.x, pos.x);
    EXPECT_EQ(viewport.WorkPos.y, pos.y);
    EXPECT_EQ(viewport.WorkSize.x, size.x);
    EXPECT_EQ(viewport.WorkSize.y, size.y);
  };
  touch_->new_frame(frame);
  touch_->new_frame(frame);  // a tick the app skipped
  ImGui::NewFrame();
  expect_work_rect({10.0f, 20.0f}, {385.0f, 250.0f});
  ImGui::Render();

  frame.seconds = 2.0;
  touch_->new_frame(frame);
  ImGui::NewFrame();
  expect_work_rect({10.0f, 20.0f}, {385.0f, 250.0f});
  ImGui::Render();

  frame.seconds = 3.0;
  frame.safe_area = {};
  touch_->new_frame(frame);
  ImGui::NewFrame();
  expect_work_rect({0.0f, 0.0f}, {400.0f, 300.0f});
  ImGui::Render();
}

// The binding writes to its own context, whichever is current.
TEST_F(ImGuiTouchTest, FeedsItsOwnContext) {
  ImGuiContext* own = ImGui::GetCurrentContext();
  ImGuiContext* other = ImGui::CreateContext();
  ImGui::SetCurrentContext(other);
  touch_->new_frame(display(0.0));
  touch_->touch_began(1, {5.0f, 6.0f});
  EXPECT_NE(ImGui::GetIO().DisplaySize.x, 400.0f);
  ImGui::DestroyContext(other);
  ImGui::SetCurrentContext(own);
  EXPECT_EQ(ImGui::GetIO().DisplaySize.x, 400.0f);
  frame();
  EXPECT_EQ(ImGui::GetIO().MousePos.x, 5.0f);
}

TEST(ImGuiTouchCreateTest, RefusesANullContext) {
  auto touch = ui::ImGuiTouch::create(nullptr);
  ASSERT_FALSE(touch.ok());
  EXPECT_EQ(touch.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST(ImGuiTouchCreateTest, RefusesASecondBackendUntilTheFirstIsGone) {
  const vg_test::HeadlessImGui imgui(ImVec2(64.0f, 64.0f));
  ImGuiContext* context = ImGui::GetCurrentContext();
  const ImGuiIO& io = ImGui::GetIO();
  auto first = ui::ImGuiTouch::create(context);
  ASSERT_TRUE(first.ok()) << first.status().message();
  EXPECT_NE(io.BackendPlatformName, nullptr);
  EXPECT_NE(io.BackendPlatformUserData, nullptr);
  auto second = ui::ImGuiTouch::create(context);
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.status().domain(), vkc::Status::Code::InvalidArgument);

  first.value().reset();
  EXPECT_EQ(io.BackendPlatformName, nullptr);
  EXPECT_EQ(io.BackendPlatformUserData, nullptr);
  EXPECT_TRUE(ui::ImGuiTouch::create(context).ok());
}

// new_frame checks its precondition rather than hand ImGui a display or a
// step it asserts on only in debug builds.
TEST(ImGuiTouchDeathTest, ADisplayOutsideThePreconditionIsAProgrammerError) {
  const vg_test::HeadlessImGui imgui(ImVec2(64.0f, 64.0f));
  auto touch = ui::ImGuiTouch::create(ImGui::GetCurrentContext());
  ASSERT_TRUE(touch.ok()) << touch.status().message();
  ui::DisplayFrame empty = display(0.0);
  empty.pixels.width = 0;
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});  // route the abort message to stderr
        touch.value()->new_frame(empty);
      },
      "at least 1 x 1");
  for (const float scale :
       {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN()}) {
    ui::DisplayFrame unscaled = display(0.0);
    unscaled.pixels_per_point = scale;
    EXPECT_DEATH(
        {
          vkc::set_log_handler({});
          touch.value()->new_frame(unscaled);
        },
        "pixels_per_point");
  }
  ui::DisplayFrame timeless = display(std::numeric_limits<double>::infinity());
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});
        touch.value()->new_frame(timeless);
      },
      "finite clock");
}

}  // namespace

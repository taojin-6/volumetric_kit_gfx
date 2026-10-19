// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/imgui_touch.hpp"

#include <cfloat>
#include <cmath>

#include "volumetric_kit/core/base/check.hpp"

#include "imgui.h"
// For the context's IO and its main viewport's work inset (ImGuiViewportP),
// which has no public setter; ui_test pins what this relies on.
#include "imgui_internal.h"

namespace volumetric_kit::gfx::ui {
namespace {

// The first frame's step, as ImGui's own platform backends take it.
constexpr float kFirstStep = 1.0f / 60.0f;
// What a clock that stalls or goes back steps instead, as imgui_impl_glfw's
// does, so DeltaTime stays above 0 as ImGui::NewFrame requires.
constexpr double kLeastStep = 0.00001;

// Feeds the touch-screen source before an event, so ImGui gives the finger no
// hover before its press: it takes the position and the press on separate
// frames.
ImGuiIO& touch_io(ImGuiContext& context) {
  ImGuiIO& io = context.IO;
  io.AddMouseSourceEvent(ImGuiMouseSource_TouchScreen);
  return io;
}

}  // namespace

core::Result<std::unique_ptr<ImGuiTouch>> ImGuiTouch::create(
    ImGuiContext* context) {
  if (context == nullptr) {
    return core::Status::invalid_argument(
        "ImGuiTouch::create needs an ImGui context");
  }
  if (context->IO.BackendPlatformUserData != nullptr) {
    return core::Status::invalid_argument(
        "ImGuiTouch::create: the context already has a platform backend");
  }
  return std::unique_ptr<ImGuiTouch>(new ImGuiTouch(*context));
}

// Marks the context taken, as ImGui's own platform backends do. None of
// ImGui's platform capability flags (gamepad, cursor shapes, setting the
// pointer's position) applies to a touch screen, so BackendFlags stay as they
// are.
ImGuiTouch::ImGuiTouch(ImGuiContext& context) noexcept : context_(context) {
  ImGuiIO& io = context_.IO;
  io.BackendPlatformUserData = this;
  io.BackendPlatformName = "volumetric_kit_gfx_touch";
}

ImGuiTouch::~ImGuiTouch() {
  ImGuiIO& io = context_.IO;
  io.BackendPlatformName = nullptr;
  io.BackendPlatformUserData = nullptr;
}

void ImGuiTouch::new_frame(const DisplayFrame& frame) {
  VKC_CHECK(frame.pixels.width > 0 && frame.pixels.height > 0,
            "ImGuiTouch::new_frame needs a drawable of at least 1 x 1");
  VKC_CHECK(
      std::isfinite(frame.pixels_per_point) && frame.pixels_per_point > 0.0f,
      "ImGuiTouch::new_frame needs a finite pixels_per_point > 0");
  VKC_CHECK(std::isfinite(frame.seconds),
            "ImGuiTouch::new_frame needs a finite clock");
  ImGuiIO& io = context_.IO;
  const float scale = frame.pixels_per_point;
  io.DisplaySize = ImVec2(static_cast<float>(frame.pixels.width) / scale,
                          static_cast<float>(frame.pixels.height) / scale);
  io.DisplayFramebufferScale = ImVec2(scale, scale);

  if (seconds_) {
    const double seconds =
        frame.seconds > *seconds_ ? frame.seconds : *seconds_ + kLeastStep;
    io.DeltaTime = static_cast<float>(seconds - *seconds_);
    seconds_ = seconds;
  } else {
    io.DeltaTime = kFirstStep;
    seconds_ = frame.seconds;
  }

  // ImGui::NewFrame moves the build inset into the work inset (WorkPos,
  // WorkSize) and zeroes it, so assigning it here applies the safe area to the
  // coming frame once, however many ticks the app skipped since the last.
  ImGuiViewportP& viewport = *context_.Viewports[0];
  viewport.BuildWorkInsetMin =
      ImVec2(frame.safe_area.left, frame.safe_area.top);
  viewport.BuildWorkInsetMax =
      ImVec2(frame.safe_area.right, frame.safe_area.bottom);
}

void ImGuiTouch::touch_began(std::uintptr_t finger, glm::vec2 at) {
  if (finger_) {
    return;  // a later finger is the app's
  }
  finger_ = finger;
  ImGuiIO& io = touch_io(context_);
  io.AddMousePosEvent(at.x, at.y);
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
}

void ImGuiTouch::touch_moved(std::uintptr_t finger, glm::vec2 at) {
  if (finger_ != finger) {
    return;
  }
  touch_io(context_).AddMousePosEvent(at.x, at.y);
}

void ImGuiTouch::touch_ended(std::uintptr_t finger) {
  if (finger_ != finger) {
    return;
  }
  finger_.reset();
  ImGuiIO& io = touch_io(context_);
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
  // No hover once the finger is gone.
  io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
}

}  // namespace volumetric_kit::gfx::ui

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file ui_test_support.hpp
/// A CPU-only Dear ImGui context for the ui tests that build widgets or draw
/// lists without a device, and the mouse's clicks and drags on it.

#include "imgui.h"

namespace vg_test {

// The current ImGui context while this lives, with no renderer backend and no
// imgui.ini. Its font atlas is built on the CPU, so NewFrame's atlas assert
// holds.
class HeadlessImGui {
 public:
  explicit HeadlessImGui(ImVec2 display_size)
      : context_(ImGui::CreateContext()) {
    ImGui::SetCurrentContext(context_);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // hermetic: no imgui.ini in the CWD
    io.DisplaySize = display_size;
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  }
  ~HeadlessImGui() { ImGui::DestroyContext(context_); }
  HeadlessImGui(const HeadlessImGui&) = delete;
  HeadlessImGui& operator=(const HeadlessImGui&) = delete;

 private:
  ImGuiContext* context_;
};

// The mouse to `at` (points), then pressed and released, a call of `frame`
// (one ImGui frame) each.
template <typename Frame>
void click(ImVec2 at, Frame&& frame) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent(at.x, at.y);
  frame();
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
  frame();
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
  frame();
}

// A drag with the mouse from `from` to `to` (points), through their middle,
// released there, a call of `frame` (one ImGui frame) each step.
template <typename Frame>
void drag(ImVec2 from, ImVec2 to, Frame&& frame) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent(from.x, from.y);
  frame();
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
  frame();
  io.AddMousePosEvent(0.5f * (from.x + to.x), 0.5f * (from.y + to.y));
  frame();
  io.AddMousePosEvent(to.x, to.y);
  frame();
  io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
  frame();
}

}  // namespace vg_test

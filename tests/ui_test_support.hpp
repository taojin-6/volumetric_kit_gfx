// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file ui_test_support.hpp
/// A CPU-only Dear ImGui context for the ui tests that build widgets or draw
/// lists without a device.

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

}  // namespace vg_test

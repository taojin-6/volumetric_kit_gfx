// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/glfw/imgui_glfw.hpp"

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"

namespace volumetric_kit::gfx::glfw {
namespace {

// Makes a context current for this object's lifetime, then restores the one
// that was: ImGui's backends act on the current context, a global.
class CurrentContext {
 public:
  explicit CurrentContext(ImGuiContext* context)
      : previous_(ImGui::GetCurrentContext()) {
    ImGui::SetCurrentContext(context);
  }
  ~CurrentContext() { ImGui::SetCurrentContext(previous_); }
  CurrentContext(const CurrentContext&) = delete;
  CurrentContext& operator=(const CurrentContext&) = delete;

 private:
  ImGuiContext* previous_;
};

}  // namespace

core::Result<std::unique_ptr<ImGuiGlfw>> ImGuiGlfw::create(
    GLFWwindow* window, ImGuiContext* context) {
  if (window == nullptr) {
    return core::Status::invalid_argument("ImGuiGlfw::create needs a window");
  }
  if (context == nullptr) {
    return core::Status::invalid_argument(
        "ImGuiGlfw::create needs an ImGui context");
  }
  const CurrentContext current(context);
  if (ImGui::GetIO().BackendPlatformUserData != nullptr) {
    return core::Status::invalid_argument(
        "ImGuiGlfw::create: the context already has a platform backend");
  }
  if (!ImGui_ImplGlfw_InitForVulkan(window, /*install_callbacks=*/true)) {
    return core::Status::unsupported("ImGui_ImplGlfw_InitForVulkan failed");
  }
  return std::unique_ptr<ImGuiGlfw>(new ImGuiGlfw(*window, *context));
}

ImGuiGlfw::~ImGuiGlfw() {
  const CurrentContext current(&context_);
  ImGui_ImplGlfw_Shutdown();
}

void ImGuiGlfw::new_frame() {
  ImGui::SetCurrentContext(&context_);
  ImGui_ImplGlfw_NewFrame();
}

}  // namespace volumetric_kit::gfx::glfw

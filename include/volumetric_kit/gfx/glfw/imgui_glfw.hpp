// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file imgui_glfw.hpp
/// @brief Dear ImGui's GLFW platform backend (`imgui_impl_glfw`) on one
///        ImGui context: the input and window-size half of the overlay that
///        @ref ui::ImGuiOverlay draws.

#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/gfx/glfw/export.hpp"

struct GLFWwindow;
struct ImGuiContext;

namespace volumetric_kit::gfx::glfw {

/// @brief Binds `imgui_impl_glfw` to a window for one ImGui context, and
///        shuts it down on destruction.
///
/// Each frame, @ref new_frame feeds the window's size and input to the
/// context; then @ref ui::ImGuiOverlay::new_frame begins the ImGui frame.
/// With `install_callbacks`, the backend chains the window's callbacks set
/// before @ref create, and puts them back when this is destroyed.
///
/// @warning The window and the context must outlive this: declare it after
///          the overlay that owns the context, so it is destroyed first.
///
/// @code
/// auto overlay = ui::ImGuiOverlay::create(device, instance, overlay_config);
/// if (!overlay) return overlay.status();
/// VKC_ASSIGN(std::unique_ptr<glfw::ImGuiGlfw> platform,
///            glfw::ImGuiGlfw::create(window, overlay.value().context()));
/// // ... per frame:
/// platform->new_frame();         // io.DisplaySize and input
/// overlay.value().new_frame();   // ImGui::NewFrame
/// ImGui::ShowDemoWindow();
/// overlay.value().render(frame.cmd);
/// @endcode
class VG_GLFW_API ImGuiGlfw {
 public:
  /// @brief Initialize `imgui_impl_glfw` for @p window on @p context.
  ///
  /// The current ImGui context is the same after the call as before it.
  /// @param window             The window whose input and size the context
  ///                           takes.
  /// @param context            The ImGui context, e.g.
  ///                           @ref ui::ImGuiOverlay::context; it must have
  ///                           no platform backend yet.
  /// @param install_callbacks  Install the backend's GLFW callbacks, which
  ///                           chain the window's earlier ones. Without them,
  ///                           the caller forwards input to the backend.
  /// @return The binding; `core::Status::Code::InvalidArgument` for a null
  ///         @p window or @p context, or a @p context that already has a
  ///         platform backend; `core::Status::Code::Unsupported` if the
  ///         backend fails to initialize.
  static core::Result<std::unique_ptr<ImGuiGlfw>> create(
      GLFWwindow* window, ImGuiContext* context, bool install_callbacks = true);

  /// @brief Shut the backend down on its context, putting back the window's
  ///        callbacks; the current ImGui context is the same after as before.
  ~ImGuiGlfw();
  ImGuiGlfw(const ImGuiGlfw&) = delete;
  ImGuiGlfw& operator=(const ImGuiGlfw&) = delete;
  ImGuiGlfw(ImGuiGlfw&&) = delete;
  ImGuiGlfw& operator=(ImGuiGlfw&&) = delete;

  /// @brief Make @ref context current and run `ImGui_ImplGlfw_NewFrame`:
  ///        the window's framebuffer size and scale, the frame time, and the
  ///        mouse and gamepad state.
  void new_frame();

  /// @return The context the backend is bound to.
  ImGuiContext* context() const noexcept { return &context_; }
  /// @return The window the backend reads.
  GLFWwindow* window() const noexcept { return &window_; }

 private:
  ImGuiGlfw(GLFWwindow& window, ImGuiContext& context) noexcept
      : window_(window), context_(context) {}

  GLFWwindow& window_;
  ImGuiContext& context_;
};

}  // namespace volumetric_kit::gfx::glfw

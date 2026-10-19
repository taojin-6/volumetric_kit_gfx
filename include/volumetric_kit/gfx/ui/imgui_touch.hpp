// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file imgui_touch.hpp
/// @brief Dear ImGui's platform half for a window system ImGui has no backend
///        for, fed by the app: the display, the clock and the touches of one
///        ImGui context.

#include <cstdint>
#include <memory>
#include <optional>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

struct ImGuiContext;

namespace volumetric_kit::gfx::ui {

/// @brief Distances in from a display's edges, in points.
struct EdgeInsets {
  float top = 0.0f;     ///< From the top edge.
  float left = 0.0f;    ///< From the left edge.
  float bottom = 0.0f;  ///< From the bottom edge.
  float right = 0.0f;   ///< From the right edge.
};

/// @brief What the window system reports for one frame.
struct DisplayFrame {
  /// The drawable, in pixels.
  VkExtent2D pixels{};
  /// UIKit's `contentScaleFactor`; Android's density.
  float pixels_per_point = 1.0f;
  /// What the system keeps clear, in points (UIKit's `safeAreaInsets`,
  /// Android's `WindowInsets`): excluded from the main viewport's
  /// `WorkPos` / `WorkSize`.
  EdgeInsets safe_area;
  /// A monotonic clock, in seconds: `CADisplayLink.timestamp`.
  double seconds = 0.0;
};

/// @brief ImGui's platform half for a window system that ImGui has no backend
///        for (UIKit now, Android later): the app feeds it the display, the
///        clock and touches. It is the touch sibling of `glfw::ImGuiGlfw`.
///
/// The first finger down is ImGui's pointer until it lifts or is cancelled.
/// Later fingers are ignored, so a pinch or a two-finger pan reaches only the
/// app. The finger is a touch-screen pointer: it has no hover, and its position
/// is cleared on lift. Each frame: @ref new_frame, then
/// @ref ImGuiOverlay::new_frame. After that frame, `WantCaptureMouse` says
/// whether a finger that landed before it is on a panel.
///
/// **Threads.** Only the window's thread (main on Apple), as for the context.
/// @warning The context must outlive this: declare it after the overlay that
///          owns the context, so it is destroyed first.
///
/// @code
/// VKC_ASSIGN(std::unique_ptr<ui::ImGuiTouch> touch,
///            ui::ImGuiTouch::create(overlay.value().context()));
/// // From the view's touch callbacks, in points:
/// touch->touch_began(reinterpret_cast<std::uintptr_t>(uitouch), at);
/// // ... per frame:
/// touch->new_frame({pixels, content_scale, safe_area, link_timestamp});
/// overlay.value().new_frame();
/// const bool on_panel = ImGui::GetIO().WantCaptureMouse;
/// @endcode
class VG_UI_API ImGuiTouch {
 public:
  /// @brief Bind to @p context as its platform backend.
  /// @param context  The ImGui context, e.g. @ref ImGuiOverlay::context.
  /// @return The binding; `core::Status::Code::InvalidArgument` for a null
  ///         @p context, or one that already has a platform backend (as
  ///         `glfw::ImGuiGlfw::create` refuses).
  static core::Result<std::unique_ptr<ImGuiTouch>> create(
      ImGuiContext* context);

  /// @brief Unbind from the context, which may then take another backend.
  ~ImGuiTouch();
  ImGuiTouch(const ImGuiTouch&) = delete;
  ImGuiTouch& operator=(const ImGuiTouch&) = delete;
  ImGuiTouch(ImGuiTouch&&) = delete;
  ImGuiTouch& operator=(ImGuiTouch&&) = delete;

  /// @brief This frame's display and clock. The safe area is assigned (never
  ///        added) to the work inset each call, so a tick the app skips does
  ///        not apply it twice; it replaces any inset a main menu bar reserved.
  /// @param frame  The drawable, its scale, its safe area and the clock. The
  ///               first frame steps 1/60 s; a clock that stalls or goes back
  ///               steps 10 us.
  /// @pre Both pixel sides > 0; `pixels_per_point` finite and > 0; `seconds`
  ///      finite.
  void new_frame(const DisplayFrame& frame);

  /// @brief A finger touched down; it becomes the pointer if none is.
  /// @param finger  Stable while down: a UITouch's address, an Android pointer
  ///                id.
  /// @param at      In points, from the view's top-left.
  void touch_began(std::uintptr_t finger, glm::vec2 at);
  /// @brief A finger moved.
  /// @param finger  As @ref touch_began.
  /// @param at      As @ref touch_began.
  void touch_moved(std::uintptr_t finger, glm::vec2 at);
  /// @brief The finger lifted, or the system cancelled it (a recognizer took
  ///        it).
  /// @param finger  As @ref touch_began.
  void touch_ended(std::uintptr_t finger);

 private:
  explicit ImGuiTouch(ImGuiContext& context) noexcept;

  ImGuiContext& context_;
  /// The pointer's finger, while one is down.
  std::optional<std::uintptr_t> finger_;
  /// The clock as of the last @ref new_frame; empty before the first.
  std::optional<double> seconds_;
};

}  // namespace volumetric_kit::gfx::ui

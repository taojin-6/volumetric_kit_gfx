// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file docked_panel.hpp
/// @brief A panel docked at the right or the bottom of a window, beside its
///        scene: its layout, its handle and its header, around content of
///        the caller's.

#include <cstdint>
#include <utility>
#include <vector>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief The window edge a @ref DockedPanel docks at.
enum class DockSide : uint8_t {
  Right = 0,   ///< Beside the scene.
  Bottom = 1,  ///< Below the scene.
};

/// @brief What a shell keeps of a @ref DockedPanel across a rebuild of its
///        view.
struct DockedPanelState {
  bool open = false;          ///< Whether the panel shows.
  float right_share = 0.4f;   ///< Its share of the safe width, docked right.
  float bottom_share = 0.4f;  ///< Its share of the safe height, at the bottom.
};

/// @brief How a @ref DockedPanel starts, and its handle's detents.
struct DockedPanelConfig {
  /// The state it starts in. Its shares are also the default shares the
  /// dock side is chosen at.
  DockedPanelState state;
  /// The panel's least size across, in ImGui points.
  float min_points = 240.0f;
  /// The shares a drag on the handle snaps to on release, such as
  /// `{0.45f, 0.6f}` on a phone; empty, a drag resizes freely.
  std::vector<float> detents;
};

/// @brief The content's size as it lays out docked at each side, in any one
///        unit (DockedPanel::set_content_size). Zero, it has none.
struct DockedContentSize {
  glm::vec2 right{0.0f, 0.0f};   ///< Docked right.
  glm::vec2 bottom{0.0f, 0.0f};  ///< Docked at the bottom.
};

/// @brief One frame's rectangles of a @ref DockedPanel, in drawable pixels.
struct PanelLayout {
  /// The scene's: the whole drawable while the panel is closed, otherwise
  /// the drawable up to the panel's inner edge.
  VkRect2D scene{};
  /// The content's, inside the safe area past the handle's bar and below
  /// the header; empty while the panel is closed.
  VkRect2D content{};
  /// The edge the panel docks at.
  DockSide side = DockSide::Right;
};

/// @brief A panel docked at the right or the bottom of a window, beside its
///        scene, around content of the caller's.
///
/// The panel and the scene split the drawable, so nothing covers the scene:
/// an app draws its scene in @ref PanelLayout::scene, with that rectangle's
/// aspect, and its content in @ref PanelLayout::content.
/// - **Side.** The panel docks where the content (@ref set_content_size)
///   shows larger at the default share (before the least size), by more
///   than 25 %; otherwise, or with no content size, at the end of the
///   window's long axis: right on a landscape window, at the bottom on a
///   portrait one. The side is chosen again only when the drawable or the
///   content size changes.
/// - **Share** of the safe area along the dock axis, held to 25-60 % and at
///   least @ref DockedPanelConfig::min_points across (60 % at most).
/// - **Chrome** (@ref draw_chrome), ImGui windows that take the pointer: the
///   handle, a bar on the panel's inner edge, drawn 6 points across and
///   taking the pointer over a frame height (44 points on a touch overlay);
///   and the header across the panel's top: the title, the caller's items
///   and a close button.
/// - **Safe area.** The panel reaches the drawable's edge; its handle,
///   header and content stay in ImGui's work area, the safe area that
///   `ImGuiTouch` sets.
///
/// Rectangles are in drawable pixels; the chrome is in ImGui points, which
/// `DisplayFramebufferScale` converts.
///
/// @code
/// ui::DockedPanel panel;  // closed
/// // each frame, after ImGuiOverlay::new_frame:
/// const ui::PanelLayout& layout = panel.layout(drawable);
/// panel.draw_chrome("Inspector", [] { ImGui::Button("Reset"); });
/// // the scene in layout.scene; then, open, the content in layout.content.
/// @endcode
class VG_UI_API DockedPanel {
 public:
  /// The least and the most share of the safe area along the dock axis.
  static constexpr float kMinShare = 0.25f;
  static constexpr float kMaxShare = 0.6f;

  /// @brief A panel in @p config 's state, with no content size.
  explicit DockedPanel(const DockedPanelConfig& config = {});

  /// @return Whether the panel shows.
  bool open() const noexcept { return state_.open; }
  /// @brief Show or hide the panel, from the next @ref layout.
  void set_open(bool open) noexcept { state_.open = open; }
  /// @return The state, for a shell to carry across a rebuild.
  DockedPanelState state() const noexcept { return state_; }

  /// @brief The content's size at each side, which picks the side.
  void set_content_size(const DockedContentSize& size) noexcept;

  /// @brief Lay out @p drawable for this frame, inside ImGui's work area.
  /// @param drawable  The drawable's size, in pixels.
  /// @return The rectangles, which stay valid until the next call.
  /// @pre An ImGui frame is open, with its context current.
  const PanelLayout& layout(VkExtent2D drawable);
  /// @return The rectangles as of the last @ref layout.
  const PanelLayout& laid_out() const noexcept { return layout_; }

  /// @brief Draw the handle and the header, as ImGui windows named after
  ///        @p title. A drag on the handle moves the panel's edge with the
  ///        pointer, snapping to the detents on release; a drag released
  ///        below 25 % closes the panel, and a click changes nothing. The
  ///        close button closes it. Closed, it draws nothing.
  /// @param title  The header's text, its first 64 characters unique among
  ///               windows.
  /// @param items  A callable, `void()`, that draws the caller's items in
  ///               the header's window on the title's line. It is called in
  ///               place, never stored or copied, so its captures cost
  ///               nothing. The close button, a frame height square, follows
  ///               them at the header's right end.
  /// @pre @ref layout ran this frame; a panel it found closed draws nothing.
  template <typename Items>
  void draw_chrome(const char* title, Items&& items) {
    if (begin_chrome(title)) {
      std::forward<Items>(items)();
    }
    end_chrome();
  }
  /// @brief Draw the handle and the header, with no items of the caller's.
  /// @param title  The header's text, its first 64 characters unique among
  ///               windows.
  /// @pre @ref layout ran this frame; a panel it found closed draws nothing.
  void draw_chrome(const char* title) {
    draw_chrome(title, [] {});
  }

  /// @return Whether @p point, in drawable pixels, is in the content of the
  ///         last layout, with the panel open.
  bool over_content(glm::vec2 point) const noexcept;

 private:
  // The share the panel takes on `side`.
  float& share(DockSide side) noexcept;
  // draw_chrome's halves. The first draws the handle and opens the header's
  // window with the title, and returns whether the caller's items go in it;
  // the second draws the close button and closes the window.
  bool begin_chrome(const char* title);
  void end_chrome();

  DockedPanelConfig config_;
  DockedPanelState state_;
  DockedContentSize content_size_;
  PanelLayout layout_;
  // What the side was chosen for.
  VkExtent2D chosen_for_{0, 0};
  bool content_changed_ = true;
  // This frame's chrome, in drawable pixels: the safe area, the panel's inner
  // edge along the dock axis, and the header's height.
  float safe_[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // x0, y0, x1, y1
  float edge_ = 0.0f;
  float header_ = 0.0f;
  bool laid_out_open_ = false;  // whether the last layout was of it open
  // Whether begin_chrome opened the header's window, and whether ImGui draws
  // it.
  bool header_begun_ = false;
  bool header_drawn_ = false;
  // A drag on the handle: the share it started from, and that share moved
  // by the pointer.
  float drag_from_ = 0.0f;
  float drag_share_ = 0.0f;
};

}  // namespace volumetric_kit::gfx::ui

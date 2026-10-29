// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_panel.hpp
/// @brief A panel docked at the right or the bottom of a window, beside its
///        scene, showing pictures through one pan-and-zoom view.

#include <cstdint>
#include <vector>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief The window edge an @ref ImagePanel docks at.
enum class DockSide : uint8_t {
  Right = 0,   ///< Beside the scene; pictures stacked top to bottom.
  Bottom = 1,  ///< Below the scene; pictures side by side.
};

/// @brief What a shell keeps of an @ref ImagePanel across a rebuild of its
///        view.
struct ImagePanelState {
  bool open = false;          ///< Whether the panel shows.
  float right_share = 0.4f;   ///< Its share of the safe width, docked right.
  float bottom_share = 0.4f;  ///< Its share of the safe height, at the bottom.
};

/// @brief How an @ref ImagePanel starts, and its handle's detents.
struct ImagePanelConfig {
  /// The state it starts in. Its shares are also the default shares the
  /// dock side is chosen at.
  ImagePanelState state;
  /// The panel's least size across, in ImGui points.
  float min_points = 240.0f;
  /// The shares a drag on the handle snaps to on release, such as
  /// `{0.45f, 0.6f}` on a phone; empty, a drag resizes freely.
  std::vector<float> detents;
};

/// @brief One frame's rectangles of an @ref ImagePanel, in drawable pixels.
struct PanelLayout {
  /// The scene's: the whole drawable while the panel is closed, otherwise
  /// the drawable up to the panel's inner edge.
  VkRect2D scene{};
  /// The pictures' viewport, inside the safe area below the panel's header;
  /// empty while the panel is closed.
  VkRect2D pictures{};
  /// The edge the panel docks at.
  DockSide side = DockSide::Right;
};

/// @brief A panel docked at the right or the bottom of a window, beside its
///        scene, showing pictures through one pan-and-zoom view.
///
/// The panel and the scene split the drawable, so nothing covers the scene:
/// an app draws its scene in @ref PanelLayout::scene, with that rectangle's
/// aspect, and the pictures through @ref view into
/// @ref PanelLayout::pictures.
/// - **Side.** The panel docks where the pictures show larger at the default
///   share (before the least size), by more than 25 %; otherwise at the end
///   of the window's long axis: right on a landscape window, at the bottom
///   on a portrait one.
///   The side is chosen again only when the drawable or the pictures change.
/// - **Share** of the safe area along the dock axis, held to 25-60 % and at
///   least @ref ImagePanelConfig::min_points across (60 % at most).
/// - **Pictures** run along the panel: stacked when docked right, side by
///   side at the bottom, @ref kGap picture pixels apart, as one image of the
///   view. The view keeps its zoom and centre across a resize or a rotation
///   (where a new side rearranges the pictures, the same point of the same
///   picture stays at the centre), and fits again when the pictures' size or
///   count changes.
/// - **Chrome** (@ref draw_chrome), ImGui windows that take the pointer: the
///   handle, a bar on the panel's inner edge, drawn 6 points and taking the
///   pointer over a frame height (44 points on a touch overlay); and the
///   header across the panel's top, the title then Fit, 1:1 and close.
/// - **Safe area.** The panel reaches the drawable's edge; its handle,
///   header and pictures stay in ImGui's work area, the safe area that
///   `ImGuiTouch` sets.
///
/// Rectangles are in drawable pixels, so @ref view maps pictures to drawable
/// pixels; the chrome is in ImGui points, which `DisplayFramebufferScale`
/// converts.
///
/// @code
/// ui::ImagePanel panel;  // closed
/// panel.set_pictures({1920.0f, 1080.0f}, 2);
/// // each frame, after ImGuiOverlay::new_frame:
/// const ui::PanelLayout& layout = panel.layout(drawable);
/// panel.draw_chrome("Pictures");
/// // the scene in layout.scene; then, open, picture p through panel.view(),
/// // its top-left at panel.picture_origin(p). Input over the pictures:
/// if (!ImGui::GetIO().WantCaptureMouse && panel.over_pictures(cursor)) {
///   panel.view().zoom_about(cursor, 1.25f);
/// }
/// @endcode
class VG_UI_API ImagePanel {
 public:
  /// The least and the most share of the safe area along the dock axis.
  static constexpr float kMinShare = 0.25f;
  static constexpr float kMaxShare = 0.6f;
  /// The gap between pictures, in picture pixels.
  static constexpr float kGap = 16.0f;

  /// @brief A panel in @p config 's state, with no pictures.
  explicit ImagePanel(const ImagePanelConfig& config = {});

  /// @return Whether the panel shows.
  bool open() const noexcept { return state_.open; }
  /// @brief Show or hide the panel.
  void set_open(bool open) noexcept { state_.open = open; }

  /// @brief Show @p count pictures of @p size.
  /// @param size   Each picture's size, in its pixels.
  /// @param count  The number of pictures; 0 shows none.
  void set_pictures(glm::vec2 size, uint32_t count) noexcept;

  /// @brief Lay out @p drawable for this frame, inside ImGui's work area.
  /// @param drawable  The drawable's size, in pixels.
  /// @return The rectangles, which stay valid until the next call.
  /// @pre An ImGui frame is open, with its context current.
  const PanelLayout& layout(VkExtent2D drawable);

  /// @brief Draw the handle and the header, as ImGui windows named after
  ///        @p title. A drag on the handle moves the panel's edge with the
  ///        pointer, snapping to the detents on release; a drag released
  ///        below 25 % closes the panel, and a click changes nothing. Fit
  ///        fits the view, 1:1 shows one picture pixel a drawable pixel, and
  ///        close closes. Closed, it draws nothing.
  /// @param title  The header's text, its first 64 characters unique among
  ///               windows.
  /// @pre @ref layout ran this frame; a panel it found closed draws nothing.
  void draw_chrome(const char* title);

  /// @return The pictures' view, in drawable pixels.
  camera::ImageView2D& view() noexcept { return view_; }
  /// @return The pictures' view, in drawable pixels.
  const camera::ImageView2D& view() const noexcept { return view_; }

  /// @return Picture @p i 's top-left in the view's image coordinates.
  glm::vec2 picture_origin(uint32_t i) const noexcept;

  /// @return Whether @p point, in drawable pixels, is in the pictures'
  ///         viewport of the last layout, with the panel open.
  bool over_pictures(glm::vec2 point) const noexcept;

  /// @brief A double tap or click at @p point, in drawable pixels: fitted,
  ///        zoom to one picture pixel a drawable pixel about it; otherwise
  ///        fit.
  void double_tap(glm::vec2 point) noexcept;

  /// @return The state, for a shell to carry across a rebuild.
  ImagePanelState state() const noexcept { return state_; }

 private:
  // The share the panel takes on `side`.
  float& share(DockSide side) noexcept;

  ImagePanelConfig config_;
  ImagePanelState state_;
  glm::vec2 picture_size_{0.0f, 0.0f};
  uint32_t picture_count_ = 0;
  camera::ImageView2D view_;
  PanelLayout layout_;
  // What the side was chosen for, and what the view's image was set for:
  // its size, and the pictures and side it arranges.
  VkExtent2D chosen_for_{0, 0};
  bool pictures_changed_ = true;
  glm::vec2 image_size_{0.0f, 0.0f};
  glm::vec2 view_picture_size_{0.0f, 0.0f};
  uint32_t view_count_ = 0;
  DockSide view_side_ = DockSide::Right;
  // This frame's chrome, in drawable pixels: the safe area, the panel's inner
  // edge along the dock axis, and the header's height.
  float safe_[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // x0, y0, x1, y1
  float edge_ = 0.0f;
  float header_ = 0.0f;
  bool laid_out_open_ = false;  // whether the last layout was of it open
  // A drag on the handle: the share it started from, and that share moved
  // by the pointer.
  float drag_from_ = 0.0f;
  float drag_share_ = 0.0f;
};

}  // namespace volumetric_kit::gfx::ui

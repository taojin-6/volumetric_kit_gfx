// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_panel.hpp
/// @brief A panel docked beside a window's scene (DockedPanel), showing
///        pictures through one pan-and-zoom view.

#include <cstdint>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/ui/docked_panel.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief A panel docked beside a window's scene (@ref DockedPanel, whose
///        rules it lays out and draws its chrome by), showing pictures
///        through one pan-and-zoom view.
///
/// An app draws its scene in @ref PanelLayout::scene, with that rectangle's
/// aspect, and the pictures through @ref view into @ref PanelLayout::content.
/// - **Side.** The pictures' size at each side is the panel's content size,
///   so it docks where they show larger.
/// - **Pictures** run along the panel: stacked when docked right, side by
///   side at the bottom, @ref kGap picture pixels apart, as one image of the
///   view. The view keeps its zoom and centre across a resize or a rotation,
///   and fits again when the pictures' size or count changes. Where a new
///   side rearranges the pictures, a fitted view (zoom 1) fits again; a
///   zoomed one keeps its zoom and the same point of the same picture at the
///   centre, or, from a gap or beyond the pictures, the nearest point of the
///   nearest picture.
/// - **Header:** the title, then Fit, 1:1 and close.
///
/// Rectangles are in drawable pixels, so @ref view maps pictures to drawable
/// pixels.
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
  /// The gap between pictures, in picture pixels.
  static constexpr float kGap = 16.0f;

  /// @brief A panel in @p config 's state, with no pictures.
  explicit ImagePanel(const DockedPanelConfig& config = {});

  /// @return Whether the panel shows.
  bool open() const noexcept { return dock_.open(); }
  /// @brief Show or hide the panel.
  void set_open(bool open) noexcept { dock_.set_open(open); }

  /// @brief Show @p count pictures of @p size.
  /// @param size   Each picture's size, in its pixels.
  /// @param count  The number of pictures; 0 shows none.
  void set_pictures(glm::vec2 size, uint32_t count) noexcept;

  /// @brief Lay out @p drawable for this frame (DockedPanel::layout), and
  ///        the view in the content.
  /// @param drawable  The drawable's size, in pixels.
  /// @return The rectangles, which stay valid until the next call.
  /// @pre An ImGui frame is open, with its context current.
  const PanelLayout& layout(VkExtent2D drawable);

  /// @brief Draw the handle and the header (DockedPanel::draw_chrome), with
  ///        Fit, which fits the view, and 1:1, which shows one picture pixel
  ///        a drawable pixel.
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
  bool over_pictures(glm::vec2 point) const noexcept {
    return dock_.over_content(point);
  }

  /// @brief A double tap or click at @p point, in drawable pixels: fitted,
  ///        zoom to one picture pixel a drawable pixel about it; otherwise
  ///        fit.
  void double_tap(glm::vec2 point) noexcept;

  /// @return The state, for a shell to carry across a rebuild.
  DockedPanelState state() const noexcept { return dock_.state(); }

 private:
  DockedPanel dock_;
  glm::vec2 picture_size_{0.0f, 0.0f};
  uint32_t picture_count_ = 0;
  camera::ImageView2D view_;
  // What the view's image was set for: its size, and the pictures and side
  // it arranges.
  glm::vec2 image_size_{0.0f, 0.0f};
  glm::vec2 view_picture_size_{0.0f, 0.0f};
  uint32_t view_count_ = 0;
  DockSide view_side_ = DockSide::Right;
};

}  // namespace volumetric_kit::gfx::ui

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_panel.hpp
/// @brief Pictures in a docked panel's content, through one pan-and-zoom
///        view.

#include <cstdint>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/ui/docked_panel.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief Pictures in the content of a @ref DockedPanel, a program's own or
///        a host's, through one pan-and-zoom view.
///
/// It holds no dock. Each frame a program gives the dock the pictures' size
/// (@ref content_size), arranges them in the content the dock laid out
/// (@ref arrange), and puts Fit and 1:1 in the dock's header
/// (@ref header_items).
/// - **Side.** The pictures' arrangement at each side is the content size,
///   so the panel docks where they show larger.
/// - **Pictures** run along the panel: stacked when docked right, side by
///   side at the bottom, @ref kGap picture pixels apart, as one image of the
///   view. The view keeps its zoom and centre across a resize or a rotation,
///   and fits again when the pictures' size or count changes. Where a new
///   side rearranges the pictures, a fitted view (zoom 1) fits again; a
///   zoomed one keeps its zoom and the same point of the same picture at the
///   centre, or, from a gap or beyond the pictures, the nearest point of the
///   nearest picture.
///
/// The content is in drawable pixels, so @ref view maps pictures to drawable
/// pixels.
///
/// @code
/// ui::DockedPanel dock;  // closed
/// ui::ImagePanel pictures;
/// pictures.set_pictures({1920.0f, 1080.0f}, 2);
/// // each frame, after ImGuiOverlay::new_frame:
/// dock.set_content_size(pictures.content_size());
/// const ui::PanelLayout& layout = dock.layout(drawable);
/// pictures.arrange(layout.content, layout.side);
/// dock.draw_chrome("Pictures", [&pictures] { pictures.header_items(); });
/// // open, picture p through pictures.view(), its top-left at
/// // pictures.picture_origin(p). Input over the pictures:
/// if (!ImGui::GetIO().WantCaptureMouse && dock.over_content(cursor)) {
///   pictures.view().zoom_about(cursor, 1.25f);
/// }
/// @endcode
class VG_UI_API ImagePanel {
 public:
  /// The gap between pictures, in picture pixels.
  static constexpr float kGap = 16.0f;

  /// @brief Show @p count pictures of @p size.
  /// @param size   Each picture's size, in its pixels.
  /// @param count  The number of pictures; 0 shows none.
  void set_pictures(glm::vec2 size, uint32_t count) noexcept;

  /// @return The pictures' arrangement at each side, in picture pixels, for
  ///         DockedPanel::set_content_size; with no pictures, none.
  DockedContentSize content_size() const noexcept;

  /// @brief Arrange the pictures along a panel docked at @p side, and the
  ///        view in @p content.
  /// @param content  The pictures' rectangle, in drawable pixels, such as
  ///                 PanelLayout::content. Empty, as a closed panel's is, the
  ///                 view stays as it is.
  /// @param side     The edge the panel docks at.
  void arrange(const VkRect2D& content, DockSide side) noexcept;

  /// @brief Fit, which fits the view, and 1:1, which shows one picture pixel
  ///        a drawable pixel, right-aligned before the close button.
  /// @pre In DockedPanel::draw_chrome's items, in the header's window.
  void header_items();

  /// @return The pictures' view, in drawable pixels.
  camera::ImageView2D& view() noexcept { return view_; }
  /// @return The pictures' view, in drawable pixels.
  const camera::ImageView2D& view() const noexcept { return view_; }

  /// @return Picture @p i 's top-left in the view's image coordinates, as
  ///         the last @ref arrange laid them out.
  glm::vec2 picture_origin(uint32_t i) const noexcept;

  /// @brief A double tap or click at @p point, in drawable pixels: fitted,
  ///        zoom to one picture pixel a drawable pixel about it; otherwise
  ///        fit.
  void double_tap(glm::vec2 point) noexcept;

 private:
  glm::vec2 picture_size_{0.0f, 0.0f};
  uint32_t picture_count_ = 0;
  DockSide side_ = DockSide::Right;  // the last arrange's
  camera::ImageView2D view_;
  // What the view's image was set for: its size, and the pictures and side
  // it arranges.
  glm::vec2 image_size_{0.0f, 0.0f};
  glm::vec2 view_picture_size_{0.0f, 0.0f};
  uint32_t view_count_ = 0;
  DockSide view_side_ = DockSide::Right;
};

}  // namespace volumetric_kit::gfx::ui

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_overlay.hpp
/// @brief Marks given in image coordinates -- points, polylines, labels and a
///        cell grid -- drawn with Dear ImGui over an image shown through a
///        @ref camera::ImageView2D.

#include <cstddef>
#include <cstdint>

#include <glm/vec2.hpp>

#include "imgui.h"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief Where texel centres sit in the image coordinates of the marks.
enum class PixelCenters : uint8_t {
  /// Texel `(i, j)`'s centre is at `(i, j)`: OpenCV's and recon's convention.
  Integer,
  /// Its centre is at `(i + 0.5, j + 0.5)`: Vulkan's and
  /// @ref camera::ImageView2D's.
  Half,
};

/// @brief Draws marks over an image on an ImGui draw list, so each lands on
///        the texel it names however the view pans and zooms.
///
/// The one home of the mapping from an image point to ImGui's display
/// coordinates: `view.image_to_target(p + 0.5) / io.DisplayFramebufferScale`
/// for @ref PixelCenters::Integer (no `+ 0.5` for `Half`), since the view maps
/// Vulkan image coordinates to framebuffer pixels and ImGui draws in window
/// coordinates. Every mark is clipped to the view's viewport, and a label
/// wholly outside it is not drawn; nor is a mark with a non-finite coordinate.
/// Sizes -- radius, thickness, label offset -- are in display pixels, so marks
/// keep their size as the image zooms.
///
/// The overlay copies the view and borrows the draw list, so make one per
/// image per frame.
///
/// @code
/// overlay.new_frame();
/// ui::ImageOverlay marks(view, ui::PixelCenters::Integer);
/// marks.grid(8, 6, coverage_fills, IM_COL32(255, 255, 255, 64));
/// for (const glm::vec2& corner : corners) {
///   marks.point(corner, IM_COL32(0, 255, 0, 255));
/// }
/// marks.polyline(outline, 4, IM_COL32(255, 255, 0, 255), 2.0f, true);
/// marks.label(corners[0], "0", IM_COL32_WHITE, IM_COL32(0, 0, 0, 160));
/// target->begin(cmd, {});
/// overlay.render(cmd);
/// target->end(cmd);
/// @endcode
class VG_UI_API ImageOverlay {
 public:
  /// @brief Marks over @p view's image, drawn into @p draw_list.
  /// @param view       Where the image sits; copied.
  /// @param centers    The convention of the image coordinates marks are
  ///                   given in.
  /// @param draw_list  The list to draw into; null draws into the background
  ///                   list, behind every ImGui window. Borrowed.
  /// @pre An ImGui frame is active (`NewFrame` was called) with its context
  ///      current. Its `io.DisplayFramebufferScale` is read here; a
  ///      non-positive or non-finite component counts as 1.
  ImageOverlay(const camera::ImageView2D& view, PixelCenters centers,
               ImDrawList* draw_list = nullptr);

  /// @brief Map an image point to ImGui display coordinates.
  /// @param image_point  In image coordinates, under the overlay's
  ///                     @ref PixelCenters.
  /// @return Its position in display (window) coordinates.
  glm::vec2 to_display(glm::vec2 image_point) const noexcept;

  /// @brief A filled dot centred on an image point.
  /// @param image_point  The dot's centre, in image coordinates.
  /// @param color        Its colour (`IM_COL32`).
  /// @param radius       Its radius, in display pixels.
  void point(glm::vec2 image_point, ImU32 color, float radius = 3.0f) const;

  /// @brief Line segments through image points.
  /// @param points     The vertices, in image coordinates; fewer than 2 draw
  ///                   nothing.
  /// @param count      The number of @p points.
  /// @param color      The line colour (`IM_COL32`).
  /// @param thickness  The line width, in display pixels.
  /// @param closed     Join the last vertex back to the first.
  void polyline(const glm::vec2* points, size_t count, ImU32 color,
                float thickness = 1.0f, bool closed = false) const;

  /// @brief Text whose top-left corner is @p offset display pixels from an
  ///        image point.
  /// @param image_point  The anchor, in image coordinates.
  /// @param text         The text; null draws nothing.
  /// @param color        Its colour (`IM_COL32`).
  /// @param background   A box filled behind the text; 0 (transparent) for
  ///                     none.
  /// @param offset       From the anchor to the text, in display pixels.
  void label(glm::vec2 image_point, const char* text, ImU32 color,
             ImU32 background = 0,
             glm::vec2 offset = glm::vec2(0.0f, 0.0f)) const;

  /// @brief Divide the whole image into @p columns x @p rows equal cells,
  ///        fill each and outline them.
  ///
  /// The cells span the image's edges whatever its @ref PixelCenters: cell
  /// `(c, r)` covers texel columns `[c * w / columns, (c + 1) * w / columns)`
  /// of a `w`-texel-wide image, and rows likewise.
  /// @param columns     The number of cells across; non-positive draws
  ///                    nothing.
  /// @param rows        The number of cells down; non-positive draws nothing.
  /// @param fills       `columns * rows` fill colours in row-major order, a
  ///                    transparent one leaving its cell unfilled; null fills
  ///                    none.
  /// @param line_color  The colour of the cell borders; transparent draws
  ///                    none.
  /// @param thickness   The border width, in display pixels.
  void grid(int columns, int rows, const ImU32* fills, ImU32 line_color,
            float thickness = 1.0f) const;

 private:
  /// Display coordinates of a point in Vulkan image coordinates (texel edges
  /// on integers), where the grid's cell borders are.
  glm::vec2 edge_to_display(glm::vec2 image_point) const noexcept;
  /// The viewport's corners in display coordinates: where marks are clipped.
  glm::vec2 viewport_min() const noexcept;
  glm::vec2 viewport_max() const noexcept;

  camera::ImageView2D view_;
  /// Added to a mark's image point to give Vulkan image coordinates.
  glm::vec2 center_offset_{0.0f, 0.0f};
  glm::vec2 framebuffer_scale_{1.0f, 1.0f};
  ImDrawList* draw_list_ = nullptr;
};

}  // namespace volumetric_kit::gfx::ui

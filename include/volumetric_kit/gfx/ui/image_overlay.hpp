// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_overlay.hpp
/// @brief Marks given in image coordinates -- points, polylines, labels, a
///        cell grid and a caption -- drawn with Dear ImGui over an image, or a
///        region of it, shown through a @ref camera::ImageView2D.

#include <cstddef>
#include <cstdint>

#include <glm/vec2.hpp>

#include "imgui.h"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief Where texel centers sit in the image coordinates of the marks.
enum class PixelCenters : uint8_t {
  /// Texel `(i, j)`'s center is at `(i, j)`: OpenCV's and recon's convention.
  Integer,
  /// Its center is at `(i + 0.5, j + 0.5)`: Vulkan's and
  /// @ref camera::ImageView2D's.
  Half,
};

/// @brief A rectangle of an image that an @ref ImageOverlay draws over --
///        one camera's region of a mosaic, say -- in the image's texels.
///
/// In @ref camera::ImageView2D's image coordinates, texel edges on integers:
/// the region's texel `(i, j)` is the image's `(origin.x + i, origin.y + j)`.
/// A non-finite component, or a size that is not positive, leaves nothing to
/// draw in.
///
/// @code
/// // Camera 1's 3840 x 2160 picture, right of camera 0's past a 16-texel
/// // gutter.
/// const ui::OverlayRegion tile{{3856.0f, 0.0f}, {3840.0f, 2160.0f}};
/// @endcode
struct OverlayRegion {
  /// Its top-left corner, in the image's texels.
  glm::vec2 origin{0.0f, 0.0f};
  /// Its size, in texels.
  glm::vec2 size{0.0f, 0.0f};
};

/// @brief Draws marks over an image, or a region of it, on an ImGui draw
///        list, so each lands on the texel it names however the view pans and
///        zooms.
///
/// Marks are given in the region's own coordinates -- a camera's pixels, for
/// its region of a mosaic -- and the region is the whole image unless one is
/// given. The one home of the mapping from a mark to ImGui's display
/// coordinates: `view.image_to_target(region.origin + p + 0.5) /
/// io.DisplayFramebufferScale` for @ref PixelCenters::Integer (no `+ 0.5` for
/// `Half`), since the view maps Vulkan image coordinates to framebuffer pixels
/// and ImGui draws in window coordinates. Every mark is clipped to the region
/// within the view's viewport (@ref clip_min, @ref clip_max), and none is
/// drawn when that is empty; nor is a label wholly outside it, or a mark with
/// a non-finite coordinate. Sizes -- radius, thickness, label offset -- are in
/// display pixels, so marks keep their size as the image zooms; a radius or
/// thickness is clamped to [0, 10000], NaN to 0.
///
/// The overlay copies the view and borrows the draw list, so make one per
/// region per frame.
///
/// @code
/// overlay.new_frame();
/// ui::ImageOverlay marks(view, ui::PixelCenters::Integer, tile);
/// marks.grid(8, 6, coverage_fills.data(), coverage_fills.size(),
///            IM_COL32(255, 255, 255, 64));
/// for (const glm::vec2& corner : corners) {
///   marks.point(corner, IM_COL32(0, 255, 0, 255));
/// }
/// marks.polyline(outline, 4, IM_COL32(255, 255, 0, 255), 2.0f, true);
/// marks.label(corners[0], "0", IM_COL32_WHITE, IM_COL32(0, 0, 0, 160));
/// marks.caption("camera 1", IM_COL32_WHITE, IM_COL32(0, 0, 0, 160));
/// target->begin(cmd, {});
/// overlay.render(cmd);
/// target->end(cmd);
/// @endcode
class VG_UI_API ImageOverlay {
 public:
  /// @brief Marks over the whole of @p view's image, drawn into
  ///        @p draw_list.
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

  /// @brief Marks over @p region of @p view's image, given in the region's
  ///        coordinates and drawn into @p draw_list.
  /// @param view       Where the image sits; copied.
  /// @param centers    The convention of the region coordinates marks are
  ///                   given in.
  /// @param region     The part of the image the marks are over.
  /// @param draw_list  The list to draw into; null draws into the background
  ///                   list, behind every ImGui window. Borrowed.
  /// @pre As for the overlay of the whole image.
  ImageOverlay(const camera::ImageView2D& view, PixelCenters centers,
               const OverlayRegion& region, ImDrawList* draw_list = nullptr);

  /// @brief Map a point of the region to ImGui display coordinates.
  /// @param image_point  In the region's coordinates, under the overlay's
  ///                     @ref PixelCenters.
  /// @return Its position in display (window) coordinates.
  glm::vec2 to_display(glm::vec2 image_point) const noexcept;

  /// @brief Map a point of the region in Vulkan's coordinates -- texel edges
  ///        on integers, whatever the overlay's @ref PixelCenters -- to ImGui
  ///        display coordinates: where its corners and cell borders land.
  ///
  /// For @ref PixelCenters::Integer, `edge_to_display(p)` is
  /// `to_display(p - 0.5)` -- the region's corner, `(0, 0)`, is OpenCV's
  /// `(-0.5, -0.5)`; for `Half` the two are the same.
  /// @param edge_point  In the region's coordinates, texel edges on integers.
  /// @return Its position in display (window) coordinates.
  glm::vec2 edge_to_display(glm::vec2 edge_point) const noexcept;

  /// @return The top-left corner of the rectangle marks are clipped to -- the
  ///         region within the view's viewport, on whole target pixels: those
  ///         whose centers it covers, which show its texels -- in display
  ///         coordinates. Finite, whatever the region.
  glm::vec2 clip_min() const noexcept { return clip_min_; }
  /// @return Its bottom-right corner, in display coordinates: no lower than
  ///         @ref clip_min, and equal to it on an axis the rectangle is empty
  ///         across.
  glm::vec2 clip_max() const noexcept { return clip_max_; }

  /// @brief A filled dot centered on a point of the region.
  /// @param image_point  The dot's center, in the region's coordinates.
  /// @param color        Its color (`IM_COL32`).
  /// @param radius       Its radius, in display pixels.
  void point(glm::vec2 image_point, ImU32 color, float radius = 3.0f) const;

  /// @brief Line segments through points of the region.
  /// @param points     The vertices, in the region's coordinates; fewer
  ///                   than 2 draw nothing.
  /// @param count      The number of @p points.
  /// @param color      The line color (`IM_COL32`).
  /// @param thickness  The line width, in display pixels.
  /// @param closed     Join the last vertex back to the first.
  void polyline(const glm::vec2* points, size_t count, ImU32 color,
                float thickness = 1.0f, bool closed = false) const;

  /// @brief Text whose top-left corner is @p offset display pixels from a
  ///        point of the region.
  ///
  /// The corner is truncated to whole display pixels, as ImGui does to text
  /// unless the list has `ImDrawListFlags_TextNoPixelSnap`, so the background
  /// box stays even around the text.
  /// @param image_point  The anchor, in the region's coordinates.
  /// @param text         The text; null or empty draws nothing.
  /// @param color        Its color (`IM_COL32`).
  /// @param background   A box filled behind the text; 0 (transparent) for
  ///                     none.
  /// @param offset       From the anchor to the text, in display pixels.
  void label(glm::vec2 image_point, const char* text, ImU32 color,
             ImU32 background = 0,
             glm::vec2 offset = glm::vec2(0.0f, 0.0f)) const;

  /// @brief Text pinned inside the top-left corner of the clip rectangle:
  ///        the region's, while it is in view, and the viewport's once the
  ///        view pans past it, so a title stays in sight.
  ///
  /// The text starts 4 display pixels in from that corner, snapped as a
  /// label's is, so its background box -- 2 pixels around it -- stays inside
  /// the clip rectangle wherever the view is; text too large for the
  /// rectangle is clipped. It is drawn when called, over the marks before it.
  /// @param text        The text; null or empty draws nothing.
  /// @param color       Its color (`IM_COL32`).
  /// @param background  A box filled behind the text; 0 (transparent) for
  ///                    none.
  void caption(const char* text, ImU32 color, ImU32 background = 0) const;

  /// @brief Divide the region into @p columns x @p rows equal cells, fill
  ///        each and outline them.
  ///
  /// The cells span the region's edges whatever its @ref PixelCenters: cell
  /// `(c, r)` covers texel columns `[c * w / columns, (c + 1) * w / columns)`
  /// of a `w`-texel-wide region, and rows likewise. Inner borders are
  /// centered on the cell edges; the outer ones run along the region's
  /// outermost pixels, inside the clip, so each keeps its whole width.
  /// @param columns     The number of cells across; non-positive draws
  ///                    nothing.
  /// @param rows        The number of cells down; non-positive draws nothing.
  /// @param fills       The cells' fill colors in row-major order, a
  ///                    transparent one leaving its cell unfilled; null fills
  ///                    none.
  /// @param fill_count  The number of @p fills; cells past it are unfilled.
  /// @param line_color  The color of the cell borders; transparent draws
  ///                    none.
  /// @param thickness   The border width, in display pixels.
  void grid(int columns, int rows, const ImU32* fills, size_t fill_count,
            ImU32 line_color, float thickness = 1.0f) const;

 private:
  /// Whether the clip rectangle is empty (or NaN), so nothing is drawn.
  bool clipped_out() const noexcept;
  /// Draws text at a display point, over its background box, clipped.
  void text_at(glm::vec2 display_point, const char* text, ImU32 color,
               ImU32 background) const;

  camera::ImageView2D view_;
  OverlayRegion region_;
  /// Added to a mark's region point to give Vulkan coordinates.
  glm::vec2 center_offset_{0.0f, 0.0f};
  glm::vec2 framebuffer_scale_{1.0f, 1.0f};
  ImDrawList* draw_list_ = nullptr;
  glm::vec2 clip_min_{0.0f, 0.0f};
  glm::vec2 clip_max_{0.0f, 0.0f};
};

}  // namespace volumetric_kit::gfx::ui

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_view_2d.hpp
/// @brief A pan-and-zoom view of a 2D image in a viewport: the mapping
///        between image coordinates and target pixels.

#include <glm/vec2.hpp>

#include "volumetric_kit/gfx/camera/export.hpp"

namespace volumetric_kit::gfx::camera {

/// @brief Where a 2D image sits in a viewport: fitted, panned and zoomed.
///
/// The 2D counterpart of @ref OrbitCamera. Maps image coordinates to target
/// pixels as `target = origin() + image * scale()`, which is what
/// `pipelines::ImageDraw` takes, and back again for picking: an overlay drawn
/// through @ref image_to_target stays on its texels however the view moves.
///
/// The state is a zoom relative to the fit and the image point at the
/// viewport's center, so resizing the viewport (@ref set_viewport) keeps what
/// is shown and scales it with the window -- a fitted view stays fitted.
/// Zoom 1 fits the whole image in the viewport with its aspect kept.
///
/// Coordinates follow Vulkan's: image texel `(i, j)` covers
/// `[i, i + 1) x [j, j + 1)`, its center at `(i + 0.5, j + 0.5)`; OpenCV's
/// convention, centers on integers, is half a texel off it. Target pixels are
/// the render target's: on a high-DPI display they are framebuffer pixels, so
/// scale a window-coordinate cursor position by the framebuffer-to-window
/// ratio before passing it to @ref zoom_about or @ref target_to_image.
///
/// @code
/// camera::ImageView2D view({3840.0f, 2160.0f}, {0.0f, 0.0f},
///                          {1280.0f, 720.0f});
/// view.zoom_about(cursor, 1.25f);  // scroll: zoom in about the cursor
/// view.pan(drag_delta);            // drag
/// view.set_viewport({0.0f, 0.0f}, framebuffer_size);  // window resized
/// draw.origin = view.origin();
/// draw.scale = view.scale();
/// @endcode
class VG_CAMERA_API ImageView2D {
 public:
  /// @brief A 1 x 1 image fitted to a 1 x 1 viewport.
  ImageView2D() = default;

  /// @brief A view of @p image_size fitted to the viewport, centered.
  /// @param image_size       The image's size in texels; positive.
  /// @param viewport_origin  The viewport's top-left corner, in target pixels.
  /// @param viewport_size    The viewport's size, in target pixels.
  ImageView2D(glm::vec2 image_size, glm::vec2 viewport_origin,
              glm::vec2 viewport_size) noexcept;

  /// @brief Move or resize the viewport, keeping the zoom (relative to the
  ///        fit) and the image point at its center.
  /// @param origin  The viewport's top-left corner, in target pixels.
  /// @param size    Its size, in target pixels; a zero size (a minimized
  ///                window) gives a zero @ref scale until it grows again.
  void set_viewport(glm::vec2 origin, glm::vec2 size) noexcept;

  /// @brief Show another image, fitted.
  /// @param size  The image's size in texels; positive.
  void set_image_size(glm::vec2 size) noexcept;

  /// @brief Fit the whole image in the viewport, centered (zoom 1).
  void fit() noexcept;

  /// @brief Zoom by @p factor, keeping the image point under @p target_point
  ///        where it is.
  ///
  /// The zoom is clamped to the limits set by @ref set_zoom_limits.
  /// @param target_point  The fixed point, in target pixels: the cursor.
  /// @param factor        Above 1 zooms in, below 1 out; positive.
  void zoom_about(glm::vec2 target_point, float factor) noexcept;

  /// @brief Move the image by @p delta target pixels.
  /// @param delta  The move, in target pixels: a drag's.
  void pan(glm::vec2 delta) noexcept;

  /// @brief Bound how far @ref zoom_about zooms.
  /// @param min_zoom   The smallest zoom, relative to the fit; positive.
  /// @param max_scale  The largest scale, in target pixels per texel; a fit
  ///                   larger than it is still allowed.
  void set_zoom_limits(float min_zoom, float max_scale) noexcept;

  /// @return Target pixels per texel (0 for an empty viewport).
  float scale() const noexcept;
  /// @return Where image coordinate (0, 0) lands, in target pixels.
  glm::vec2 origin() const noexcept;
  /// @return The zoom relative to the fit: 1 fits the image.
  float zoom() const noexcept { return zoom_; }
  /// @return The image point at the viewport's center.
  glm::vec2 center() const noexcept { return center_; }
  /// @return The image's size in texels.
  glm::vec2 image_size() const noexcept { return image_size_; }
  /// @return The viewport's top-left corner, in target pixels.
  glm::vec2 viewport_origin() const noexcept { return viewport_origin_; }
  /// @return The viewport's size, in target pixels.
  glm::vec2 viewport_size() const noexcept { return viewport_size_; }

  /// @brief Map an image point to target pixels.
  /// @param image_point  In image coordinates.
  /// @return Its position in target pixels.
  glm::vec2 image_to_target(glm::vec2 image_point) const noexcept;

  /// @brief Map a target pixel position to image coordinates.
  /// @param target_point  In target pixels.
  /// @return The image point there; @ref center for an empty viewport.
  glm::vec2 target_to_image(glm::vec2 target_point) const noexcept;

 private:
  float fit_scale() const noexcept;
  float clamp_zoom(float zoom) const noexcept;

  glm::vec2 image_size_{1.0f, 1.0f};
  glm::vec2 viewport_origin_{0.0f, 0.0f};
  glm::vec2 viewport_size_{1.0f, 1.0f};
  float zoom_ = 1.0f;
  glm::vec2 center_{0.5f, 0.5f};
  float min_zoom_ = 0.25f;
  float max_scale_ = 64.0f;
};

}  // namespace volumetric_kit::gfx::camera

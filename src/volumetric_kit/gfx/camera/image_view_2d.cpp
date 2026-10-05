// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/camera/image_view_2d.hpp"

#include <algorithm>
#include <cmath>

#include <glm/common.hpp>

namespace volumetric_kit::gfx::camera {

ImageView2D::ImageView2D(glm::vec2 image_size, glm::vec2 viewport_origin,
                         glm::vec2 viewport_size) noexcept
    : viewport_origin_(viewport_origin),
      viewport_size_(glm::max(viewport_size, glm::vec2{0.0f, 0.0f})) {
  set_image_size(image_size);
}

void ImageView2D::set_viewport(glm::vec2 origin, glm::vec2 size) noexcept {
  viewport_origin_ = origin;
  viewport_size_ = glm::max(size, glm::vec2{0.0f, 0.0f});
  // A larger viewport raises the fit, and with it the scale of a zoom
  // relative to the fit, which may then pass max_scale_. An empty one (a
  // minimized window) has no fit to clamp against: keep the zoom for when it
  // grows again.
  if (fit_scale() > 0.0f) {
    zoom_ = clamp_zoom(zoom_);
  }
}

void ImageView2D::set_image_size(glm::vec2 size) noexcept {
  // A non-positive size would make every mapping divide by zero; keep 1 x 1.
  image_size_ =
      glm::vec2{size.x > 0.0f ? size.x : 1.0f, size.y > 0.0f ? size.y : 1.0f};
  fit();
}

void ImageView2D::fit() noexcept {
  zoom_ = 1.0f;
  center_ = image_size_ * 0.5f;
}

void ImageView2D::zoom_about(glm::vec2 target_point, float factor) noexcept {
  const float s = scale();
  if (!(factor > 0.0f) || !std::isfinite(factor) || s <= 0.0f) {
    return;
  }
  const glm::vec2 fixed = target_to_image(target_point);
  zoom_ = clamp_zoom(zoom_ * factor);
  // Choose the center so `fixed` still lands on target_point:
  // target_point = viewport center + (fixed - center) * new scale.
  const glm::vec2 viewport_center = viewport_origin_ + viewport_size_ * 0.5f;
  center_ = fixed - (target_point - viewport_center) / scale();
}

void ImageView2D::pan(glm::vec2 delta) noexcept {
  const float s = scale();
  if (s > 0.0f) {
    center_ -= delta / s;
  }
}

void ImageView2D::set_zoom_limits(float min_zoom, float max_scale) noexcept {
  if (min_zoom > 0.0f) {
    // The fit is always allowed, so the smallest zoom is at most 1.
    min_zoom_ = std::min(min_zoom, 1.0f);
  }
  if (max_scale > 0.0f) {
    max_scale_ = max_scale;
  }
  zoom_ = clamp_zoom(zoom_);
}

float ImageView2D::scale() const noexcept { return fit_scale() * zoom_; }

glm::vec2 ImageView2D::origin() const noexcept {
  return viewport_origin_ + viewport_size_ * 0.5f - center_ * scale();
}

glm::vec2 ImageView2D::image_to_target(glm::vec2 image_point) const noexcept {
  return origin() + image_point * scale();
}

glm::vec2 ImageView2D::target_to_image(glm::vec2 target_point) const noexcept {
  const float s = scale();
  if (s <= 0.0f) {
    return center_;
  }
  return (target_point - origin()) / s;
}

float ImageView2D::fit_scale() const noexcept {
  return std::min(viewport_size_.x / image_size_.x,
                  viewport_size_.y / image_size_.y);
}

float ImageView2D::clamp_zoom(float zoom) const noexcept {
  // The largest zoom reaches max_scale_ pixels per texel, but never forbids
  // the fit itself, which a small image in a large viewport exceeds; nor does
  // the smallest, which set_zoom_limits keeps at most 1.
  const float fit = fit_scale();
  const float max_zoom = fit > 0.0f ? std::max(1.0f, max_scale_ / fit) : 1.0f;
  return std::clamp(zoom, min_zoom_, max_zoom);
}

}  // namespace volumetric_kit::gfx::camera

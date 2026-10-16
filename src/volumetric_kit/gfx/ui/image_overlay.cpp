// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/image_overlay.hpp"

#include <algorithm>
#include <cmath>

#include <glm/common.hpp>

namespace volumetric_kit::gfx::ui {

namespace {

/// Padding of a label's background box around its text, in display pixels.
constexpr float kLabelPad = 2.0f;

/// The largest radius or thickness passed to ImGui, in display pixels: ImGui
/// converts sizes to int, which is undefined for NaN or one past INT_MAX.
constexpr float kMaxSize = 10000.0f;

/// How far a caption's text starts in from the clip rectangle's corner, in
/// display pixels: its background padding, and a margin wider than the pixel
/// snap, which moves text up to a pixel back toward the corner.
constexpr float kCaptionInset = 2.0f * kLabelPad;

bool finite(glm::vec2 p) noexcept {
  return std::isfinite(p.x) && std::isfinite(p.y);
}

ImVec2 im(glm::vec2 p) noexcept { return ImVec2(p.x, p.y); }

float usable_scale(float s) noexcept {
  return s > 0.0f && std::isfinite(s) ? s : 1.0f;
}

/// A radius or thickness clamped to [0, kMaxSize], NaN to 0.
float usable_size(float size) noexcept {
  return size > 0.0f ? std::min(size, kMaxSize) : 0.0f;
}

/// A non-finite region becomes an empty one, so the clip stays finite.
OverlayRegion usable_region(const OverlayRegion& region) noexcept {
  return finite(region.origin) && finite(region.size) ? region
                                                      : OverlayRegion{};
}

/// Clips a draw list to a rectangle (within its current clip) while in scope.
class ClipScope {
 public:
  ClipScope(ImDrawList& list, glm::vec2 min, glm::vec2 max) : list_(list) {
    list_.PushClipRect(im(min), im(max), true);
  }
  ~ClipScope() { list_.PopClipRect(); }
  ClipScope(const ClipScope&) = delete;
  ClipScope& operator=(const ClipScope&) = delete;

 private:
  ImDrawList& list_;
};

}  // namespace

ImageOverlay::ImageOverlay(const camera::ImageView2D& view,
                           PixelCenters centers, ImDrawList* draw_list)
    : ImageOverlay(view, centers,
                   OverlayRegion{{0.0f, 0.0f}, view.image_size()}, draw_list) {}

ImageOverlay::ImageOverlay(const camera::ImageView2D& view,
                           PixelCenters centers, const OverlayRegion& region,
                           ImDrawList* draw_list)
    : view_(view),
      region_(usable_region(region)),
      center_offset_(centers == PixelCenters::Integer ? glm::vec2(0.5f, 0.5f)
                                                      : glm::vec2(0.0f, 0.0f)),
      draw_list_(draw_list != nullptr ? draw_list
                                      : ImGui::GetBackgroundDrawList()) {
  const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
  framebuffer_scale_ = {usable_scale(scale.x), usable_scale(scale.y)};
  // Region within viewport. An empty one -- out of view, or a size that is
  // not positive -- keeps its maximum at its minimum.
  const glm::vec2 viewport_min = view_.viewport_origin() / framebuffer_scale_;
  const glm::vec2 viewport_max =
      (view_.viewport_origin() + view_.viewport_size()) / framebuffer_scale_;
  clip_min_ = glm::max(edge_to_display({0.0f, 0.0f}), viewport_min);
  clip_max_ = glm::max(clip_min_,
                       glm::min(edge_to_display(region_.size), viewport_max));
}

glm::vec2 ImageOverlay::to_display(glm::vec2 image_point) const noexcept {
  return edge_to_display(image_point + center_offset_);
}

glm::vec2 ImageOverlay::edge_to_display(glm::vec2 edge_point) const noexcept {
  return view_.image_to_target(region_.origin + edge_point) /
         framebuffer_scale_;
}

void ImageOverlay::point(glm::vec2 image_point, ImU32 color,
                         float radius) const {
  const glm::vec2 p = to_display(image_point);
  if (clipped_out() || !finite(p)) {
    return;
  }
  const ClipScope clip(*draw_list_, clip_min_, clip_max_);
  draw_list_->AddCircleFilled(im(p), usable_size(radius), color);
}

void ImageOverlay::polyline(const glm::vec2* points, size_t count, ImU32 color,
                            float thickness, bool closed) const {
  if (points == nullptr || count < 2 || clipped_out()) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    const glm::vec2 p = to_display(points[i]);
    if (!finite(p)) {
      draw_list_->PathClear();
      return;
    }
    draw_list_->PathLineTo(im(p));
  }
  const ClipScope clip(*draw_list_, clip_min_, clip_max_);
  draw_list_->PathStroke(color, usable_size(thickness),
                         closed ? ImDrawFlags_Closed : ImDrawFlags_None);
}

void ImageOverlay::label(glm::vec2 image_point, const char* text, ImU32 color,
                         ImU32 background, glm::vec2 offset) const {
  if (text == nullptr || *text == '\0' || clipped_out()) {
    return;
  }
  const glm::vec2 p = to_display(image_point) + offset;
  if (!finite(p)) {
    return;
  }
  text_at(p, text, color, background);
}

void ImageOverlay::caption(const char* text, ImU32 color,
                           ImU32 background) const {
  if (text == nullptr || *text == '\0' || clipped_out()) {
    return;
  }
  text_at(clip_min_ + kCaptionInset, text, color, background);
}

void ImageOverlay::grid(int columns, int rows, const ImU32* fills,
                        size_t fill_count, ImU32 line_color,
                        float thickness) const {
  if (columns <= 0 || rows <= 0 || clipped_out()) {
    return;
  }
  const float width = usable_size(thickness);
  const glm::vec2 size = region_.size;
  const auto corner = [&](int column, int row) {
    return im(edge_to_display(
        {size.x * static_cast<float>(column) / static_cast<float>(columns),
         size.y * static_cast<float>(row) / static_cast<float>(rows)}));
  };
  const ClipScope clip(*draw_list_, clip_min_, clip_max_);
  const size_t across = static_cast<size_t>(columns);
  const size_t filled =
      fills != nullptr
          ? std::min(fill_count, across * static_cast<size_t>(rows))
          : 0;
  for (size_t cell = 0; cell < filled; ++cell) {
    const int column = static_cast<int>(cell % across);
    const int row = static_cast<int>(cell / across);
    // AddRectFilled skips a transparent fill.
    draw_list_->AddRectFilled(corner(column, row), corner(column + 1, row + 1),
                              fills[cell]);
  }
  // Strokes centered on the edges (AddLine would shift them half a pixel); a
  // transparent color adds nothing.
  const auto border = [&](ImVec2 from, ImVec2 to) {
    draw_list_->PathLineTo(from);
    draw_list_->PathLineTo(to);
    draw_list_->PathStroke(line_color, width, ImDrawFlags_None);
  };
  for (int column = 0; column <= columns; ++column) {
    border(corner(column, 0), corner(column, rows));
  }
  for (int row = 0; row <= rows; ++row) {
    border(corner(0, row), corner(columns, row));
  }
}

bool ImageOverlay::clipped_out() const noexcept {
  return !(clip_max_.x > clip_min_.x && clip_max_.y > clip_min_.y);
}

void ImageOverlay::text_at(glm::vec2 display_point, const char* text,
                           ImU32 color, ImU32 background) const {
  glm::vec2 p = display_point;
  // Where ImGui will draw the text (IM_TRUNC), so the box pads it evenly.
  if ((draw_list_->Flags & ImDrawListFlags_TextNoPixelSnap) == 0) {
    p = glm::trunc(p);
  }
  ImFont* font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const ImVec2 size = ImGui::CalcTextSize(text);
  const glm::vec2 box_min = p - kLabelPad;
  const glm::vec2 box_max = p + glm::vec2(size.x, size.y) + kLabelPad;

  const ClipScope clip(*draw_list_, clip_min_, clip_max_);
  // Wholly outside the clip, it adds nothing -- and ImGui's text layout, which
  // truncates the position to an int, never sees a far-off one.
  const ImVec2 list_min = draw_list_->GetClipRectMin();
  const ImVec2 list_max = draw_list_->GetClipRectMax();
  if (box_max.x <= list_min.x || box_min.x >= list_max.x ||
      box_max.y <= list_min.y || box_min.y >= list_max.y) {
    return;
  }
  draw_list_->AddRectFilled(im(box_min), im(box_max), background);
  draw_list_->AddText(font, font_size, im(p), color, text);
}

}  // namespace volumetric_kit::gfx::ui

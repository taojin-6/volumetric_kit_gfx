// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/image_overlay.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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

/// The target pixel edge before the first pixel whose center is at or past
/// `x`: a rectangle so rounded holds the pixels whose centers it covers, one
/// on its top or left edge included, as Vulkan rasterizes.
glm::vec2 pixel_edge(glm::vec2 x) noexcept { return glm::ceil(x - 0.5f); }

/// Display coordinates for target pixel edges `min` to `max` on an axis --
/// whole numbers, `max >= min` -- at `scale` target pixels per display pixel.
/// ImGui's Vulkan backend multiplies a clip rectangle by the scale and
/// truncates its corner and its size, so each edge is nudged up until those
/// give back exactly `min` and `max - min`, at any scale.
glm::vec2 display_edges(float min, float max, float scale) noexcept {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float lo = min / scale;
  for (int nudge = 0; nudge < 8 && lo * scale < min; ++nudge) {
    lo = std::nextafter(lo, kInf);
  }
  if (!(max > min)) {
    return {lo, lo};
  }
  float hi = max / scale;
  for (int nudge = 0; nudge < 8; ++nudge) {
    // Separate statements, as the backend's, so no product is fused.
    const float start = lo * scale;
    const float end = hi * scale;
    if (end - start >= max - min) {
      break;
    }
    hi = std::nextafter(hi, kInf);
  }
  return {lo, hi};
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
      region_(region),
      center_offset_(centers == PixelCenters::Integer ? glm::vec2(0.5f, 0.5f)
                                                      : glm::vec2(0.0f, 0.0f)),
      draw_list_(draw_list != nullptr ? draw_list
                                      : ImGui::GetBackgroundDrawList()) {
  const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
  framebuffer_scale_ = {usable_scale(scale.x), usable_scale(scale.y)};
  // Region within viewport, on the target pixels whose centers lie in it, as
  // the image's own are. An empty one -- out of view, a size that is not
  // positive, or not finite -- keeps its maximum at its minimum.
  const OverlayRegion clip = usable_region(region_);
  const glm::vec2 viewport_min = view_.viewport_origin();
  const glm::vec2 viewport_max = viewport_min + view_.viewport_size();
  glm::vec2 min =
      pixel_edge(glm::max(view_.image_to_target(clip.origin), viewport_min));
  glm::vec2 max = glm::max(
      min, pixel_edge(glm::min(view_.image_to_target(clip.origin + clip.size),
                               viewport_max)));
  if (!finite(min) || !finite(max)) {
    min = finite(viewport_min) ? pixel_edge(viewport_min) : glm::vec2(0.0f);
    max = min;
  }
  for (int axis = 0; axis < 2; ++axis) {
    const glm::vec2 edges =
        display_edges(min[axis], max[axis], framebuffer_scale_[axis]);
    clip_min_[axis] = edges[0];
    clip_max_[axis] = edges[1];
  }
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
    return edge_to_display(
        {size.x * static_cast<float>(column) / static_cast<float>(columns),
         size.y * static_cast<float>(row) / static_cast<float>(rows)});
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
    draw_list_->AddRectFilled(im(corner(column, row)),
                              im(corner(column + 1, row + 1)), fills[cell]);
  }
  // Every border spans the region's pixels -- its clip, unless the viewport
  // cuts it. Inner ones are strokes centered on the cell edges (AddLine would
  // shift them half a pixel); the outer ones run along the outermost pixels,
  // so the clip leaves them their whole width at any phase. A transparent
  // color adds nothing.
  const glm::vec2 lo =
      pixel_edge(view_.image_to_target(region_.origin)) / framebuffer_scale_;
  const glm::vec2 hi =
      pixel_edge(view_.image_to_target(region_.origin + size)) /
      framebuffer_scale_;
  const float half = 0.5f * width;
  const auto border = [&](glm::vec2 from, glm::vec2 to) {
    draw_list_->PathLineTo(im(from));
    draw_list_->PathLineTo(im(to));
    draw_list_->PathStroke(line_color, width, ImDrawFlags_None);
  };
  for (int column = 0; column <= columns; ++column) {
    const float x = column == 0         ? lo.x + half
                    : column == columns ? hi.x - half
                                        : corner(column, 0).x;
    border({x, lo.y}, {x, hi.y});
  }
  for (int row = 0; row <= rows; ++row) {
    const float y = row == 0      ? lo.y + half
                    : row == rows ? hi.y - half
                                  : corner(0, row).y;
    border({lo.x, y}, {hi.x, y});
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

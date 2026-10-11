// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/image_overlay.hpp"

#include <cmath>

namespace volumetric_kit::gfx::ui {

namespace {

/// Padding of a label's background box around its text, in display pixels.
constexpr float kLabelPad = 2.0f;

bool finite(glm::vec2 p) noexcept {
  return std::isfinite(p.x) && std::isfinite(p.y);
}

ImVec2 im(glm::vec2 p) noexcept { return ImVec2(p.x, p.y); }

float usable_scale(float s) noexcept {
  return s > 0.0f && std::isfinite(s) ? s : 1.0f;
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
    : view_(view),
      center_offset_(centers == PixelCenters::Integer ? glm::vec2(0.5f, 0.5f)
                                                      : glm::vec2(0.0f, 0.0f)),
      draw_list_(draw_list != nullptr ? draw_list
                                      : ImGui::GetBackgroundDrawList()) {
  const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
  framebuffer_scale_ = {usable_scale(scale.x), usable_scale(scale.y)};
}

glm::vec2 ImageOverlay::to_display(glm::vec2 image_point) const noexcept {
  return edge_to_display(image_point + center_offset_);
}

glm::vec2 ImageOverlay::edge_to_display(glm::vec2 image_point) const noexcept {
  return view_.image_to_target(image_point) / framebuffer_scale_;
}

void ImageOverlay::point(glm::vec2 image_point, ImU32 color,
                         float radius) const {
  const glm::vec2 p = to_display(image_point);
  if (!finite(p)) {
    return;
  }
  const ClipScope clip(*draw_list_, viewport_min(), viewport_max());
  draw_list_->AddCircleFilled(im(p), radius, color);
}

void ImageOverlay::polyline(const glm::vec2* points, size_t count, ImU32 color,
                            float thickness, bool closed) const {
  if (points == nullptr || count < 2) {
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
  const ClipScope clip(*draw_list_, viewport_min(), viewport_max());
  draw_list_->PathStroke(color, thickness,
                         closed ? ImDrawFlags_Closed : ImDrawFlags_None);
}

void ImageOverlay::label(glm::vec2 image_point, const char* text, ImU32 color,
                         ImU32 background, glm::vec2 offset) const {
  if (text == nullptr) {
    return;
  }
  const glm::vec2 p = to_display(image_point) + offset;
  if (!finite(p)) {
    return;
  }
  ImFont* font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const ImVec2 size = ImGui::CalcTextSize(text);
  const glm::vec2 box_min = p - kLabelPad;
  const glm::vec2 box_max = p + glm::vec2(size.x, size.y) + kLabelPad;

  const ClipScope clip(*draw_list_, viewport_min(), viewport_max());
  // Wholly outside the clip, it adds nothing -- and ImGui's text layout, which
  // truncates the position to an int, never sees a far-off one.
  const ImVec2 clip_min = draw_list_->GetClipRectMin();
  const ImVec2 clip_max = draw_list_->GetClipRectMax();
  if (box_max.x <= clip_min.x || box_min.x >= clip_max.x ||
      box_max.y <= clip_min.y || box_min.y >= clip_max.y) {
    return;
  }
  draw_list_->AddRectFilled(im(box_min), im(box_max), background);
  draw_list_->AddText(font, font_size, im(p), color, text);
}

void ImageOverlay::grid(int columns, int rows, const ImU32* fills,
                        ImU32 line_color, float thickness) const {
  if (columns <= 0 || rows <= 0) {
    return;
  }
  const glm::vec2 size = view_.image_size();
  const auto corner = [&](int column, int row) {
    return im(edge_to_display(
        {size.x * static_cast<float>(column) / static_cast<float>(columns),
         size.y * static_cast<float>(row) / static_cast<float>(rows)}));
  };
  const ClipScope clip(*draw_list_, viewport_min(), viewport_max());
  if (fills != nullptr) {
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        // AddRectFilled skips a transparent fill.
        draw_list_->AddRectFilled(
            corner(column, row), corner(column + 1, row + 1),
            fills[static_cast<size_t>(row) * static_cast<size_t>(columns) +
                  static_cast<size_t>(column)]);
      }
    }
  }
  // Strokes centred on the edges (AddLine would shift them half a pixel); a
  // transparent colour adds nothing.
  const auto border = [&](ImVec2 from, ImVec2 to) {
    draw_list_->PathLineTo(from);
    draw_list_->PathLineTo(to);
    draw_list_->PathStroke(line_color, thickness, ImDrawFlags_None);
  };
  for (int column = 0; column <= columns; ++column) {
    border(corner(column, 0), corner(column, rows));
  }
  for (int row = 0; row <= rows; ++row) {
    border(corner(0, row), corner(columns, row));
  }
}

glm::vec2 ImageOverlay::viewport_min() const noexcept {
  return view_.viewport_origin() / framebuffer_scale_;
}

glm::vec2 ImageOverlay::viewport_max() const noexcept {
  return (view_.viewport_origin() + view_.viewport_size()) / framebuffer_scale_;
}

}  // namespace volumetric_kit::gfx::ui

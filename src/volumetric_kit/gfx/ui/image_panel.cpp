// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/image_panel.hpp"

#include <algorithm>
#include <cmath>

#include <glm/common.hpp>

#include "imgui.h"

namespace volumetric_kit::gfx::ui {
namespace {

// The pictures as one image, along the panel docked at `side`.
glm::vec2 composite(DockSide side, glm::vec2 size, uint32_t count) {
  const float n = static_cast<float>(count);
  const float gaps = (n - 1.0f) * ImagePanel::kGap;
  return side == DockSide::Right ? glm::vec2(size.x, n * size.y + gaps)
                                 : glm::vec2(n * size.x + gaps, size.y);
}

// Picture `i`'s top-left in that image.
glm::vec2 origin_of(DockSide side, glm::vec2 size, uint32_t i) {
  const float at = static_cast<float>(i);
  return side == DockSide::Right
             ? glm::vec2(0.0f, at * (size.y + ImagePanel::kGap))
             : glm::vec2(at * (size.x + ImagePanel::kGap), 0.0f);
}

// Where `point` of the image along `from` lies in the image along `to`: the
// same point of the picture it is in, or the nearest point of the nearest
// picture (from a gap, the picture on its side of the gap's middle), so it
// lies inside the new image.
glm::vec2 moved(glm::vec2 point, DockSide from, DockSide to, glm::vec2 size,
                uint32_t count) {
  const bool right = from == DockSide::Right;
  const float step = (right ? size.y : size.x) + ImagePanel::kGap;
  const float at = std::floor(
      ((right ? point.y : point.x) + 0.5f * ImagePanel::kGap) / step);
  const auto i = static_cast<uint32_t>(
      std::clamp(at, 0.0f, static_cast<float>(count - 1)));
  const glm::vec2 in =
      glm::clamp(point - origin_of(from, size, i), glm::vec2(0.0f, 0.0f), size);
  return in + origin_of(to, size, i);
}

// Whether `view` fits its image: zoom 1, give or take rounding.
bool fitted(const camera::ImageView2D& view) {
  constexpr float kFitted = 1e-3f;
  return std::abs(view.zoom() - 1.0f) <= kFitted;
}

}  // namespace

void ImagePanel::set_pictures(glm::vec2 size, uint32_t count) noexcept {
  picture_size_ = size;
  picture_count_ = count;
}

DockedContentSize ImagePanel::content_size() const noexcept {
  const bool has_pictures =
      picture_count_ > 0 && picture_size_.x > 0.0f && picture_size_.y > 0.0f;
  if (!has_pictures) {
    return {};
  }
  return {composite(DockSide::Right, picture_size_, picture_count_),
          composite(DockSide::Bottom, picture_size_, picture_count_)};
}

void ImagePanel::arrange(const VkRect2D& content, DockSide side) noexcept {
  if (content.extent.width == 0 || content.extent.height == 0) {
    return;
  }
  side_ = side;
  // The viewport first, so a new image fits the panel as it is now.
  view_.set_viewport({static_cast<float>(content.offset.x),
                      static_cast<float>(content.offset.y)},
                     {static_cast<float>(content.extent.width),
                      static_cast<float>(content.extent.height)});
  const bool has_pictures =
      picture_count_ > 0 && picture_size_.x > 0.0f && picture_size_.y > 0.0f;
  if (!has_pictures) {
    return;
  }
  const glm::vec2 image = composite(side, picture_size_, picture_count_);
  if (image == image_size_) {
    return;
  }
  // New pictures, or a fitted view of the same ones, fit again.
  const bool kept = picture_size_ == view_picture_size_ &&
                    picture_count_ == view_count_ && !fitted(view_);
  const float zoom = view_.zoom();
  const glm::vec2 centre = kept ? moved(view_.center(), view_side_, side,
                                        picture_size_, picture_count_)
                                : glm::vec2(0.0f, 0.0f);
  view_.set_image_size(image);  // fitted
  if (kept) {
    // The same pictures along the other side, zoomed: the zoom, and the
    // centre's point moved with its picture.
    view_.zoom_about(view_.viewport_origin() + 0.5f * view_.viewport_size(),
                     zoom);
    view_.pan((view_.center() - centre) * view_.scale());
  }
  image_size_ = image;
  view_picture_size_ = picture_size_;
  view_count_ = picture_count_;
  view_side_ = side;
}

void ImagePanel::header_items() {
  // Right-aligned before the dock's close button, a frame height square an
  // item spacing on.
  const ImGuiStyle& style = ImGui::GetStyle();
  const float buttons = ImGui::CalcTextSize("Fit").x +
                        ImGui::CalcTextSize("1:1").x +
                        4.0f * style.FramePadding.x + ImGui::GetFrameHeight() +
                        2.0f * style.ItemSpacing.x;
  const float room = ImGui::GetContentRegionAvail().x - buttons;
  if (room > 0.0f) {
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + room);
  }
  if (ImGui::Button("Fit")) {
    view_.fit();
  }
  ImGui::SameLine();
  if (ImGui::Button("1:1") && view_.scale() > 0.0f) {
    view_.zoom_about(view_.viewport_origin() + 0.5f * view_.viewport_size(),
                     1.0f / view_.scale());
  }
}

glm::vec2 ImagePanel::picture_origin(uint32_t i) const noexcept {
  return origin_of(side_, picture_size_, i);
}

void ImagePanel::double_tap(glm::vec2 point) noexcept {
  if (fitted(view_) && view_.scale() > 0.0f) {
    view_.zoom_about(point, 1.0f / view_.scale());
  } else {
    view_.fit();
  }
}

}  // namespace volumetric_kit::gfx::ui

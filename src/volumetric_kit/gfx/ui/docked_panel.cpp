// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/docked_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "imgui.h"

namespace volumetric_kit::gfx::ui {
namespace {

// How much larger the content must show on one side to dock there.
constexpr float kPreference = 1.25f;
constexpr float kBackgroundAlpha = 0.6f;

// A rectangle of whole pixels from [x0, x1) x [y0, y1), rounded; empty where
// it is inverted.
VkRect2D pixel_rect(float x0, float y0, float x1, float y1) {
  const float left = std::round(x0);
  const float top = std::round(y0);
  const float right = std::max(std::round(x1), left);
  const float bottom = std::max(std::round(y1), top);
  VkRect2D rect{};
  rect.offset = {static_cast<int32_t>(left), static_cast<int32_t>(top)};
  rect.extent = {static_cast<uint32_t>(right - left),
                 static_cast<uint32_t>(bottom - top)};
  return rect;
}

// The panel's size across an axis `axis` pixels long, at `share`, in whole
// pixels: the share held to [kMinShare, kMaxShare], then at least `least`,
// then at most kMaxShare of the axis.
float across(float share, float axis, float least) {
  const float held =
      std::clamp(share, DockedPanel::kMinShare, DockedPanel::kMaxShare) * axis;
  return std::round(
      std::min(std::max(held, least), DockedPanel::kMaxShare * axis));
}

// Whether `size` is a size: both sides above 0.
bool sized(glm::vec2 size) { return size.x > 0.0f && size.y > 0.0f; }

// The scale `size` fits `rect` at.
float fit_scale(const VkRect2D& rect, glm::vec2 size) {
  return std::min(static_cast<float>(rect.extent.width) / size.x,
                  static_cast<float>(rect.extent.height) / size.y);
}

// This frame's metrics, in drawable pixels: the safe area (x0, y0, x1, y1),
// the bar across each axis, the panel's least size across each axis, and the
// header's height.
struct Metrics {
  VkExtent2D drawable{};
  float safe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  glm::vec2 bar{0.0f, 0.0f};
  glm::vec2 least{0.0f, 0.0f};
  float header = 0.0f;
};

// The scene, the content and the panel's inner edge, docked at `side` at
// `share`, and at least the panel's least size across where `least`.
struct Geometry {
  VkRect2D scene{};
  VkRect2D content{};
  float edge = 0.0f;
};

Geometry geometry(DockSide side, float share, const Metrics& m,
                  bool least = true) {
  const float* safe = m.safe;
  const auto width = static_cast<float>(m.drawable.width);
  const auto height = static_cast<float>(m.drawable.height);
  Geometry g;
  if (side == DockSide::Right) {
    g.edge =
        safe[2] - across(share, safe[2] - safe[0], least ? m.least.x : 0.0f);
    g.scene = pixel_rect(0.0f, 0.0f, g.edge, height);
    g.content =
        pixel_rect(g.edge + m.bar.x, safe[1] + m.header, safe[2], safe[3]);
  } else {
    g.edge =
        safe[3] - across(share, safe[3] - safe[1], least ? m.least.y : 0.0f);
    g.scene = pixel_rect(0.0f, 0.0f, width, g.edge);
    g.content =
        pixel_rect(safe[0], g.edge + m.bar.y + m.header, safe[2], safe[3]);
  }
  return g;
}

Metrics metrics(VkExtent2D drawable, float min_points) {
  const ImGuiIO& io = ImGui::GetIO();
  const ImGuiStyle& style = ImGui::GetStyle();
  const ImVec2 s = io.DisplayFramebufferScale;
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  Metrics m;
  m.drawable = drawable;
  const auto width = static_cast<float>(drawable.width);
  const auto height = static_cast<float>(drawable.height);
  m.safe[0] = std::clamp(std::round(viewport->WorkPos.x * s.x), 0.0f, width);
  m.safe[1] = std::clamp(std::round(viewport->WorkPos.y * s.y), 0.0f, height);
  m.safe[2] =
      std::clamp(std::round((viewport->WorkPos.x + viewport->WorkSize.x) * s.x),
                 m.safe[0], width);
  m.safe[3] =
      std::clamp(std::round((viewport->WorkPos.y + viewport->WorkSize.y) * s.y),
                 m.safe[1], height);
  m.bar = {std::round(DockedPanel::kBarPoints * s.x),
           std::round(DockedPanel::kBarPoints * s.y)};
  m.least = {min_points * s.x, min_points * s.y};
  m.header =
      std::round((ImGui::GetFrameHeight() + 2.0f * style.FramePadding.y) * s.y);
  return m;
}

}  // namespace

DockedPanel::DockedPanel(const DockedPanelConfig& config)
    : config_(config), state_(config.state) {}

float& DockedPanel::share(DockSide side) noexcept {
  return side == DockSide::Right ? state_.right_share : state_.bottom_share;
}

void DockedPanel::set_content_size(const DockedContentSize& size) noexcept {
  if (size.right != content_size_.right ||
      size.bottom != content_size_.bottom) {
    content_size_ = size;
    content_changed_ = true;
  }
}

const PanelLayout& DockedPanel::layout(VkExtent2D drawable) {
  const Metrics m = metrics(drawable, config_.min_points);

  // The side, chosen again only when the drawable or the content changes:
  // where the content fits larger at the default shares as they stand, by
  // more than kPreference, else at the end of the long axis.
  if (content_changed_ || drawable.width != chosen_for_.width ||
      drawable.height != chosen_for_.height) {
    DockSide side =
        drawable.width >= drawable.height ? DockSide::Right : DockSide::Bottom;
    if (sized(content_size_.right) && sized(content_size_.bottom)) {
      const float right = fit_scale(
          geometry(DockSide::Right, config_.state.right_share, m, false)
              .content,
          content_size_.right);
      const float bottom = fit_scale(
          geometry(DockSide::Bottom, config_.state.bottom_share, m, false)
              .content,
          content_size_.bottom);
      if (right > kPreference * bottom) {
        side = DockSide::Right;
      } else if (bottom > kPreference * right) {
        side = DockSide::Bottom;
      }
    }
    layout_.side = side;
    chosen_for_ = drawable;
    content_changed_ = false;
  }

  std::copy(std::begin(m.safe), std::end(m.safe), std::begin(safe_));
  header_ = m.header;
  laid_out_open_ = state_.open;
  if (!state_.open) {
    layout_.scene = {{0, 0}, drawable};
    layout_.content = {};
    return layout_;
  }
  const Geometry g = geometry(layout_.side, share(layout_.side), m);
  layout_.scene = g.scene;
  layout_.content = g.content;
  edge_ = g.edge;
  return layout_;
}

void DockedPanel::draw_chrome(const char* title,
                              const std::function<void()>& items) {
  if (!state_.open || !laid_out_open_) {
    return;
  }
  ImGuiIO& io = ImGui::GetIO();
  const ImGuiStyle& style = ImGui::GetStyle();
  const ImVec2 s = io.DisplayFramebufferScale;
  const bool right = layout_.side == DockSide::Right;
  // The windows' names, on the stack: ImGui names windows globally.
  constexpr int kTitleChars = 64;
  char handle_name[kTitleChars + 16];
  char header_name[kTitleChars + 16];
  std::snprintf(handle_name, sizeof(handle_name), "%.*s##handle", kTitleChars,
                title);
  std::snprintf(header_name, sizeof(header_name), "%.*s##header", kTitleChars,
                title);
  constexpr ImGuiWindowFlags kFlags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollWithMouse;

  // The handle: a window a frame height across the bar, along the panel's
  // safe length. Into the panel it reaches no further than the header's
  // padding, so it never covers the header's items; the rest of it lies
  // over the scene.
  const float bar =
      right ? std::round(kBarPoints * s.x) : std::round(kBarPoints * s.y);
  const float hit = std::max(ImGui::GetFrameHeight(), kBarPoints);
  const float padding = right ? style.WindowPadding.x : style.FramePadding.y;
  const float inward = std::min(0.5f * (hit - kBarPoints), padding);
  const float start = (edge_ + bar) / (right ? s.x : s.y) + inward - hit;
  const ImVec2 handle_pos =
      right ? ImVec2(start, safe_[1] / s.y) : ImVec2(safe_[0] / s.x, start);
  const ImVec2 handle_size = right ? ImVec2(hit, (safe_[3] - safe_[1]) / s.y)
                                   : ImVec2((safe_[2] - safe_[0]) / s.x, hit);
  ImGui::SetNextWindowPos(handle_pos);
  ImGui::SetNextWindowSize(handle_size);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1.0f, 1.0f));
  const bool handle_drawn = ImGui::Begin(
      handle_name, nullptr, kFlags | ImGuiWindowFlags_NoBackground);
  ImGui::PopStyleVar(3);
  if (handle_drawn) {
    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
    ImGui::InvisibleButton("##drag", handle_size);
    float& current = share(layout_.side);
    if (ImGui::IsItemActivated()) {
      drag_from_ = current;
      drag_share_ = current;
    }
    const float length = right ? safe_[2] - safe_[0] : safe_[3] - safe_[1];
    if (ImGui::IsItemActive() && length > 0.0f) {
      // The share the press found, moved by the pointer since, of the safe
      // length along the dock axis: the edge keeps its place under the
      // pointer.
      const ImVec2 travel =
          ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
      drag_share_ =
          drag_from_ - (right ? travel.x * s.x : travel.y * s.y) / length;
      current = std::clamp(drag_share_, kMinShare, kMaxShare);
    }
    if (ImGui::IsItemDeactivated()) {
      // Zero within ImGui's drag threshold: a click or a tap, which changes
      // nothing.
      const ImVec2 dragged = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
      if (dragged.x == 0.0f && dragged.y == 0.0f) {
        current = drag_from_;
      } else if (drag_share_ < kMinShare) {
        state_.open = false;
        current = drag_from_;
      } else if (!config_.detents.empty()) {
        current = *std::min_element(
            config_.detents.begin(), config_.detents.end(),
            [&](float a, float b) {
              return std::abs(a - drag_share_) < std::abs(b - drag_share_);
            });
      }
    }
    const ImGuiCol colour = ImGui::IsItemActive()    ? ImGuiCol_SeparatorActive
                            : ImGui::IsItemHovered() ? ImGuiCol_SeparatorHovered
                                                     : ImGuiCol_Separator;
    const ImVec2 bar_min = right ? ImVec2(edge_ / s.x, handle_pos.y)
                                 : ImVec2(handle_pos.x, edge_ / s.y);
    const ImVec2 bar_max =
        right ? ImVec2((edge_ + bar) / s.x, handle_pos.y + handle_size.y)
              : ImVec2(handle_pos.x + handle_size.x, (edge_ + bar) / s.y);
    ImGui::GetWindowDrawList()->AddRectFilled(bar_min, bar_max,
                                              ImGui::GetColorU32(colour));
  }
  ImGui::End();

  // The header, across the panel's safe top: the title, the caller's items,
  // and a close button at the right.
  const ImVec2 header_pos = right ? ImVec2((edge_ + bar) / s.x, safe_[1] / s.y)
                                  : ImVec2(safe_[0] / s.x, (edge_ + bar) / s.y);
  const float header_width =
      right ? (safe_[2] - edge_ - bar) / s.x : (safe_[2] - safe_[0]) / s.x;
  ImGui::SetNextWindowPos(header_pos);
  ImGui::SetNextWindowSize(ImVec2(header_width, header_ / s.y));
  ImGui::SetNextWindowBgAlpha(kBackgroundAlpha);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                      ImVec2(style.WindowPadding.x, style.FramePadding.y));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1.0f, 1.0f));
  // Kept behind the handle: a press on the header never raises it over the
  // handle's reach.
  const bool header_drawn =
      ImGui::Begin(header_name, nullptr,
                   kFlags | ImGuiWindowFlags_NoScrollbar |
                       ImGuiWindowFlags_NoBringToFrontOnFocus);
  ImGui::PopStyleVar(2);
  if (header_drawn) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(title);
    if (items) {
      ImGui::SameLine();
      items();
    }
    const float frame = ImGui::GetFrameHeight();
    ImGui::SameLine();
    const float room = ImGui::GetContentRegionAvail().x - frame;
    if (room > 0.0f) {
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + room);
    }
    if (ImGui::Button("##close", ImVec2(frame, frame))) {
      state_.open = false;
    }
    // A cross, which the default font has no glyph for.
    const ImVec2 lo = ImGui::GetItemRectMin();
    const ImVec2 hi = ImGui::GetItemRectMax();
    const float inset = 0.3f * frame;
    ImDrawList* list = ImGui::GetWindowDrawList();
    const ImU32 ink = ImGui::GetColorU32(ImGuiCol_Text);
    list->AddLine(ImVec2(lo.x + inset, lo.y + inset),
                  ImVec2(hi.x - inset, hi.y - inset), ink, 1.5f);
    list->AddLine(ImVec2(lo.x + inset, hi.y - inset),
                  ImVec2(hi.x - inset, lo.y + inset), ink, 1.5f);
  }
  ImGui::End();
}

bool DockedPanel::over_content(glm::vec2 point) const noexcept {
  if (!state_.open || !laid_out_open_) {
    return false;
  }
  const VkRect2D& r = layout_.content;
  const auto x0 = static_cast<float>(r.offset.x);
  const auto y0 = static_cast<float>(r.offset.y);
  return point.x >= x0 && point.y >= y0 &&
         point.x < x0 + static_cast<float>(r.extent.width) &&
         point.y < y0 + static_cast<float>(r.extent.height);
}

}  // namespace volumetric_kit::gfx::ui

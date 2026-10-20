// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/collapsible_panel.hpp"

#include <string>

#include "volumetric_kit/core/base/check.hpp"

#include "imgui.h"
// For RenderArrow and RenderText, with which ImGui's own ArrowButton and
// TreeNode draw their arrow and label.
#include "imgui_internal.h"

namespace volumetric_kit::gfx::ui {
namespace {

constexpr ImGuiWindowFlags kFlags =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
    ImGuiWindowFlags_NoFocusOnAppearing;
constexpr float kBackgroundAlpha = 0.6f;

// The header's button, @p height tall, in TreeNode's layout: the arrow in a
// font-size box after the padding, the title two paddings on. A press flips
// @p open, and the arrow shows the new state.
void header_button(const char* title, float height, bool* open) {
  const ImGuiStyle& style = ImGui::GetStyle();
  const float title_x = ImGui::GetFontSize() + 3.0f * style.FramePadding.x;
  const float width = title_x + ImGui::CalcTextSize(title, nullptr, true).x +
                      style.FramePadding.x;
  if (ImGui::Button("##header", ImVec2(width, height))) {
    *open = !*open;
  }
  const ImVec2 min = ImGui::GetItemRectMin();
  ImGui::RenderArrow(
      ImGui::GetWindowDrawList(),
      ImVec2(min.x + style.FramePadding.x, min.y + style.FramePadding.y),
      ImGui::GetColorU32(ImGuiCol_Text),
      *open ? ImGuiDir_Down : ImGuiDir_Right);
  ImGui::RenderText(ImVec2(min.x + title_x, min.y + style.FramePadding.y),
                    title);
}

}  // namespace

bool begin_collapsible_panel(const char* id, const char* title, ImVec2 pos,
                             bool* open) {
  VKC_CHECK(id != nullptr && title != nullptr && open != nullptr,
            "begin_collapsible_panel needs an id, a title and a state");
  // The header window has no padding, so the button is all of it: no part of
  // the header takes input that the button does not.
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowBgAlpha(kBackgroundAlpha);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  const bool header_drawn = ImGui::Begin(id, nullptr, kFlags);
  ImGui::PopStyleVar();
  // The body goes below this frame's position by the frame height, not by the
  // header's size, which a new window takes a frame to measure.
  const ImVec2 header = ImGui::GetWindowPos();
  const float height = ImGui::GetFrameHeight();
  if (header_drawn) {
    header_button(title, height, open);
  }
  ImGui::End();
  if (!*open) {
    return false;
  }

  ImGui::SetNextWindowPos(ImVec2(header.x, header.y + height));
  ImGui::SetNextWindowBgAlpha(kBackgroundAlpha);
  if (ImGui::Begin((std::string(id) + "##body").c_str(), nullptr,
                   kFlags | ImGuiWindowFlags_NoInputs)) {
    return true;
  }
  ImGui::End();
  return false;
}

void end_collapsible_panel() { ImGui::End(); }

}  // namespace volumetric_kit::gfx::ui

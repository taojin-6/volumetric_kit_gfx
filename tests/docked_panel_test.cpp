// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// DockedPanel on CPU-only ImGui frames: the side with and without a content
// size, and the caller's header items. Its shares, handle and close are
// ImagePanel's, which image_panel_test drives.

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>

#include <glm/vec2.hpp>

#include "imgui.h"
#include "imgui_internal.h"  // FindWindowByName, for the header's rect
#include "ui_test_support.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/ui/docked_panel.hpp"
#include "volumetric_kit/gfx/ui/imgui_touch.hpp"

namespace {

namespace ui = volumetric_kit::gfx::ui;

constexpr const char* kTitle = "Inspector";

// An iPad each way up: its pixels at 2 pixels a point, and its safe area.
constexpr VkExtent2D kPortrait{1668, 2420};
constexpr VkExtent2D kLandscape{2420, 1668};
const ui::EdgeInsets kSafe{24, 0, 20, 0};

ui::DockedPanelConfig opened() {
  ui::DockedPanelConfig config;
  config.state.open = true;
  return config;
}

class DockedPanelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto touch = ui::ImGuiTouch::create(ImGui::GetCurrentContext());
    ASSERT_TRUE(touch.ok()) << touch.status().message();
    touch_ = std::move(touch).value();
  }

  // One frame of `panel` on `pixels`: its layout, then its chrome with
  // `items`.
  ui::PanelLayout frame(ui::DockedPanel& panel, VkExtent2D pixels,
                        const std::function<void()>& items = nullptr) {
    seconds_ += 1.0 / 60.0;
    touch_->new_frame({pixels, 2.0f, kSafe, seconds_});
    ImGui::NewFrame();
    const ui::PanelLayout layout = panel.layout(pixels);
    panel.draw_chrome(kTitle, items);
    ImGui::Render();
    return layout;
  }

  vg_test::HeadlessImGui imgui_{ImVec2(64.0f, 64.0f)};
  std::unique_ptr<ui::ImGuiTouch> touch_;
  double seconds_ = 0.0;
};

// With no content size the panel docks at the end of the long axis. A tall
// content shows larger docked right on a portrait iPad, by more than 25 %,
// so it docks there; cleared, the long axis again.
TEST_F(DockedPanelTest, TheSideFollowsTheContentElseTheLongAxis) {
  ui::DockedPanel panel(opened());
  EXPECT_EQ(frame(panel, kLandscape).side, ui::DockSide::Right);
  EXPECT_EQ(frame(panel, kPortrait).side, ui::DockSide::Bottom);

  const glm::vec2 tall{1024.0f, 2048.0f};
  panel.set_content_size({tall, tall});
  const ui::PanelLayout right = frame(panel, kPortrait);
  EXPECT_EQ(right.side, ui::DockSide::Right);
  EXPECT_EQ(right.scene.extent.height, kPortrait.height);
  EXPECT_GT(right.content.offset.x,
            static_cast<int32_t>(right.scene.extent.width));

  panel.set_content_size({});
  EXPECT_EQ(frame(panel, kPortrait).side, ui::DockSide::Bottom);
}

// The caller's items sit in the header's window on the title's line, before
// the close button at its right end; closed, they are not drawn.
TEST_F(DockedPanelTest, TheHeaderHoldsTheCallersItems) {
  ui::DockedPanel panel(opened());
  int drawn = 0;
  ImRect item;
  const char* window = nullptr;
  const auto items = [&]() {
    ++drawn;
    ImGui::Button("Reset");
    item = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    window = ImGui::GetCurrentWindow()->Name;
  };
  frame(panel, kLandscape, items);
  ASSERT_EQ(drawn, 1);
  EXPECT_EQ(std::string(window), std::string(kTitle) + "##header");
  const ImGuiWindow* header =
      ImGui::FindWindowByName((std::string(kTitle) + "##header").c_str());
  ASSERT_NE(header, nullptr);
  EXPECT_TRUE(header->Rect().Contains(item));
  const float title = ImGui::CalcTextSize(kTitle).x;
  EXPECT_GT(item.Min.x, header->Rect().Min.x + title);
  EXPECT_LT(item.Max.x,
            header->Rect().Max.x - ImGui::GetFrameHeight());  // the close's

  panel.set_open(false);
  frame(panel, kLandscape, items);
  EXPECT_EQ(drawn, 1);
}

}  // namespace

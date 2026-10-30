// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// DockedPanel on CPU-only ImGui frames: the side, by the content's size at
// each side and the default shares, else the long axis; and the header, the
// caller's items and the close button at its right end. Its shares, handle
// and close are also driven, with pictures in the content, by
// image_panel_test.

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>

#include <glm/vec2.hpp>

#include "imgui.h"
#include "imgui_internal.h"  // FindWindowByName, for the chrome's rects
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

ImRect window_rect(const char* suffix) {
  const ImGuiWindow* window =
      ImGui::FindWindowByName((std::string(kTitle) + suffix).c_str());
  EXPECT_NE(window, nullptr) << suffix;
  return window != nullptr ? window->Rect() : ImRect();
}

class DockedPanelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto touch = ui::ImGuiTouch::create(ImGui::GetCurrentContext());
    ASSERT_TRUE(touch.ok()) << touch.status().message();
    touch_ = std::move(touch).value();
  }

  // One frame of `panel` on `pixels`: its layout, then its chrome with
  // `items_`.
  ui::PanelLayout frame(ui::DockedPanel& panel, VkExtent2D pixels) {
    seconds_ += 1.0 / 60.0;
    touch_->new_frame({pixels, 2.0f, kSafe, seconds_});
    ImGui::NewFrame();
    const ui::PanelLayout layout = panel.layout(pixels);
    panel.draw_chrome(kTitle, [this]() {
      if (items_) {
        items_();
      }
    });
    ImGui::Render();
    return layout;
  }

  vg_test::HeadlessImGui imgui_{ImVec2(64.0f, 64.0f)};
  std::unique_ptr<ui::ImGuiTouch> touch_;
  double seconds_ = 0.0;
  std::function<void()> items_;  // the header's items; empty, none
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

// Each side's fit takes that side's content size. On a landscape iPad at
// 40 %, the right's content is 956 x 1530 pixels and the bottom's
// 2420 x 570. A content that is tall docked right and wide at the bottom
// fits 0.38 right and 0.57 at the bottom, so it docks at the bottom, against
// the long axis; swapped, or with the right's size at both, right. A small
// one right fits 1.53 there, so right; with the bottom's size at both, the
// bottom.
TEST_F(DockedPanelTest, EachSideFitsItsOwnContentSize) {
  const glm::vec2 tall{1000.0f, 4000.0f};
  const glm::vec2 wide{4000.0f, 1000.0f};
  const glm::vec2 small{500.0f, 1000.0f};
  struct Case {
    ui::DockedContentSize size;
    ui::DockSide side;
  };
  const auto R = ui::DockSide::Right;
  const auto B = ui::DockSide::Bottom;
  const Case cases[] = {
      {{tall, wide}, B},   // 0.57 at the bottom, 0.38 right
      {{wide, tall}, R},   // swapped
      {{tall, tall}, R},   // the right's size at both
      {{small, wide}, R},  // 1.53 right, 0.57 at the bottom
      {{wide, wide}, B},   // the bottom's size at both
  };
  ui::DockedPanel panel(opened());
  for (const Case& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "right " << c.size.right.x << " x " << c.size.right.y
                 << ", bottom " << c.size.bottom.x << " x " << c.size.bottom.y);
    panel.set_content_size(c.size);
    EXPECT_EQ(frame(panel, kLandscape).side, c.side);
  }
}

// The side is chosen at the default shares, not at one the handle has
// dragged to. Dragged to 60 % on a landscape iPad, the right's content is
// 1440 pixels wide, and a content 2500 x 500 docked right fits 0.576 there,
// within 25 % of the bottom's 0.57 (the long axis: right); at the default
// 40 % it fits 0.38, so the bottom wins.
TEST_F(DockedPanelTest, TheSideIsChosenAtTheDefaultShares) {
  ui::DockedPanel panel(opened());
  frame(panel, kLandscape);
  ASSERT_EQ(frame(panel, kLandscape).side, ui::DockSide::Right);
  // 242 points, 484 pixels: 20 % of the safe width.
  const ImVec2 handle = window_rect("##handle").GetCenter();
  vg_test::drag(handle, ImVec2(handle.x - 242.0f, handle.y),
                [&] { frame(panel, kLandscape); });
  ASSERT_NEAR(panel.state().right_share, 0.6f, 1e-3f);

  panel.set_content_size({{2500.0f, 500.0f}, {4000.0f, 1000.0f}});
  EXPECT_EQ(frame(panel, kLandscape).side, ui::DockSide::Bottom);
  EXPECT_NEAR(panel.state().right_share, 0.6f, 1e-3f);
}

// The caller's items sit in the header's window on the title's line, before
// the close button at its right end; closed, they are not drawn.
TEST_F(DockedPanelTest, TheHeaderHoldsTheCallersItems) {
  ui::DockedPanel panel(opened());
  int drawn = 0;
  ImRect item;
  const char* window = nullptr;
  items_ = [&]() {
    ++drawn;
    ImGui::Button("Reset");
    item = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    window = ImGui::GetCurrentWindow()->Name;
  };
  frame(panel, kLandscape);
  ASSERT_EQ(drawn, 1);
  EXPECT_EQ(std::string(window), std::string(kTitle) + "##header");
  const ImRect header = window_rect("##header");
  EXPECT_TRUE(header.Contains(item));
  const float title = ImGui::CalcTextSize(kTitle).x;
  EXPECT_GT(item.Min.x, header.Min.x + title);
  EXPECT_LT(item.Max.x, header.Max.x - ImGui::GetFrameHeight());  // the close

  panel.set_open(false);
  frame(panel, kLandscape);
  EXPECT_EQ(drawn, 1);
}

// The close button is at the header's right end, however far the items
// before it reach: with a left-aligned item and with none, a click there
// closes the panel.
TEST_F(DockedPanelTest, TheCloseButtonIsAtTheHeadersRightEnd) {
  for (const bool with_item : {true, false}) {
    SCOPED_TRACE(with_item ? "after an item" : "with no items");
    items_ = nullptr;
    if (with_item) {
      items_ = [] { ImGui::Button("Reset"); };
    }
    ui::DockedPanel panel(opened());
    frame(panel, kLandscape);
    frame(panel, kLandscape);
    const ImRect header = window_rect("##header");
    const float frame_height = ImGui::GetFrameHeight();
    const ImVec2 close(
        header.Max.x - ImGui::GetStyle().WindowPadding.x - 0.5f * frame_height,
        header.GetCenter().y);
    vg_test::click(close, [&] { frame(panel, kLandscape); });
    EXPECT_FALSE(panel.open());
  }
}

}  // namespace

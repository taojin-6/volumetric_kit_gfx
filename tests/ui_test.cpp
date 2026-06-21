// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImGuiOverlay: empty/default + argument validation, the move-only lifecycle
// (it owns an ImGui context + imgui_impl_vulkan backend), and an end-to-end
// new_frame -> build UI -> render into an offscreen target's dynamic-rendering
// scope. The GPU cases skip when no Vulkan device is present; the headless path
// exercises the real backend (font-atlas upload included) without a window.

#include <gtest/gtest.h>

#include <utility>

#include "imgui.h"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"
#include "vulkan_test_fixture.hpp"

namespace ui = volumetric_kit::gfx::ui;

namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

// --- Default / empty (no device needed) -------------------------------------

TEST(ImGuiOverlayTest, DefaultConstructedIsEmpty) {
  ui::ImGuiOverlay overlay;
  EXPECT_FALSE(overlay.valid());
  EXPECT_EQ(overlay.context(), nullptr);
}

class ImGuiOverlayDeviceTest : public VulkanDeviceTest {
 protected:
  static vg::RenderTargetLayout color_layout() {
    vg::RenderTargetLayout layout;
    layout.color_formats[0] = kFormat;
    layout.color_count = 1;
    return layout;
  }

  static ui::ImGuiOverlayConfig make_config(const vg::RenderTargetLayout& l) {
    ui::ImGuiOverlayConfig config;
    config.layout = l;
    config.min_image_count = 2;
    config.image_count = 2;
    return config;
  }

  vg::Allocator make_allocator() {
    auto allocator = vg::Allocator::create(instance_->handle(), *device_);
    EXPECT_TRUE(allocator.ok()) << allocator.status().message();
    return std::move(allocator).value();
  }

  vg::OffscreenTarget make_target(vg::Allocator& allocator,
                                  VkExtent2D extent = {64, 64}) {
    vg::OffscreenTargetDesc desc;
    desc.extent = extent;
    desc.color_format = kFormat;
    auto target = vg::OffscreenTarget::create(allocator, desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    return std::move(target).value();
  }

  ui::ImGuiOverlay make_overlay(const vg::RenderTargetLayout& layout) {
    auto overlay = ui::ImGuiOverlay::create(*device_, instance_->handle(),
                                            make_config(layout));
    EXPECT_TRUE(overlay.ok()) << overlay.status().message();
    return std::move(overlay).value();
  }
};

// --- Validation: rejected before any ImGui context is created ---------------

TEST_F(ImGuiOverlayDeviceTest, NullInstanceRejected) {
  auto overlay = ui::ImGuiOverlay::create(*device_, VK_NULL_HANDLE,
                                          make_config(color_layout()));
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(ImGuiOverlayDeviceTest, EmptyLayoutRejected) {
  // Default layout has color_count == 0 -- ImGui has no attachment to target.
  auto overlay =
      ui::ImGuiOverlay::create(*device_, instance_->handle(), make_config({}));
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(ImGuiOverlayDeviceTest, LowImageCountRejected) {
  ui::ImGuiOverlayConfig config = make_config(color_layout());
  config.min_image_count = 1;  // ImGui requires >= 2
  auto overlay =
      ui::ImGuiOverlay::create(*device_, instance_->handle(), config);
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(ImGuiOverlayDeviceTest, ImageCountBelowMinRejected) {
  ui::ImGuiOverlayConfig config = make_config(color_layout());
  config.min_image_count = 3;
  config.image_count = 2;  // < min_image_count
  auto overlay =
      ui::ImGuiOverlay::create(*device_, instance_->handle(), config);
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vg::Status::Code::InvalidArgument);
}

// --- Move-only lifecycle ----------------------------------------------------

TEST_F(ImGuiOverlayDeviceTest, MoveConstructLeavesSourceEmpty) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTarget target = make_target(allocator);
  ui::ImGuiOverlay source = make_overlay(target.layout());
  ASSERT_TRUE(source.valid());
  ImGuiContext* context = source.context();

  ui::ImGuiOverlay moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.context(), context);   // context transferred...
  EXPECT_FALSE(source.valid());          // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.context(), nullptr);  // ...and cleared in the source
}

TEST_F(ImGuiOverlayDeviceTest, MoveAssignOverLiveLeavesSourceEmpty) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTarget target = make_target(allocator);
  ui::ImGuiOverlay dst = make_overlay(target.layout());
  ui::ImGuiOverlay src = make_overlay(target.layout());
  ImGuiContext* src_context = src.context();

  dst = std::move(src);  // shuts down dst's backend + context, adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_EQ(dst.context(), src_context);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(ImGuiOverlayDeviceTest, SelfMoveAssignIsSafe) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTarget target = make_target(allocator);
  ui::ImGuiOverlay overlay = make_overlay(target.layout());
  ImGuiContext* context = overlay.context();

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the
  // self-move guard must keep the overlay (and its context) intact.
  ui::ImGuiOverlay* alias = &overlay;
  overlay = std::move(*alias);
  EXPECT_TRUE(overlay.valid());
  EXPECT_EQ(overlay.context(), context);
}

// --- End-to-end: render the UI into an offscreen target ---------------------

TEST_F(ImGuiOverlayDeviceTest, RendersIntoOffscreenTargetDynamicRendering) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTarget target = make_target(allocator);
  ui::ImGuiOverlay overlay = make_overlay(target.layout());

  // No platform backend here, so set DisplaySize ourselves (ImGui::NewFrame
  // requires it). create() left the overlay's context current.
  ImGui::SetCurrentContext(overlay.context());
  ImGui::GetIO().DisplaySize = ImVec2(64.0f, 64.0f);

  overlay.new_frame();
  ImGui::ShowDemoWindow();  // a non-trivial draw list exercising the font atlas

  auto pool = vg::CommandPool::create(device(), device_->graphics_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();
  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();

  // Dynamic rendering does not transition the image; move it to the attachment
  // layout before begin() (as the offscreen-target test does).
  VkImageMemoryBarrier to_color{};
  to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_color.srcAccessMask = 0;
  to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.image = target.color_image();
  to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(raw, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &to_color);

  vg::RenderTargetBeginInfo begin_info;
  begin_info.clear_color.float32[3] = 1.0f;  // opaque black
  const vg::RenderTarget rt = target.target();
  rt.begin(raw, begin_info);
  overlay.render(raw);  // records ImGui draws inside the rendering scope
  rt.end(raw);

  ASSERT_TRUE(cmd.value().end().ok());
  submit_and_wait(raw);  // validation-layer-clean completion is the assertion

  // Idle before the overlay (and its backend pipeline/pool) tears down at scope
  // exit, per ImGuiOverlay's teardown contract.
  vkDeviceWaitIdle(device());
}

}  // namespace

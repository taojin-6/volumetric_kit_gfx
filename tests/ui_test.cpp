// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImGuiOverlay: empty/default + argument validation, the move-only lifecycle
// (it owns an ImGui context + imgui_impl_vulkan backend), and an end-to-end
// new_frame -> build UI -> render into an offscreen target's dynamic-rendering
// scope. The GPU cases skip when no Vulkan device is present; the headless path
// exercises the real backend (font-atlas upload included) without a window.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "imgui.h"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/frame_metrics.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"
#include "volumetric_kit/gfx/ui/metrics_panel.hpp"
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

TEST_F(ImGuiOverlayDeviceTest, ZeroSamplesRejected) {
  ui::ImGuiOverlayConfig config = make_config(color_layout());
  config.layout.samples = static_cast<VkSampleCountFlagBits>(0);  // invalid
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
  ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini side-effect
  ImGui::GetIO().DisplaySize = ImVec2(64.0f, 64.0f);

  overlay.new_frame();
  ImGui::ShowDemoWindow();  // a non-trivial draw list exercising the font atlas
  // A deterministic opaque-white rect over the whole 64x64 viewport (on the
  // background draw list, behind the demo window) so the readback below can
  // assert ImGui actually rendered, not merely that recording was clean.
  ImGui::GetBackgroundDrawList()->AddRectFilled(
      ImVec2(0.0f, 0.0f), ImVec2(64.0f, 64.0f), IM_COL32_WHITE);

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
  target.record_readback(raw);  // copy the rendered image into the host buffer

  ASSERT_TRUE(cmd.value().end().ok());
  submit_and_wait(raw);

  // The white background rect fills the viewport, so the center texel must be
  // opaque white -- proof ImGui drew (validation-clean recording alone would
  // pass even if nothing was rasterized).
  const auto* px = static_cast<const uint8_t*>(target.pixels());
  ASSERT_NE(px, nullptr);
  constexpr size_t kCenter = (32 * 64 + 32) * 4;  // RGBA8, tightly packed
  EXPECT_EQ(px[kCenter + 0], 255);                // R
  EXPECT_EQ(px[kCenter + 1], 255);                // G
  EXPECT_EQ(px[kCenter + 2], 255);                // B
  EXPECT_EQ(px[kCenter + 3], 255);                // A

  // Idle before the overlay (and its backend pipeline/pool) tears down at scope
  // exit, per ImGuiOverlay's teardown contract.
  vkDeviceWaitIdle(device());
}

// --- Metrics panel: CPU-only widget building (no device) --------------------

// Builds a headless ImGui frame (no renderer backend) around `build`, returning
// the total vertex count of the resulting draw data — a proxy for "the panel
// produced geometry". The font atlas is built on the CPU so NewFrame's
// atlas-built assert holds without a backend.
template <class Build>
int panel_draw_vertices(Build&& build) {
  ImGuiContext* ctx = ImGui::CreateContext();
  ImGui::SetCurrentContext(ctx);
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;  // hermetic: no imgui.ini in the CWD
  io.DisplaySize = ImVec2(320.0f, 240.0f);
  io.DeltaTime = 1.0f / 60.0f;
  unsigned char* pixels = nullptr;
  int width = 0;
  int height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);  // forces atlas build

  // Two frames: a brand-new auto-sizing window is rendered hidden on its first
  // frame while ImGui measures its content, emitting geometry only once
  // settled.
  int vertices = 0;
  for (int frame = 0; frame < 2; ++frame) {
    ImGui::NewFrame();
    build();
    ImGui::Render();
    vertices = ImGui::GetDrawData()->TotalVtxCount;
  }
  ImGui::DestroyContext(ctx);
  return vertices;
}

vg::FrameMetrics sample_metrics() {
  vg::FrameMetrics metrics;
  metrics.fps = 90.0;
  metrics.cpu_frame_ms = 11.0;
  metrics.memory_used_bytes = 256u * 1024u * 1024u;
  metrics.memory_budget_bytes = 1024u * 1024u * 1024u;
  vg::FrameMetrics::Section gpu_stage;
  gpu_stage.name = "render";
  gpu_stage.cpu_ms = 0.8;
  gpu_stage.gpu_ms = 1.2;
  gpu_stage.has_gpu = true;
  metrics.sections.push_back(gpu_stage);
  vg::FrameMetrics::Section cpu_stage;
  cpu_stage.name = "cull";
  cpu_stage.cpu_ms = 0.3;
  metrics.sections.push_back(cpu_stage);
  return metrics;
}

// Geometry for a bare titled window with no body -- the title bar, border, and
// background ImGui emits for any window. The panel must exceed this to prove it
// drew its body and not merely a window frame; the default title matches the
// panel's so the decoration cancels out of the comparisons below.
int bare_window_vertices() {
  return panel_draw_vertices([] {
    ImGui::Begin("Performance");
    ImGui::End();
  });
}

// The populated panel (fps + memory lines + a per-stage table mixing a GPU and
// a CPU-only stage) emits strictly more geometry than the empty-metrics panel,
// which has neither the memory line nor the table. Both windows share identical
// decoration, so the surplus is body content -- a plain `> 0` would pass on the
// title bar alone.
TEST(MetricsPanelTest, ProducesGeometryForPopulatedMetrics) {
  const int empty =
      panel_draw_vertices([] { ui::draw_metrics_panel(vg::FrameMetrics{}); });
  const int populated =
      panel_draw_vertices([] { ui::draw_metrics_panel(sample_metrics()); });
  EXPECT_GT(populated, empty);
}

// Empty metrics (no stages, no budget) still draw a real body: the fps line and
// the "(no timed stages)" placeholder push the window past bare decoration --
// the empty-sections and no-budget branches.
TEST(MetricsPanelTest, HandlesEmptyMetrics) {
  const int empty =
      panel_draw_vertices([] { ui::draw_metrics_panel(vg::FrameMetrics{}); });
  EXPECT_GT(empty, bare_window_vertices());
}

}  // namespace

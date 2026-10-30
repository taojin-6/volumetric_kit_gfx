// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImGuiOverlay: empty/default + argument validation, the UI scale, the
// move-only lifecycle (it owns an ImGui context + imgui_impl_vulkan backend),
// and an end-to-end new_frame -> build UI -> render into an offscreen target's
// dynamic-rendering scope. The GPU cases skip when no Vulkan device is present;
// the headless path exercises the real backend (font-atlas upload included)
// without a window. The metrics panel and the stage table are read back as the
// text ImGui's log takes down, on a context with no device.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <glm/gtc/packing.hpp>

#include "color_test_support.hpp"
#include "gfx_test_support.hpp"
#include "imgui.h"
#include "imgui_internal.h"  // FindWindowByName, for a window's laid-out rect
#include "ui_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/gfx/core/frame_metrics.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/ui/collapsible_panel.hpp"
#include "volumetric_kit/gfx/ui/image_overlay.hpp"
#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"
#include "volumetric_kit/gfx/ui/metrics_panel.hpp"

namespace ui = volumetric_kit::gfx::ui;

namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

// --- Default / empty (no device needed) -------------------------------------

TEST(ImGuiOverlayTest, DefaultConstructedIsEmpty) {
  ui::ImGuiOverlay overlay;
  EXPECT_FALSE(overlay.valid());
  EXPECT_EQ(overlay.context(), nullptr);
}

class ImGuiOverlayDeviceTest : public vg_test::RendererDeviceTest {
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

  vg::OffscreenTarget make_target(
      VkExtent2D extent = {64, 64}, VkFormat format = kFormat,
      VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
    vg::OffscreenTargetDesc desc;
    desc.extent = extent;
    desc.color_format = format;
    desc.samples = samples;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    return std::move(target).value();
  }

  ui::ImGuiOverlay make_overlay(const vg::RenderTargetLayout& layout) {
    auto overlay = ui::ImGuiOverlay::create(device(), instance().handle(),
                                            make_config(layout));
    EXPECT_TRUE(overlay.ok()) << overlay.status().message();
    return std::move(overlay).value();
  }

  // Renders the frame built since overlay.new_frame() over opaque black into
  // `target` and reads it back to target.pixels().
  void render(const vg::OffscreenTarget& target, ui::ImGuiOverlay& overlay) {
    auto pool =
        vkc::CommandPool::create(device().handle(), device().queue_family());
    ASSERT_TRUE(pool.ok()) << pool.status().message();
    auto cmd = pool.value().allocate_primary();
    ASSERT_TRUE(cmd.ok()) << cmd.status().message();
    ASSERT_TRUE(
        cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    const VkCommandBuffer raw = cmd.value().handle();

    // Dynamic rendering does not transition the image; move it to the
    // attachment layout before begin() (as the offscreen-target test does).
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
    target.record_readback(raw);  // copy the rendered image to the host buffer

    ASSERT_TRUE(cmd.value().end().ok());
    const vkc::Status submitted = device().submit_and_wait(raw);
    ASSERT_TRUE(submitted.ok()) << submitted.message();
  }
};

// --- Validation: rejected before any ImGui context is created ---------------

// A device made to another library's requirements (the core's defaults: no
// dynamic rendering, which the backend's pipeline is built for) is refused.
using ImGuiOverlayForeignDeviceTest = vkc::test::VulkanDeviceTest;

TEST_F(ImGuiOverlayForeignDeviceTest, DeviceWithoutRendererFloorRejected) {
  ui::ImGuiOverlayConfig config;
  config.layout.color_formats[0] = kFormat;
  config.layout.color_count = 1;
  config.min_image_count = 2;
  config.image_count = 2;
  auto overlay =
      ui::ImGuiOverlay::create(device(), instance().handle(), config);
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vkc::Status::Code::Unsupported)
      << overlay.status().message();
}

TEST_F(ImGuiOverlayDeviceTest, NullInstanceRejected) {
  auto overlay = ui::ImGuiOverlay::create(device(), VK_NULL_HANDLE,
                                          make_config(color_layout()));
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(ImGuiOverlayDeviceTest, EmptyLayoutRejected) {
  // Default layout has color_count == 0 -- ImGui has no attachment to target.
  auto overlay =
      ui::ImGuiOverlay::create(device(), instance().handle(), make_config({}));
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(ImGuiOverlayDeviceTest, LowImageCountRejected) {
  ui::ImGuiOverlayConfig config = make_config(color_layout());
  config.min_image_count = 1;  // ImGui requires >= 2
  auto overlay =
      ui::ImGuiOverlay::create(device(), instance().handle(), config);
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(ImGuiOverlayDeviceTest, ImageCountBelowMinRejected) {
  ui::ImGuiOverlayConfig config = make_config(color_layout());
  config.min_image_count = 3;
  config.image_count = 2;  // < min_image_count
  auto overlay =
      ui::ImGuiOverlay::create(device(), instance().handle(), config);
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(ImGuiOverlayDeviceTest, ZeroSamplesRejected) {
  ui::ImGuiOverlayConfig config = make_config(color_layout());
  config.layout.samples = static_cast<VkSampleCountFlagBits>(0);  // invalid
  auto overlay =
      ui::ImGuiOverlay::create(device(), instance().handle(), config);
  ASSERT_FALSE(overlay.ok());
  EXPECT_EQ(overlay.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(ImGuiOverlayDeviceTest, ScaleNotFiniteAndPositiveRejected) {
  for (const float scale :
       {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity()}) {
    SCOPED_TRACE(::testing::Message() << "scale " << scale);
    ui::ImGuiOverlayConfig config = make_config(color_layout());
    config.scale = scale;
    auto overlay =
        ui::ImGuiOverlay::create(device(), instance().handle(), config);
    ASSERT_FALSE(overlay.ok());
    EXPECT_EQ(overlay.status().domain(), vkc::Status::Code::InvalidArgument);
  }
}

TEST_F(ImGuiOverlayDeviceTest, MinFrameHeightNotFiniteAndNonNegativeRejected) {
  for (const float height : {-1.0f, std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity()}) {
    SCOPED_TRACE(::testing::Message() << "min_frame_height " << height);
    ui::ImGuiOverlayConfig config = make_config(color_layout());
    config.min_frame_height = height;
    auto overlay =
        ui::ImGuiOverlay::create(device(), instance().handle(), config);
    ASSERT_FALSE(overlay.ok());
    EXPECT_EQ(overlay.status().domain(), vkc::Status::Code::InvalidArgument);
  }
}

// --- Scale ------------------------------------------------------------------

// The scale multiplies the style's sizes and the font ImGui draws with, from
// what an overlay at scale 1 has.
TEST_F(ImGuiOverlayDeviceTest, ScaleSizesTheStyleAndTheFont) {
  vg::OffscreenTarget target = make_target();
  struct Sizes {
    ImGuiStyle style;
    float font = 0.0f;
  };
  const auto sizes_at = [&](float scale) {
    ui::ImGuiOverlayConfig config = make_config(target.layout());
    config.scale = scale;
    auto overlay =
        ui::ImGuiOverlay::create(device(), instance().handle(), config);
    EXPECT_TRUE(overlay.ok()) << overlay.status().message();
    Sizes sizes;
    if (!overlay.ok()) return sizes;
    ImGui::GetIO().DisplaySize = ImVec2(64.0f, 64.0f);
    overlay.value().new_frame();
    sizes.style = ImGui::GetStyle();
    sizes.font = ImGui::GetFontSize();
    render(target, overlay.value());
    vkDeviceWaitIdle(device().handle());
    return sizes;
  };
  const Sizes one = sizes_at(1.0f);
  const Sizes two = sizes_at(2.0f);
  ASSERT_GT(one.font, 0.0f);
  EXPECT_FLOAT_EQ(two.font, 2.0f * one.font);
  const auto expect_doubled = [](ImVec2 at_one, ImVec2 at_two) {
    EXPECT_FLOAT_EQ(at_two.x, 2.0f * at_one.x);
    EXPECT_FLOAT_EQ(at_two.y, 2.0f * at_one.y);
  };
  expect_doubled(one.style.WindowPadding, two.style.WindowPadding);
  expect_doubled(one.style.FramePadding, two.style.FramePadding);
  expect_doubled(one.style.ItemSpacing, two.style.ItemSpacing);
  ASSERT_GT(one.style.ScrollbarRounding, 0.0f);
  EXPECT_FLOAT_EQ(two.style.ScrollbarRounding,
                  2.0f * one.style.ScrollbarRounding);
  ASSERT_GT(one.style.TabRounding, 0.0f);
  EXPECT_FLOAT_EQ(two.style.TabRounding, 2.0f * one.style.TabRounding);
  // Scale 1 is ImGui's default style.
  const ImGuiStyle defaults;
  expect_doubled(defaults.WindowPadding, two.style.WindowPadding);
}

// A phone's sizes: the least frame height raises the frames, a collapsible
// panel's header among them, to the whole unit that reaches it, and keeps the
// text the scale gives. One the frames already reach, or 0, keeps the style
// the scale gives.
TEST_F(ImGuiOverlayDeviceTest, MinFrameHeightRaisesTheFramesNotTheText) {
  constexpr float kScale = 1.3f;
  constexpr float kMinFrameHeight = 44.0f;
  vg::OffscreenTarget target = make_target({256, 256});
  struct Sizes {
    ImVec2 frame_padding;
    float font = 0.0f;
    float frame = 0.0f;
    float header = 0.0f;
  };
  const auto sizes_at = [&](float min_frame_height) {
    ui::ImGuiOverlayConfig config = make_config(target.layout());
    config.scale = kScale;
    config.min_frame_height = min_frame_height;
    auto overlay =
        ui::ImGuiOverlay::create(device(), instance().handle(), config);
    EXPECT_TRUE(overlay.ok()) << overlay.status().message();
    Sizes sizes;
    if (!overlay.ok()) return sizes;
    ImGui::GetIO().DisplaySize = ImVec2(256.0f, 256.0f);
    // Two frames: the header's window takes its size on the second.
    bool open = true;
    for (int frame = 0; frame < 2; ++frame) {
      overlay.value().new_frame();
      sizes.frame_padding = ImGui::GetStyle().FramePadding;
      sizes.font = ImGui::GetFontSize();
      sizes.frame = ImGui::GetFrameHeight();
      if (ui::begin_collapsible_panel("##panel", "Stats", ImVec2(0.0f, 0.0f),
                                      &open)) {
        ImGui::TextUnformatted("bitrate");
        ui::end_collapsible_panel();
      }
      render(target, overlay.value());
    }
    const ImGuiWindow* header = ImGui::FindWindowByName("##panel");
    EXPECT_NE(header, nullptr);
    if (header != nullptr) sizes.header = header->Size.y;
    vkDeviceWaitIdle(device().handle());
    return sizes;
  };
  const Sizes kept = sizes_at(0.0f);
  const Sizes raised = sizes_at(kMinFrameHeight);
  const Sizes reached = sizes_at(kept.frame);

  ImGuiStyle scaled;
  scaled.ScaleAllSizes(kScale);
  EXPECT_EQ(kept.frame_padding.x, scaled.FramePadding.x);
  EXPECT_EQ(kept.frame_padding.y, scaled.FramePadding.y);
  EXPECT_EQ(reached.frame_padding.y, scaled.FramePadding.y);
  ASSERT_LT(kept.frame, kMinFrameHeight);

  EXPECT_GE(raised.frame, kMinFrameHeight);
  EXPECT_LT(raised.frame, kMinFrameHeight + 2.0f);
  EXPECT_EQ(raised.frame_padding.x, kept.frame_padding.x);
  EXPECT_EQ(raised.font, kept.font);
  EXPECT_GE(raised.header, raised.frame);
  EXPECT_GE(kept.header, kept.frame);
}

// --- Move-only lifecycle ----------------------------------------------------

TEST_F(ImGuiOverlayDeviceTest, MoveConstructLeavesSourceEmpty) {
  vg::OffscreenTarget target = make_target();
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
  vg::OffscreenTarget target = make_target();
  ui::ImGuiOverlay dst = make_overlay(target.layout());
  ui::ImGuiOverlay src = make_overlay(target.layout());
  ImGuiContext* src_context = src.context();

  dst = std::move(src);  // shuts down dst's backend + context, adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_EQ(dst.context(), src_context);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(ImGuiOverlayDeviceTest, SelfMoveAssignIsSafe) {
  vg::OffscreenTarget target = make_target();
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
  vg::OffscreenTarget target = make_target();
  ui::ImGuiOverlay overlay = make_overlay(target.layout());

  // No platform backend here, so set DisplaySize ourselves (ImGui::NewFrame
  // requires it). create() left the overlay's context current.
  ImGui::SetCurrentContext(overlay.context());
  ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini side-effect
  ImGui::GetIO().DisplaySize = ImVec2(64.0f, 64.0f);

  overlay.new_frame();
  ImGui::ShowDemoWindow();  // a non-trivial draw list exercising the font atlas
  // Put the following rect beyond the 16-bit vertex range. Its readback also
  // checks that the Vulkan backend binds and reads the 32-bit index buffer.
  std::vector<glm::vec2> points(33000);
  for (size_t i = 0; i < points.size(); ++i) {
    points[i] = {static_cast<float>(i % 32), static_cast<float>((i / 32) % 32)};
  }
  const ui::ImageOverlay marks(
      vg::camera::ImageView2D({64.0f, 64.0f}, {0.0f, 0.0f}, {64.0f, 64.0f}),
      ui::PixelCenters::Half);
  marks.polyline(points.data(), points.size(), IM_COL32(255, 0, 0, 255));
  ASSERT_GT(ImGui::GetBackgroundDrawList()->VtxBuffer.Size, 65536);
  // A deterministic opaque-white rect over the whole 64x64 viewport (on the
  // background draw list, behind the demo window) so the readback below can
  // assert ImGui actually rendered, not merely that recording was clean.
  ImGui::GetBackgroundDrawList()->AddRectFilled(
      ImVec2(0.0f, 0.0f), ImVec2(64.0f, 64.0f), IM_COL32_WHITE);
  ASSERT_NO_FATAL_FAILURE(render(target, overlay));

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
  vkDeviceWaitIdle(device().handle());
}

// The overlay draws into a multisampled target, validation-clean: whole
// pixels as at one sample, and a rectangle's edge through the middle of a
// pixel blends.
TEST_F(ImGuiOverlayDeviceTest, RendersIntoAMultisampledTarget) {
  vg::OffscreenTarget target =
      make_target({64, 64}, kFormat, vg::kWindowSamples);
  ASSERT_EQ(target.layout().samples, vg::kWindowSamples);
  ui::ImGuiOverlay overlay = make_overlay(target.layout());
  ImGui::GetIO().DisplaySize = ImVec2(64.0f, 64.0f);
  overlay.new_frame();
  ImDrawList* draw = ImGui::GetBackgroundDrawList();
  draw->AddRectFilled(ImVec2(0.0f, 0.0f), ImVec2(64.0f, 64.0f), IM_COL32_WHITE);
  draw->AddRectFilled(ImVec2(10.5f, 0.0f), ImVec2(64.0f, 64.0f),
                      IM_COL32(255, 0, 0, 255));
  ASSERT_NO_FATAL_FAILURE(render(target, overlay));

  const auto* px = static_cast<const uint8_t*>(target.pixels());
  ASSERT_NE(px, nullptr);
  const auto green = [&](uint32_t x) {
    return px[(size_t{32} * 64 + x) * 4 + 1];
  };
  EXPECT_EQ(green(5), 255) << "white";
  EXPECT_EQ(green(32), 0) << "red";
  EXPECT_GT(green(10), 0) << "the edge pixel blends";
  EXPECT_LT(green(10), 255) << "the edge pixel blends";
  vkDeviceWaitIdle(device().handle());
}

// ImGui's colours show as authored on every target: a UNORM target stores and
// blends their bytes as given; on an sRGB or float target the overlay decodes
// them first, so the target blends and stores their linear light, which an
// sRGB one encodes back. The right half is drawn translucent over black.
TEST_F(ImGuiOverlayDeviceTest, ColoursShowAsAuthoredOnEveryTarget) {
  constexpr uint8_t kColour[3] = {200, 100, 50};
  constexpr uint8_t kAlpha = 128;
  struct Case {
    VkFormat format;
    double scale;      // a stored value per unit: 255 for a byte, 1 for a half
    double tolerance;  // in stored values
  };
  // What `format` stores, in [0, 1], for colour byte `byte` at `alpha`.
  const auto expected = [](VkFormat format, uint8_t byte, double alpha) {
    const double light = vg_test::srgb_to_linear(byte / 255.0) * alpha;
    switch (format) {
      case VK_FORMAT_R8G8B8A8_UNORM:
        return byte / 255.0 * alpha;
      case VK_FORMAT_R8G8B8A8_SRGB:
        return vg_test::linear_to_srgb(light);
      default:
        return light;
    }
  };
  for (const Case c : {Case{VK_FORMAT_R8G8B8A8_UNORM, 255.0, 1.0},
                       Case{VK_FORMAT_R8G8B8A8_SRGB, 255.0, 1.0},
                       Case{VK_FORMAT_R16G16B16A16_SFLOAT, 1.0, 2e-3}}) {
    SCOPED_TRACE(::testing::Message() << "format " << c.format);
    vg::OffscreenTarget target = make_target({64, 64}, c.format);
    ui::ImGuiOverlay overlay = make_overlay(target.layout());
    ImGui::GetIO().DisplaySize = ImVec2(64.0f, 64.0f);
    overlay.new_frame();
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    draw->AddRectFilled(ImVec2(0.0f, 0.0f), ImVec2(32.0f, 64.0f),
                        IM_COL32(kColour[0], kColour[1], kColour[2], 255));
    draw->AddRectFilled(ImVec2(32.0f, 0.0f), ImVec2(64.0f, 64.0f),
                        IM_COL32(kColour[0], kColour[1], kColour[2], kAlpha));
    ASSERT_NO_FATAL_FAILURE(render(target, overlay));

    const auto* px = static_cast<const uint8_t*>(target.pixels());
    ASSERT_NE(px, nullptr);
    // Channel `ch` of texel (x, 32), as stored: a byte or a half.
    const auto stored = [&](uint32_t x, int ch) -> double {
      const size_t texel = size_t{32} * 64 + x;
      if (c.format != VK_FORMAT_R16G16B16A16_SFLOAT) {
        return px[texel * 4 + ch];
      }
      uint16_t half = 0;
      std::memcpy(&half, px + texel * 8 + ch * 2, sizeof(half));
      return glm::unpackHalf1x16(half);
    };
    struct Half {
      uint32_t x;
      double alpha;
    };
    for (const Half h : {Half{16, 1.0}, Half{48, kAlpha / 255.0}}) {
      SCOPED_TRACE(::testing::Message() << "alpha " << h.alpha);
      for (int ch = 0; ch < 3; ++ch) {
        EXPECT_NEAR(stored(h.x, ch),
                    c.scale * expected(c.format, kColour[ch], h.alpha),
                    c.tolerance);
      }
      EXPECT_NEAR(stored(h.x, 3), c.scale, c.tolerance);
    }
    vkDeviceWaitIdle(device().handle());
  }
}

// A region's grid, rendered: at any sub-pixel phase, with or without
// anti-aliasing, each outer border is drawn as strongly as an inner line --
// its whole width lies inside the region -- and nothing lands outside the
// region's pixels, those whose centers lie in it.
TEST_F(ImGuiOverlayDeviceTest, RegionGridKeepsEveryBorder) {
  constexpr int kSize = 256;
  vg::OffscreenTarget target = make_target({kSize, kSize});
  ui::ImGuiOverlay overlay = make_overlay(target.layout());
  ImGui::SetCurrentContext(overlay.context());
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::GetIO().DisplaySize = ImVec2(kSize, kSize);
  overlay.new_frame();

  struct Style {
    ImDrawListFlags flags;
    float thickness;
  };
  constexpr ImDrawListFlags kAntiAliased = ImDrawListFlags_AntiAliasedLines;
  const Style styles[] = {
      {0, 1.0f},
      {0, 2.0f},
      {kAntiAliased, 1.0f},
      {kAntiAliased | ImDrawListFlags_AntiAliasedLinesUseTex, 1.0f},
  };
  const float phases[] = {0.0f, 0.3f, 0.5f, 0.7f};
  // 1 target pixel per texel. A 32 x 32 region per style (row) and phase
  // (column), 64 texels apart, with a 2 x 2 grid: inner lines through its
  // middle.
  const vg::camera::ImageView2D view({kSize, kSize}, {0.0f, 0.0f},
                                     {kSize, kSize});
  const auto region = [&](int row, int column) {
    const float phase = phases[column];
    return ui::OverlayRegion{
        {64.0f * static_cast<float>(column) + 16.0f + phase,
         64.0f * static_cast<float>(row) + 16.0f + phase},
        {32.0f, 32.0f}};
  };
  ImDrawList& list = *ImGui::GetBackgroundDrawList();
  const ImDrawListFlags flags = list.Flags;
  for (int row = 0; row < 4; ++row) {
    list.Flags =
        (flags & ~(kAntiAliased | ImDrawListFlags_AntiAliasedLinesUseTex)) |
        styles[row].flags;
    for (int column = 0; column < 4; ++column) {
      ui::ImageOverlay(view, ui::PixelCenters::Half, region(row, column))
          .grid(2, 2, nullptr, 0, IM_COL32_WHITE, styles[row].thickness);
    }
  }
  list.Flags = flags;
  ASSERT_NO_FATAL_FAILURE(render(target, overlay));

  const auto* px = static_cast<const uint8_t*>(target.pixels());
  ASSERT_NE(px, nullptr);
  // The red channel at `i` along the scan line `at`, across x or down y, and
  // its sum over [from, to].
  const auto red = [&](bool across, int at, int i) {
    const size_t x = static_cast<size_t>(across ? i : at);
    const size_t y = static_cast<size_t>(across ? at : i);
    return static_cast<int>(px[(y * kSize + x) * 4]);
  };
  const auto sum = [&](bool across, int at, int from, int to) {
    int total = 0;
    for (int i = from; i <= to; ++i) total += red(across, at, i);
    return total;
  };
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      SCOPED_TRACE(::testing::Message()
                   << "style " << row << ", phase " << phases[column]);
      const ui::OverlayRegion r = region(row, column);
      for (const bool across : {true, false}) {
        // The region's 32 pixels on this axis, from the first whose center
        // is in it, and a scan line between its inner lines.
        const float origin = across ? r.origin.x : r.origin.y;
        const int first = static_cast<int>(std::ceil(origin - 0.5f));
        const int last = first + 31;
        const int at = static_cast<int>(std::ceil(
                           (across ? r.origin.y : r.origin.x) - 0.5f)) +
                       8;
        const int inner = static_cast<int>(origin + 16.0f);
        const int line = sum(across, at, inner - 2, inner + 2);
        EXPECT_GE(4 * sum(across, at, first, first + 2), 3 * line);
        EXPECT_GE(4 * sum(across, at, last - 2, last), 3 * line);
        EXPECT_EQ(red(across, at, first - 1), 0);
        EXPECT_EQ(red(across, at, last + 1), 0);
      }
    }
  }
  vkDeviceWaitIdle(device().handle());
}

// Intersecting the overlay with a draw-list clip or the framebuffer must keep
// the last row and column, including after panning the whole-image overload.
// An oversized mark also exposes any pixels leaking beyond the intended clip.
TEST_F(ImGuiOverlayDeviceTest, ImageMarksKeepPixelsAtClipEdges) {
  constexpr int kSize = 256;
  const ImVec2 scales[] = {
      {1.0f, 1.0f}, {1.1f, 1.1f}, {1.75f, 1.75f}, {1.1f, 1.75f}, {1.75f, 1.1f}};
  for (const ImVec2 scale : scales) {
    for (const bool whole_image : {false, true}) {
      for (const bool nested_clip : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "scale " << scale.x << ", " << scale.y
                     << ", whole image " << whole_image << ", nested clip "
                     << nested_clip);
        vg::OffscreenTarget target = make_target({kSize, kSize});
        ui::ImGuiOverlay overlay = make_overlay(target.layout());
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(kSize / scale.x, kSize / scale.y);
        io.DisplayFramebufferScale = scale;
        overlay.new_frame();
        ImDrawList& list = *ImGui::GetBackgroundDrawList();
        list.Flags &= ~ImDrawListFlags_AntiAliasedFill;

        // These origins round upward when converted back to target pixels
        // at 1.1 and 1.75 respectively. Clamping the far edge used to lose
        // a pixel when the backend truncated the width or height.
        const glm::vec2 origin{scale.x == 1.75f ? 227.0f : 147.0f,
                               scale.y == 1.75f ? 227.0f : 147.0f};
        vg::camera::ImageView2D view({kSize, kSize}, {0.0f, 0.0f},
                                     {kSize, kSize});
        if (whole_image) view.pan(origin);
        const ui::ImageOverlay marks =
            whole_image
                ? ui::ImageOverlay(view, ui::PixelCenters::Half)
                : ui::ImageOverlay(
                      view, ui::PixelCenters::Half,
                      ui::OverlayRegion{origin, glm::vec2(kSize) - origin});
        // Change the caller's clip after constructing the overlay: it is
        // the final intersection at each mark that matters.
        // 240 projects back exactly at both fractional scales, so this
        // isolates subtracting the nudged start from a clipped integer end.
        const int end = nested_clip ? 240 : kSize;
        if (nested_clip) {
          list.PushClipRect(ImVec2(0.0f, 0.0f),
                            ImVec2(end / scale.x, end / scale.y), true);
        }
        marks.point({0.0f, 0.0f}, IM_COL32_WHITE, 512.0f);
        if (nested_clip) list.PopClipRect();
        ASSERT_NO_FATAL_FAILURE(render(target, overlay));

        const auto* px = static_cast<const uint8_t*>(target.pixels());
        ASSERT_NE(px, nullptr);
        int missing = 0;
        int outside = 0;
        for (int y = 0; y < kSize; ++y) {
          for (int x = 0; x < kSize; ++x) {
            const bool inside =
                x >= origin.x && x < end && y >= origin.y && y < end;
            const uint8_t red = px[(y * kSize + x) * 4];
            missing += inside && red != 255;
            outside += !inside && red != 0;
          }
        }
        EXPECT_EQ(missing, 0);
        EXPECT_EQ(outside, 0);
        vkDeviceWaitIdle(device().handle());
      }
    }
  }
}

// render() may submit on the device's queue itself, to update a texture (the
// font atlas, on the first frame), so then it holds the device's submit mutex
// -- the one every other submit on that queue holds -- whether or not the
// queue is shared. A frame that draws no new glyph updates no texture: it only
// records, and takes no lock.
TEST_F(ImGuiOverlayDeviceTest, RenderHoldsTheSubmitMutexOnlyToUpdateATexture) {
  vg::OffscreenTarget target = make_target();
  ui::ImGuiOverlay overlay = make_overlay(target.layout());
  ImGui::SetCurrentContext(overlay.context());
  ImGui::GetIO().DisplaySize = ImVec2(64.0f, 64.0f);

  auto pool =
      vkc::CommandPool::create(device().handle(), device().queue_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  const vg::RenderTarget rt = target.target();
  for (const bool uploads : {true, false}) {
    SCOPED_TRACE(uploads ? "atlas upload" : "no texture update");
    overlay.new_frame();
    // The same text each frame, in a window sized up front so ImGui draws it
    // on the first: the second frame adds no glyph to the atlas.
    ImGui::SetNextWindowSize(ImVec2(60.0f, 40.0f));
    ImGui::Begin("w");
    ImGui::TextUnformatted("glyphs");
    ImGui::End();
    if (!uploads) {
      for (const ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
        ASSERT_EQ(tex->Status, ImTextureStatus_OK);
      }
    }

    auto cmd = pool.value().allocate_primary();
    ASSERT_TRUE(cmd.ok()) << cmd.status().message();
    ASSERT_TRUE(
        cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    const VkCommandBuffer raw = cmd.value().handle();
    rt.begin(raw, vg::RenderTargetBeginInfo{});

    std::unique_lock<std::mutex> held(*device().submit_mutex());
    std::future<void> rendered = std::async(
        std::launch::async, [&overlay, raw] { overlay.render(raw); });
    // An upload waits for as long as the mutex is held; a frame without one
    // finishes under it.
    if (uploads) {
      EXPECT_EQ(rendered.wait_for(std::chrono::milliseconds(100)),
                std::future_status::timeout);
    } else {
      EXPECT_EQ(rendered.wait_for(std::chrono::seconds(10)),
                std::future_status::ready);
    }
    held.unlock();
    rendered.get();

    rt.end(raw);
    ASSERT_TRUE(cmd.value().end().ok());
  }
  vkDeviceWaitIdle(
      device().handle());  // the backend's own upload, before teardown
}

// --- Metrics panel: CPU-only widget building (no device) --------------------

// What a headless ImGui frame (no renderer backend) around `build` drew: the
// total vertex count of its draw data -- a proxy for "the panel produced
// geometry" -- and the windows it rendered.
struct Drawn {
  int vertices = 0;
  int windows = 0;
};

template <class Build>
Drawn panel_draw(Build&& build) {
  const vg_test::HeadlessImGui imgui(ImVec2(320.0f, 240.0f));

  // Two frames: a brand-new auto-sizing window is rendered hidden on its first
  // frame while ImGui measures its content, emitting geometry only once
  // settled.
  Drawn drawn;
  for (int frame = 0; frame < 2; ++frame) {
    ImGui::NewFrame();
    build();
    ImGui::Render();
    drawn.vertices = ImGui::GetDrawData()->TotalVtxCount;
    drawn.windows = ImGui::GetIO().MetricsRenderWindows;
  }
  return drawn;
}

vg::FrameMetrics sample_metrics() {
  vg::FrameMetrics metrics;
  metrics.fps = 90.0;
  metrics.cpu_frame_ms = 11.0;
  metrics.memory_used_bytes = 256u * 1024u * 1024u;
  metrics.memory_budget_bytes = 1024u * 1024u * 1024u;
  vkc::StageRow gpu_stage;
  gpu_stage.name = "render";
  gpu_stage.cpu_ms = 0.8;
  gpu_stage.gpu_ms = 1.2;
  gpu_stage.has_gpu = true;
  metrics.sections.push_back(gpu_stage);
  vkc::StageRow cpu_stage;
  cpu_stage.name = "cull";
  cpu_stage.cpu_ms = 0.3;
  metrics.sections.push_back(cpu_stage);
  return metrics;
}

// A bare titled window with no body -- the title bar, border, and background
// ImGui emits for any window. The panel must exceed its geometry to prove it
// drew its body and not merely a window frame; the default title matches the
// panel's so the decoration cancels out of the comparisons below.
Drawn bare_window() {
  return panel_draw([] {
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
  const Drawn empty =
      panel_draw([] { ui::draw_metrics_panel(vg::FrameMetrics{}); });
  const Drawn populated =
      panel_draw([] { ui::draw_metrics_panel(sample_metrics()); });
  EXPECT_GT(populated.vertices, empty.vertices);
}

// Empty metrics (no stages, no budget) still draw a real body: the fps line and
// the "(no timed stages)" placeholder push the window past bare decoration --
// the empty-sections and no-budget branches.
TEST(MetricsPanelTest, HandlesEmptyMetrics) {
  const Drawn empty =
      panel_draw([] { ui::draw_metrics_panel(vg::FrameMetrics{}); });
  EXPECT_GT(empty.vertices, bare_window().vertices);
}

// The stage table draws into the caller's window and opens none: the frame
// renders the bare window alone, with more geometry, with rows or without.
TEST(StageTableTest, DrawsIntoTheCallersWindow) {
  const Drawn bare = bare_window();
  ASSERT_EQ(bare.windows, 1);
  for (const std::vector<vkc::StageRow>& rows :
       {sample_metrics().sections, std::vector<vkc::StageRow>{}}) {
    SCOPED_TRACE(::testing::Message() << rows.size() << " rows");
    const Drawn table = panel_draw([&] {
      ImGui::Begin("Performance");
      ui::draw_stage_table("stages", rows);
      ImGui::End();
    });
    EXPECT_EQ(table.windows, bare.windows);
    EXPECT_GT(table.vertices, bare.vertices);
  }
}

// --- Stage table: the text each cell draws (no device) ----------------------

// The text a headless ImGui frame around `build` drew, as ImGui's log takes it
// down: a table's cells each open with a `|`, and its rows close with one.
// Logging starts in the frame's implicit window, so it spans every window
// `build` opens; two frames, as in panel_draw.
template <class Build>
std::string panel_text(Build&& build) {
  const vg_test::HeadlessImGui imgui(ImVec2(800.0f, 600.0f));
  std::string text;
  for (int frame = 0; frame < 2; ++frame) {
    ImGui::NewFrame();
    ImGui::LogToBuffer();
    build();
    text = ImGui::GetCurrentContext()->LogBuffer.c_str();
    ImGui::LogFinish();
    ImGui::Render();
  }
  return text;
}

std::string table_text(const std::vector<vkc::StageRow>& rows,
                       const ui::StageWindow& window = {}) {
  return panel_text([&] {
    ImGui::Begin("Performance");
    ui::draw_stage_table("stages", rows, window);
    ImGui::End();
  });
}

using Cells = std::vector<std::vector<std::string>>;

// The logged table's rows, its header first, as trimmed cell text: each row is
// `columns` cells between `|`s, and the text up to the next row's first `|`.
Cells table_cells(const std::string& text, std::size_t columns) {
  std::vector<std::string> fields;
  for (std::size_t at = text.find('|'); at != std::string::npos;) {
    const std::size_t next = text.find('|', at + 1);
    std::string field =
        text.substr(at + 1, next == std::string::npos ? next : next - at - 1);
    const std::size_t first = field.find_first_not_of(" \n");
    const std::size_t last = field.find_last_not_of(" \n");
    fields.push_back(first == std::string::npos
                         ? std::string()
                         : field.substr(first, last - first + 1));
    at = next;
  }
  Cells rows;
  for (std::size_t i = 0; i + columns < fields.size(); i += columns + 1) {
    rows.emplace_back(
        fields.begin() + static_cast<std::ptrdiff_t>(i),
        fields.begin() + static_cast<std::ptrdiff_t>(i + columns));
  }
  return rows;
}

vkc::StageRow stage(const char* name, double cpu_ms, double wait_ms,
                    vkc::DeviceTime device, double gpu_ms = 0.0) {
  vkc::StageRow row;
  row.name = name;
  row.cpu_ms = cpu_ms;
  row.wait_ms = wait_ms;
  row.device = device;
  row.gpu_ms = gpu_ms;
  row.has_gpu = device == vkc::DeviceTime::Measured;
  return row;
}

// Each device state has a cell of its own: a time, `n/t`, `n/a`, or blank for
// a host-only row, never a meaningless 0.000. No row waited, so there is no
// wait column.
TEST(StageTableTest, EachDeviceStateHasItsOwnCell) {
  const std::string text =
      table_text({stage("render", 1.0, 0.0, vkc::DeviceTime::Measured, 1.2),
                  stage("mesh", 0.5, 0.0, vkc::DeviceTime::NotTimed),
                  stage("decode", 0.25, 0.0, vkc::DeviceTime::Unavailable),
                  stage("cull", 0.3, 0.0, vkc::DeviceTime::None)});
  EXPECT_EQ(table_cells(text, 3), (Cells{{"Stage", "host ms", "GPU ms"},
                                         {"render", "1.000", "1.200"},
                                         {"mesh", "0.500", "n/t"},
                                         {"decode", "0.250", "n/a"},
                                         {"cull", "0.300", ""}}))
      << text;
}

// Host is busy time, never a wait: a row's wait has a column of its own, which
// comes with the first row that waited, and a row that only waited has no
// host time.
TEST(StageTableTest, HostIsBusyTimeAndAWaitHasItsOwnColumn) {
  const std::string text =
      table_text({stage("read", 2.0, 0.5, vkc::DeviceTime::None),
                  stage("wait: GPU", 1.5, 1.5, vkc::DeviceTime::None),
                  stage("draw", 0.75, 0.0, vkc::DeviceTime::Measured, 0.5)});
  EXPECT_EQ(table_cells(text, 4),
            (Cells{{"Stage", "host ms", "wait ms", "GPU ms"},
                   {"read", "1.500", "0.500", ""},
                   {"wait: GPU", "", "1.500", ""},
                   {"draw", "0.750", "", "0.500"}}))
      << text;
}

// A row built positionally, before `device` existed, says measured through
// has_gpu alone.
TEST(StageTableTest, ARowSettingOnlyHasGpuReadsMeasured) {
  const std::string text = table_text({vkc::StageRow{"pass", 0.25, 0.5, true}});
  EXPECT_EQ(table_cells(text, 3),
            (Cells{{"Stage", "host ms", "GPU ms"}, {"pass", "0.250", "0.500"}}))
      << text;
}

// The line above the table states what the figures cover, its unit "frame"
// unless named, then the application's note; nothing by default.
TEST(StageTableTest, TheStatusLineStatesTheWindow) {
  const std::vector<vkc::StageRow> rows = {
      stage("decode", 1.0, 0.0, vkc::DeviceTime::Measured, 2.0)};
  ui::StageWindow window;
  window.frames = 150;
  window.timed = 120;
  window.unit = "stream frame";
  EXPECT_NE(table_text(rows, window)
                .find("per stream frame · last 150 · GPU timed in 120"),
            std::string::npos);
  window.unit = nullptr;
  EXPECT_NE(
      table_text(rows, window).find("per frame · last 150 · GPU timed in 120"),
      std::string::npos);
  window.note = "clocks calibrated ±2 µs";
  EXPECT_NE(table_text(rows, window)
                .find("per frame · last 150 · GPU timed in 120 · clocks "
                      "calibrated ±2 µs"),
            std::string::npos);
  // A note alone is the whole line.
  ui::StageWindow off;
  off.note = "GPU timing off";
  const std::string alone = table_text(rows, off);
  EXPECT_NE(alone.find("GPU timing off"), std::string::npos) << alone;
  EXPECT_EQ(alone.find("GPU timed"), std::string::npos) << alone;
  EXPECT_EQ(alone.find(" · GPU timing off"), std::string::npos) << alone;
  EXPECT_EQ(table_text(rows).find("GPU timed"), std::string::npos);
}

// The panel states the rate, the period it gives and the recording time apart,
// and one frame's GPU timing where a stage has a device half.
TEST(MetricsPanelTest, StatesTheFrameAndWhetherItsGpuWasTimed) {
  const auto text_of = [](const vg::FrameMetrics& metrics) {
    return panel_text([&] { ui::draw_metrics_panel(metrics); });
  };
  const std::string text = text_of(sample_metrics());
  EXPECT_NE(text.find("90.0 fps · 11.11 ms a frame · 11.00 ms recording"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("per frame · last 1 · GPU timed in 1"), std::string::npos)
      << text;

  vg::FrameMetrics untimed = sample_metrics();
  untimed.sections[0].has_gpu = false;
  untimed.sections[0].device = vkc::DeviceTime::NotTimed;
  EXPECT_NE(text_of(untimed).find("per frame · last 1 · GPU timed in 0"),
            std::string::npos);

  vg::FrameMetrics host_only;
  host_only.sections.push_back(stage("cull", 0.3, 0.0, vkc::DeviceTime::None));
  EXPECT_EQ(text_of(host_only).find("GPU timed"), std::string::npos);
}

// --- Collapsible panel: CPU-only widget building (no device) ----------------

// Frames of a collapsible panel on a CPU-only context, with the mouse.
class CollapsiblePanelTest : public ::testing::Test {
 protected:
  static constexpr const char* kId = "##panel";
  static constexpr ImVec2 kPos{10.0f, 20.0f};

  // One frame of the panel, with two lines of body when it is open. Returns
  // what begin returned, and keeps the frame height, the font and title sizes
  // and, when begin returned true, the body's window and rect. Begin leaves
  // the body's window open when it returns true, and none otherwise.
  bool frame() {
    ImGui::NewFrame();
    frame_height_ = ImGui::GetFrameHeight();
    font_size_ = ImGui::GetFontSize();
    title_width_ = ImGui::CalcTextSize(title_).x;
    const ImVector<ImGuiWindowStackData>& stack =
        ImGui::GetCurrentContext()->CurrentWindowStack;
    const int depth = stack.Size;
    const bool drawn = ui::begin_collapsible_panel(kId, title_, kPos, &open_);
    EXPECT_EQ(stack.Size, depth + (drawn ? 1 : 0));
    if (drawn) {
      body_ = ImGui::GetCurrentWindow();
      body_min_ = ImGui::GetWindowPos();
      body_size_ = ImGui::GetWindowSize();
      ImGui::TextUnformatted("bitrate 11.8 Mbit/s");
      ImGui::TextUnformatted("rate 29.8 / 30 fps");
      ui::end_collapsible_panel();
    }
    ImGui::Render();
    return drawn;
  }

  // The header's window, as ImGui last laid it out and drew it.
  static ImGuiWindow* header_window() {
    ImGuiWindow* window = ImGui::FindWindowByName(kId);
    EXPECT_NE(window, nullptr);
    return window;
  }

  static ImRect header() {
    const ImGuiWindow* window = header_window();
    return window != nullptr ? window->Rect() : ImRect();
  }

  // The positions of the header's vertices, x then y.
  static std::vector<float> header_vertices() {
    std::vector<float> xy;
    if (const ImGuiWindow* window = header_window()) {
      for (const ImDrawVert& vertex : window->DrawList->VtxBuffer) {
        xy.push_back(vertex.pos.x);
        xy.push_back(vertex.pos.y);
      }
    }
    return xy;
  }

  // The mouse moved to `at`, then pressed and released, a frame each: ImGui
  // takes them on separate frames anyway.
  void click(ImVec2 at) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(at.x, at.y);
    frame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    frame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    frame();
  }

  vg_test::HeadlessImGui imgui_{ImVec2(320.0f, 240.0f)};
  const char* title_ = "Stats";
  bool open_ = true;
  float frame_height_ = 0.0f;
  float font_size_ = 0.0f;
  float title_width_ = 0.0f;
  ImGuiWindow* body_ = nullptr;
  ImVec2 body_min_;
  ImVec2 body_size_;
};

TEST_F(CollapsiblePanelTest, AClickOnTheHeaderClosesItAndAnotherOpensIt) {
  frame();
  frame();
  const ImVec2 middle = header().GetCenter();
  click(middle);
  EXPECT_FALSE(open_);
  click(middle);
  EXPECT_TRUE(open_);
}

TEST_F(CollapsiblePanelTest, ClosedItReturnsFalseAndDrawsTheHeaderAlone) {
  open_ = false;
  EXPECT_FALSE(frame());
  EXPECT_FALSE(frame());
  EXPECT_EQ(ImGui::GetIO().MetricsRenderWindows, 1);
  open_ = true;
  EXPECT_TRUE(frame());
  EXPECT_TRUE(frame());
  EXPECT_EQ(ImGui::GetIO().MetricsRenderWindows, 2);
}

// The header takes the mouse; over the body it is the app's, as over nothing.
TEST_F(CollapsiblePanelTest, OnlyTheHeaderTakesTheMouse) {
  frame();
  frame();
  ImGuiIO& io = ImGui::GetIO();
  const auto capture_at = [&](ImVec2 at) {
    io.AddMousePosEvent(at.x, at.y);
    frame();
    frame();
    return io.WantCaptureMouse;
  };
  EXPECT_TRUE(capture_at(header().GetCenter()));
  EXPECT_FALSE(capture_at(ImVec2(body_min_.x + 0.5f * body_size_.x,
                                 body_min_.y + 0.5f * body_size_.y)));
  EXPECT_FALSE(capture_at(ImVec2(300.0f, 220.0f)));
}

// The body sits directly below the header from the first frame, before ImGui
// has measured the header's window, and stays there.
TEST_F(CollapsiblePanelTest, TheBodyStartsAtTheHeadersBottomFromTheFirstFrame) {
  ASSERT_TRUE(frame());
  const ImVec2 first = body_min_;
  ASSERT_TRUE(frame());
  const ImRect settled = header();
  EXPECT_EQ(settled.Min.x, kPos.x);
  EXPECT_EQ(settled.Min.y, kPos.y);
  EXPECT_EQ(first.x, settled.Min.x);
  EXPECT_EQ(first.y, settled.Max.y);
  EXPECT_EQ(body_min_.y, first.y);
}

// The header is at least a frame tall, as ImGui's style sets it or as
// ImGuiOverlayConfig::min_frame_height raises it.
TEST_F(CollapsiblePanelTest, TheHeaderIsAtLeastAFrameTall) {
  for (const float padding : {ImGui::GetStyle().FramePadding.y, 14.0f}) {
    SCOPED_TRACE(::testing::Message() << "FramePadding.y " << padding);
    ImGui::GetStyle().FramePadding.y = padding;
    frame();
    frame();
    EXPECT_GE(header().GetHeight(), frame_height_);
  }
}

// The header is one button wide enough for the arrow and the title, both
// drawn, and the arrow turns with the state.
TEST_F(CollapsiblePanelTest, TheHeaderHoldsTheArrowAndTheTitle) {
  frame();
  frame();
  EXPECT_GE(header().GetWidth(), font_size_ + title_width_ +
                                     2.0f * ImGui::GetStyle().FramePadding.x);
  const std::vector<float> open = header_vertices();
  open_ = false;
  frame();
  frame();
  const std::vector<float> closed = header_vertices();
  EXPECT_EQ(closed.size(), open.size());
  EXPECT_NE(closed, open);
  title_ = "";
  frame();
  frame();
  EXPECT_LT(header_vertices().size(), closed.size());
}

// Both windows draw a 60 % background: the first vertex of each.
TEST_F(CollapsiblePanelTest, BothWindowsHaveA60PercentBackground) {
  frame();
  frame();
  ASSERT_NE(body_, nullptr);
  for (const ImGuiWindow* window : {header_window(), body_}) {
    ASSERT_NE(window, nullptr);
    ASSERT_FALSE(window->DrawList->VtxBuffer.empty());
    EXPECT_EQ(window->DrawList->VtxBuffer[0].col >> IM_COL32_A_SHIFT,
              static_cast<ImU32>(IM_F32_TO_INT8_SAT(0.6f)));
  }
}

// When ImGui skips a window, as ConfigDebugBeginReturnValueOnce has it skip
// each new one, begin returns false with every window ended (frame checks),
// and the panel draws from the next frame, its body below its header.
TEST_F(CollapsiblePanelTest, WhenImGuiSkipsItBeginReturnsFalse) {
  ImGui::GetIO().ConfigDebugBeginReturnValueOnce = true;
  EXPECT_FALSE(frame());
  EXPECT_TRUE(open_);
  EXPECT_TRUE(frame());
  EXPECT_TRUE(frame());
  EXPECT_EQ(body_min_.x, header().Min.x);
  EXPECT_EQ(body_min_.y, header().Max.y);
}

}  // namespace

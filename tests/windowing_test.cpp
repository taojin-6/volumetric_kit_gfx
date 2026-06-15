// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Windowing tier on a headless surface (VK_EXT_headless_surface): swapchain
// creation, the FrameLoop acquire -> render -> present choreography, recreate,
// and the move-only lifecycle. The whole suite skips when the runner has no
// headless surface (e.g. MoltenVK) or no present-capable device, so it provides
// real coverage on Linux CI (lavapipe) without needing a display.

#include <gtest/gtest.h>

#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace vg = volumetric_kit::gfx;
namespace win = volumetric_kit::gfx::windowing;

namespace {

bool instance_has_headless_surface() {
  uint32_t count = 0;
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) !=
      VK_SUCCESS) {
    return false;
  }
  std::vector<VkExtensionProperties> props(count);
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, props.data()) !=
      VK_SUCCESS) {
    return false;
  }
  for (const VkExtensionProperties& p : props) {
    if (std::strcmp(p.extensionName, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) ==
        0) {
      return true;
    }
  }
  return false;
}

class WindowingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!instance_has_headless_surface()) {
      GTEST_SKIP() << "VK_EXT_headless_surface unavailable (e.g. MoltenVK)";
    }
    vg::InstanceConfig icfg;
    icfg.enable_validation = true;
    icfg.extra_instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                                      VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
    auto instance = vg::Instance::create(icfg);
    if (!instance.ok()) {
      GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
    }
    instance_.emplace(std::move(instance).value());

    auto surface = win::Surface::headless(instance_->handle());
    if (!surface.ok()) {
      GTEST_SKIP() << "headless surface: " << surface.status().message();
    }
    surface_ = std::move(surface).value();

    auto physical = instance_->select_physical_device(surface_.handle());
    if (!physical.ok()) {
      GTEST_SKIP() << "no present-capable device: "
                   << physical.status().message();
    }

    vg::DeviceConfig dcfg;
    dcfg.needs_present = true;
    auto device = vg::Device::create(instance_->handle(), physical.value(),
                                     dcfg, surface_.handle());
    ASSERT_TRUE(device.ok()) << device.status().message();
    device_.emplace(std::move(device).value());
  }

  win::Swapchain make_swapchain(VkExtent2D extent = {256, 256}) {
    win::SwapchainConfig cfg;
    cfg.extent = extent;
    auto sc = win::Swapchain::create(*device_, surface_.handle(), cfg);
    EXPECT_TRUE(sc.ok()) << sc.status().message();
    return std::move(sc).value();
  }

  // Drive `count` clear-only frames through the loop. A fixed headless extent
  // never goes out of date, so any non-OK status is a real failure.
  vg::Status run_frames(win::FrameLoop& loop, int count) {
    for (int i = 0; i < count; ++i) {
      auto frame = loop.begin_frame();
      if (!frame.ok()) {
        return frame.status();
      }
      vg::RenderTargetBeginInfo begin;
      begin.clear_color.float32[0] = 0.1f;
      begin.clear_color.float32[3] = 1.0f;
      frame.value().target->begin(frame.value().cmd, begin);
      frame.value().target->end(frame.value().cmd);
      vg::Status end = loop.end_frame(frame.value());
      if (!end.ok()) {
        return end;
      }
    }
    return {};
  }

  std::optional<vg::Instance> instance_;
  win::Surface surface_;
  std::optional<vg::Device> device_;
};

TEST_F(WindowingTest, CreatesSwapchainWithRenderTargets) {
  win::Swapchain sc = make_swapchain();
  EXPECT_TRUE(sc.valid());
  EXPECT_GE(sc.image_count(), 1u);
  EXPECT_NE(sc.format(), VK_FORMAT_UNDEFINED);
  const vg::RenderTargetLayout layout = sc.layout();
  EXPECT_EQ(layout.color_count, 1u);
  EXPECT_EQ(layout.color_formats[0], sc.format());
  EXPECT_TRUE(sc.render_target(0).valid());
}

TEST_F(WindowingTest, FrameLoopRendersAndPresents) {
  win::Swapchain sc = make_swapchain();
  auto loop = win::FrameLoop::create(*device_, sc, /*frames_in_flight=*/2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();
  EXPECT_EQ(loop.value().frames_in_flight(), 2u);

  const vg::Status status = run_frames(loop.value(), /*count=*/8);
  EXPECT_TRUE(status.ok()) << status.message();

  vkDeviceWaitIdle(device_->handle());
}

TEST_F(WindowingTest, RecreateKeepsFormatAndLayout) {
  win::Swapchain sc = make_swapchain({256, 256});
  const VkFormat format = sc.format();
  const vg::RenderTargetLayout before = sc.layout();

  ASSERT_TRUE(sc.recreate({320, 240}).ok());
  EXPECT_EQ(sc.format(), format);
  EXPECT_TRUE(sc.layout().compatible_with(before));

  auto loop = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();
  EXPECT_TRUE(run_frames(loop.value(), 4).ok());
  vkDeviceWaitIdle(device_->handle());
}

TEST_F(WindowingTest, SwapchainMoveLeavesSourceEmpty) {
  win::Swapchain src = make_swapchain();
  ASSERT_TRUE(src.valid());

  win::Swapchain moved(std::move(src));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(WindowingTest, FrameLoopMoveLeavesSourceEmpty) {
  win::Swapchain sc = make_swapchain();
  auto created = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(created.ok()) << created.status().message();

  win::FrameLoop moved(std::move(created).value());
  EXPECT_TRUE(moved.valid());
  // The moved-to loop owns the resources and drives frames; confirm it works.
  EXPECT_TRUE(run_frames(moved, 2).ok());
  vkDeviceWaitIdle(device_->handle());
}

}  // namespace

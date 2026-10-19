// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// windowing::metal_surface over a bare CAMetalLayer, as an iOS view backs
// itself with one: the surface is made, its current extent is the layer's
// bounds times its contentsScale, and a swapchain is built on it. The layer is
// never on screen: there is no NSApplication or window, and no image is
// acquired or presented. Apple only.

#import <QuartzCore/CAMetalLayer.h>

#include <gtest/gtest.h>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/surface_source.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace win = volumetric_kit::gfx::windowing;

namespace {

using MetalSurfaceTest = vg_test::RendererTest;

TEST_F(MetalSurfaceTest, MakesASurfaceSizedByTheLayer) {
  // 160 x 90 points at 2 pixels per point. The test sets contentsScale, as a
  // view does, and never drawableSize, which the swapchain writes.
  CAMetalLayer* layer = [CAMetalLayer layer];
  layer.bounds = CGRectMake(0, 0, 160, 90);
  layer.contentsScale = 2.0;
  const VkExtent2D pixels{320, 180};

  const vkc::Result<win::SurfaceSource> source =
      win::metal_surface((__bridge const void*)layer);
  ASSERT_TRUE(source.ok()) << source.status().message();
  auto instance = vkc::Instance::create(vg_test::surface_instance_config(
      source.value().instance_extensions, active_validation()));
  ASSERT_TRUE(instance.ok()) << instance.status().message();
  // MoltenVK refuses only a null layer, and asks nothing of the window server,
  // so a failure here is the factory's.
  const vkc::Result<VkSurfaceKHR> made =
      source.value().make_surface(instance.value().handle());
  ASSERT_TRUE(made.ok()) << made.status().message();
  const win::Surface surface(instance.value().handle(), made.value());

  vkc::DeviceRequirements requirements = vg::device_requirements();
  requirements.needs_present = true;
  auto physical =
      instance.value().select_physical_device(requirements, surface.handle());
  ASSERT_TRUE(physical.ok()) << physical.status().message();
  VkSurfaceCapabilitiesKHR capabilities{};
  ASSERT_EQ(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                physical.value().handle(), surface.handle(), &capabilities),
            VK_SUCCESS);
  EXPECT_EQ(capabilities.currentExtent.width, pixels.width);
  EXPECT_EQ(capabilities.currentExtent.height, pixels.height);

  auto device = vkc::Device::create(instance.value(), physical.value(),
                                    requirements, surface.handle());
  ASSERT_TRUE(device.ok()) << device.status().message();
  win::SwapchainConfig config;
  config.extent = pixels;
  auto swapchain =
      win::Swapchain::create(device.value(), surface.handle(), config);
  ASSERT_TRUE(swapchain.ok()) << swapchain.status().message();
  EXPECT_EQ(swapchain.value().extent().width, pixels.width);
  EXPECT_EQ(swapchain.value().extent().height, pixels.height);
}

}  // namespace

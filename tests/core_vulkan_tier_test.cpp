// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// gfx's headers compile beside the core's vulkan tier, included first: an
// application that also uses the core's compute tier sees both, as recon will
// once it adopts that tier. That order used to fail twice over: naming
// core::to_string in gfx's namespace took the tier's to_string(VkResult), which
// clashed with gfx's, and swapchain_stale's unqualified vk_result(status) also
// found the tier's by argument-dependent lookup on the Status.

#include <gtest/gtest.h>

#include <optional>

#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

TEST(CoreVulkanTier, GfxBridgeWorksBesideIt) {
  const vg::Status stale = vg::vk_error(VK_ERROR_OUT_OF_DATE_KHR, "present");
  EXPECT_TRUE(vg::windowing::swapchain_stale(stale));
  EXPECT_EQ(vg::vk_result(stale), VK_ERROR_OUT_OF_DATE_KHR);
  // The tier reads gfx's status the same way: they share the Status type.
  EXPECT_EQ(vkc::vk_result(stale), VK_ERROR_OUT_OF_DATE_KHR);
  EXPECT_EQ(vg::to_string(VK_ERROR_DEVICE_LOST), "VK_ERROR_DEVICE_LOST");
  EXPECT_EQ(to_string(stale.domain()), "Backend");
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/vk_query.hpp
/// Internal helpers for swapchain.cpp: the Vulkan "enumerate (count, then
/// fill)" idiom for a surface's formats and present modes, each returning a
/// `core::Result` so a real failure surfaces, and whether a device's limits
/// take a window's sample count. The instance and physical-device queries
/// live in the core's vulkan tier. Not a public header.

#include <cstdint>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"

namespace volumetric_kit::gfx {

/// @brief Whether a device renders @ref kWindowSamples into a framebuffer:
///        its colour and, with a depth attachment, its depth sample counts.
///
/// Vulkan requires the count in both limits; a portability driver may report
/// less.
/// @param limits  The device's limits.
/// @param depth   Whether the framebuffer has a depth attachment.
/// @return `true` when every attachment the framebuffer has takes the count.
inline bool window_samples_supported(const VkPhysicalDeviceLimits& limits,
                                     bool depth) noexcept {
  return (limits.framebufferColorSampleCounts & kWindowSamples) != 0 &&
         (!depth ||
          (limits.framebufferDepthSampleCounts & kWindowSamples) != 0);
}

/// @return The surface formats @p device offers on @p surface (empty if none),
///         or a non-OK `core::Status` if the query fails. `VK_INCOMPLETE` on
///         the fill call is a success code (the list changed size between the
///         count and fill — a display reconfiguration): the entries actually
///         written are trusted, not treated as a failure.
inline core::Result<std::vector<VkSurfaceFormatKHR>> surface_formats(
    VkPhysicalDevice device, VkSurfaceKHR surface) {
  uint32_t count = 0;
  VKC_VK_TRY(
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr));
  std::vector<VkSurfaceFormatKHR> formats(count);
  if (count == 0) {
    return formats;  // caller reports "unsupported"; skip the empty fill call
  }
  const VkResult filled = vkGetPhysicalDeviceSurfaceFormatsKHR(
      device, surface, &count, formats.data());
  if (filled != VK_SUCCESS && filled != VK_INCOMPLETE) {
    return core::vk_error(filled, "vkGetPhysicalDeviceSurfaceFormatsKHR");
  }
  formats.resize(count);
  return formats;
}

/// @return The present modes @p device offers on @p surface (empty if none), or
///         a non-OK `core::Status` if the query fails. Tolerates
///         `VK_INCOMPLETE` on the fill call exactly as @ref surface_formats
///         does.
inline core::Result<std::vector<VkPresentModeKHR>> surface_present_modes(
    VkPhysicalDevice device, VkSurfaceKHR surface) {
  uint32_t count = 0;
  VKC_VK_TRY(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count,
                                                       nullptr));
  std::vector<VkPresentModeKHR> modes(count);
  if (count == 0) {
    return modes;
  }
  const VkResult filled = vkGetPhysicalDeviceSurfacePresentModesKHR(
      device, surface, &count, modes.data());
  if (filled != VK_SUCCESS && filled != VK_INCOMPLETE) {
    return core::vk_error(filled, "vkGetPhysicalDeviceSurfacePresentModesKHR");
  }
  modes.resize(count);
  return modes;
}

}  // namespace volumetric_kit::gfx

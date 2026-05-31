// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/vk_query.hpp
/// Internal helpers shared by instance.cpp and device.cpp: the Vulkan
/// "enumerate (count, then fill)" idiom and the physical-device queue-family /
/// extension queries. Each enumerator checks the `VkResult` it would otherwise
/// drop, and the queue-family lookups return one consistent
/// `std::optional<uint32_t>` shape. Not a public header.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @return Whether @p name appears in @p available (exact match).
inline bool has_extension(const std::vector<VkExtensionProperties>& available,
                          const char* name) {
  return std::any_of(available.begin(), available.end(),
                     [&](const VkExtensionProperties& e) {
                       return std::strcmp(e.extensionName, name) == 0;
                     });
}

/// @return The available instance extensions, or empty if enumeration fails.
inline std::vector<VkExtensionProperties> instance_extensions() {
  uint32_t count = 0;
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) !=
      VK_SUCCESS) {
    return {};
  }
  std::vector<VkExtensionProperties> exts(count);
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data()) !=
      VK_SUCCESS) {
    exts.resize(count);  // VK_INCOMPLETE: trust only what was written
  }
  return exts;
}

/// @return The available instance layers, or empty if enumeration fails.
inline std::vector<VkLayerProperties> instance_layers() {
  uint32_t count = 0;
  if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) {
    return {};
  }
  std::vector<VkLayerProperties> layers(count);
  if (vkEnumerateInstanceLayerProperties(&count, layers.data()) != VK_SUCCESS) {
    layers.resize(count);
  }
  return layers;
}

/// @return The device extensions for @p device, or empty if enumeration fails.
inline std::vector<VkExtensionProperties> device_extensions(
    VkPhysicalDevice device) {
  uint32_t count = 0;
  if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) !=
      VK_SUCCESS) {
    return {};
  }
  std::vector<VkExtensionProperties> exts(count);
  if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                           exts.data()) != VK_SUCCESS) {
    exts.resize(count);
  }
  return exts;
}

/// @return The physical devices for @p instance, or empty if none / on failure.
inline std::vector<VkPhysicalDevice> physical_devices(VkInstance instance) {
  uint32_t count = 0;
  if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) {
    return {};
  }
  std::vector<VkPhysicalDevice> devices(count);
  if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) !=
      VK_SUCCESS) {
    devices.resize(count);
  }
  return devices;
}

/// @return The first graphics-capable queue family index, or `std::nullopt`.
inline std::optional<uint32_t> find_graphics_family(VkPhysicalDevice device) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
  for (uint32_t i = 0; i < count; ++i) {
    if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      return i;
    }
  }
  return std::nullopt;
}

/// @return The first queue family that can present to @p surface, or
///         `std::nullopt`. A failed support query is treated as "this family
///         cannot present" rather than trusting an unwritten result.
inline std::optional<uint32_t> find_present_family(VkPhysicalDevice device,
                                                   VkSurfaceKHR surface) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  for (uint32_t i = 0; i < count; ++i) {
    VkBool32 supported = VK_FALSE;
    if (vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &supported) ==
            VK_SUCCESS &&
        supported == VK_TRUE) {
      return i;
    }
  }
  return std::nullopt;
}

/// @return Whether @p format on @p device supports every bit in @p flags for
/// the
///         requested @p tiling: `VK_IMAGE_TILING_OPTIMAL` checks the optimal-
///         tiling features, otherwise the linear-tiling features.
inline bool format_supports(VkPhysicalDevice device, VkFormat format,
                            VkImageTiling tiling, VkFormatFeatureFlags flags) {
  VkFormatProperties props{};
  vkGetPhysicalDeviceFormatProperties(device, format, &props);
  const VkFormatFeatureFlags supported = tiling == VK_IMAGE_TILING_OPTIMAL
                                             ? props.optimalTilingFeatures
                                             : props.linearTilingFeatures;
  return (supported & flags) == flags;
}

}  // namespace volumetric_kit::gfx

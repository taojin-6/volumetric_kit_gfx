// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/vk_query.hpp
/// Internal helpers shared by instance.cpp / device.cpp / swapchain.cpp: the
/// Vulkan "enumerate (count, then fill)" idiom and the physical-device
/// queue-family / extension / surface queries. Each enumerator checks the
/// `VkResult` it would otherwise drop, and the queue-family lookups each return
/// their find as a `std::optional` (a `GraphicsFamily` for graphics, a bare
/// index for present). The surface enumerators return a @ref Result so a real
/// failure surfaces (unlike the instance/device ones, whose callers treat a
/// failed enumeration as "feature absent"). Not a public header.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "volumetric_kit/gfx/core/result.hpp"
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

/// @return The surface formats @p device offers on @p surface (empty if none),
///         or a non-OK @ref Status if the query fails. `VK_INCOMPLETE` on the
///         fill call is a success code (the list changed size between the count
///         and fill — a display reconfiguration): the entries actually written
///         are trusted, not treated as a failure.
inline Result<std::vector<VkSurfaceFormatKHR>> surface_formats(
    VkPhysicalDevice device, VkSurfaceKHR surface) {
  uint32_t count = 0;
  VG_VK_TRY(
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr));
  std::vector<VkSurfaceFormatKHR> formats(count);
  if (count == 0) {
    return formats;  // caller reports "unsupported"; skip the empty fill call
  }
  const VkResult filled = vkGetPhysicalDeviceSurfaceFormatsKHR(
      device, surface, &count, formats.data());
  if (filled != VK_SUCCESS && filled != VK_INCOMPLETE) {
    return vk_error(filled, "vkGetPhysicalDeviceSurfaceFormatsKHR");
  }
  formats.resize(count);
  return formats;
}

/// @return The present modes @p device offers on @p surface (empty if none), or
///         a non-OK @ref Status if the query fails. Tolerates `VK_INCOMPLETE`
///         on the fill call exactly as @ref surface_formats does.
inline Result<std::vector<VkPresentModeKHR>> surface_present_modes(
    VkPhysicalDevice device, VkSurfaceKHR surface) {
  uint32_t count = 0;
  VG_VK_TRY(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count,
                                                      nullptr));
  std::vector<VkPresentModeKHR> modes(count);
  if (count == 0) {
    return modes;
  }
  const VkResult filled = vkGetPhysicalDeviceSurfacePresentModesKHR(
      device, surface, &count, modes.data());
  if (filled != VK_SUCCESS && filled != VK_INCOMPLETE) {
    return vk_error(filled, "vkGetPhysicalDeviceSurfacePresentModesKHR");
  }
  modes.resize(count);
  return modes;
}

/// The chosen graphics queue family: its index and the `timestampValidBits`
/// the driver reports for it (0 == that queue cannot write timestamps, which
/// MoltenVK may report).
struct GraphicsFamily {
  uint32_t index;
  uint32_t timestamp_valid_bits;
};

/// @return The first graphics-capable queue family (its index and
///         `timestampValidBits`), or `std::nullopt`. The single
///         `vkGetPhysicalDeviceQueueFamilyProperties` fill already carries the
///         family's timestamp support back, so no extra driver round-trip is
///         needed.
inline std::optional<GraphicsFamily> find_graphics_family(
    VkPhysicalDevice device) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
  for (uint32_t i = 0; i < count; ++i) {
    if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      return GraphicsFamily{i, families[i].timestampValidBits};
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

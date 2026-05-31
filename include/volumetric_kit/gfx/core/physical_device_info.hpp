// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file physical_device_info.hpp
/// @brief Read-only capability and format queries for a chosen physical device.

#include <string>
#include <vector>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Cached, read-only capabilities of one `VkPhysicalDevice`.
///
/// Supported device extensions, the core `VkPhysicalDeviceFeatures2`, and the
/// device properties/limits are captured once by @ref query. Format queries hit
/// the driver live (they are cheap, and there are hundreds of formats). Use
/// this before @ref Device::create to pre-check the extensions and features you
/// intend to request, and for swapchain / volume / interop format selection.
///
/// A default-constructed instance is empty (`handle()` is `VK_NULL_HANDLE`); it
/// is the moved-from state of @ref Device::caps and supports no real queries.
///
/// @warning The `VkPhysicalDevice` passed to @ref query must outlive every
///          format query (those re-enter the driver with the stored handle).
///
/// @code
/// PhysicalDeviceInfo caps = instance.query_physical_device(physical);
/// if (caps.format_supports(VK_FORMAT_R16G16B16A16_SFLOAT,
///                          VK_IMAGE_TILING_OPTIMAL,
///                          VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
///   // ... safe to allocate an HDR color target in this format ...
/// }
/// @endcode
class VG_CORE_API PhysicalDeviceInfo {
 public:
  /// @brief Capture the capabilities of @p physical.
  /// @param physical  The physical device to inspect (belonging to a >= 1.1
  ///                  instance, so `vkGetPhysicalDeviceFeatures2` is
  ///                  available).
  /// @return The captured info.
  static PhysicalDeviceInfo query(VkPhysicalDevice physical);

  /// @brief Construct an empty info (owns nothing; `handle()` is null).
  PhysicalDeviceInfo() = default;

  /// @return The inspected physical device (`VK_NULL_HANDLE` when empty).
  VkPhysicalDevice handle() const noexcept { return physical_; }

  /// @return Whether @p name is in the device's supported extension list.
  bool supports_device_extension(const char* name) const;

  /// @return The cached core (1.0) feature set.
  const VkPhysicalDeviceFeatures& features() const noexcept {
    return features2_.features;
  }
  /// @return The cached `VkPhysicalDeviceFeatures2`. Only the core 1.0 set is
  ///         captured (its `pNext` is null); chain version/extension feature
  ///         structs yourself to query those.
  const VkPhysicalDeviceFeatures2& features2() const noexcept {
    return features2_;
  }
  /// @return The cached device properties (apiVersion, deviceType, limits,
  /// ...).
  const VkPhysicalDeviceProperties& properties() const noexcept {
    return properties_;
  }
  /// @return The cached device limits (shorthand for `properties().limits`).
  const VkPhysicalDeviceLimits& limits() const noexcept {
    return properties_.limits;
  }

  /// @brief The buffer / linear-tiling / optimal-tiling feature flags for a
  ///        format (a live `vkGetPhysicalDeviceFormatProperties` query).
  /// @param format  The format to inspect.
  VkFormatProperties format_properties(VkFormat format) const;

  /// @brief Whether @p format supports every bit in @p features for @p tiling.
  /// @param format    The format to test.
  /// @param tiling    `VK_IMAGE_TILING_OPTIMAL` checks `optimalTilingFeatures`;
  ///                  `VK_IMAGE_TILING_LINEAR` checks `linearTilingFeatures`.
  /// @param features  The required feature bits (all must be present).
  /// @return `true` if every requested bit is supported.
  bool format_supports(VkFormat format, VkImageTiling tiling,
                       VkFormatFeatureFlags features) const;

 private:
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  // Extension names are stored as owned strings (not VkExtensionProperties) so
  // this type stays trivially copyable/movable and self-contained.
  std::vector<std::string> extension_names_;
  VkPhysicalDeviceFeatures2 features2_{};
  VkPhysicalDeviceProperties properties_{};
};

}  // namespace volumetric_kit::gfx

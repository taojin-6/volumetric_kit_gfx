// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/physical_device_info.hpp"

#include "volumetric_kit/gfx/core/impl/vk_query.hpp"

namespace volumetric_kit::gfx {

PhysicalDeviceInfo PhysicalDeviceInfo::query(VkPhysicalDevice physical) {
  PhysicalDeviceInfo info;
  info.physical_ = physical;
  for (const VkExtensionProperties& e : device_extensions(physical)) {
    info.extension_names_.emplace_back(e.extensionName);
  }
  info.features2_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  vkGetPhysicalDeviceFeatures2(physical, &info.features2_);  // pNext stays null
  vkGetPhysicalDeviceProperties(physical, &info.properties_);
  return info;
}

bool PhysicalDeviceInfo::supports_device_extension(const char* name) const {
  for (const std::string& e : extension_names_) {
    if (e == name) {
      return true;
    }
  }
  return false;
}

VkFormatProperties PhysicalDeviceInfo::format_properties(
    VkFormat format) const {
  VkFormatProperties props{};
  vkGetPhysicalDeviceFormatProperties(physical_, format, &props);
  return props;
}

bool PhysicalDeviceInfo::format_supports(VkFormat format, VkImageTiling tiling,
                                         VkFormatFeatureFlags features) const {
  return ::volumetric_kit::gfx::format_supports(physical_, format, tiling,
                                                features);
}

}  // namespace volumetric_kit::gfx

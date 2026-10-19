// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Before the first include, so vulkan.h declares VK_EXT_metal_surface. Outside
// Objective-C, vulkan_metal.h types CAMetalLayer as void, so this file builds
// on every OS and needs no Metal API.
#define VK_USE_PLATFORM_METAL_EXT

#include "volumetric_kit/gfx/windowing/surface_source.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "volumetric_kit/core/vulkan/vk_result.hpp"

namespace volumetric_kit::gfx::windowing {

core::Result<SurfaceSource> headless_surface() {
  std::uint32_t count = 0;
  VKC_VK_TRY(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
  std::vector<VkExtensionProperties> offered(count);
  VKC_VK_TRY(
      vkEnumerateInstanceExtensionProperties(nullptr, &count, offered.data()));
  offered.resize(count);
  const bool found = std::any_of(
      offered.begin(), offered.end(), [](const VkExtensionProperties& offer) {
        return std::strcmp(offer.extensionName,
                           VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) == 0;
      });
  if (!found) {
    return core::Status::unsupported(
        "the Vulkan loader offers no VK_EXT_headless_surface");
  }

  SurfaceSource source;
  source.instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                                VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
  // An extension entry point the link-time loader exports no symbol for, so
  // it is resolved on the instance, which gives none without the extension.
  source.make_surface = [](VkInstance instance) -> core::Result<VkSurfaceKHR> {
    auto create = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT"));
    if (create == nullptr) {
      return core::Status::unsupported(
          "VK_EXT_headless_surface is not enabled on this instance");
    }
    VkHeadlessSurfaceCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VKC_VK_TRY(create(instance, &info, nullptr, &surface));
    return surface;
  };
  return source;
}

core::Result<SurfaceSource> metal_surface(const void* metal_layer) {
  if (metal_layer == nullptr) {
    return core::Status::invalid_argument("metal_surface needs a CAMetalLayer");
  }
  SurfaceSource source;
  source.instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                                VK_EXT_METAL_SURFACE_EXTENSION_NAME};
  // Resolved on the instance, as the headless entry point is, so this links
  // where the loader exports no such symbol (Linux).
  source.make_surface =
      [metal_layer](VkInstance instance) -> core::Result<VkSurfaceKHR> {
    auto create = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateMetalSurfaceEXT"));
    if (create == nullptr) {
      return core::Status::unsupported(
          "VK_EXT_metal_surface is not enabled on this instance");
    }
    VkMetalSurfaceCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    info.pLayer = static_cast<const CAMetalLayer*>(metal_layer);
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VKC_VK_TRY(create(instance, &info, nullptr, &surface));
    return surface;
  };
  return source;
}

}  // namespace volumetric_kit::gfx::windowing

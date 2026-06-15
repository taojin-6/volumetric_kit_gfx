// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/windowing/surface.hpp"

namespace volumetric_kit::gfx::windowing {

Surface::Surface(VkInstance instance, VkSurfaceKHR surface) noexcept
    : instance_(instance), surface_(surface) {}

Surface::~Surface() { destroy(); }

Surface::Surface(Surface&& other) noexcept
    : instance_(other.instance_), surface_(other.surface_) {
  other.instance_ = VK_NULL_HANDLE;
  other.surface_ = VK_NULL_HANDLE;
}

Surface& Surface::operator=(Surface&& other) noexcept {
  if (this != &other) {
    destroy();
    instance_ = other.instance_;
    surface_ = other.surface_;
    other.instance_ = VK_NULL_HANDLE;
    other.surface_ = VK_NULL_HANDLE;
  }
  return *this;
}

Result<Surface> Surface::headless(VkInstance instance) {
  // vkCreateHeadlessSurfaceEXT is an extension entry point the link-time loader
  // does not export as a symbol, so resolve it dynamically: a driver without
  // VK_EXT_headless_surface (e.g. MoltenVK) then yields Unsupported instead of
  // a link error, and the headless tests skip cleanly.
  auto create = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT"));
  if (create == nullptr) {
    return Status::unsupported(
        "VK_EXT_headless_surface is not available on this instance");
  }
  VkHeadlessSurfaceCreateInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VG_VK_TRY(create(instance, &info, nullptr, &surface));
  return Surface(instance, surface);
}

void Surface::destroy() noexcept {
  if (surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
  }
  instance_ = VK_NULL_HANDLE;
}

}  // namespace volumetric_kit::gfx::windowing

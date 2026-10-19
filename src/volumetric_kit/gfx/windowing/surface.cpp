// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/windowing/surface.hpp"

#include "volumetric_kit/gfx/windowing/surface_source.hpp"

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

core::Result<Surface> Surface::headless(VkInstance instance) {
  VKC_ASSIGN(const SurfaceSource source, headless_surface());
  VKC_ASSIGN(const VkSurfaceKHR surface, source.make_surface(instance));
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

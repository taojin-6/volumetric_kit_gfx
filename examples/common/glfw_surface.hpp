// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file glfw_surface.hpp
/// @brief Shared example helper: a WindowedApp::SurfaceFactory backed by GLFW,
///        so the windowed examples do not each repeat the same
///        glfwCreateWindowSurface plumbing. The library tier stays
///        window-system-free; this GLFW glue lives consumer-side.

// windowed_app.hpp pulls in core/vulkan.hpp, so Vulkan is declared before GLFW
// below and glfw3.h then exposes glfwCreateWindowSurface. Self-contained, so
// this header may be included in any order.
#include "volumetric_kit/gfx/app/windowed_app.hpp"

#include <GLFW/glfw3.h>

namespace example {

/// @brief A @ref volumetric_kit::gfx::app::WindowedApp surface factory that
///        creates the `VkSurfaceKHR` for @p window via
///        `glfwCreateWindowSurface` on the app's instance.
/// @param window  The GLFW window to present into; must outlive the app.
/// @return A factory returning the surface, or a Vulkan-domain @ref
///         volumetric_kit::gfx::Status when GLFW fails to create one.
inline volumetric_kit::gfx::app::WindowedApp::SurfaceFactory
glfw_surface_factory(GLFWwindow* window) {
  return [window](
             VkInstance instance) -> volumetric_kit::gfx::Result<VkSurfaceKHR> {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const VkResult result =
        glfwCreateWindowSurface(instance, window, nullptr, &surface);
    if (result != VK_SUCCESS) {
      return volumetric_kit::gfx::vk_error(result, "glfwCreateWindowSurface");
    }
    return surface;
  };
}

}  // namespace example

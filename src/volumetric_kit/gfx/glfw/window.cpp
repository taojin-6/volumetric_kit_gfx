// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/glfw/window.hpp"

// Before GLFW, so glfw3.h sees Vulkan and declares glfwCreateWindowSurface.
#include "volumetric_kit/core/vulkan/vulkan.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "volumetric_kit/core/vulkan/vk_result.hpp"

namespace volumetric_kit::gfx::glfw {

core::Result<std::vector<const char*>> instance_extensions() {
  // Drop an error left by an earlier call, so the reason read below is this
  // call's own: GLFW returns NULL with no error when the platform simply has
  // no Vulkan surface.
  glfwGetError(nullptr);
  std::uint32_t count = 0;
  const char** names = glfwGetRequiredInstanceExtensions(&count);
  if (names == nullptr || count == 0) {
    const char* reason = nullptr;
    glfwGetError(&reason);
    return core::Status::unsupported(
        std::string("GLFW offers no Vulkan surface: ") +
        (reason != nullptr ? reason : "the platform has no Vulkan surface"));
  }
  return std::vector<const char*>(names, names + count);
}

app::WindowedApp::SurfaceFactory glfw_surface_factory(GLFWwindow* window) {
  return [window](VkInstance instance) -> core::Result<VkSurfaceKHR> {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VKC_VK_TRY(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    return surface;
  };
}

VkExtent2D framebuffer_extent(GLFWwindow* window) {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  return {static_cast<std::uint32_t>(std::max(1, width)),
          static_cast<std::uint32_t>(std::max(1, height))};
}

glm::vec2 cursor_in_framebuffer(GLFWwindow* window) {
  double x = 0.0;
  double y = 0.0;
  glfwGetCursorPos(window, &x, &y);
  int window_width = 0;
  int window_height = 0;
  glfwGetWindowSize(window, &window_width, &window_height);
  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
  const float scale_x = window_width > 0
                            ? static_cast<float>(framebuffer_width) /
                                  static_cast<float>(window_width)
                            : 1.0f;
  const float scale_y = window_height > 0
                            ? static_cast<float>(framebuffer_height) /
                                  static_cast<float>(window_height)
                            : 1.0f;
  return {static_cast<float>(x) * scale_x, static_cast<float>(y) * scale_y};
}

}  // namespace volumetric_kit::gfx::glfw

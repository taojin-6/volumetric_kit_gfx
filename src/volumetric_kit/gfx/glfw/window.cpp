// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/glfw/window.hpp"

// Before GLFW, so glfw3.h sees Vulkan and declares glfwCreateWindowSurface.
#include "volumetric_kit/core/vulkan/vulkan.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/glfw/impl/framebuffer_pixels.hpp"

namespace volumetric_kit::gfx::glfw {

namespace {

// GLFW's last error, after `call`, as an Unsupported status.
core::Status glfw_failure(const char* call) {
  const char* reason = nullptr;
  glfwGetError(&reason);
  return core::Status::unsupported(std::string(call) + " failed: " +
                                   (reason != nullptr ? reason : "no reason"));
}

}  // namespace

core::Result<std::unique_ptr<Library>> Library::init() {
  glfwGetError(nullptr);  // the reason read on failure is this call's own
  if (glfwInit() != GLFW_TRUE) {
    return glfw_failure("glfwInit");
  }
  return std::unique_ptr<Library>(new Library());
}

Library::~Library() { glfwTerminate(); }

void WindowDeleter::operator()(GLFWwindow* window) const noexcept {
  glfwDestroyWindow(window);
}

core::Result<Window> create_window(int width, int height, const char* title) {
  if (width <= 0 || height <= 0 || title == nullptr) {
    return core::Status::invalid_argument(
        "glfw::create_window needs a positive size and a title");
  }
  glfwGetError(nullptr);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Vulkan, not OpenGL
  GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (window == nullptr) {
    return glfw_failure("glfwCreateWindow");
  }
  return Window(window);
}

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
    if (window == nullptr) {
      return core::Status::invalid_argument(
          "glfw_surface_factory needs a window");
    }
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VKC_VK_TRY(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    return surface;
  };
}

core::Result<windowing::SurfaceSource> surface_source(GLFWwindow* window) {
  if (window == nullptr) {
    return core::Status::invalid_argument(
        "glfw::surface_source needs a window");
  }
  VKC_ASSIGN(std::vector<const char*> extensions, instance_extensions());
  return windowing::SurfaceSource{std::move(extensions),
                                  glfw_surface_factory(window)};
}

VkExtent2D framebuffer_size(GLFWwindow* window) {
  VKC_CHECK(window != nullptr, "glfw::framebuffer_size needs a window");
  glm::ivec2 framebuffer{0, 0};
  glfwGetFramebufferSize(window, &framebuffer.x, &framebuffer.y);
  return detail::to_framebuffer_size(
      framebuffer, glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE);
}

VkExtent2D framebuffer_extent(GLFWwindow* window) {
  VKC_CHECK(window != nullptr, "glfw::framebuffer_extent needs a window");
  const VkExtent2D size = framebuffer_size(window);
  return {std::max(1u, size.width), std::max(1u, size.height)};
}

glm::vec2 cursor_in_framebuffer(GLFWwindow* window) {
  VKC_CHECK(window != nullptr, "glfw::cursor_in_framebuffer needs a window");
  glm::dvec2 cursor{0.0, 0.0};
  glfwGetCursorPos(window, &cursor.x, &cursor.y);
  glm::ivec2 window_size{0, 0};
  glfwGetWindowSize(window, &window_size.x, &window_size.y);
  glm::ivec2 framebuffer{0, 0};
  glfwGetFramebufferSize(window, &framebuffer.x, &framebuffer.y);
  return detail::to_framebuffer_pixels(cursor, window_size, framebuffer);
}

}  // namespace volumetric_kit::gfx::glfw

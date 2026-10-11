// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file window.hpp
/// @brief A GLFW window as the app tier takes one: its instance extensions,
///        its surface, its framebuffer size and the cursor in framebuffer
///        pixels.

#include <vector>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/glfw/export.hpp"

struct GLFWwindow;

namespace volumetric_kit::gfx::glfw {

/// @brief The instance extensions GLFW needs to make a surface:
///        `VK_KHR_surface` and the platform's.
///
/// @code
/// auto extensions = glfw::instance_extensions();
/// if (!extensions) return extensions.status();
/// app::WindowedAppConfig config;
/// config.instance_extensions = extensions.value();
/// @endcode
/// @return GLFW's names, valid until `glfwTerminate`;
///         `core::Status::Code::Unsupported`, with GLFW's reason, when GLFW
///         is not initialized, finds no Vulkan loader, or the platform cannot
///         make a Vulkan surface.
VG_GLFW_API core::Result<std::vector<const char*>> instance_extensions();

/// @brief A @ref app::WindowedApp surface factory that makes @p window's
///        surface with `glfwCreateWindowSurface`.
///
/// @code
/// auto app = app::WindowedApp::create(config,
///                                     glfw::glfw_surface_factory(window));
/// @endcode
/// @param window  The window to present to; it must outlive the surface.
/// @return The factory. It returns the surface;
///         `core::Status::Code::InvalidArgument` for a null @p window; or
///         GLFW's `VkResult` as a backend `core::Status` -- e.g. when the
///         instance lacks @ref instance_extensions.
VG_GLFW_API app::WindowedApp::SurfaceFactory glfw_surface_factory(
    GLFWwindow* window);

/// @brief The window's framebuffer size in pixels, as GLFW reports it: 0 x 0
///        while the window is minimized.
///
/// The extent `begin_frame` takes: on 0 x 0 it skips the tick, with no
/// swapchain rebuild and no queue drain.
/// @code
/// auto frame = app.begin_frame(glfw::framebuffer_size(window));
/// @endcode
/// @param window  The window.
/// @pre @p window is non-null.
/// @return Its framebuffer size; zero while minimized.
VG_GLFW_API VkExtent2D framebuffer_size(GLFWwindow* window);

/// @brief The window's framebuffer size in pixels, never zero.
///
/// A window may report 0 x 0 while minimized; this reads 1 x 1 then, for a
/// use that cannot take zero: an initial `swapchain.extent`, a viewport or an
/// aspect ratio. A frame loop takes @ref framebuffer_size instead.
/// @code
/// app::WindowedAppConfig config;
/// config.swapchain.extent = glfw::framebuffer_extent(window);
/// @endcode
/// @param window  The window.
/// @pre @p window is non-null.
/// @return Its framebuffer extent, at least 1 x 1.
VG_GLFW_API VkExtent2D framebuffer_extent(GLFWwindow* window);

/// @brief The cursor in framebuffer pixels, from the window's top-left.
///
/// GLFW reports the cursor in window coordinates, which a high-DPI display
/// scales down from the framebuffer's; this scales them back, so the result
/// is in the pixels a @ref camera::ImageView2D maps.
/// @code
/// view.zoom_about(glfw::cursor_in_framebuffer(window), 1.1f);
/// @endcode
/// @param window  The window.
/// @pre @p window is non-null.
/// @return The cursor position; unscaled while the window has no size.
VG_GLFW_API glm::vec2 cursor_in_framebuffer(GLFWwindow* window);

}  // namespace volumetric_kit::gfx::glfw

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file window.hpp
/// @brief A GLFW window as the app tier takes one: GLFW's initialization, the
///        window, its surface source, its framebuffer size and the cursor in
///        framebuffer pixels.

#include <memory>
#include <vector>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/glfw/export.hpp"
#include "volumetric_kit/gfx/windowing/surface_source.hpp"

struct GLFWwindow;

namespace volumetric_kit::gfx::glfw {

/// @brief GLFW, initialized while this lives: `glfwInit`, then
///        `glfwTerminate`.
///
/// Set any init hint, such as the platform, before @ref init.
///
/// @code
/// VKC_ASSIGN(std::unique_ptr<glfw::Library> library, glfw::Library::init());
/// VKC_ASSIGN(glfw::Window window, glfw::create_window(1280, 720, "viewer"));
/// // ... the window is destroyed before the library ...
/// @endcode
class VG_GLFW_API Library {
 public:
  /// @return The guard; `core::Status::Code::Unsupported`, with GLFW's
  ///         reason, if `glfwInit` fails.
  static core::Result<std::unique_ptr<Library>> init();

  ~Library();
  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  Library(Library&&) = delete;
  Library& operator=(Library&&) = delete;

 private:
  Library() = default;
};

/// @brief Destroys a GLFW window: `glfwDestroyWindow`.
struct VG_GLFW_API WindowDeleter {
  /// @param window  The window to destroy.
  void operator()(GLFWwindow* window) const noexcept;
};

/// @brief An owned GLFW window.
using Window = std::unique_ptr<GLFWwindow, WindowDeleter>;

/// @brief A window with no client API, for Vulkan to present to. Other
///        window hints stay as the caller set them.
///
/// @code
/// auto window = glfw::create_window(1280, 720, "viewer");
/// if (!window) return window.status();
/// config.swapchain.extent = glfw::framebuffer_extent(window.value().get());
/// @endcode
/// @param width   The window's width in screen coordinates; > 0.
/// @param height  The window's height in screen coordinates; > 0.
/// @param title   Its title, UTF-8; non-null.
/// @pre A @ref Library lives, and outlives the window.
/// @return The window; `core::Status::Code::InvalidArgument` for a
///         non-positive size or a null @p title; or
///         `core::Status::Code::Unsupported`, with GLFW's reason, if GLFW
///         cannot make one.
VG_GLFW_API core::Result<Window> create_window(int width, int height,
                                               const char* title);

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

/// @brief GLFW's @ref windowing::SurfaceSource for @p window.
///
/// @code
/// VKC_ASSIGN(windowing::SurfaceSource surface, glfw::surface_source(window));
/// VKC_ASSIGN(std::unique_ptr<core::SharedDevice> shared,
///            app::make_shared_device(surface, compute, config));
/// @endcode
/// @param window  The window to present to; it must outlive the surface.
/// @return @ref instance_extensions with @ref glfw_surface_factory;
///         `core::Status::Code::InvalidArgument` for a null @p window; or
///         @ref instance_extensions' failure.
VG_GLFW_API core::Result<windowing::SurfaceSource> surface_source(
    GLFWwindow* window);

/// @brief The window's framebuffer size in pixels: 0 x 0 while GLFW reports
///        it iconified.
///
/// macOS and X11 keep an iconified window's size, so this reads
/// `GLFW_ICONIFIED` too; Wayland reports no iconified state, so a minimized
/// window there reads its size. The extent `begin_frame` takes: on 0 x 0 it
/// skips the tick, with no swapchain rebuild and no queue drain.
/// @code
/// auto frame = app.begin_frame(glfw::framebuffer_size(window));
/// @endcode
/// @param window  The window.
/// @pre @p window is non-null.
/// @return Its framebuffer size.
VG_GLFW_API VkExtent2D framebuffer_size(GLFWwindow* window);

/// @brief The window's framebuffer size in pixels, never zero.
///
/// @ref framebuffer_size, but 1 x 1 where that reads 0 x 0, for a use that
/// cannot take zero: an initial `swapchain.extent`, a viewport or an aspect
/// ratio. A frame loop takes @ref framebuffer_size instead.
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

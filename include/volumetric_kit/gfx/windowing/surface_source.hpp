// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file surface_source.hpp
/// @brief A window system's surface as the tiers above take it: its instance
///        extensions and the factory that makes it, from one constructor per
///        window system.

#include <functional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/windowing/export.hpp"

namespace volumetric_kit::gfx::windowing {

/// @brief Makes a `VkSurfaceKHR` on the instance it is given; the caller owns
///        the result. The type of `core::SharedDeviceConfig::make_surface` and
///        of `app::WindowedApp::SurfaceFactory`.
using SurfaceFactory = std::function<core::Result<VkSurfaceKHR>(VkInstance)>;

/// @brief A window system's surface, as everything above `windowing` takes it.
///
/// One constructor per window system makes it, so its extensions and its
/// factory cannot disagree. Each returns a `Result`, refusing what it can see
/// is wrong before any instance exists:
/// - `glfw::surface_source(window)`: a GLFW window (X11, Wayland, macOS);
/// - @ref metal_surface: a `CAMetalLayer` (iOS);
/// - @ref headless_surface: no window (tests, CI).
///
/// @code
/// VKC_ASSIGN(windowing::SurfaceSource surface, glfw::surface_source(window));
/// VKC_ASSIGN(std::unique_ptr<core::SharedDevice> shared,
///            app::make_shared_device(surface, compute, config));
/// VKC_ASSIGN(app::WindowedApp app,
///            app::WindowedApp::adopt(*shared, app_config));
/// @endcode
struct SurfaceSource {
  /// `VK_KHR_surface` and the platform's.
  std::vector<const char*> instance_extensions;
  /// Called once, on the thread that owns the window or layer.
  SurfaceFactory make_surface;
};

/// @brief `VK_EXT_headless_surface`'s source, for tests and CI. The one home
///        of the headless factory: @ref Surface::headless wraps it.
///
/// @code
/// auto headless = windowing::headless_surface();
/// if (!headless.ok()) GTEST_SKIP() << headless.status().message();
/// @endcode
/// @return The source; `core::Status::Code::Unsupported` where the loader
///         does not offer the extension (NVIDIA's ICD; Lavapipe and MoltenVK
///         offer it), or the failed `VkResult` of asking it. Its factory
///         returns the surface, `Unsupported` on an instance without the
///         extension, or the `VkResult`.
VG_WINDOWING_API core::Result<SurfaceSource> headless_surface();

/// @brief `VK_EXT_metal_surface`'s source over a `CAMetalLayer`: the Vulkan
///        call GLFW makes for a macOS window, and no Metal API.
///
/// @code
/// // Objective-C++, on the main thread:
/// layer.contentsScale = view.window.screen.nativeScale;
/// VKC_ASSIGN(windowing::SurfaceSource surface,
///            windowing::metal_surface((__bridge const void*)layer));
/// @endcode
/// @param metal_layer  A `CAMetalLayer*`; it must outlive the surface.
/// @return The source; `core::Status::Code::InvalidArgument` for a null
///         @p metal_layer. Its factory returns the surface, `Unsupported`
///         where `vkCreateMetalSurfaceEXT` does not resolve (an instance
///         without the extension, or a loader without it), or the `VkResult`.
/// @warning Make the surface, and build and drive the swapchain, on the thread
///          that owns the layer (main). MoltenVK writes the layer's properties
///          as it builds, `drawableSize` among them: the caller sets
///          `contentsScale` and never `drawableSize`.
VG_WINDOWING_API core::Result<SurfaceSource> metal_surface(
    const void* metal_layer);

}  // namespace volumetric_kit::gfx::windowing

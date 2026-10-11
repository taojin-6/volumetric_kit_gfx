// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file shared_device_config.hpp
/// @brief The `core::SharedDeviceConfig` for one device that a windowed
///        renderer and a compute library both adopt, from any window system.
///
/// The embedder's half of the create/adopt seam (DECISIONS.md, "A `VkDevice`
/// may be created *or adopted*"). gfx learns only the compute library's
/// `core::DeviceRequirements`, never the library.

#include <vector>

#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/core/vulkan/shared_device.hpp"
#include "volumetric_kit/gfx/app/export.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"

namespace volumetric_kit::gfx::app {

/// @brief What the embedder sets for a device its windowed renderer shares
///        with a compute library; the window system and the two libraries
///        supply the rest.
///
/// @code
/// app::WindowDeviceConfig config;
/// config.instance.app_name = "viewer";
/// config.instance.enable_validation = true;
/// config.graphics = app_config.device;  // what WindowedApp::adopt verifies
/// @endcode
struct WindowDeviceConfig {
  /// The instance: its name, validation, and any extensions beyond the window
  /// system's, which are added. Validation here is the only way to validate an
  /// adopted renderer: @ref WindowedApp::adopt ignores its own.
  core::InstanceConfig instance;
  /// The renderer's requirements: the @ref WindowedAppConfig::device the app
  /// is adopted with, which presentation is added to.
  core::DeviceRequirements graphics = device_requirements();
};

/// @brief The union of a windowed renderer's and a compute library's needs,
///        as the `core::SharedDeviceConfig` that builds their one device. Pure:
///        no window-system or Vulkan call.
///
/// @code
/// core::SharedDeviceConfig shared = app::shared_device_config(
///     {VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME},
///     compute, config);
/// shared.make_surface = make_metal_surface;  // the window system's
/// VKC_ASSIGN(std::unique_ptr<core::SharedDevice> device,
///            core::SharedDevice::create(shared));
/// @endcode
/// @param surface_extensions  The window system's instance extensions:
///                            `VK_KHR_surface` and the platform's.
/// @param compute             The compute library's requirements, passed on
///                            unchanged.
/// @param config              The embedder's instance and renderer settings.
/// @return @p config's instance with each of @p surface_extensions it does not
///         already name added once, by string; @p compute; and @p config's
///         graphics requirements with `needs_present` set. `make_surface` is
///         left empty for the caller.
VG_APP_API core::SharedDeviceConfig shared_device_config(
    const std::vector<const char*>& surface_extensions,
    const core::DeviceRequirements& compute, const WindowDeviceConfig& config);

}  // namespace volumetric_kit::gfx::app

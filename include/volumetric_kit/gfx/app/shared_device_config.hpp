// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file shared_device_config.hpp
/// @brief One device that a windowed renderer and a compute library both
///        adopt, from any window system: its `core::SharedDeviceConfig`, and
///        the device itself.
///
/// The embedder's half of the create/adopt seam (DECISIONS.md, "A `VkDevice`
/// may be created *or adopted*"). gfx learns only the compute library's
/// `core::DeviceRequirements`, never the library.

#include <memory>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/core/vulkan/shared_device.hpp"
#include "volumetric_kit/gfx/app/export.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/windowing/surface_source.hpp"

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
/// VKC_ASSIGN(windowing::SurfaceSource surface,
///            windowing::metal_surface(layer));
/// core::SharedDeviceConfig shared =
///     app::shared_device_config(surface.instance_extensions, compute, config);
/// shared.make_surface = surface.make_surface;
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

/// @brief Build one instance and device that satisfy the renderer and a
///        compute library, and @p surface's surface on them:
///        @ref shared_device_config, then `core::SharedDevice::create`.
///
/// The renderer adopts it through @ref WindowedApp::adopt, which takes the
/// surface; the compute library adopts `core::SharedDevice::compute_payload()`
/// through `core::Device::adopt`. The device's summary is logged at `Info`,
/// and a warning when both libraries share one queue
/// (`core::QueuePlan::SharedQueue`).
///
/// @code
/// app::WindowDeviceConfig config;
/// config.instance.app_name = "viewer";
/// config.graphics = app_config.device;
/// VKC_ASSIGN(std::unique_ptr<core::SharedDevice> shared,
///            app::make_shared_device(surface, fusion_requirements, config));
/// VKC_ASSIGN(app::WindowedApp app,
///            app::WindowedApp::adopt(*shared, app_config));
/// @endcode
/// @param surface  The window system's source; its window or layer must
///                 outlive the device.
/// @param compute  The compute library's requirements.
/// @param config   The instance and the renderer's requirements.
/// @return The device, or `core::SharedDevice::create`'s failure, the surface
///         factory's among them. Treat a failure as fatal rather than falling
///         back to two devices, which would give up the shared `VkBuffer` the
///         device exists for.
/// @warning Both adopters, and everything made on them, must be destroyed
///          before the returned device.
VG_APP_API core::Result<std::unique_ptr<core::SharedDevice>> make_shared_device(
    const windowing::SurfaceSource& surface,
    const core::DeviceRequirements& compute, const WindowDeviceConfig& config);

}  // namespace volumetric_kit::gfx::app

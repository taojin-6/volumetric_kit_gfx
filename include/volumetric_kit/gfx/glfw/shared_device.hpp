// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file shared_device.hpp
/// @brief One `VkDevice` for a GLFW window's renderer and a compute library,
///        for both to adopt.
///
/// The embedder's half of the create/adopt seam (DECISIONS.md, "A `VkDevice`
/// may be created *or adopted*"), for a GLFW window. gfx learns only the
/// compute library's @ref core::DeviceRequirements, never the library.

#include <memory>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/core/vulkan/shared_device.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/glfw/export.hpp"

struct GLFWwindow;

namespace volumetric_kit::gfx::glfw {

/// @brief What the embedder sets for @ref make_shared_device; the rest comes
///        from the window and the two libraries.
struct SharedDeviceConfig {
  /// The instance: its name, validation, and any extensions beyond GLFW's,
  /// which are added. Validation here is the only way to validate an adopted
  /// renderer: @ref app::WindowedApp::adopt ignores its own.
  core::InstanceConfig instance;
  /// The renderer's requirements: the @ref app::WindowedAppConfig::device the
  /// app is adopted with, which presentation is added to.
  core::DeviceRequirements graphics = device_requirements();
};

/// @brief The `core::SharedDeviceConfig` @ref make_shared_device builds, with
///        no GLFW or Vulkan call.
///
/// @code
/// const core::SharedDeviceConfig shared = glfw::shared_device_config(
///     {"VK_KHR_surface", "VK_EXT_headless_surface"}, compute, config);
/// // shared.graphics.needs_present is set; config.graphics is unchanged.
/// @endcode
/// @param surface_extensions  The window system's instance extensions, from
///                            @ref instance_extensions.
/// @param compute             The compute library's requirements, passed on
///                            unchanged.
/// @param config              The embedder's instance and renderer settings.
/// @return @p config's instance with @p surface_extensions added once each,
///         @p compute, and @p config's graphics requirements with
///         `needs_present` set. `make_surface` is left empty.
VG_GLFW_API core::SharedDeviceConfig shared_device_config(
    const std::vector<const char*>& surface_extensions,
    const core::DeviceRequirements& compute, const SharedDeviceConfig& config);

/// @brief Build one instance and device that satisfy the renderer and a
///        compute library, and @p window's surface on them.
///
/// The renderer adopts @ref core::SharedDevice::graphics_payload through
/// @ref app::WindowedApp::adopt, taking the surface with
/// @ref core::SharedDevice::release_surface; the compute library adopts
/// @ref core::SharedDevice::compute_payload through `core::Device::adopt`.
/// The device's summary is logged at `Info`, and a warning when both
/// libraries share one queue (@ref core::QueuePlan::SharedQueue).
///
/// @code
/// glfw::SharedDeviceConfig config;
/// config.instance.app_name = "viewer";
/// config.graphics = app_config.device;
/// VKC_ASSIGN(std::unique_ptr<core::SharedDevice> shared,
///            glfw::make_shared_device(window, fusion_requirements, config));
/// VKC_ASSIGN(app::WindowedApp app,
///            app::WindowedApp::adopt(
///                shared->graphics_payload(), app_config,
///                [&](VkInstance) -> core::Result<VkSurfaceKHR> {
///                  return shared->release_surface();
///                }));
/// @endcode
/// @param window   The window the renderer presents to; it must outlive the
///                 device.
/// @param compute  The compute library's requirements.
/// @param config   The instance and the renderer's requirements.
/// @return The device; `core::Status::Code::InvalidArgument` for a null
///         @p window; @ref instance_extensions' failure; or
///         `core::SharedDevice::create`'s. Treat a failure as fatal rather
///         than falling back to two devices, which would give up the shared
///         `VkBuffer` the device exists for.
/// @warning Both adopters, and everything made on them, must be destroyed
///          before the returned device.
VG_GLFW_API core::Result<std::unique_ptr<core::SharedDevice>>
make_shared_device(GLFWwindow* window, const core::DeviceRequirements& compute,
                   const SharedDeviceConfig& config);

}  // namespace volumetric_kit::gfx::glfw

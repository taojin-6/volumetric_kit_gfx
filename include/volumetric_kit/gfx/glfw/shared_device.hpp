// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file shared_device.hpp
/// @brief One `VkDevice` for a GLFW window's renderer and a compute library,
///        for both to adopt: @ref app::shared_device_config with GLFW's
///        instance extensions and surface.

#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/shared_device.hpp"
#include "volumetric_kit/gfx/app/shared_device_config.hpp"
#include "volumetric_kit/gfx/glfw/export.hpp"

struct GLFWwindow;

namespace volumetric_kit::gfx::glfw {

/// @brief Build one instance and device that satisfy the renderer and a
///        compute library, and @p window's surface on them.
///
/// The renderer adopts `core::SharedDevice::graphics_payload()` through
/// @ref app::WindowedApp::adopt, taking the surface with
/// `core::SharedDevice::release_surface()`; the compute library adopts
/// `core::SharedDevice::compute_payload()` through `core::Device::adopt`.
/// The device's summary is logged at `Info`, and a warning when both
/// libraries share one queue (`core::QueuePlan::SharedQueue`).
///
/// @code
/// app::WindowDeviceConfig config;
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
                   const app::WindowDeviceConfig& config);

}  // namespace volumetric_kit::gfx::glfw

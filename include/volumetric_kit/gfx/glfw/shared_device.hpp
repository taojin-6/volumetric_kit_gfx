// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file shared_device.hpp
/// @brief One `VkDevice` for a GLFW window's renderer and a compute library,
///        for both to adopt: @ref app::make_shared_device on the window's
///        @ref surface_source.

#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/shared_device.hpp"
#include "volumetric_kit/gfx/app/shared_device_config.hpp"
#include "volumetric_kit/gfx/glfw/export.hpp"

struct GLFWwindow;

namespace volumetric_kit::gfx::glfw {

/// @brief @ref app::make_shared_device on @p window's @ref surface_source.
///
/// @code
/// app::WindowDeviceConfig config;
/// config.instance.app_name = "viewer";
/// config.graphics = app_config.device;
/// VKC_ASSIGN(std::unique_ptr<core::SharedDevice> shared,
///            glfw::make_shared_device(window, fusion_requirements, config));
/// VKC_ASSIGN(app::WindowedApp app,
///            app::WindowedApp::adopt(*shared, app_config));
/// @endcode
/// @param window   The window the renderer presents to; it must outlive the
///                 device.
/// @param compute  The compute library's requirements.
/// @param config   The instance and the renderer's requirements.
/// @return As @ref app::make_shared_device; or @ref surface_source's failure,
///         `core::Status::Code::InvalidArgument` for a null @p window among
///         them.
/// @warning Both adopters, and everything made on them, must be destroyed
///          before the returned device.
VG_GLFW_API core::Result<std::unique_ptr<core::SharedDevice>>
make_shared_device(GLFWwindow* window, const core::DeviceRequirements& compute,
                   const app::WindowDeviceConfig& config);

}  // namespace volumetric_kit::gfx::glfw

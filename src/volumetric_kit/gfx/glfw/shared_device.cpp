// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/glfw/shared_device.hpp"

#include "volumetric_kit/gfx/glfw/window.hpp"
#include "volumetric_kit/gfx/windowing/surface_source.hpp"

namespace volumetric_kit::gfx::glfw {

core::Result<std::unique_ptr<core::SharedDevice>> make_shared_device(
    GLFWwindow* window, const core::DeviceRequirements& compute,
    const app::WindowDeviceConfig& config) {
  VKC_ASSIGN(const windowing::SurfaceSource surface, surface_source(window));
  return app::make_shared_device(surface, compute, config);
}

}  // namespace volumetric_kit::gfx::glfw

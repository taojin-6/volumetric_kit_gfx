// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/glfw/shared_device.hpp"

#include <utility>
#include <vector>

#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/glfw/window.hpp"

namespace volumetric_kit::gfx::glfw {

core::Result<std::unique_ptr<core::SharedDevice>> make_shared_device(
    GLFWwindow* window, const core::DeviceRequirements& compute,
    const app::WindowDeviceConfig& config) {
  if (window == nullptr) {
    return core::Status::invalid_argument("make_shared_device needs a window");
  }
  VKC_ASSIGN(const std::vector<const char*> extensions, instance_extensions());
  core::SharedDeviceConfig shared =
      app::shared_device_config(extensions, compute, config);
  shared.make_surface = glfw_surface_factory(window);
  VKC_ASSIGN(std::unique_ptr<core::SharedDevice> device,
             core::SharedDevice::create(shared));
  log_message(core::LogLevel::Info, "shared device: " + device->summary());
  if (device->plan() == core::QueuePlan::SharedQueue) {
    log_message(core::LogLevel::Warning,
                "shared device: one queue shared under a mutex, so compute "
                "and rendering serialize");
  }
  return device;
}

}  // namespace volumetric_kit::gfx::glfw

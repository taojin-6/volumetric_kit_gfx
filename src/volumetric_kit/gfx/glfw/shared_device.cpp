// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/glfw/shared_device.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/glfw/window.hpp"

namespace volumetric_kit::gfx::glfw {

core::SharedDeviceConfig shared_device_config(
    const std::vector<const char*>& surface_extensions,
    const core::DeviceRequirements& compute, const SharedDeviceConfig& config) {
  core::SharedDeviceConfig shared;
  shared.instance = config.instance;
  for (const char* name : surface_extensions) {
    const bool listed = std::any_of(
        shared.instance.extensions.begin(), shared.instance.extensions.end(),
        [name](const char* listed_name) {
          return std::strcmp(listed_name, name) == 0;
        });
    if (!listed) {
      shared.instance.extensions.push_back(name);
    }
  }
  shared.compute = compute;
  shared.graphics = config.graphics;
  shared.graphics.needs_present = true;
  return shared;
}

core::Result<std::unique_ptr<core::SharedDevice>> make_shared_device(
    GLFWwindow* window, const core::DeviceRequirements& compute,
    const SharedDeviceConfig& config) {
  if (window == nullptr) {
    return core::Status::invalid_argument("make_shared_device needs a window");
  }
  VKC_ASSIGN(const std::vector<const char*> extensions, instance_extensions());
  core::SharedDeviceConfig shared =
      shared_device_config(extensions, compute, config);
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

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/app/shared_device_config.hpp"

#include <algorithm>
#include <cstring>

namespace volumetric_kit::gfx::app {

core::SharedDeviceConfig shared_device_config(
    const std::vector<const char*>& surface_extensions,
    const core::DeviceRequirements& compute, const WindowDeviceConfig& config) {
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

}  // namespace volumetric_kit::gfx::app

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/app/headless_app.hpp"

#include <memory>
#include <utility>

namespace volumetric_kit::gfx::app {

Result<HeadlessApp> HeadlessApp::create(const HeadlessAppConfig& config) {
  // Built at its resting place, like WindowedApp::create: a failure at any
  // step unwinds the partial State in reverse member order.
  auto state = std::make_unique<State>();

  InstanceConfig instance_config;
  instance_config.app_name = config.app_name;
  instance_config.enable_validation = config.enable_validation;
  instance_config.extensions = config.instance_extensions;
  VG_ASSIGN(Instance instance, Instance::create(instance_config));
  state->instance.emplace(std::move(instance));

  // No surface anywhere in the chain: selection needs no present-capable
  // queue family and the device enables no swapchain extension.
  if (config.device.needs_present) {
    return Status::invalid_argument(
        "HeadlessApp::create: a headless app has no surface to present to; "
        "leave device.needs_present false");
  }
  // One set of requirements threads through selection and creation, so the
  // device chosen is one the create accepts. The device learns from the
  // instance whether VK_EXT_debug_utils was enabled, so labels work or no-op.
  VG_ASSIGN(PhysicalDeviceInfo physical,
            state->instance->select_physical_device(config.device));
  VG_ASSIGN(Device device,
            Device::create(*state->instance, physical, config.device));
  state->device.emplace(std::move(device));

  VG_ASSIGN(Allocator allocator,
            Allocator::create(state->instance->handle(), *state->device));
  state->allocator.emplace(std::move(allocator));

  HeadlessApp app;
  app.state_ = std::move(state);
  return app;
}

}  // namespace volumetric_kit::gfx::app

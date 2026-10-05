// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/app/headless_app.hpp"

#include <memory>
#include <utility>

namespace volumetric_kit::gfx::app {

core::Result<HeadlessApp> HeadlessApp::create(const HeadlessAppConfig& config) {
  // No surface anywhere in the chain: selection needs no present-capable
  // queue family and the device enables no swapchain extension. Refused before
  // the instance exists, like every argument check.
  if (config.device.needs_present) {
    return core::Status::invalid_argument(
        "HeadlessApp::create: a headless app has no surface to present to; "
        "leave device.needs_present false");
  }
  // The renderer's floor, whatever config.device holds: a caller that built it
  // from DeviceRequirements{} rather than device_requirements() still gets a
  // device every gfx type can run on. One set of requirements then threads
  // through selection and creation, so the device chosen is one the create
  // accepts.
  VKC_ASSIGN(const core::DeviceRequirements reqs,
             core::merge(device_requirements(), config.device));

  // Built at its resting place, like WindowedApp::create: a failure at any
  // step unwinds the partial State in reverse member order.
  auto state = std::make_unique<State>();

  core::InstanceConfig instance_config;
  instance_config.app_name = config.app_name;
  instance_config.enable_validation = config.enable_validation;
  instance_config.extensions = config.instance_extensions;
  VKC_ASSIGN(core::Instance instance, core::Instance::create(instance_config));
  state->instance.emplace(std::move(instance));

  // The device learns from the instance whether VK_EXT_debug_utils was
  // enabled, so labels work or no-op.
  VKC_ASSIGN(core::PhysicalDeviceInfo physical,
             state->instance->select_physical_device(reqs));
  VKC_ASSIGN(core::Device device,
             core::Device::create(*state->instance, physical, reqs));
  state->device.emplace(std::move(device));

  VKC_ASSIGN(
      core::Allocator allocator,
      core::Allocator::create(state->instance->handle(), *state->device));
  state->allocator.emplace(std::move(allocator));

  HeadlessApp app;
  app.state_ = std::move(state);
  return app;
}

}  // namespace volumetric_kit::gfx::app

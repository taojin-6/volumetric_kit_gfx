// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/app/windowed_app.hpp"

#include <memory>
#include <utility>

namespace volumetric_kit::gfx::app {

Result<WindowedApp> WindowedApp::create(const WindowedAppConfig& config,
                                        const SurfaceFactory& create_surface) {
  if (!create_surface) {
    return Status::invalid_argument(
        "WindowedApp::create: create_surface must be callable");
  }
  // Build into the final State up front: the later steps borrow the earlier
  // members by address (loop -> swapchain/device, swapchain -> device/
  // allocator), so each must be created at its resting place, never moved
  // afterwards. A failure at any step returns through VG_ASSIGN and the
  // partially built State unwinds in reverse member order.
  auto state = std::make_unique<State>();

  InstanceConfig instance_config;
  instance_config.app_name = config.app_name;
  instance_config.enable_validation = config.enable_validation;
  instance_config.extra_instance_extensions = config.instance_extensions;
  VG_ASSIGN(Instance instance, Instance::create(instance_config));
  state->instance.emplace(std::move(instance));

  VG_ASSIGN(VkSurfaceKHR raw_surface,
            create_surface(state->instance->handle()));
  if (raw_surface == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "WindowedApp::create: the surface factory returned VK_NULL_HANDLE");
  }
  state->surface = windowing::Surface(state->instance->handle(), raw_surface);

  // One surface threads through selection, DeviceConfig::needs_present, and
  // Device::create, so the three stay consistent by construction.
  VG_ASSIGN(VkPhysicalDevice physical,
            state->instance->select_physical_device(state->surface.handle()));
  DeviceConfig device_config;
  device_config.needs_present = true;
  VG_ASSIGN(Device device,
            Device::create(state->instance->handle(), physical, device_config,
                           state->surface.handle()));
  state->device.emplace(std::move(device));

  VG_ASSIGN(Allocator allocator,
            Allocator::create(state->instance->handle(), *state->device));
  state->allocator.emplace(std::move(allocator));

  // A depth-configured swapchain allocates its per-image depth attachments
  // through the allocator; a color-only one takes none.
  Allocator* depth_allocator =
      config.swapchain.depth_format != VK_FORMAT_UNDEFINED ? &*state->allocator
                                                           : nullptr;
  VG_ASSIGN(
      windowing::Swapchain swapchain,
      windowing::Swapchain::create(*state->device, state->surface.handle(),
                                   config.swapchain, depth_allocator));
  state->swapchain = std::move(swapchain);

  VG_ASSIGN(windowing::FrameLoop frame_loop,
            windowing::FrameLoop::create(*state->device, state->swapchain,
                                         config.frames_in_flight));
  state->frame_loop = std::move(frame_loop);

  WindowedApp app;
  app.state_ = std::move(state);
  return app;
}

Result<std::optional<windowing::Frame>> WindowedApp::begin_frame(
    VkExtent2D current_extent) {
  if (state_ == nullptr) {
    return Status::invalid_argument("WindowedApp::begin_frame: empty app");
  }
  return state_->frame_loop.begin_frame(current_extent);
}

Status WindowedApp::end_frame(const windowing::Frame& frame) {
  if (state_ == nullptr) {
    return Status::invalid_argument("WindowedApp::end_frame: empty app");
  }
  return state_->frame_loop.end_frame(frame);
}

void WindowedApp::set_recreate_callback(
    std::function<Status(VkExtent2D)> callback) {
  if (state_ != nullptr) {
    state_->frame_loop.set_recreate_callback(std::move(callback));
  }
}

void WindowedApp::set_profiler(Profiler* profiler) noexcept {
  if (state_ != nullptr) {
    state_->frame_loop.set_profiler(profiler);
  }
}

Status WindowedApp::wait_idle() const {
  if (state_ == nullptr) {
    return Status::invalid_argument("WindowedApp::wait_idle: empty app");
  }
  // Queue-scoped (not device-wide): on a shared adopted device this must not
  // idle a sibling library's queues.
  return state_->device->wait_idle();
}

}  // namespace volumetric_kit::gfx::app

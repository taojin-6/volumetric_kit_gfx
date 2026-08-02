// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/app/windowed_app.hpp"

#include <memory>
#include <string>
#include <utility>

namespace volumetric_kit::gfx::app {

// Everything downstream of the device is identical on both bring-up paths, and
// each step borrows the previous one by address, so it lives here once rather
// than being written twice and drifting: surface (already in `state`) ->
// allocator -> swapchain -> frame loop, all on whatever device `state` holds.
Status WindowedApp::finish_bring_up(State& state,
                                    const WindowedAppConfig& config) {
  VG_ASSIGN(Allocator allocator,
            Allocator::create(state.instance_handle, *state.device));
  state.allocator.emplace(std::move(allocator));

  // A depth-configured swapchain allocates its per-image depth attachments
  // through the allocator; a color-only one takes none.
  Allocator* depth_allocator =
      config.swapchain.depth_format != VK_FORMAT_UNDEFINED ? &*state.allocator
                                                           : nullptr;
  VG_ASSIGN(windowing::Swapchain swapchain,
            windowing::Swapchain::create(*state.device, state.surface.handle(),
                                         config.swapchain, depth_allocator));
  state.swapchain = std::move(swapchain);

  VG_ASSIGN(windowing::FrameLoop frame_loop,
            windowing::FrameLoop::create(*state.device, state.swapchain,
                                         config.frames_in_flight));
  state.frame_loop = std::move(frame_loop);
  return {};
}

// Run the caller's factory and park the surface in `state`. Shared so both
// paths reject a null factory and a VK_NULL_HANDLE result identically.
Status WindowedApp::make_surface(State& state,
                                 const SurfaceFactory& create_surface,
                                 const char* who) {
  if (!create_surface) {
    return Status::invalid_argument(std::string(who) +
                                    ": create_surface must be callable");
  }
  VG_ASSIGN(VkSurfaceKHR raw_surface, create_surface(state.instance_handle));
  if (raw_surface == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        std::string(who) + ": the surface factory returned VK_NULL_HANDLE");
  }
  state.surface = windowing::Surface(state.instance_handle, raw_surface);
  return {};
}

Result<WindowedApp> WindowedApp::create(const WindowedAppConfig& config,
                                        const SurfaceFactory& create_surface) {
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
  state->instance_handle = state->instance->handle();

  VG_TRY(make_surface(*state, create_surface, "WindowedApp::create"));

  // One surface threads through selection, DeviceConfig::needs_present, and
  // Device::create, so the three stay consistent by construction.
  VG_ASSIGN(VkPhysicalDevice physical,
            state->instance->select_physical_device(state->surface.handle()));
  DeviceConfig device_config;
  device_config.needs_present = true;
  VG_ASSIGN(Device device,
            Device::create(state->instance_handle, physical, device_config,
                           state->surface.handle()));
  state->device.emplace(std::move(device));

  VG_TRY(finish_bring_up(*state, config));

  WindowedApp app;
  app.state_ = std::move(state);
  return app;
}

Result<WindowedApp> WindowedApp::adopt(const AdoptedDevice& adopted,
                                       const WindowedAppConfig& config,
                                       const SurfaceFactory& create_surface) {
  // A windowed app must present, so refuse a compute-only share here rather
  // than let it fail deeper as a missing present queue. Device::adopt checks
  // the rest (queues, extensions, features) against what the renderer needs.
  if (!adopted.has_present || adopted.present_queue == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "WindowedApp::adopt: adopted device must carry a present queue "
        "(set has_present and present_queue)");
  }
  if (adopted.instance == VK_NULL_HANDLE) {
    return Status::invalid_argument("WindowedApp::adopt: instance is null");
  }

  auto state = std::make_unique<State>();
  // The instance stays the embedder's: `state->instance` is left empty and
  // nothing here destroys it. Only the handle is recorded, which is all the
  // surface and the allocator need.
  state->instance_handle = adopted.instance;

  VG_TRY(make_surface(*state, create_surface, "WindowedApp::adopt"));

  // No physical-device selection: the embedder already chose one, and the
  // surface it must present to is the one just created from its instance.
  DeviceConfig device_config;
  device_config.needs_present = true;
  VG_ASSIGN(Device device, Device::adopt(adopted, device_config));
  state->device.emplace(std::move(device));

  VG_TRY(finish_bring_up(*state, config));

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

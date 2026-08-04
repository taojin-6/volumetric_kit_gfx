// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/app/windowed_app.hpp"

#include <memory>
#include <string>
#include <string_view>
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

// Bind the app to `instance` and run the caller's factory on it. Shared so
// both paths reject a null factory and a VK_NULL_HANDLE result identically,
// and so the instance is recorded in exactly one place. `state` is written
// only once the surface is in hand, so a rejected factory leaves it untouched.
Status WindowedApp::make_surface(State& state, VkInstance instance,
                                 const SurfaceFactory& create_surface,
                                 std::string_view who) {
  if (!create_surface) {
    return Status::invalid_argument(std::string(who) +
                                    ": create_surface must be callable");
  }
  VG_ASSIGN(VkSurfaceKHR raw_surface, create_surface(instance));
  if (raw_surface == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        std::string(who) + ": the surface factory returned VK_NULL_HANDLE");
  }
  state.instance_handle = instance;
  state.surface = windowing::Surface(instance, raw_surface);
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

  VG_TRY(make_surface(*state, state->instance->handle(), create_surface,
                      "WindowedApp::create"));

  // One surface threads through selection, DeviceConfig::needs_present, and
  // Device::create, so the three stay consistent by construction.
  VG_ASSIGN(VkPhysicalDevice physical,
            state->instance->select_physical_device(state->surface.handle()));
  DeviceConfig device_config = config.device;
  device_config.needs_present = true;
  // The device-level debug-utils table can only be loaded when the *instance*
  // enabled VK_EXT_debug_utils; without this every label and object name on the
  // resulting device is a silent no-op, so a capture of a validation-enabled
  // app would show no pass markers at all.
  device_config.enable_debug_utils = state->instance->debug_utils_enabled();
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
  // Vet the whole share before running the caller's factory, so a share that
  // was never going to work does not first cost the embedder a created-and-
  // immediately-destroyed window surface. Device::adopt repeats these (it is
  // callable on its own) and adds the checks that need the physical device --
  // queue families, extensions, features -- but those can only run later.
  if (adopted.instance == VK_NULL_HANDLE ||
      adopted.physical_device == VK_NULL_HANDLE ||
      adopted.device == VK_NULL_HANDLE ||
      adopted.graphics_queue == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "WindowedApp::adopt: instance, physical device, device, and graphics "
        "queue must be non-null");
  }
  // A windowed app must present, so refuse a compute-only share rather than
  // let it fail deeper as a missing present queue.
  if (!adopted.has_present || adopted.present_queue == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "WindowedApp::adopt: adopted device must carry a present queue "
        "(set has_present and present_queue)");
  }

  auto state = std::make_unique<State>();

  // The instance stays the embedder's: `state->instance` is left empty and
  // nothing here destroys it -- make_surface records only the handle, which is
  // all the surface and the allocator need.
  VG_TRY(make_surface(*state, adopted.instance, create_surface,
                      "WindowedApp::adopt"));

  // No physical-device selection: the embedder already chose one.
  DeviceConfig device_config = config.device;
  device_config.needs_present = true;
  // The instance is the embedder's, so its debug-utils state cannot be queried
  // here -- it is declared on the share instead.
  device_config.enable_debug_utils = adopted.enabled_debug_utils;
  VG_ASSIGN(Device device, Device::adopt(adopted, device_config));
  state->device.emplace(std::move(device));

  // create() gets this by construction -- it picks the present family *for*
  // this surface. Device::adopt never sees a surface, so nothing so far has
  // established that the embedder's present family can present to the one the
  // factory just returned: a device built before any window existed had to
  // pick that family blind. Ask now -- after Device::adopt has bounds-checked
  // the index, which the query itself requires -- rather than let the
  // swapchain trip VUID-VkSwapchainCreateInfoKHR-surface-271 (or, unvalidated,
  // sail into undefined behavior).
  VkBool32 present_supported = VK_FALSE;
  VG_VK_TRY(vkGetPhysicalDeviceSurfaceSupportKHR(
      adopted.physical_device, adopted.present_family, state->surface.handle(),
      &present_supported));
  if (present_supported != VK_TRUE) {
    return Status::unsupported(
        "WindowedApp::adopt: the adopted present queue family cannot present "
        "to the surface create_surface returned");
  }

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

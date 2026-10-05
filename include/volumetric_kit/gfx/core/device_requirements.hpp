// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device_requirements.hpp
/// @brief What the renderer requires of a device.
///
/// The device is volumetric_kit_core's (`core/vulkan/device.hpp`; DECISIONS.md,
/// 2026-10-04, "The device comes from volumetric_kit_core"). What gfx adds is
/// @ref device_requirements, the one statement of what the renderer needs,
/// which a standalone caller passes to `Instance::select_physical_device` and
/// `Device::create`, and an embedder merges with a compute library's
/// (`core::merge`) before building a shared device each library adopts.
///
/// @code
/// core::DeviceRequirements reqs = device_requirements();
/// reqs.needs_present = true;  // VK_KHR_swapchain and a present queue
/// VKC_ASSIGN(core::PhysicalDeviceInfo gpu,
///            instance.select_physical_device(reqs, surface));
/// VKC_ASSIGN(core::Device device,
///            core::Device::create(instance, gpu, reqs, surface));
/// @endcode

#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief The requirements the renderer brings to a device.
///
/// Vulkan 1.3 on a graphics queue, with `dynamicRendering` (every pass records
/// `vkCmdBeginRendering`) and `timelineSemaphore` (`core::TimelineSemaphore`).
/// Set `needs_present` for a swapchain -- the core then requires a present
/// queue for the surface and enables `VK_KHR_swapchain` -- and add extensions
/// or features a technique needs (`fillModeNonSolid` for a wireframe
/// pipeline, the external-memory fd extensions for CUDA interop).
/// @return The renderer's requirements.
inline core::DeviceRequirements device_requirements() {
  core::DeviceRequirements reqs;
  reqs.api_version = VK_API_VERSION_1_3;
  reqs.queue_flags = VK_QUEUE_GRAPHICS_BIT;
  reqs.timeline_semaphore = true;
  reqs.dynamic_rendering = true;
  return reqs;
}

}  // namespace volumetric_kit::gfx

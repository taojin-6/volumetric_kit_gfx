// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device.hpp
/// @brief gfx's names for volumetric_kit_core's logical device and its
///        requirements, and the requirements the renderer brings.
///
/// The device is the family's shared one (the core's
/// `volumetric_kit/core/vulkan/device.hpp`, DECISIONS.md, 2026-10-04, "The
/// device comes from volumetric_kit_core"): created from a
/// @ref PhysicalDeviceInfo and a set of @ref DeviceRequirements, or adopted
/// from an embedder that created it -- the seam a compute library such as recon
/// shares one `VkDevice` with the renderer through. A shared device is built
/// from the union of both libraries' requirements (`merge`), and each adopts it
/// with its own, which `Device::adopt` and `Device::check_enabled` hold it to.
///
/// @code
/// DeviceRequirements reqs = device_requirements();
/// reqs.needs_present = true;  // VK_KHR_swapchain and a present queue
/// VG_ASSIGN(PhysicalDeviceInfo gpu,
///           instance.select_physical_device(reqs, surface));
/// VG_ASSIGN(Device device, Device::create(instance, gpu, reqs, surface));
/// VG_TRY(device.submit_single_time(
///     [&](VkCommandBuffer cmd) { record(cmd); }));
/// @endcode

#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/gfx/core/physical_device_info.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

using core::AdoptedDevice;
using core::check_device_support;
using core::Device;
using core::DeviceRequirements;
using core::DeviceSupport;
using core::EnabledFeatures;
using core::merge;

/// @brief The requirements the renderer brings to a device.
///
/// Vulkan 1.3 on a graphics queue, with `dynamicRendering` (every pass records
/// `vkCmdBeginRendering`) and `timelineSemaphore` (@ref TimelineSemaphore).
/// Set `needs_present` for a swapchain -- the core then requires a present
/// queue for the surface and enables `VK_KHR_swapchain` -- and add extensions
/// or features a technique needs (`fillModeNonSolid` for a wireframe
/// pipeline, the external-memory fd extensions for CUDA interop).
/// @return The renderer's requirements.
inline DeviceRequirements device_requirements() {
  DeviceRequirements reqs;
  reqs.api_version = VK_API_VERSION_1_3;
  reqs.queue_flags = VK_QUEUE_GRAPHICS_BIT;
  reqs.timeline_semaphore = true;
  reqs.dynamic_rendering = true;
  return reqs;
}

}  // namespace volumetric_kit::gfx

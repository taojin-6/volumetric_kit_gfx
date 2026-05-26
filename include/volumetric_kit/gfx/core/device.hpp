// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device.hpp
/// The logical device built on a chosen physical device: queues, command pool,
/// and setup helpers. Holds no surface/swapchain (that is the windowing tier).

#include <functional>

#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/export.hpp"

namespace volumetric_kit::gfx {

struct DeviceConfig {
  /// Require (and enable VK_KHR_swapchain for) a present-capable queue. Needs a
  /// surface.
  bool needs_present = false;
  /// Enable the external-memory / -semaphore fd extensions — the CUDA interop
  /// seam.
  bool needs_external_memory = false;
  /// Core device features to enable.
  /// TODO: layer in the 1.3 feature structs (VkPhysicalDeviceVulkan13Features).
  VkPhysicalDeviceFeatures features = {};
};

/// Owns a `VkDevice`, its queues, and a command pool.
class VG_API Device {
 public:
  /// Create a logical device on `physical` (belonging to `instance`). Pass a
  /// `surface` when `config.needs_present` so a present-capable queue family
  /// can be chosen.
  ///
  /// Precondition: `instance` must outlive the returned `Device`. The device
  /// stores only handles, not ownership, so destroying the instance first is
  /// undefined behavior. Compose them so the instance is destroyed last — e.g.
  /// declare the instance before the device in an owning struct, so reverse
  /// member-destruction order tears the device down first.
  static Result<Device> create(VkInstance instance, VkPhysicalDevice physical,
                               const DeviceConfig& config,
                               VkSurfaceKHR surface = VK_NULL_HANDLE);

  ~Device();
  Device(Device&& other) noexcept;
  Device& operator=(Device&& other) noexcept;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  /// The owned logical device.
  VkDevice handle() const noexcept { return device_; }
  /// The physical device it was created on.
  VkPhysicalDevice physical_device() const noexcept { return physical_; }

  /// Index and queue of the graphics-capable family (always present).
  uint32_t graphics_family() const noexcept { return graphics_family_; }
  VkQueue graphics_queue() const noexcept { return graphics_queue_; }

  /// Whether a present-capable queue was created (i.e. config.needs_present).
  bool has_present() const noexcept { return present_queue_ != VK_NULL_HANDLE; }
  /// Present family and queue — valid only when has_present() is true.
  uint32_t present_family() const noexcept { return present_family_; }
  VkQueue present_queue() const noexcept { return present_queue_; }

  /// Command pool on the graphics family (RESET_COMMAND_BUFFER_BIT).
  VkCommandPool command_pool() const noexcept { return command_pool_; }

  /// Record + submit a one-shot command buffer on the graphics queue, blocking
  /// on a fence until it completes. For setup/uploads only — never the
  /// per-frame path.
  Status submit_single_time(
      const std::function<void(VkCommandBuffer)>& record) const;

 private:
  Device() = default;
  void destroy() noexcept;

  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;

  uint32_t graphics_family_ = 0;
  uint32_t present_family_ = 0;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  VkQueue present_queue_ = VK_NULL_HANDLE;
};

}  // namespace volumetric_kit::gfx

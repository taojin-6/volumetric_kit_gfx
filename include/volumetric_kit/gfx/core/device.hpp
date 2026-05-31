// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device.hpp
/// The logical device built on a chosen physical device: queues, command pool,
/// and setup helpers. Holds no surface/swapchain (that is the windowing tier).

#include <functional>
#include <vector>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/physical_device_info.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

struct DeviceConfig {
  /// Require (and enable VK_KHR_swapchain for) a present-capable queue. Needs a
  /// surface.
  bool needs_present = false;
  /// Enable the external-memory / -semaphore *fd* extensions — the CUDA interop
  /// seam on Linux. Win32 / other handle types are requested through
  /// @ref extra_device_extensions instead.
  bool needs_external_memory = false;
  /// Core (1.0) device features to enable (fed into
  /// `VkPhysicalDeviceFeatures2`).
  VkPhysicalDeviceFeatures features = {};
  /// Device extensions to enable beyond those implied by the flags above. Each
  /// is validated against the device's supported list; a missing one fails
  /// @ref create with @ref Status::Code::Unsupported. A name already implied by
  /// a flag (e.g. `VK_KHR_swapchain`) is de-duplicated, not passed twice.
  std::vector<const char*> extra_device_extensions;
  /// Optional caller-owned `pNext` chain of feature structs (e.g.
  /// `VkPhysicalDeviceVulkan13Features`, or any `*FeaturesKHR/EXT`) to enable.
  /// @ref create appends it to the tail of the `VkPhysicalDeviceFeatures2`
  /// chain it builds (after its own timeline-semaphore link). Query support
  /// first via
  /// @ref PhysicalDeviceInfo and request only features the device reports, or
  /// device creation fails. Each struct must set its `sType`; the pointed-to
  /// structs must outlive the @ref create call (they are consumed
  /// synchronously, not stored).
  const void* feature_chain = nullptr;
};

/// Owns a `VkDevice`, its queues, and a command pool.
class VG_CORE_API Device {
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

  /// Read-only capabilities of the physical device this was created on,
  /// captured at create() time (extensions/features/limits cached; format
  /// queries live).
  const PhysicalDeviceInfo& caps() const noexcept { return caps_; }

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
  PhysicalDeviceInfo caps_;
};

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device.hpp
/// @brief The logical device built on a chosen physical device: its queues,
///        command pool, and setup helpers.

#include <functional>
#include <vector>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/physical_device_info.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Parameters for @ref Device::create.
struct DeviceConfig {
  /// Require (and enable VK_KHR_swapchain for) a present-capable queue. Needs a
  /// surface.
  bool needs_present = false;
  /// Enable the external-memory / -semaphore *fd* extensions — the CUDA interop
  /// seam on Linux. Other POSIX-fd handle types are requested through
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
  /// synchronously, not stored). If the chain includes a
  /// `VkPhysicalDeviceVulkan12Features` (or a standalone
  /// `VkPhysicalDeviceTimelineSemaphoreFeatures`), @ref create enables
  /// `timelineSemaphore` within it rather than linking its own struct, so the
  /// chain never holds both the 1.2 aggregate and the individual struct (which
  /// Vulkan forbids).
  const void* feature_chain = nullptr;
};

/// @brief Owns a `VkDevice`, its graphics (and optional present) queues, and a
///        graphics command pool. Holds no surface/swapchain — that is the
///        windowing tier.
///
/// @warning The @ref Instance the device is created on must outlive it (see
///          @ref create): the device stores only borrowed handles.
///
/// @code
/// Result<Device> device = Device::create(instance.handle(), physical, {});
/// if (!device) return device.status();
/// VG_TRY(device.value().submit_single_time(
///     [&](VkCommandBuffer cmd) { record_uploads(cmd); }));
/// @endcode
class VG_CORE_API Device {
 public:
  /// @brief Create a logical device on @p physical (a Vulkan 1.3+ device).
  /// @param instance  The instance @p physical belongs to; it must outlive the
  ///                  returned device (kept as a lifetime contract, not used at
  ///                  creation).
  /// @param physical  The physical device to build on (must report Vulkan
  ///                  >= 1.3 and expose a graphics queue family).
  /// @param config    Queues, features, and extensions to enable.
  /// @param surface   Required when `config.needs_present`, to choose a
  ///                  present-capable queue family; otherwise ignored.
  /// @return The device on success, or a non-OK @ref Status: @ref
  ///         Status::Code::InvalidArgument for a null @p physical or a
  ///         `needs_present` without a @p surface; @ref
  ///         Status::Code::Unsupported when @p physical is below Vulkan 1.3,
  ///         lacks a required queue family, or is missing a requested
  ///         extension.
  /// @pre @p instance must outlive the returned `Device`; declare the instance
  ///      before the device so reverse member-destruction tears the device down
  ///      first.
  static Result<Device> create(VkInstance instance, VkPhysicalDevice physical,
                               const DeviceConfig& config,
                               VkSurfaceKHR surface = VK_NULL_HANDLE);

  ~Device();
  Device(Device&& other) noexcept;
  Device& operator=(Device&& other) noexcept;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  /// @return The owned logical device (`VK_NULL_HANDLE` when moved-from).
  VkDevice handle() const noexcept { return device_; }
  /// @return The physical device it was created on.
  VkPhysicalDevice physical_device() const noexcept { return physical_; }

  /// @return Read-only capabilities of the physical device, captured at
  ///         create() time (extensions/features/limits cached; format queries
  ///         live).
  const PhysicalDeviceInfo& caps() const noexcept { return caps_; }

  /// @return The graphics-capable queue family index (always present).
  uint32_t graphics_family() const noexcept { return graphics_family_; }
  /// @return The graphics queue.
  VkQueue graphics_queue() const noexcept { return graphics_queue_; }

  /// @brief The number of meaningful low-order bits in a timestamp written on
  ///        the graphics queue (the family's `timestampValidBits`), captured at
  ///        create().
  /// @return The valid-bit count, in `[0, 64]`. `0` means the graphics queue
  ///         does not support timestamps (MoltenVK may report this) — callers
  ///         must gate any GPU timing on a non-zero value rather than recording
  ///         `vkCmdWriteTimestamp` unconditionally. Convert the raw tick delta
  ///         to nanoseconds with `caps().limits().timestampPeriod` (the
  ///         ns-per-tick factor).
  uint32_t graphics_timestamp_valid_bits() const noexcept {
    return graphics_timestamp_valid_bits_;
  }

  /// @return Whether a present-capable queue was created
  /// (config.needs_present).
  bool has_present() const noexcept { return present_queue_ != VK_NULL_HANDLE; }
  /// @return The present queue family index — valid only when @ref has_present.
  uint32_t present_family() const noexcept { return present_family_; }
  /// @return The present queue — valid only when @ref has_present.
  VkQueue present_queue() const noexcept { return present_queue_; }

  /// @return The graphics-family command pool (created
  ///         `RESET_COMMAND_BUFFER_BIT`).
  VkCommandPool command_pool() const noexcept { return command_pool_; }

  /// @brief Record + submit a one-shot command buffer on the graphics queue,
  ///        blocking on a fence until it completes. For setup/uploads only —
  ///        never the per-frame path.
  /// @param record  Callback that records into the command buffer between an
  ///                implicit begin/end.
  /// @return OK once the work completes, or a non-OK @ref Status (recording,
  ///         submit, or wait failure).
  /// @note Not internally synchronized despite being `const`: it allocates from
  ///       and submits on the device's shared graphics pool/queue, which Vulkan
  ///       requires be externally synchronized. Serialize concurrent calls, or
  ///       give each thread its own @ref CommandPool.
  Status submit_single_time(
      const std::function<void(VkCommandBuffer)>& record) const;

 private:
  Device() = default;
  void destroy() noexcept;

  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;

  uint32_t graphics_family_ = 0;
  uint32_t graphics_timestamp_valid_bits_ = 0;
  uint32_t present_family_ = 0;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  VkQueue present_queue_ = VK_NULL_HANDLE;
  PhysicalDeviceInfo caps_;
};

}  // namespace volumetric_kit::gfx

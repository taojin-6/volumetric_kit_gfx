// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device.hpp
/// @brief The logical device built on a chosen physical device: its queues,
///        command pool, and setup helpers.

#include <functional>
#include <mutex>
#include <vector>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/impl/debug_utils_table.hpp"
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
  /// Resolve the `VK_EXT_debug_utils` device entry points so @ref
  /// Device::debug_utils returns an active table. Set this only from
  /// @ref Instance::debug_utils_enabled: the device-level labels can only be
  /// emitted when the instance enabled the extension, and forcing this true
  /// otherwise is unsupported (the entry points may resolve to non-null but
  /// invalid trampolines). Left false, the table is inactive and every
  /// label/object-name call becomes a no-op.
  bool enable_debug_utils = false;
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

/// @brief What the renderer needs from a `VkDevice`, published so an embedder
///        that shares one device across several libraries can merge everyone's
///        requirements, create a device satisfying the union, and hand it to
///        each library via @ref Device::adopt.
///
/// Derived from a @ref DeviceConfig by @ref Device::requirements. Expressed in
/// raw Vulkan data so the bundle carries no type a sibling library must import.
struct DeviceRequirements {
  /// Minimum device Vulkan version (the renderer targets 1.3 core).
  uint32_t api_version = VK_API_VERSION_1_3;
  /// Queue capabilities at least one assigned queue must carry.
  VkQueueFlags queue_flags = VK_QUEUE_GRAPHICS_BIT;
  /// Also needs a present-capable queue (implies `VK_KHR_swapchain`).
  bool needs_present = false;
  /// Device extensions to enable (swapchain / external-memory / caller extras).
  std::vector<const char*> device_extensions;
  /// Core (1.0) features to enable.
  VkPhysicalDeviceFeatures features = {};
  /// `timelineSemaphore` (1.2 core) — the renderer's sync primitive.
  bool timeline_semaphore = true;
  /// `dynamicRendering` (1.3 core) — required by `RenderTarget`.
  bool dynamic_rendering = true;
};

/// @brief A `VkDevice` the caller already created, plus a description of what
///        the caller ENABLED on it, handed to @ref Device::adopt.
///
/// The `enabled_*` fields exist because Vulkan gives no way to query which
/// extensions/features were enabled at device-creation time; the creator must
/// declare them so @ref Device::adopt can verify by set-comparison. The arrays
/// and structs it points to need only outlive the `adopt` call.
struct AdoptedDevice {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  /// The graphics-capable queue assigned to the renderer, and its family.
  uint32_t graphics_family = 0;
  VkQueue graphics_queue = VK_NULL_HANDLE;
  /// The present queue/family — required when `DeviceConfig::needs_present`.
  bool has_present = false;
  uint32_t present_family = 0;
  VkQueue present_queue = VK_NULL_HANDLE;
  /// When non-null, the assigned queue is shared with another library; every
  /// `vkQueueSubmit` on it must hold this mutex (Vulkan requires queue submits
  /// be externally synchronized).
  std::mutex* submit_mutex = nullptr;

  /// What the creator enabled on `device` (for `adopt`'s set-comparison
  /// verify).
  const char* const* enabled_device_extensions = nullptr;
  uint32_t enabled_device_extension_count = 0;
};

/// @brief Owns *or borrows* a `VkDevice`, its graphics (and optional present)
///        queues, and a graphics command pool. Holds no surface/swapchain —
///        that is the windowing tier.
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

  /// @brief Adopt a `VkDevice` an embedder already created, **without owning
  ///        it** — the returned device's destructor leaves the `VkDevice`
  ///        alone (it still creates and owns its own command pool). Use this to
  ///        run the renderer on a device shared with a sibling library.
  /// @param adopted  The existing handles, the queue assigned to the renderer,
  ///                 and what the creator enabled on the device.
  /// @param config   The same config the renderer would pass to @ref create;
  ///                 its needs are validated against @p adopted.
  /// @return The (non-owning) device, or a non-OK @ref Status:
  ///         @ref Status::Code::InvalidArgument for null handles or a
  ///         `needs_present` config without a present queue; @ref
  ///         Status::Code::Unsupported when @p adopted is below Vulkan 1.3, its
  ///         assigned queue family lacks the required capabilities, or a
  ///         required extension/feature was not enabled on it.
  /// @pre The instance/device in @p adopted must outlive the returned `Device`.
  static Result<Device> adopt(const AdoptedDevice& adopted,
                              const DeviceConfig& config);

  /// @brief The device requirements implied by @p config — the set an embedder
  ///        merges with other libraries' to build one shared device.
  /// @param config  The renderer's device configuration.
  /// @return The extensions, features, queue capabilities, and API version the
  ///         renderer needs enabled.
  static DeviceRequirements requirements(const DeviceConfig& config);

  ~Device();
  Device(Device&& other) noexcept;
  Device& operator=(Device&& other) noexcept;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  /// @return The owned logical device (`VK_NULL_HANDLE` when moved-from).
  VkDevice handle() const noexcept { return device_; }
  /// @return The physical device it was created on.
  VkPhysicalDevice physical_device() const noexcept { return physical_; }

  /// @return Whether this wrapper owns (and will destroy) the `VkDevice`.
  ///         `false` for a device obtained through @ref adopt.
  bool owns_device() const noexcept { return owns_device_; }

  /// @return The mutex guarding submits on a shared queue, or `nullptr` when
  /// the
  ///         queue is exclusively this device's. Submit sites that record on
  ///         @ref graphics_queue must hold it when non-null.
  std::mutex* submit_mutex() const noexcept { return submit_mutex_; }

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

  /// @brief The resolved `VK_EXT_debug_utils` entry points for this device.
  /// @return The cached table. It is active (and labels/object names emit) only
  ///         when `config.enable_debug_utils` was set and the extension is
  ///         enabled on the instance; otherwise it is inactive and every
  ///         @ref DebugLabelScope / @ref QueueLabelScope / @ref set_object_name
  ///         built from it is a no-op.
  const DebugUtilsTable& debug_utils() const noexcept { return debug_utils_; }

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

  /// @brief Submit an already-recorded, ended command buffer on the graphics
  ///        queue and block on a throwaway fence until it completes.
  ///
  /// The command-buffer-owning tail of @ref submit_single_time, exposed so a
  /// caller that records incrementally (e.g. @ref UploadBatch) shares
  /// one submit+fence+wait implementation. Ownership of @p cmd stays with the
  /// caller; it must be in the executable (ended) state.
  /// @param cmd  A recorded, ended command buffer.
  /// @return OK once the work completes, or a non-OK @ref Status (submit or
  ///         wait failure).
  /// @note Same external-synchronization caveat as @ref submit_single_time.
  Status submit_and_wait(VkCommandBuffer cmd) const;

 private:
  Device() = default;
  void destroy() noexcept;

  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  // False when the device was adopted (@ref adopt): destroy() then tears down
  // only the command pool this wrapper made and leaves the VkDevice to its
  // owner. Borrowed, not owned; nulled/reset on every ownership transfer.
  bool owns_device_ = true;
  std::mutex* submit_mutex_ = nullptr;

  uint32_t graphics_family_ = 0;
  uint32_t graphics_timestamp_valid_bits_ = 0;
  uint32_t present_family_ = 0;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  VkQueue present_queue_ = VK_NULL_HANDLE;
  PhysicalDeviceInfo caps_;
  // Plain resolved PFNs (trivially copyable): the move ctor / assignment /
  // reset below copy and clear it alongside the handles, with no special
  // handling.
  DebugUtilsTable debug_utils_;
};

}  // namespace volumetric_kit::gfx

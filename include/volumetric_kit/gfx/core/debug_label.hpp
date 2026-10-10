// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file debug_label.hpp
/// @brief The cross-tool GPU capture-label scopes: `VK_EXT_debug_utils`
///        command-buffer and queue regions, opened and closed by RAII.
///
/// One emitter serves every standard capture tool — RenderDoc, Nsight Graphics,
/// Nsight Systems, and Xcode's Metal frame debugger all consume the same
/// `VK_EXT_debug_utils` labels, so there is no per-tool code.
/// @ref DebugLabelScope nests a region inside a command buffer (where most
/// captures group draws); @ref QueueLabelScope nests a region on a queue's
/// submit timeline (where Nsight Systems shows it). Name objects with the
/// core's `Device::set_object_name`.
///
/// Both scopes emit only where the device's instance enabled the extension
/// (the core's `Device::debug_labels_available`); elsewhere they are inert —
/// no labels are emitted and no error is raised. Labels carry no color, as the
/// core's do not.

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/export.hpp"

namespace volumetric_kit::core {
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx {

/// @brief A nested debug-label region inside a command buffer, opened on
///        construction and closed on destruction.
///
/// The region groups the commands recorded between construction and destruction
/// under @p name in a GPU capture. It records through the core's
/// `Device::begin_debug_label` / `end_debug_label`. An inert
/// (default-constructed) scope emits nothing.
///
/// @warning The command buffer must stay in the recording state for the scope's
///          whole lifetime: the destructor records
///          `vkCmdEndDebugUtilsLabelEXT` into it. Destroy the scope before
///          ending the command buffer, and keep the device alive, unmoved, for
///          the scope's lifetime.
///
/// @code
/// {
///   DebugLabelScope pass(device, cmd, "shadow pass");
///   record_shadow_draws(cmd);
/// }  // region ends here
/// @endcode
class VG_CORE_API DebugLabelScope {
 public:
  /// @brief An inert scope: emits nothing and ends nothing.
  DebugLabelScope() noexcept = default;

  /// @brief Open a labelled region in @p cmd, where @p device has debug
  ///        labels.
  /// @param device  The device @p cmd was allocated from.
  /// @param cmd     The recording command buffer to mark up.
  /// @param name    The region label shown in the capture. It must outlive the
  ///                scope, whose end hands it back to the core's
  ///                `Device::end_debug_label`. A null name makes the scope
  ///                inert — Vulkan requires a non-null label name.
  DebugLabelScope(const core::Device& device, VkCommandBuffer cmd,
                  const char* name) noexcept;

  ~DebugLabelScope();
  DebugLabelScope(DebugLabelScope&& other) noexcept;
  DebugLabelScope& operator=(DebugLabelScope&& other) noexcept;
  DebugLabelScope(const DebugLabelScope&) = delete;
  DebugLabelScope& operator=(const DebugLabelScope&) = delete;

  /// @return Whether this scope has an open region it will end on destruction.
  bool active() const noexcept { return device_ != nullptr; }

 private:
  void close() noexcept;

  const core::Device* device_ = nullptr;  // borrowed; null when inert
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  const char* name_ = nullptr;
};

/// @brief A nested debug-label region on a device's queue timeline, opened on
///        construction and closed on destruction.
///
/// The queue counterpart of @ref DebugLabelScope: it marks up the device's
/// queue itself rather than a command buffer, so the region appears on the
/// submit timeline (where Nsight Systems shows queue work) rather than inside
/// a captured frame. Vulkan requires the queue be externally synchronized for
/// both label calls, so each holds the device's `submit_mutex` — the mutex the
/// core's submits, and another library sharing the queue, hold. An inert
/// scope emits nothing.
///
/// The core's device does not resolve the queue-label entry points, so each
/// scope looks up its two with `vkGetDeviceProcAddr`.
///
/// @warning Keep the device alive, unmoved, for the scope's lifetime: the
///          destructor records `vkQueueEndDebugUtilsLabelEXT` on its queue,
///          under its mutex. Do not open or close a scope while holding that
///          mutex; it is not recursive.
///
/// @code
/// {
///   QueueLabelScope frame(device, "frame 42");
///   device.queue_submit(1, &submit, fence);
/// }  // region ends here
/// @endcode
class VG_CORE_API QueueLabelScope {
 public:
  /// @brief An inert scope: emits nothing and ends nothing.
  QueueLabelScope() noexcept = default;

  /// @brief Open a labelled region on @p device's queue, where @p device has
  ///        debug labels.
  /// @param device  The device whose queue (`Device::queue`) to mark up.
  /// @param name    The region label shown in the capture (the driver copies
  ///                it). A null name makes the scope inert — Vulkan requires a
  ///                non-null label name.
  QueueLabelScope(const core::Device& device, const char* name) noexcept;

  ~QueueLabelScope();
  QueueLabelScope(QueueLabelScope&& other) noexcept;
  QueueLabelScope& operator=(QueueLabelScope&& other) noexcept;
  QueueLabelScope(const QueueLabelScope&) = delete;
  QueueLabelScope& operator=(const QueueLabelScope&) = delete;

  /// @return Whether this scope has an open region it will end on destruction.
  bool active() const noexcept { return end_ != nullptr; }

 private:
  void close() noexcept;

  const core::Device* device_ = nullptr;  // borrowed; null when inert
  PFN_vkQueueEndDebugUtilsLabelEXT end_ = nullptr;
};

}  // namespace volumetric_kit::gfx

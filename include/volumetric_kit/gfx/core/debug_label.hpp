// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file debug_label.hpp
/// @brief The cross-tool GPU capture-label surface: `VK_EXT_debug_utils`
///        command/queue label scopes and object naming.
///
/// One emitter serves every standard capture tool — RenderDoc, Nsight Graphics,
/// Nsight Systems, and Xcode's Metal frame debugger all consume the same
/// `VK_EXT_debug_utils` labels and object names, so there is no per-tool code.
/// @ref DebugLabelScope nests a region inside a command buffer (where most
/// captures group draws); @ref QueueLabelScope nests a region on a queue's
/// submit timeline (where Nsight Systems shows it); @ref set_object_name gives
/// a handle a readable name in a capture.
///
/// Every entry point routes through a @ref DebugUtilsTable. When the extension
/// is not enabled the table is inactive, and every operation here compiles to a
/// branch-to-noop — no labels are emitted and no error is raised.

#include <cstdint>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/impl/debug_utils_table.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief A nested debug-label region inside a command buffer, opened on
///        construction and closed on destruction.
///
/// The region groups the commands recorded between construction and destruction
/// under @p name in a GPU capture. An inert (default-constructed) scope, or one
/// built against an inactive @ref DebugUtilsTable, emits nothing.
///
/// @warning The command buffer must stay in the recording state for the scope's
///          whole lifetime: the destructor records
///          `vkCmdEndDebugUtilsLabelEXT` into it. Destroy the scope before
///          ending the command buffer, and keep the @ref DebugUtilsTable
///          (i.e. the device it came from) alive for the scope's lifetime.
///
/// @code
/// {
///   DebugLabelScope pass(cmd, device.debug_utils(), "shadow pass");
///   record_shadow_draws(cmd);
/// }  // region ends here
/// @endcode
class VG_CORE_API DebugLabelScope {
 public:
  /// @brief An inert scope: emits nothing and ends nothing.
  DebugLabelScope() noexcept = default;

  /// @brief Open a labelled region in @p cmd (when @p table is active).
  /// @param cmd    The recording command buffer to mark up.
  /// @param table  The resolved entry points; an inactive table makes this a
  ///               no-op.
  /// @param name   The region label shown in the capture (must outlive only
  ///               this call — the driver copies it). A null name makes the
  ///               scope inert — Vulkan requires a non-null label name.
  /// @param color  Optional RGBA tint in `[0, 1]` as a 4-float array; a null
  ///               pointer leaves the color unset.
  DebugLabelScope(VkCommandBuffer cmd, const DebugUtilsTable& table,
                  const char* name, const float color[4] = nullptr) noexcept;

  ~DebugLabelScope();
  DebugLabelScope(DebugLabelScope&& other) noexcept;
  DebugLabelScope& operator=(DebugLabelScope&& other) noexcept;
  DebugLabelScope(const DebugLabelScope&) = delete;
  DebugLabelScope& operator=(const DebugLabelScope&) = delete;

  /// @return Whether this scope has an open region it will end on destruction.
  bool active() const noexcept { return end_ != nullptr; }

 private:
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  PFN_vkCmdEndDebugUtilsLabelEXT end_ = nullptr;
};

/// @brief A nested debug-label region on a queue's submit timeline, opened on
///        construction and closed on destruction.
///
/// The queue counterpart of @ref DebugLabelScope: it marks up the queue itself
/// rather than a command buffer, so the region appears on the submit timeline
/// (where Nsight Systems shows queue work) rather than inside a captured frame.
/// An inert scope, or one built against an inactive @ref DebugUtilsTable, emits
/// nothing.
///
/// @warning Keep the @ref DebugUtilsTable (the device it came from) alive for
///          the scope's lifetime; the destructor records
///          `vkQueueEndDebugUtilsLabelEXT` on the queue.
///
/// @code
/// {
///   QueueLabelScope frame(device.graphics_queue(), device.debug_utils(),
///                         "frame 42");
///   vkQueueSubmit(...);
/// }  // region ends here
/// @endcode
class VG_CORE_API QueueLabelScope {
 public:
  /// @brief An inert scope: emits nothing and ends nothing.
  QueueLabelScope() noexcept = default;

  /// @brief Open a labelled region on @p queue (when @p table is active).
  /// @param queue  The queue whose timeline to mark up.
  /// @param table  The resolved entry points; an inactive table makes this a
  ///               no-op.
  /// @param name   The region label shown in the capture (the driver copies
  ///               it). A null name makes the scope inert — Vulkan requires a
  ///               non-null label name.
  /// @param color  Optional RGBA tint in `[0, 1]` as a 4-float array; a null
  ///               pointer leaves the color unset.
  QueueLabelScope(VkQueue queue, const DebugUtilsTable& table, const char* name,
                  const float color[4] = nullptr) noexcept;

  ~QueueLabelScope();
  QueueLabelScope(QueueLabelScope&& other) noexcept;
  QueueLabelScope& operator=(QueueLabelScope&& other) noexcept;
  QueueLabelScope(const QueueLabelScope&) = delete;
  QueueLabelScope& operator=(const QueueLabelScope&) = delete;

  /// @return Whether this scope has an open region it will end on destruction.
  bool active() const noexcept { return end_ != nullptr; }

 private:
  VkQueue queue_ = VK_NULL_HANDLE;
  PFN_vkQueueEndDebugUtilsLabelEXT end_ = nullptr;
};

/// @brief Give a Vulkan object a human-readable name in GPU captures.
/// @param device  The device that owns @p handle.
/// @param table   The resolved entry points; an inactive table makes this a
///                no-op.
/// @param type    The object's `VkObjectType` (e.g. `VK_OBJECT_TYPE_IMAGE`).
/// @param handle  The object handle, reinterpreted to `uint64_t`. A
///                `VK_NULL_HANDLE` (0) handle is ignored.
/// @param name    The name to attach (the driver copies it).
/// @note A no-op when @p table is inactive or @p handle is null; the named
///       object then simply shows its address in a capture instead.
VG_CORE_API void set_object_name(VkDevice device, const DebugUtilsTable& table,
                                 VkObjectType type, uint64_t handle,
                                 const char* name);

}  // namespace volumetric_kit::gfx

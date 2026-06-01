// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file command_pool.hpp
/// @brief A `VkCommandPool` on one queue family; allocates command buffers.

#include <cstdint>

#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Owns a `VkCommandPool` on one queue family and allocates command
///        buffers from it.
///
/// Vulkan requires a command pool — and the buffers allocated from it — to be
/// externally synchronized, so give each recording thread its own
/// `CommandPool` rather than sharing one. Created against a `VkDevice` (which
/// must outlive the pool); the pool in turn must outlive every @ref
/// CommandBuffer it allocates.
///
/// @code
/// Result<CommandPool> pool =
///     CommandPool::create(device.handle(), device.graphics_family());
/// if (!pool) return pool.status();
/// Result<CommandBuffer> cmd = pool.value().allocate_primary();
/// @endcode
class VG_CORE_API CommandPool {
 public:
  /// @brief Create a command pool on @p queue_family.
  /// @param device        The logical device that owns the pool.
  /// @param queue_family  The queue family buffers from this pool submit on.
  /// @param flags         Pool create flags; the default makes each buffer
  ///                      individually resettable (so it can be re-recorded
  ///                      each frame).
  /// @return The pool on success, or a non-OK @ref Status.
  static Result<CommandPool> create(
      VkDevice device, uint32_t queue_family,
      VkCommandPoolCreateFlags flags =
          VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  ~CommandPool() = default;
  CommandPool(CommandPool&& other) noexcept;
  CommandPool& operator=(CommandPool&& other) noexcept;
  CommandPool(const CommandPool&) = delete;
  CommandPool& operator=(const CommandPool&) = delete;

  /// @return The underlying `VkCommandPool` (`VK_NULL_HANDLE` when empty).
  VkCommandPool handle() const noexcept { return pool_.get(); }

  /// @return The queue family the pool was created on (valid when @ref valid).
  uint32_t queue_family() const noexcept { return queue_family_; }

  /// @return `true` if this owns a pool.
  bool valid() const noexcept { return pool_.valid(); }

  /// @brief Allocate one primary command buffer from this pool.
  /// @return The command buffer on success, or a non-OK @ref Status. It frees
  ///         itself back to this pool, which must outlive it.
  /// @note Not thread-safe: allocating from (and recording buffers of) one pool
  ///       must be externally synchronized.
  Result<CommandBuffer> allocate_primary();

 private:
  CommandPool() = default;

  UniqueHandle<VkCommandPool, vkDestroyCommandPool> pool_;
  uint32_t queue_family_ = 0;
};

}  // namespace volumetric_kit::gfx

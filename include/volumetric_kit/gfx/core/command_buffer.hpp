// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file command_buffer.hpp
/// @brief A primary `VkCommandBuffer` owned against the pool it was allocated
///        from.

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Owns a primary `VkCommandBuffer` and frees it back to its pool.
///
/// Produced by @ref CommandPool::allocate_primary; rarely constructed directly.
/// This is the recording handle passed to a pass's `record(cmd, …)` and the
/// surface imgui / overlay code records into. Record by passing @ref handle to
/// the `vkCmd*` calls between @ref begin and @ref end. A default-constructed
/// `CommandBuffer` is empty (`valid()` is false) and safe to move-assign into.
///
/// @warning The producing @ref CommandPool (and the device it belongs to) must
///          outlive this buffer: destruction frees it back through that pool,
///          so destroying — or move-assigning over — the pool while a buffer it
///          produced is still alive is undefined behavior. Not thread-safe with
///          respect to its pool: allocating, freeing, and recording buffers
///          from one pool must be externally synchronized (a Vulkan
///          requirement) — use one @ref CommandPool per recording thread.
///
/// @code
/// Result<CommandBuffer> cmd = pool.allocate_primary();
/// if (!cmd) return cmd.status();
/// VG_TRY(cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));
/// vkCmdFillBuffer(cmd.value().handle(), buffer, 0, VK_WHOLE_SIZE, 0u);
/// VG_TRY(cmd.value().end());
/// // ... submit cmd.value().handle() on a queue of the pool's family ...
/// @endcode
class VG_CORE_API CommandBuffer {
 public:
  /// @brief Construct an empty buffer (owns nothing; `valid()` is false).
  CommandBuffer() noexcept = default;

  /// @brief Adopt @p command_buffer, freed back to @p pool on @p device.
  ///        Produced by @ref CommandPool::allocate_primary.
  /// @param device          The device @p pool belongs to.
  /// @param pool            The pool @p command_buffer was allocated from.
  /// @param command_buffer  The command buffer to take ownership of.
  CommandBuffer(VkDevice device, VkCommandPool pool,
                VkCommandBuffer command_buffer) noexcept;

  ~CommandBuffer();
  CommandBuffer(CommandBuffer&& other) noexcept;
  CommandBuffer& operator=(CommandBuffer&& other) noexcept;
  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;

  /// @return The underlying `VkCommandBuffer` (`VK_NULL_HANDLE` when empty).
  VkCommandBuffer handle() const noexcept { return command_buffer_; }

  /// @return `true` if this owns a command buffer.
  bool valid() const noexcept { return command_buffer_ != VK_NULL_HANDLE; }

  /// @brief Begin recording. With a pool created `RESET_COMMAND_BUFFER_BIT`,
  ///        this implicitly resets the buffer, so it may be re-recorded each
  ///        frame.
  /// @param flags  `VkCommandBufferUsageFlags` (e.g.
  ///               `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT`).
  /// @return OK on success, or a non-OK @ref Status.
  Status begin(VkCommandBufferUsageFlags flags = 0);

  /// @brief Finish recording.
  /// @return OK on success, or a non-OK @ref Status.
  Status end();

 private:
  void destroy() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
};

}  // namespace volumetric_kit::gfx

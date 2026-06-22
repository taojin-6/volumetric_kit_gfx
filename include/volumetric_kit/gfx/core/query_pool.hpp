// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file query_pool.hpp
/// @brief RAII wrapper for a `VkQueryPool` of timestamp queries — the GPU-side
///        clock the renderer reads to time submitted work.

#include <cstdint>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Owns a `VkQueryPool` and the operations to reset it, write timestamps
///        into it, and read the raw 64-bit tick values back.
///
/// A timestamp query records the GPU clock when a recorded
/// `vkCmdWriteTimestamp` drains to the given pipeline stage. The elapsed time
/// between two ticks is
/// `(end - begin) * VkPhysicalDeviceLimits::timestampPeriod` nanoseconds. A
/// queue family advertises how many low bits of each tick are meaningful
/// through `VkQueueFamilyProperties::timestampValidBits`; where that is zero,
/// timestamps are unsupported on that family and the read-back values carry no
/// signal — check it before trusting a measurement.
///
/// @warning The @p device passed to @ref create must outlive the pool: the
///          destructor frees through it, so destroying the device first is
///          undefined behavior. For destruction gated on in-flight GPU work,
///          hand the pool to @ref RetireQueue instead of letting it fall out of
///          scope while a submission still references it.
///
/// @code
/// Result<QueryPool> pool = QueryPool::create(device, /*query_count=*/2);
/// if (!pool) return pool.status();
/// // Record into a command buffer:
/// pool.value().cmd_reset(cmd, 0, 2);
/// pool.value().cmd_write_timestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0);
/// // ... work ...
/// pool.value().cmd_write_timestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
///                                  1);
/// // After the submission retires:
/// uint64_t ticks[2] = {};
/// if (pool.value().read_results(0, 2, ticks).ok()) { /* ticks[1] - ticks[0] */
/// }
/// @endcode
class VG_CORE_API QueryPool {
 public:
  /// @brief Create a query pool of @p query_count queries.
  /// @param device       The logical device that owns the pool.
  /// @param query_count  Number of queries in the pool; must be non-zero.
  /// @param type         The query type (default: `VK_QUERY_TYPE_TIMESTAMP`).
  /// @return The pool on success, or a non-OK @ref Status:
  ///         @ref Status::Code::InvalidArgument for a null @p device or a zero
  ///         @p query_count, or a Vulkan failure from `vkCreateQueryPool`.
  static Result<QueryPool> create(VkDevice device, uint32_t query_count,
                                  VkQueryType type = VK_QUERY_TYPE_TIMESTAMP);

  QueryPool() = default;
  ~QueryPool() = default;
  QueryPool(QueryPool&& other) noexcept;
  QueryPool& operator=(QueryPool&& other) noexcept;
  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;

  /// @return The underlying `VkQueryPool` handle (`VK_NULL_HANDLE` when empty).
  VkQueryPool handle() const noexcept { return handle_.get(); }

  /// @return The number of queries in the pool (`0` when empty).
  uint32_t query_count() const noexcept { return query_count_; }

  /// @return `true` if this owns a pool.
  bool valid() const noexcept { return handle_.valid(); }

  /// @brief Record `vkCmdResetQueryPool` to return @p count queries to the
  ///        unavailable state.
  /// @param cmd    The command buffer to record into (in the recording state).
  /// @param first  The first query to reset.
  /// @param count  How many queries to reset.
  /// @pre A query must be reset (outside a render pass) before it is written;
  ///      reading or writing a query that was never reset is undefined.
  void cmd_reset(VkCommandBuffer cmd, uint32_t first,
                 uint32_t count) const noexcept;

  /// @brief Record `vkCmdWriteTimestamp` to write the GPU clock into query
  ///        @p index once @p stage drains.
  /// @param cmd    The command buffer to record into (in the recording state).
  /// @param stage  The pipeline stage whose completion the timestamp captures.
  /// @param index  The query index to write (must be `< query_count()`).
  void cmd_write_timestamp(VkCommandBuffer cmd, VkPipelineStageFlagBits stage,
                           uint32_t index) const noexcept;

  /// @brief Copy @p count raw 64-bit tick values, starting at @p first, into
  ///        @p out via `vkGetQueryPoolResults`.
  /// @param first  The first query to read.
  /// @param count  How many queries to read (@p out must hold @p count values).
  /// @param out    Destination for @p count `uint64_t` ticks.
  /// @return OK once every requested result is available; a non-OK @ref Status
  ///         carrying `VK_NOT_READY` if any is not yet — this does not wait
  ///         (no `VK_QUERY_RESULT_WAIT_BIT`), so call it only after the
  ///         submission that wrote the timestamps has retired.
  Status read_results(uint32_t first, uint32_t count, uint64_t* out) const;

 private:
  UniqueHandle<VkQueryPool, vkDestroyQueryPool> handle_;
  uint32_t query_count_ = 0;
};

}  // namespace volumetric_kit::gfx

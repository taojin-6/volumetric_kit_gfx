// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file retire_queue.hpp
/// @brief Deferred destruction of GPU resources, keyed on `VkFence` completion.

#include <cstddef>
#include <functional>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/retire_list.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Runs deferred deleters once the GPU fence guarding each resource
/// signals.
///
/// When a resource may still be read by in-flight GPU work, it cannot be freed
/// at the call site. Enqueue its deleter against the `VkFence` that will be
/// signaled when that work finishes, then call @ref poll each frame to release
/// the entries whose fence has signaled. This realizes the locked
/// "double-buffer + release via a `VkFence`-completion CPU token" rule (see
/// `CLAUDE.md`) without any cross-API GPU event. Fences are observed, not
/// owned. The GPU-independent bookkeeping lives in @ref RetireList.
///
/// @code
/// RetireQueue retire(device);
/// // Defer freeing `buffer` until `frame_fence` signals:
/// retire.push(frame_fence, [buf = buffer]() { destroy(buf); });
/// retire.poll();  // once per frame
/// @endcode
class VG_CORE_API RetireQueue {
 public:
  /// @brief Construct a queue that observes fences belonging to @p device.
  /// @param device  The logical device whose fences gate the deleters.
  explicit RetireQueue(VkDevice device) noexcept;

  /// @brief Runs every still-pending deleter, then destroys the queue.
  /// @pre The device is idle (e.g. after `vkDeviceWaitIdle` or @ref drain), so
  /// the
  ///      remaining resources are safe to release without waiting on their
  ///      fences.
  ~RetireQueue();

  /// @brief Take over @p other's pending deleters; @p other is left empty.
  RetireQueue(RetireQueue&& other) noexcept;
  /// @brief Run this queue's still-pending deleters, then take over @p other's.
  /// @pre As at destruction, the device is idle, so the resources this queue
  /// still
  ///      holds are safe to release without waiting on their fences.
  RetireQueue& operator=(RetireQueue&& other) noexcept;
  RetireQueue(const RetireQueue&) = delete;
  RetireQueue& operator=(const RetireQueue&) = delete;

  /// @brief Defer @p deleter until @p fence is signaled.
  /// @param fence    A fence (observed, not owned) signaled when the GPU work
  /// that last
  ///                 touched the resource completes. Must stay valid until the
  ///                 deleter runs (via @ref poll, @ref drain, or destruction).
  /// @param deleter  Invoked exactly once, from @ref poll, @ref drain, or
  /// destruction.
  ///                 Must not throw: it may run from a `noexcept` context
  ///                 (destruction or move-assignment), where an escaping
  ///                 exception calls `std::terminate`.
  /// @note Not thread-safe: serialize @ref push against @ref poll / @ref drain.
  void push(VkFence fence, std::function<void()> deleter);

  /// @brief Run the deleters whose fence is currently signaled; keep the rest.
  /// @return The number of deleters run.
  /// @note Not thread-safe: serialize @ref push against @ref poll / @ref drain.
  std::size_t poll();

  /// @brief Wait for every pending fence to signal, then run all deleters.
  /// @note Not thread-safe: serialize @ref push against @ref poll / @ref drain.
  void drain();

  /// @return The number of deleters still pending.
  std::size_t pending() const noexcept { return list_.pending(); }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  RetireList<VkFence> list_;
};

}  // namespace volumetric_kit::gfx

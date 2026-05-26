// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sync.hpp
/// @brief RAII wrappers for the Vulkan synchronization primitives the renderer
///        waits on.
///
/// @ref Fence is host-side — the CPU blocks until submitted GPU work completes.
/// @ref Semaphore is a binary GPU-queue-to-queue primitive. @ref
/// TimelineSemaphore is a counter-based primitive the host and GPU can both
/// signal and wait on (Vulkan 1.2 core; @ref Device::create enables the feature
/// when supported).

#include <cstdint>

#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/export.hpp"

namespace volumetric_kit::gfx {

/// @brief A `VkFence`: the host waits on it for submitted GPU work to finish.
///
/// @code
/// Result<Fence> fence = Fence::create(device);
/// if (!fence) return fence.status();
/// // ... submit work that signals fence.value().handle() ...
/// fence.value().wait();
/// @endcode
class VG_API Fence {
 public:
  /// @brief Create a fence.
  /// @param device    The logical device that owns the fence.
  /// @param signaled  Whether the fence begins in the signaled state.
  /// @return The fence on success, or a non-OK @ref Status.
  static Result<Fence> create(VkDevice device, bool signaled = false);

  ~Fence();
  Fence(Fence&& other) noexcept;
  Fence& operator=(Fence&& other) noexcept;
  Fence(const Fence&) = delete;
  Fence& operator=(const Fence&) = delete;

  /// @return The underlying `VkFence` handle.
  VkFence handle() const noexcept { return fence_; }

  /// @brief Block until the fence is signaled or the timeout elapses.
  /// @param timeout_ns  Maximum wait, in nanoseconds (default: wait forever).
  /// @return OK once signaled; a non-OK @ref Status carrying `VK_TIMEOUT` if
  /// the
  ///         timeout elapses first — a timeout is reported, not a hard error.
  Status wait(uint64_t timeout_ns = UINT64_MAX) const;

  /// @brief Return the fence to the unsignaled state.
  /// @return OK on success, or a non-OK @ref Status.
  Status reset();

  /// @return `true` if the fence is currently signaled (non-blocking query).
  bool is_signaled() const;

 private:
  Fence() = default;
  void destroy() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
};

/// @brief A binary `VkSemaphore`: orders work between GPU queue submissions.
class VG_API Semaphore {
 public:
  /// @brief Create a binary semaphore.
  /// @param device  The logical device that owns the semaphore.
  /// @return The semaphore on success, or a non-OK @ref Status.
  static Result<Semaphore> create(VkDevice device);

  ~Semaphore();
  Semaphore(Semaphore&& other) noexcept;
  Semaphore& operator=(Semaphore&& other) noexcept;
  Semaphore(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;

  /// @return The underlying `VkSemaphore` handle.
  VkSemaphore handle() const noexcept { return semaphore_; }

 private:
  Semaphore() = default;
  void destroy() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

/// @brief A timeline `VkSemaphore`: a monotonically increasing 64-bit counter
/// that
///        both the host and GPU queues can signal and wait on.
///
/// Unlike a binary @ref Semaphore, the host can read, signal, and wait on the
/// value directly — the basis for the in-flight-frame cap and CPU/GPU
/// hand-offs. Requires the device's `timelineSemaphore` feature (core in
/// Vulkan 1.2), which
/// @ref Device::create enables when supported; creating one on a device without
/// it is invalid use, reported by the validation layers.
///
/// @code
/// Result<TimelineSemaphore> timeline = TimelineSemaphore::create(device);
/// if (!timeline) return timeline.status();
/// timeline.value().signal(1);    // host raises the counter
/// timeline.value().wait(1);      // returns once the counter reaches >= 1
/// @endcode
class VG_API TimelineSemaphore {
 public:
  /// @brief Create a timeline semaphore.
  /// @param device         The logical device that owns the semaphore.
  /// @param initial_value  The counter's starting value.
  /// @return The semaphore on success, or a non-OK @ref Status (e.g. when the
  ///         device lacks the `timelineSemaphore` feature).
  static Result<TimelineSemaphore> create(VkDevice device,
                                          uint64_t initial_value = 0);

  ~TimelineSemaphore();
  TimelineSemaphore(TimelineSemaphore&& other) noexcept;
  TimelineSemaphore& operator=(TimelineSemaphore&& other) noexcept;
  TimelineSemaphore(const TimelineSemaphore&) = delete;
  TimelineSemaphore& operator=(const TimelineSemaphore&) = delete;

  /// @return The underlying `VkSemaphore` handle.
  VkSemaphore handle() const noexcept { return semaphore_; }

  /// @return The current counter value, or a non-OK @ref Status.
  Result<uint64_t> value() const;

  /// @brief Host-signal the counter to @p value.
  /// @param value  The new counter value; must exceed the current value.
  /// @return OK on success, or a non-OK @ref Status.
  Status signal(uint64_t value);

  /// @brief Block until the counter reaches at least @p value or the timeout
  /// elapses.
  /// @param value       The counter value to wait for.
  /// @param timeout_ns  Maximum wait, in nanoseconds (default: wait forever).
  /// @return OK once reached; a non-OK @ref Status carrying `VK_TIMEOUT` on
  /// timeout.
  Status wait(uint64_t value, uint64_t timeout_ns = UINT64_MAX) const;

 private:
  TimelineSemaphore() = default;
  void destroy() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

}  // namespace volumetric_kit::gfx

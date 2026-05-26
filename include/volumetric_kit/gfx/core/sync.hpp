// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sync.hpp
/// @brief RAII wrappers for the Vulkan synchronization primitives the renderer
///        waits on.
///
/// @ref Fence is host-side — the CPU blocks until submitted GPU work completes.
/// @ref Semaphore is a binary GPU-queue-to-queue primitive. Timeline semaphores
/// (counter-based CPU/GPU sync) arrive in a later stage, together with the
/// Vulkan 1.2 device-feature enablement they require.

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

}  // namespace volumetric_kit::gfx

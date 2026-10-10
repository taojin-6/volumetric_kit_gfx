// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file retire_queue.hpp
/// @brief Deferred destruction of GPU resources, keyed on the values of a
///        timeline semaphore.

#include <cstddef>
#include <cstdint>
#include <functional>

#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/retire_list.hpp"

namespace volumetric_kit::gfx {

/// @brief Runs deferred deleters once a timeline semaphore reaches the value
///        guarding each one.
///
/// A resource the GPU may still read cannot be freed where the caller is done
/// with it. Push its deleter with the timeline value the last work using it
/// sets -- the number of the last frame that drew it, on
/// @ref windowing::FrameLoop::timeline -- and call @ref poll once per frame,
/// which runs every deleter whose value the timeline has reached. This is
/// gfx's one deferred-release mechanism: the CPU completion token of the
/// double-buffer rule in `AGENTS.md`. The GPU-independent bookkeeping lives
/// in @ref RetireList.
///
/// @warning The timeline is borrowed by address: it must outlive this queue
///          and stay where it is (@ref windowing::FrameLoop::timeline does).
///          The `core::Device` a queued deleter frees through must outlive it
///          too: destruction drains (waits for + runs) every pending deleter.
///          Compose so the queue is destroyed first. A
///          `core::Buffer` or `core::Image` keeps its allocator's state alive
///          itself, so the `core::Allocator` need not outlive the queue.
///
/// The deleter is a `std::function`, so it must be copyable: capture copyable
/// state (raw handles, a `shared_ptr`), not a move-only `core::Buffer` /
/// `core::Image` by value.
///
/// @code
/// RetireQueue retire(loop.timeline());
/// // per frame, after begin_frame:
/// retire.poll();
/// // free `view` once every frame up to this one has completed:
/// VkDevice dev = device.handle();
/// retire.push(frame.number,
///             [dev, view]() { vkDestroyImageView(dev, view, nullptr); });
/// @endcode
class VG_CORE_API RetireQueue {
 public:
  /// @brief Construct a queue that observes @p timeline.
  /// @param timeline  The timeline whose values gate the deleters; borrowed
  ///                  by address (see the class warning).
  explicit RetireQueue(const core::TimelineSemaphore& timeline) noexcept;

  /// @brief Waits for each pending value, runs its deleter, then destroys the
  ///        queue (as in @ref drain). Values already reached -- the common
  ///        idle-at-teardown case -- return at once, so a queue drain
  ///        beforehand is not required to free safely. A value nothing will
  ///        set blocks here (see @ref push); call @ref reclaim first for a
  ///        no-wait teardown when the device is already idle or lost.
  ~RetireQueue();

  /// @brief Take over @p other's timeline and pending deleters; @p other is
  ///        left empty.
  RetireQueue(RetireQueue&& other) noexcept;
  /// @brief Drains this queue (waits for + runs its own pending deleters, as
  ///        in @ref drain), then takes over @p other's timeline and pending
  ///        deleters.
  RetireQueue& operator=(RetireQueue&& other) noexcept;
  RetireQueue(const RetireQueue&) = delete;
  RetireQueue& operator=(const RetireQueue&) = delete;

  /// @brief Defer @p deleter until the timeline reaches @p value.
  /// @param value    A value the timeline reaches once the GPU work that last
  ///                 touches the resource completes, such as that frame's
  ///                 `windowing::Frame::number`. Work that sets it must be
  ///                 submitted before the queue is drained.
  /// @param deleter  Invoked exactly once, from @ref poll, @ref drain,
  ///                 @ref reclaim or destruction. Must not throw: it may run
  ///                 from a `noexcept` context (destruction or
  ///                 move-assignment), where an escaping exception calls
  ///                 `std::terminate`.
  /// @pre The queue observes a timeline (not moved from; checked with
  ///      `VKC_CHECK`).
  /// @note Not thread-safe: serialize @ref push against @ref poll /
  ///       @ref drain.
  void push(std::uint64_t value, std::function<void()> deleter);

  /// @brief Run the deleters whose value the timeline has reached; keep the
  ///        rest. Reads the timeline once, and only when a deleter is
  ///        pending.
  /// @return The number of deleters run: none when the read fails (device
  ///         lost), which leaves them to @ref drain or @ref reclaim.
  /// @note Not thread-safe: serialize @ref push against @ref poll /
  ///       @ref drain.
  std::size_t poll();

  /// @brief Wait for the timeline to reach every pending value, then run all
  ///        deleters. Logs a warning (and frees anyway) if a wait fails, e.g.
  ///        on device loss. Blocks indefinitely on a value nothing sets --
  ///        use @ref reclaim instead when the device is idle or lost.
  /// @note Not thread-safe: serialize @ref push against @ref poll /
  ///       @ref drain.
  void drain();

  /// @brief Run every pending deleter immediately WITHOUT waiting for its
  ///        value, then clear -- a no-wait forced reclaim for teardown.
  /// @pre Every guarding value is reached, or the device is idle (e.g. after
  ///      `core::Device::wait_idle`) or lost; the deleters run without
  ///      consulting the timeline, so running early while the GPU still reads
  ///      a resource is undefined behavior.
  /// @note Not thread-safe: serialize @ref push against @ref poll /
  ///       @ref drain / @ref reclaim.
  void reclaim();

  /// @return The number of deleters still pending.
  std::size_t pending() const noexcept { return list_.pending(); }

 private:
  const core::TimelineSemaphore* timeline_ = nullptr;  // borrowed
  RetireList<std::uint64_t> list_;
};

}  // namespace volumetric_kit::gfx

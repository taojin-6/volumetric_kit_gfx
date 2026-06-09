// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file retire_list.hpp
/// @brief A GPU-free list of deferred deleters, each run once when its key
/// reports
///        ready.
///
/// Readiness and waiting are supplied by the caller as callables, so the
/// ordering, run-once, compaction, and drain logic is free of Vulkan and
/// unit-testable on any host. @ref volumetric_kit::gfx::RetireQueue
/// instantiates it with `Key = VkFence` and feeds in `vkGetFenceStatus` /
/// `vkWaitForFences`.

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace volumetric_kit::gfx {

/// @brief Holds `(key, deleter)` entries and runs each deleter exactly once,
/// when the
///        caller's predicate reports the key ready.
/// @tparam Key  Identifies the work guarding an entry (a `VkFence` in
/// practice).
///
/// Each deleter runs once, from @ref poll, @ref drain, or @ref run_all.
/// Destruction and move-assignment do not run pending deleters; they release
/// the closures (and so any captured resources) without invoking the deleter
/// bodies, so call @ref run_all or @ref drain first when a body must execute.
/// @ref run_all is `noexcept` — a throwing deleter calls `std::terminate`, so
/// deleters must not throw. The list owns its deleters and is not thread-safe:
/// serialize @ref push against @ref poll / @ref drain.
///
/// @code
/// RetireList<int> list;
/// // Free `handle` once frame `frame_id`'s GPU work has completed:
/// list.push(frame_id, [h = handle]() { destroy(h); });
/// list.poll([](int id) { return id <= last_completed_frame; });  // ready ones
/// @endcode
template <class Key>
class RetireList {
 public:
  RetireList() = default;
  ~RetireList() = default;
  RetireList(RetireList&&) noexcept = default;
  // Hand-written (not defaulted) to guard self-move: a defaulted move-assign
  // would `entries_ = std::move(entries_)`, a std::vector self-move that is
  // valid-but-unspecified and can silently drop every pending deleter.
  RetireList& operator=(RetireList&& other) noexcept {
    if (this != &other) {
      entries_ = std::move(other.entries_);
      ready_scratch_ = std::move(other.ready_scratch_);
    }
    return *this;
  }
  RetireList(const RetireList&) = delete;
  RetireList& operator=(const RetireList&) = delete;

  /// @brief Append a deferred deleter guarded by @p key.
  /// @param key      Identifies the GPU work that must finish before @p deleter
  /// runs.
  /// @param deleter  Invoked once, later, when @p key reports ready. Must not
  /// throw.
  void push(Key key, std::function<void()> deleter);

  /// @brief Run the deleters whose key @p is_ready reports ready; keep the
  /// rest.
  /// @param is_ready  Callable `bool(const Key&)`.
  /// @return The number of deleters run.
  template <class ReadyFn>
  std::size_t poll(ReadyFn is_ready);

  /// @brief Wait on each entry's key via @p wait, then run every deleter.
  /// @param wait  Callable `void(const Key&)` that blocks until the key is
  /// ready.
  template <class WaitFn>
  void drain(WaitFn wait);

  /// @brief Run every pending deleter unconditionally (no readiness check),
  /// then clear.
  /// @pre Every guarding key is already ready (e.g. the device is idle); the
  /// deleters
  ///      run without consulting it.
  void run_all() noexcept;

  /// @return The number of deleters still pending.
  std::size_t pending() const noexcept { return entries_.size(); }

 private:
  // Run each detached deleter once, skipping any empty target.
  static void run_each(std::vector<std::function<void()>>& deleters);

  struct Entry {
    Key key;
    std::function<void()> deleter;
  };

  std::vector<Entry> entries_;

  // Reused across poll() calls so the per-frame poll does not reallocate the
  // ready-deleter buffer each time; see poll() for the re-entrancy handling.
  std::vector<std::function<void()>> ready_scratch_;
};

}  // namespace volumetric_kit::gfx

#include "volumetric_kit/gfx/core/impl/retire_list.hpp"

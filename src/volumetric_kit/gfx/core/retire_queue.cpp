// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/retire_queue.hpp"

#include <string>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/gfx/core/log.hpp"

namespace volumetric_kit::gfx {

RetireQueue::RetireQueue(const core::TimelineSemaphore& timeline) noexcept
    : timeline_(&timeline) {}

// Wait for each pending value, then run its deleter -- never free a resource
// the GPU might still be reading. drain() returns at once for values already
// reached (the common idle-at-teardown case), so this is safe without a queue
// drain first.
RetireQueue::~RetireQueue() { drain(); }

RetireQueue::RetireQueue(RetireQueue&& other) noexcept
    : timeline_(other.timeline_), list_(std::move(other.list_)) {
  other.timeline_ = nullptr;  // a moved-from queue observes no timeline
}

RetireQueue& RetireQueue::operator=(RetireQueue&& other) noexcept {
  if (this != &other) {
    drain();  // wait + run this queue's own deleters before adopting other's
    timeline_ = other.timeline_;
    list_ = std::move(other.list_);
    other.timeline_ = nullptr;
  }
  return *this;
}

void RetireQueue::push(std::uint64_t value, std::function<void()> deleter) {
  VKC_CHECK(timeline_ != nullptr,
            "RetireQueue::push on a queue that observes no timeline (moved "
            "from)");
  list_.push(value, std::move(deleter));
}

std::size_t RetireQueue::poll() {
  if (list_.pending() == 0) return 0;
  // A failed read (device lost) defers everything; drain or reclaim frees it
  // at teardown.
  const core::Result<std::uint64_t> reached = timeline_->value();
  if (!reached.ok()) return 0;
  const std::uint64_t completed = reached.value();
  return list_.poll(
      [completed](std::uint64_t value) { return value <= completed; });
}

void RetireQueue::drain() {
  // Pending deleters imply a timeline: push refuses a queue without one.
  bool failed = false;
  list_.drain([this, &failed](std::uint64_t value) {
    if (failed) return;  // the device is lost: free the rest without waiting
    const core::Status waited = timeline_->wait(value);
    if (!waited.ok()) {
      // The guarded work won't complete; the deleter still runs to reclaim
      // the resource, but record it -- this noexcept teardown path has no
      // other observability hook.
      failed = true;
      log_message(core::LogLevel::Warning,
                  "RetireQueue::drain: waiting for the timeline failed (" +
                      waited.message() + "); freeing the resources regardless");
    }
  });
}

void RetireQueue::reclaim() { list_.run_all(); }

}  // namespace volumetric_kit::gfx

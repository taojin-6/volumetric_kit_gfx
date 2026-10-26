// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/retire_queue.hpp"

#include <cstdio>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/core/log.hpp"

namespace volumetric_kit::gfx {

RetireQueue::RetireQueue(const core::TimelineSemaphore& timeline) noexcept
    : timeline_(&timeline) {}

RetireQueue::~RetireQueue() { drain_or_preserve(); }

RetireQueue::RetireQueue(RetireQueue&& other) noexcept
    : timeline_(other.timeline_), list_(std::move(other.list_)) {
  other.timeline_ = nullptr;
}

RetireQueue& RetireQueue::operator=(RetireQueue&& other) noexcept {
  if (this != &other) {
    drain_or_preserve();
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
  if (!list_) list_ = std::make_unique<RetireList<std::uint64_t>>();
  list_->push(value, std::move(deleter));
}

std::size_t RetireQueue::poll() try {
  if (pending() == 0) return 0;
  const core::Result<std::uint64_t> reached = timeline_->value();
  if (!reached.ok()) return 0;
  const std::uint64_t completed = reached.value();
  return list_->poll(
      [completed](std::uint64_t value) { return value <= completed; });
} catch (...) {
  // Reads and scratch allocation may fail. Neither moves a pending owner
  // before it succeeds; leave the entries for a later poll/drain.
  return 0;
}

core::Status RetireQueue::drain() noexcept try {
  if (pending() == 0) return {};
  core::Status status;
  bool lost = false;
  list_->drain_until([this, &status, &lost](std::uint64_t value) {
    if (lost) return true;
    status = timeline_->wait(value);
    lost = core::vk_result(status) == VK_ERROR_DEVICE_LOST;
    return status.ok() || lost;
  });
  return status;
} catch (...) {
  return core::exception_status("RetireQueue::drain",
                                core::Status::Code::OutOfMemory);
}

void RetireQueue::drain_or_preserve() noexcept {
  const core::Status status = drain();
  if (status.ok() || pending() == 0) return;
  // Destroying a closure can free its captured GPU resource even without
  // calling its body. Preserve the whole owner; allocating a replacement
  // here would itself be unsafe when the wait failed under memory pressure.
  (void)list_.release();
  constexpr const char* message =
      "RetireQueue: timeline completion is unknown; preserving pending "
      "resources";
  try {
    log_message(core::LogLevel::Warning, message);
  } catch (...) {
    // A sink can allocate or throw. Neither may escape this teardown path.
    std::fputs("[vg warning] ", stderr);
    std::fputs(message, stderr);
    std::fputc('\n', stderr);
  }
}

void RetireQueue::reclaim() {
  if (list_) list_->run_all();
}

}  // namespace volumetric_kit::gfx

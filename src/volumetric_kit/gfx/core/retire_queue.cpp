// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/retire_queue.hpp"

#include <cstdint>
#include <utility>

#include "volumetric_kit/gfx/core/log.hpp"

namespace volumetric_kit::gfx {

RetireQueue::RetireQueue(VkDevice device) noexcept : device_(device) {}

// Wait for each pending fence, then run its deleter — never free a resource the
// GPU might still be reading. drain() returns immediately for already-signaled
// fences (the common idle-at-teardown case), so this is safe without requiring
// the caller to vkDeviceWaitIdle first.
RetireQueue::~RetireQueue() { drain(); }

RetireQueue::RetireQueue(RetireQueue&& other) noexcept
    : device_(other.device_), list_(std::move(other.list_)) {
  other.device_ = VK_NULL_HANDLE;  // moved-from queue observes no device
}

RetireQueue& RetireQueue::operator=(RetireQueue&& other) noexcept {
  if (this != &other) {
    drain();  // wait + run this queue's own deleters before adopting other's
    device_ = other.device_;
    list_ = std::move(other.list_);
    other.device_ = VK_NULL_HANDLE;
  }
  return *this;
}

void RetireQueue::push(VkFence fence, std::function<void()> deleter) {
  list_.push(fence, std::move(deleter));
}

std::size_t RetireQueue::poll() {
  // Only VK_SUCCESS releases. VK_NOT_READY and hard errors (e.g.
  // VK_ERROR_DEVICE_LOST) both defer; on device loss the resources are
  // reclaimed by @ref drain / destruction at teardown.
  return list_.poll([this](VkFence fence) {
    return vkGetFenceStatus(device_, fence) == VK_SUCCESS;
  });
}

void RetireQueue::drain() {
  list_.drain([this](VkFence fence) {
    VkResult result = vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
      // Device loss (or an invalid fence) means the guarded work won't
      // complete; the deleter still runs to reclaim the resource, but record it
      // -- this noexcept teardown path has no other observability hook.
      log_message(LogLevel::Warning,
                  "RetireQueue::drain: vkWaitForFences did not return "
                  "VK_SUCCESS; freeing the resource regardless");
    }
  });
}

void RetireQueue::reclaim() { list_.run_all(); }

}  // namespace volumetric_kit::gfx

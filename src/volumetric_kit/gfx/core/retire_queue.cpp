// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/retire_queue.hpp"

#include <cstdint>
#include <utility>

namespace volumetric_kit::gfx {

RetireQueue::RetireQueue(VkDevice device) noexcept : device_(device) {}

RetireQueue::~RetireQueue() { list_.run_all(); }

RetireQueue::RetireQueue(RetireQueue&& other) noexcept
    : device_(other.device_), list_(std::move(other.list_)) {
  other.device_ = VK_NULL_HANDLE;  // moved-from queue observes no device
}

RetireQueue& RetireQueue::operator=(RetireQueue&& other) noexcept {
  if (this != &other) {
    list_.run_all();  // release deleters already queued here (device assumed
                      // idle)
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
    vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
  });
}

}  // namespace volumetric_kit::gfx

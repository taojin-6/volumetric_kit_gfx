// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/sync.hpp"

namespace volumetric_kit::gfx {

// --- Fence ------------------------------------------------------------------

Result<Fence> Fence::create(VkDevice device, bool signaled) {
  VkFenceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (signaled) {
    info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  }

  Fence fence;
  fence.device_ = device;
  VG_VK_TRY(vkCreateFence(device, &info, nullptr, &fence.fence_));
  return fence;
}

Status Fence::wait(uint64_t timeout_ns) const {
  VkResult result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeout_ns);
  if (result != VK_SUCCESS) {
    // VK_TIMEOUT lands here too: surfaced as a non-OK Status the caller can
    // distinguish via status.code(), not treated as a hard failure.
    return vk_error(result, "vkWaitForFences");
  }
  return Status{};
}

Status Fence::reset() {
  VG_VK_TRY(vkResetFences(device_, 1, &fence_));
  return Status{};
}

bool Fence::is_signaled() const {
  return vkGetFenceStatus(device_, fence_) == VK_SUCCESS;
}

Fence::Fence(Fence&& other) noexcept
    : device_(other.device_), fence_(other.fence_) {
  other.fence_ = VK_NULL_HANDLE;
}

Fence& Fence::operator=(Fence&& other) noexcept {
  if (this != &other) {
    destroy();
    device_ = other.device_;
    fence_ = other.fence_;
    other.fence_ = VK_NULL_HANDLE;
  }
  return *this;
}

Fence::~Fence() { destroy(); }

void Fence::destroy() noexcept {
  if (fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_, fence_, nullptr);
    fence_ = VK_NULL_HANDLE;
  }
}

// --- Semaphore --------------------------------------------------------------

Result<Semaphore> Semaphore::create(VkDevice device) {
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  Semaphore semaphore;
  semaphore.device_ = device;
  VG_VK_TRY(vkCreateSemaphore(device, &info, nullptr, &semaphore.semaphore_));
  return semaphore;
}

Semaphore::Semaphore(Semaphore&& other) noexcept
    : device_(other.device_), semaphore_(other.semaphore_) {
  other.semaphore_ = VK_NULL_HANDLE;
}

Semaphore& Semaphore::operator=(Semaphore&& other) noexcept {
  if (this != &other) {
    destroy();
    device_ = other.device_;
    semaphore_ = other.semaphore_;
    other.semaphore_ = VK_NULL_HANDLE;
  }
  return *this;
}

Semaphore::~Semaphore() { destroy(); }

void Semaphore::destroy() noexcept {
  if (semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(device_, semaphore_, nullptr);
    semaphore_ = VK_NULL_HANDLE;
  }
}

// --- TimelineSemaphore ------------------------------------------------------

Result<TimelineSemaphore> TimelineSemaphore::create(VkDevice device,
                                                    uint64_t initial_value) {
  VkSemaphoreTypeCreateInfo type_info{};
  type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type_info.initialValue = initial_value;

  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = &type_info;

  TimelineSemaphore semaphore;
  semaphore.device_ = device;
  VG_VK_TRY(vkCreateSemaphore(device, &info, nullptr, &semaphore.semaphore_));
  return semaphore;
}

Result<uint64_t> TimelineSemaphore::value() const {
  uint64_t value = 0;
  VG_VK_TRY(vkGetSemaphoreCounterValue(device_, semaphore_, &value));
  return value;
}

Status TimelineSemaphore::signal(uint64_t value) {
  VkSemaphoreSignalInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  info.semaphore = semaphore_;
  info.value = value;
  VG_VK_TRY(vkSignalSemaphore(device_, &info));
  return Status{};
}

Status TimelineSemaphore::wait(uint64_t value, uint64_t timeout_ns) const {
  VkSemaphoreWaitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore_;
  info.pValues = &value;

  VkResult result = vkWaitSemaphores(device_, &info, timeout_ns);
  if (result != VK_SUCCESS) {
    // VK_TIMEOUT surfaces here as a non-OK Status, distinguishable via code().
    return vk_error(result, "vkWaitSemaphores");
  }
  return Status{};
}

TimelineSemaphore::TimelineSemaphore(TimelineSemaphore&& other) noexcept
    : device_(other.device_), semaphore_(other.semaphore_) {
  other.semaphore_ = VK_NULL_HANDLE;
}

TimelineSemaphore& TimelineSemaphore::operator=(
    TimelineSemaphore&& other) noexcept {
  if (this != &other) {
    destroy();
    device_ = other.device_;
    semaphore_ = other.semaphore_;
    other.semaphore_ = VK_NULL_HANDLE;
  }
  return *this;
}

TimelineSemaphore::~TimelineSemaphore() { destroy(); }

void TimelineSemaphore::destroy() noexcept {
  if (semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(device_, semaphore_, nullptr);
    semaphore_ = VK_NULL_HANDLE;
  }
}

}  // namespace volumetric_kit::gfx

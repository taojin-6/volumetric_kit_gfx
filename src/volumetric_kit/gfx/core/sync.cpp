// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/sync.hpp"

namespace volumetric_kit::gfx {

// The move/destroy lifecycle for all three primitives lives in UniqueHandle
// (see unique_handle.hpp); here we only create the handle and expose the
// domain operations.

// --- Fence ------------------------------------------------------------------

Result<Fence> Fence::create(VkDevice device, bool signaled) {
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument("Fence::create: device is null");
  }
  VkFenceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (signaled) {
    info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  }

  VkFence handle = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateFence(device, &info, nullptr, &handle));

  Fence fence;
  fence.handle_ = UniqueHandle<VkFence, vkDestroyFence>(device, handle);
  return fence;
}

Status Fence::wait(uint64_t timeout_ns) const {
  VkFence fence = handle_.get();
  VkResult result =
      vkWaitForFences(handle_.device(), 1, &fence, VK_TRUE, timeout_ns);
  if (result != VK_SUCCESS) {
    // VK_TIMEOUT lands here too: surfaced as a non-OK Status the caller can
    // distinguish via status.code(), not treated as a hard failure.
    return vk_error(result, "vkWaitForFences");
  }
  return Status{};
}

Status Fence::reset() {
  VkFence fence = handle_.get();
  VG_VK_TRY(vkResetFences(handle_.device(), 1, &fence));
  return Status{};
}

bool Fence::is_signaled() const {
  return vkGetFenceStatus(handle_.device(), handle_.get()) == VK_SUCCESS;
}

// --- Semaphore --------------------------------------------------------------

Result<Semaphore> Semaphore::create(VkDevice device) {
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument("Semaphore::create: device is null");
  }
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkSemaphore handle = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateSemaphore(device, &info, nullptr, &handle));

  Semaphore semaphore;
  semaphore.handle_ =
      UniqueHandle<VkSemaphore, vkDestroySemaphore>(device, handle);
  return semaphore;
}

// --- TimelineSemaphore ------------------------------------------------------

Result<TimelineSemaphore> TimelineSemaphore::create(VkDevice device,
                                                    uint64_t initial_value) {
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "TimelineSemaphore::create: device is null");
  }
  VkSemaphoreTypeCreateInfo type_info{};
  type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type_info.initialValue = initial_value;

  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = &type_info;

  VkSemaphore handle = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateSemaphore(device, &info, nullptr, &handle));

  TimelineSemaphore semaphore;
  semaphore.handle_ =
      UniqueHandle<VkSemaphore, vkDestroySemaphore>(device, handle);
  return semaphore;
}

Result<uint64_t> TimelineSemaphore::value() const {
  uint64_t value = 0;
  VG_VK_TRY(
      vkGetSemaphoreCounterValue(handle_.device(), handle_.get(), &value));
  return value;
}

Status TimelineSemaphore::signal(uint64_t value) {
  // A host signal must strictly advance the counter
  // (VUID-VkSemaphoreSignalInfo-value-03258). Read the current value and reject
  // a non-increasing signal up front, so a deterministic misuse is a domain
  // error rather than undefined behavior in a release build (validation off).
  uint64_t current = 0;
  VG_VK_TRY(
      vkGetSemaphoreCounterValue(handle_.device(), handle_.get(), &current));
  if (value <= current) {
    return Status::invalid_argument(
        "TimelineSemaphore::signal value must exceed the current counter "
        "value");
  }

  VkSemaphoreSignalInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  info.semaphore = handle_.get();
  info.value = value;
  VG_VK_TRY(vkSignalSemaphore(handle_.device(), &info));
  return Status{};
}

Status TimelineSemaphore::wait(uint64_t value, uint64_t timeout_ns) const {
  VkSemaphore semaphore = handle_.get();
  VkSemaphoreWaitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore;
  info.pValues = &value;

  VkResult result = vkWaitSemaphores(handle_.device(), &info, timeout_ns);
  if (result != VK_SUCCESS) {
    // VK_TIMEOUT surfaces here as a non-OK Status, distinguishable via code().
    return vk_error(result, "vkWaitSemaphores");
  }
  return Status{};
}

}  // namespace volumetric_kit::gfx

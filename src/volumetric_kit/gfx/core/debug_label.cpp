// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/debug_label.hpp"

#include <mutex>

#include "volumetric_kit/core/vulkan/device.hpp"

namespace volumetric_kit::gfx {

DebugLabelScope::DebugLabelScope(const core::Device& device,
                                 VkCommandBuffer cmd,
                                 const char* name) noexcept {
  // The conditions under which the core's begin_debug_label emits, so
  // active() says whether a region is open.
  if (!device.debug_labels_available() || cmd == VK_NULL_HANDLE ||
      name == nullptr) {
    return;
  }
  device.begin_debug_label(cmd, name);
  device_ = &device;
  cmd_ = cmd;
  name_ = name;
}

void DebugLabelScope::close() noexcept {
  if (device_ != nullptr) {
    device_->end_debug_label(cmd_, name_);
  }
}

DebugLabelScope::~DebugLabelScope() { close(); }

DebugLabelScope::DebugLabelScope(DebugLabelScope&& other) noexcept
    : device_(other.device_), cmd_(other.cmd_), name_(other.name_) {
  other.device_ = nullptr;
  other.cmd_ = VK_NULL_HANDLE;
  other.name_ = nullptr;
}

DebugLabelScope& DebugLabelScope::operator=(DebugLabelScope&& other) noexcept {
  if (this != &other) {
    close();  // close our own region before adopting the source's
    device_ = other.device_;
    cmd_ = other.cmd_;
    name_ = other.name_;
    other.device_ = nullptr;
    other.cmd_ = VK_NULL_HANDLE;
    other.name_ = nullptr;
  }
  return *this;
}

QueueLabelScope::QueueLabelScope(const core::Device& device,
                                 const char* name) noexcept {
  if (!device.debug_labels_available() || name == nullptr) {
    return;
  }
  // TODO: take queue labels from the core's Device once it records them, so
  // the entry points are resolved once per device, by the core.
  const auto begin = reinterpret_cast<PFN_vkQueueBeginDebugUtilsLabelEXT>(
      vkGetDeviceProcAddr(device.handle(), "vkQueueBeginDebugUtilsLabelEXT"));
  const auto end = reinterpret_cast<PFN_vkQueueEndDebugUtilsLabelEXT>(
      vkGetDeviceProcAddr(device.handle(), "vkQueueEndDebugUtilsLabelEXT"));
  if (begin == nullptr || end == nullptr) {
    return;
  }
  VkDebugUtilsLabelEXT label{};
  label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
  label.pLabelName = name;
  {
    const std::lock_guard<std::mutex> lock(*device.submit_mutex());
    begin(device.queue(), &label);
  }
  device_ = &device;
  end_ = end;
}

void QueueLabelScope::close() noexcept {
  if (end_ != nullptr) {
    const std::lock_guard<std::mutex> lock(*device_->submit_mutex());
    end_(device_->queue());
  }
}

QueueLabelScope::~QueueLabelScope() { close(); }

QueueLabelScope::QueueLabelScope(QueueLabelScope&& other) noexcept
    : device_(other.device_), end_(other.end_) {
  other.device_ = nullptr;
  other.end_ = nullptr;
}

QueueLabelScope& QueueLabelScope::operator=(QueueLabelScope&& other) noexcept {
  if (this != &other) {
    close();  // close our own region before adopting the source's
    device_ = other.device_;
    end_ = other.end_;
    other.device_ = nullptr;
    other.end_ = nullptr;
  }
  return *this;
}

}  // namespace volumetric_kit::gfx

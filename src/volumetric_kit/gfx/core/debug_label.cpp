// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/debug_label.hpp"

#include <utility>

namespace volumetric_kit::gfx {
namespace {

// Fill a VkDebugUtilsLabelEXT from a name and an optional RGBA color. The
// label's color array is always written (the struct has no "unset" state); a
// null color leaves it zeroed, which tools render as no tint.
VkDebugUtilsLabelEXT make_label(const char* name, const float color[4]) {
  VkDebugUtilsLabelEXT label{};
  label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
  label.pLabelName = name;
  if (color != nullptr) {
    label.color[0] = color[0];
    label.color[1] = color[1];
    label.color[2] = color[2];
    label.color[3] = color[3];
  }
  return label;
}

}  // namespace

DebugLabelScope::DebugLabelScope(VkCommandBuffer cmd,
                                 const DebugUtilsTable& table, const char* name,
                                 const float color[4]) noexcept {
  // A null name is refused (left inert): VkDebugUtilsLabelEXT::pLabelName must
  // be non-null, unlike the optional object name in set_object_name.
  if (!table.active() || cmd == VK_NULL_HANDLE || name == nullptr) {
    return;
  }
  VkDebugUtilsLabelEXT label = make_label(name, color);
  table.cmd_begin(cmd, &label);
  cmd_ = cmd;
  end_ = table.cmd_end;
}

DebugLabelScope::~DebugLabelScope() {
  if (end_ != nullptr) {
    end_(cmd_);
  }
}

DebugLabelScope::DebugLabelScope(DebugLabelScope&& other) noexcept
    : cmd_(other.cmd_), end_(other.end_) {
  other.cmd_ = VK_NULL_HANDLE;
  other.end_ = nullptr;
}

DebugLabelScope& DebugLabelScope::operator=(DebugLabelScope&& other) noexcept {
  if (this != &other) {
    if (end_ != nullptr) {
      end_(cmd_);  // close our own region before adopting the source's
    }
    cmd_ = other.cmd_;
    end_ = other.end_;
    other.cmd_ = VK_NULL_HANDLE;
    other.end_ = nullptr;
  }
  return *this;
}

QueueLabelScope::QueueLabelScope(VkQueue queue, const DebugUtilsTable& table,
                                 const char* name,
                                 const float color[4]) noexcept {
  // A null name is refused (left inert): VkDebugUtilsLabelEXT::pLabelName must
  // be non-null, unlike the optional object name in set_object_name.
  if (!table.active() || queue == VK_NULL_HANDLE || name == nullptr) {
    return;
  }
  VkDebugUtilsLabelEXT label = make_label(name, color);
  table.queue_begin(queue, &label);
  queue_ = queue;
  end_ = table.queue_end;
}

QueueLabelScope::~QueueLabelScope() {
  if (end_ != nullptr) {
    end_(queue_);
  }
}

QueueLabelScope::QueueLabelScope(QueueLabelScope&& other) noexcept
    : queue_(other.queue_), end_(other.end_) {
  other.queue_ = VK_NULL_HANDLE;
  other.end_ = nullptr;
}

QueueLabelScope& QueueLabelScope::operator=(QueueLabelScope&& other) noexcept {
  if (this != &other) {
    if (end_ != nullptr) {
      end_(queue_);  // close our own region before adopting the source's
    }
    queue_ = other.queue_;
    end_ = other.end_;
    other.queue_ = VK_NULL_HANDLE;
    other.end_ = nullptr;
  }
  return *this;
}

void set_object_name(VkDevice device, const DebugUtilsTable& table,
                     VkObjectType type, uint64_t handle, const char* name) {
  if (table.set_name == nullptr || handle == 0) {
    return;
  }
  VkDebugUtilsObjectNameInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType = type;
  info.objectHandle = handle;
  info.pObjectName = name;
  table.set_name(device, &info);
}

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/debug_utils_table.hpp
/// Internal helper: the device-level `VK_EXT_debug_utils` entry points the
/// capture-label surface emits through. The labels and object names this table
/// drives are a single standard that RenderDoc, Nsight Graphics, Nsight
/// Systems, and Xcode's Metal frame debugger all consume — one emitter, no
/// per-tool code. Cached on @ref volumetric_kit::gfx::Device so each
/// command/queue scope reads a resolved pointer instead of re-querying the
/// loader. Not a public header.

#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// The device-level `VK_EXT_debug_utils` function pointers, resolved once per
/// device. Every member is null when the extension is not enabled (or any
/// individual fetch fails), so @ref active reports a single all-or-nothing
/// state and each call site branches to a no-op on it.
struct DebugUtilsTable {
  PFN_vkCmdBeginDebugUtilsLabelEXT cmd_begin = nullptr;
  PFN_vkCmdEndDebugUtilsLabelEXT cmd_end = nullptr;
  // Queue-timeline labels surface on Nsight Systems' submit timeline, where
  // command-buffer labels do not appear.
  PFN_vkQueueBeginDebugUtilsLabelEXT queue_begin = nullptr;
  PFN_vkQueueEndDebugUtilsLabelEXT queue_end = nullptr;
  PFN_vkSetDebugUtilsObjectNameEXT set_name = nullptr;

  /// @return Whether the entry points resolved — the all-or-nothing gate every
  ///         emitter checks before touching a pointer.
  bool active() const noexcept { return cmd_begin != nullptr; }

  /// @brief Resolve the device-level entry points for @p device.
  /// @param device  The logical device to resolve against.
  /// @param instance_debug_utils_enabled  Whether `VK_EXT_debug_utils` is
  ///        enabled on the instance the device belongs to (see
  ///        @ref Instance::debug_utils_enabled). When false, no entry point can
  ///        be resolved, so the load returns an all-null table immediately.
  /// @return A fully resolved table, or an all-null one when the extension is
  ///         disabled or any single fetch returns null (partial resolution is
  ///         treated as unavailable rather than left half-populated).
  static DebugUtilsTable load(VkDevice device,
                              bool instance_debug_utils_enabled) {
    DebugUtilsTable table;
    if (!instance_debug_utils_enabled || device == VK_NULL_HANDLE) {
      return table;
    }
    // Device-level resolution: vkGetDeviceProcAddr returns pointers that skip
    // the loader's per-device dispatch trampoline. (The instance-level
    // messenger entry points are fetched with vkGetInstanceProcAddr instead —
    // a different layer of the loader.)
    auto load_pfn = [device](const char* name) {
      return vkGetDeviceProcAddr(device, name);
    };
    table.cmd_begin = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        load_pfn("vkCmdBeginDebugUtilsLabelEXT"));
    table.cmd_end = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        load_pfn("vkCmdEndDebugUtilsLabelEXT"));
    table.queue_begin = reinterpret_cast<PFN_vkQueueBeginDebugUtilsLabelEXT>(
        load_pfn("vkQueueBeginDebugUtilsLabelEXT"));
    table.queue_end = reinterpret_cast<PFN_vkQueueEndDebugUtilsLabelEXT>(
        load_pfn("vkQueueEndDebugUtilsLabelEXT"));
    table.set_name = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        load_pfn("vkSetDebugUtilsObjectNameEXT"));

    // All-or-nothing: a partial resolution (any single null) collapses to the
    // inactive table so callers never dereference a mixed set of pointers.
    if (table.cmd_begin == nullptr || table.cmd_end == nullptr ||
        table.queue_begin == nullptr || table.queue_end == nullptr ||
        table.set_name == nullptr) {
      return DebugUtilsTable{};
    }
    return table;
  }
};

}  // namespace volumetric_kit::gfx

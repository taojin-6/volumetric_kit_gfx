// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file instance.hpp
/// The Vulkan instance + (optional) validation messenger, plus portable
/// physical-device selection. Split from the logical device so headless,
/// multi-GPU, and compute-only interop devices can be composed on one instance.

#include <string>
#include <vector>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

struct InstanceConfig {
  std::string app_name = "volumetric_kit_gfx";
  bool enable_validation = false;
  /// Surface/platform instance extensions the windowing tier supplies (e.g.
  /// from `glfwGetRequiredInstanceExtensions`). Core stays windowing-agnostic.
  std::vector<const char*> extra_instance_extensions;
};

/// Owns a `VkInstance` and, when validation is enabled, its debug messenger.
class VG_CORE_API Instance {
 public:
  /// Create the instance: turns on validation + the debug messenger when
  /// requested and available, and portability enumeration when the loader
  /// offers it (so MoltenVK devices are visible). Surface/platform extensions
  /// come from `config.extra_instance_extensions`.
  static Result<Instance> create(const InstanceConfig& config);

  ~Instance();
  Instance(Instance&& other) noexcept;
  Instance& operator=(Instance&& other) noexcept;
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;

  /// The owned VkInstance.
  VkInstance handle() const noexcept { return instance_; }
  /// Whether the validation debug messenger is active.
  bool validation_enabled() const noexcept {
    return messenger_ != VK_NULL_HANDLE;
  }

  /// Pick the best physical device: prefers discrete > integrated > virtual >
  /// CPU, requires a graphics-capable queue family, and — when `surface` is
  /// provided — a present-capable family. Errors if none qualifies.
  Result<VkPhysicalDevice> select_physical_device(
      VkSurfaceKHR surface = VK_NULL_HANDLE) const;

 private:
  Instance() = default;
  void destroy() noexcept;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
};

}  // namespace volumetric_kit::gfx

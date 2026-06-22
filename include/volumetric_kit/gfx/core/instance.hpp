// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file instance.hpp
/// @brief The Vulkan instance + (optional) validation messenger, plus portable
///        physical-device selection.

#include <string>
#include <vector>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/physical_device_info.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Parameters for @ref Instance::create.
struct InstanceConfig {
  /// Application name reported to the driver in `VkApplicationInfo`.
  std::string app_name = "volumetric_kit_gfx";
  /// Enable the Khronos validation layer + debug messenger when it is available
  /// (a no-op, with a logged warning, when the layer is not installed).
  bool enable_validation = false;
  /// Enable `VK_EXT_debug_utils` even when validation is off, whenever the
  /// extension is available. This carries the object-naming / debug-label entry
  /// points into release and profiling builds, so GPU captures (RenderDoc,
  /// Nsight, Xcode) show section labels. Validation already pulls the extension
  /// in, so this only changes behavior when @ref enable_validation is false.
  bool enable_debug_utils = false;
  /// Surface/platform instance extensions the windowing tier supplies (e.g.
  /// from `glfwGetRequiredInstanceExtensions`). Core stays windowing-agnostic.
  std::vector<const char*> extra_instance_extensions;
};

/// @brief Owns a `VkInstance` and, when validation is enabled, its debug
///        messenger; split from @ref Device so headless, multi-GPU, and
///        compute-only interop devices compose on one instance.
///
/// @code
/// Result<Instance> instance = Instance::create({.enable_validation = true});
/// if (!instance) return instance.status();
/// Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
/// if (!gpu) return gpu.status();
/// @endcode
class VG_CORE_API Instance {
 public:
  /// @brief Create the instance: turns on the validation layer + debug
  ///        messenger when requested and available, and portability enumeration
  ///        when the loader offers it (so MoltenVK devices are visible).
  /// @param config  App name, validation toggle, and surface/platform
  ///                extensions (`config.extra_instance_extensions`).
  /// @return The instance on success, or a non-OK @ref Status carrying the
  ///         `vkCreateInstance` `VkResult`.
  static Result<Instance> create(const InstanceConfig& config);

  ~Instance();
  Instance(Instance&& other) noexcept;
  Instance& operator=(Instance&& other) noexcept;
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;

  /// @return The owned `VkInstance` (`VK_NULL_HANDLE` when moved-from).
  VkInstance handle() const noexcept { return instance_; }
  /// @return Whether the validation debug messenger is active.
  bool validation_enabled() const noexcept {
    return messenger_ != VK_NULL_HANDLE;
  }
  /// @brief Report whether `VK_EXT_debug_utils` was enabled on this instance.
  /// @return `true` when the extension is enabled, so the object-naming and
  ///         debug-label entry points (`vkSetDebugUtilsObjectNameEXT`,
  ///         `vkCmdBeginDebugUtilsLabelEXT`, …) are usable. Distinct from
  ///         @ref validation_enabled: enabling debug-utils in a release build
  ///         (via `InstanceConfig::enable_debug_utils`) is what lets profiling
  ///         captures carry section labels without paying for validation.
  bool debug_utils_enabled() const noexcept { return debug_utils_; }

  /// @brief Pick the best physical device: prefers discrete > integrated >
  ///        virtual > CPU, requires a graphics-capable queue family, and — when
  ///        @p surface is provided — a present-capable family.
  /// @param surface  When non-null, restrict to devices with a queue family
  ///                 that can present to it.
  /// @return The chosen device, or @ref Status::Code::Unsupported when none
  ///         qualifies.
  /// @note Qualifies on queue families and the Vulkan 1.3 floor that
  ///       @ref Device::create also requires, so a selected device is
  ///       creatable.
  Result<VkPhysicalDevice> select_physical_device(
      VkSurfaceKHR surface = VK_NULL_HANDLE) const;

  /// @brief Capture the read-only capabilities of @p physical.
  /// @param physical  A physical device belonging to this instance.
  /// @return The captured capabilities. Use before @ref Device::create to
  ///         pre-check extensions/features, and for swapchain / volume /
  ///         interop format selection.
  PhysicalDeviceInfo query_physical_device(VkPhysicalDevice physical) const;

 private:
  Instance() = default;
  void destroy() noexcept;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  bool debug_utils_ = false;
};

}  // namespace volumetric_kit::gfx

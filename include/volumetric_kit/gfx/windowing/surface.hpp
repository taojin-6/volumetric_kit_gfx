// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file surface.hpp
/// @brief RAII owner of a `VkSurfaceKHR` — the window-system handle a
///        @ref Swapchain presents to.

#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/windowing/export.hpp"

namespace volumetric_kit::gfx::windowing {

/// @brief Owns a `VkSurfaceKHR` and destroys it on the instance it belongs to.
///
/// A window system creates the surface — pass `glfwCreateWindowSurface`'s
/// result to the adopting constructor — or, for windowless CI, @ref headless
/// creates one through `VK_EXT_headless_surface`. A @ref Swapchain presents to
/// the wrapped handle. A default-constructed `Surface` is empty (`valid()` is
/// false) and safe to move-assign into.
///
/// @warning The @p instance passed in must outlive the surface: the destructor
///          frees through it, so destroying the instance first is undefined
///          behavior.
///
/// @code
/// VkSurfaceKHR raw = VK_NULL_HANDLE;
/// glfwCreateWindowSurface(instance.handle(), window, nullptr, &raw);
/// windowing::Surface surface(instance.handle(), raw);  // adopts + frees it
/// @endcode
class VG_WINDOWING_API Surface {
 public:
  /// @brief Construct an empty surface (owns nothing; `valid()` is false).
  Surface() noexcept = default;

  /// @brief Adopt @p surface, destroyed on @p instance.
  /// @param instance  The instance @p surface was created against.
  /// @param surface   The surface handle to take ownership of.
  Surface(VkInstance instance, VkSurfaceKHR surface) noexcept;

  ~Surface();
  Surface(Surface&& other) noexcept;
  Surface& operator=(Surface&& other) noexcept;
  Surface(const Surface&) = delete;
  Surface& operator=(const Surface&) = delete;

  /// @brief Create a windowless surface via `VK_EXT_headless_surface`.
  /// @param instance  An instance created with `VK_EXT_headless_surface`
  ///                  enabled.
  /// @return The surface on success; @ref Status::Code::Unsupported when the
  ///         extension is unavailable (e.g. MoltenVK), or a Vulkan-domain
  ///         error.
  static Result<Surface> headless(VkInstance instance);

  /// @return The owned `VkSurfaceKHR` (`VK_NULL_HANDLE` when empty).
  VkSurfaceKHR handle() const noexcept { return surface_; }

  /// @return `true` if this owns a surface.
  bool valid() const noexcept { return surface_ != VK_NULL_HANDLE; }

 private:
  void destroy() noexcept;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
};

}  // namespace volumetric_kit::gfx::windowing

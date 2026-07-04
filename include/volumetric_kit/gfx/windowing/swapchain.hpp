// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file swapchain.hpp
/// @brief A `VkSwapchainKHR` plus a @ref RenderTarget per image — the
///        window-backed producer of render targets.

#include <vector>

#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/windowing/export.hpp"

namespace volumetric_kit::gfx {

class Device;

namespace windowing {

/// @brief Parameters for @ref Swapchain::create. Each preference falls back to
/// a
///        guaranteed-supported value when the surface does not offer it.
struct SwapchainConfig {
  /// Desired image size in texels; clamped to the surface's supported range, or
  /// overridden by the surface's fixed `currentExtent` when it dictates one.
  VkExtent2D extent{};
  /// Preferred color format; falls back to the surface's first supported pair.
  VkFormat preferred_format = VK_FORMAT_B8G8R8A8_SRGB;
  /// Color space paired with @ref preferred_format.
  VkColorSpaceKHR preferred_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  /// Preferred present mode; falls back to FIFO (always supported).
  VkPresentModeKHR preferred_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
  /// Requested image count; `0` selects the surface minimum + 1. Clamped to the
  /// surface's supported `[minImageCount, maxImageCount]`.
  uint32_t min_image_count = 0;
};

/// @brief Owns a `VkSwapchainKHR`, an image view per swapchain image, and a
///        @ref RenderTarget over each — the windowing-tier sibling of
///        @ref OffscreenTarget, so a pass renders into either.
///
/// Format/color-space/present-mode are chosen once at @ref create and held
/// stable across @ref recreate, so a pipeline built for @ref layout stays
/// compatible after a resize. @ref acquire_next_image / @ref present surface
/// `VK_ERROR_OUT_OF_DATE_KHR` as a non-OK @ref Status whose @ref Status::code
/// is that result, so the caller knows to @ref recreate. A default-constructed
/// `Swapchain` is empty (`valid()` is false).
///
/// @warning The @p device and surface passed to @ref create must outlive the
///          swapchain (it borrows both). Destroy the swapchain before its
///          surface and device.
///
/// @code
/// auto sc = windowing::Swapchain::create(device, surface.handle(),
///                                        {.extent = {1280, 720}});
/// if (!sc) return sc.status();
/// // ... build a pipeline for sc.value().layout(), then drive it via FrameLoop
/// @endcode
class VG_WINDOWING_API Swapchain {
 public:
  /// @brief Construct an empty swapchain (owns nothing; `valid()` is false).
  Swapchain() = default;

  /// @brief Create a swapchain on @p surface.
  /// @param device   A device created with present support
  ///                 (`DeviceConfig::needs_present`).
  /// @param surface  The surface to present to.
  /// @param config   Size and format/present-mode preferences.
  /// @return The swapchain on success, or a non-OK @ref Status:
  ///         a null @p surface or a non-present @p device returns
  ///         @ref Status::Code::InvalidArgument; a surface with no formats /
  ///         present modes returns @ref Status::Code::Unsupported; a failed
  ///         Vulkan call carries its `VkResult`.
  static Result<Swapchain> create(const Device& device, VkSurfaceKHR surface,
                                  const SwapchainConfig& config);

  ~Swapchain();
  Swapchain(Swapchain&& other) noexcept;
  Swapchain& operator=(Swapchain&& other) noexcept;
  Swapchain(const Swapchain&) = delete;
  Swapchain& operator=(const Swapchain&) = delete;

  /// @brief Acquire the next image to render into.
  /// @param image_available  Signaled when the image is ready to render to;
  ///                         a submission rendering to it must wait on this.
  /// @param timeout_ns       Maximum wait, in nanoseconds.
  /// @return The acquired image index. `VK_SUBOPTIMAL_KHR` still returns an
  ///         index (render proceeds); `VK_ERROR_OUT_OF_DATE_KHR` returns a
  ///         non-OK @ref Status carrying that code — @ref recreate and retry.
  ///         An empty swapchain returns @ref Status::Code::InvalidArgument.
  Result<uint32_t> acquire_next_image(VkSemaphore image_available,
                                      uint64_t timeout_ns = UINT64_MAX);

  /// @brief Present image @p image_index on the device's present queue.
  /// @param image_index     An index from @ref acquire_next_image.
  /// @param render_finished  Waited before presenting (signaled by the render
  ///                        submission).
  /// @return OK on success; a non-OK @ref Status carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR` when the
  ///         swapchain should be recreated, or another failed `VkResult`.
  ///         An empty swapchain returns @ref Status::Code::InvalidArgument.
  Status present(uint32_t image_index, VkSemaphore render_finished);

  /// @brief Rebuild the swapchain for @p extent (resize / out-of-date), keeping
  ///        the format and present mode. Idles the device, then hands the old
  ///        chain to `vkCreateSwapchainKHR` as `oldSwapchain` so the driver can
  ///        carry resources across the resize.
  /// @param extent  The new desired size (clamped/overridden as in @ref
  /// create).
  /// @return OK on success. A zero @p extent (minimized window) fails with
  ///         @ref Status::Code::InvalidArgument *without touching the current
  ///         swapchain* — it stays valid for a later retry; only a failed
  ///         `vkCreateSwapchainKHR` leaves the object empty (`valid()` false).
  Status recreate(VkExtent2D extent);

  /// @return The render target for swapchain image @p image_index.
  /// @pre @p image_index < @ref image_count.
  const RenderTarget& render_target(uint32_t image_index) const;

  /// @return The `VkImage` for swapchain image @p image_index — for the
  ///         layout-transition barriers dynamic rendering leaves to the caller
  ///         (e.g. @ref FrameLoop's present/attachment transitions).
  /// @pre @p image_index < @ref image_count.
  VkImage image(uint32_t image_index) const;

  /// @return The color `VkImageView` for swapchain image @p image_index — the
  ///         view `vkCmdBeginRendering` draws through. Exposed so a caller can
  ///         assemble its own @ref RenderTarget over this image (e.g. pairing
  ///         it with an externally-owned depth attachment) instead of using the
  ///         color-only @ref render_target.
  /// @pre @p image_index < @ref image_count.
  VkImageView image_view(uint32_t image_index) const;

  /// @return The format + sample signature shared by every image, for building
  /// a
  ///         compatible pipeline.
  RenderTargetLayout layout() const noexcept;

  /// @return The current image extent in texels.
  VkExtent2D extent() const noexcept { return extent_; }
  /// @return The chosen color format.
  VkFormat format() const noexcept { return format_; }
  /// @return The number of swapchain images.
  uint32_t image_count() const noexcept {
    return static_cast<uint32_t>(targets_.size());
  }
  /// @return The underlying `VkSwapchainKHR` (`VK_NULL_HANDLE` when empty).
  VkSwapchainKHR handle() const noexcept { return swapchain_; }
  /// @return `true` if this owns a swapchain.
  bool valid() const noexcept { return swapchain_ != VK_NULL_HANDLE; }

 private:
  Status select_surface_properties(const SwapchainConfig& config);
  Status build(VkExtent2D extent);
  Status create_image_resources(VkExtent2D extent);  // views + render targets
  void destroy_resources() noexcept;                 // image views + swapchain
  void reset_state() noexcept;  // null handles + zero metadata to empty
  void destroy() noexcept;

  const Device* device_ = nullptr;         // borrowed; outlives this
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;  // borrowed; outlives this
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;  // owned by the swapchain, not freed by us
  std::vector<VkImageView> views_;
  std::vector<RenderTarget> targets_;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  VkColorSpaceKHR color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  VkExtent2D extent_{};
  uint32_t requested_min_image_count_ = 0;
};

}  // namespace windowing
}  // namespace volumetric_kit::gfx

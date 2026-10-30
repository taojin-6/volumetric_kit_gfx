// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file swapchain.hpp
/// @brief A `VkSwapchainKHR` plus a @ref RenderTarget per image — the
///        window-backed producer of render targets.

#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/windowing/export.hpp"

namespace volumetric_kit::core {
class Allocator;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx {

namespace windowing {

/// @brief Parameters for @ref Swapchain::create. Each preference falls back to
/// a
///        guaranteed-supported value when the surface does not offer it.
struct SwapchainConfig {
  /// Desired image size in texels; clamped to the surface's supported range, or
  /// overridden by the surface's fixed `currentExtent` when it dictates one.
  VkExtent2D extent{};
  /// Preferred color format; falls back to an `_SRGB` format the surface
  /// offers in @ref preferred_color_space, then to its first supported pair.
  VkFormat preferred_format = VK_FORMAT_B8G8R8A8_SRGB;
  /// Color space paired with @ref preferred_format.
  VkColorSpaceKHR preferred_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  /// Preferred present mode; falls back to FIFO (always supported). FIFO, the
  /// default, paces frames to the display; MAILBOX renders as fast as the GPU
  /// allows and blocks less with a third image (@ref min_image_count).
  VkPresentModeKHR preferred_present_mode = VK_PRESENT_MODE_FIFO_KHR;
  /// Requested image count; `0` selects the surface minimum + 1. Clamped to the
  /// surface's supported `[minImageCount, maxImageCount]`.
  uint32_t min_image_count = 0;
  /// Depth attachment format, or `VK_FORMAT_UNDEFINED` (the default) for a
  /// color-only swapchain. When set, the swapchain's multisampled set also
  /// holds a depth image, so each @ref Swapchain::render_target is
  /// depth-capable.
  ///
  /// @note Depth lives for one rendering scope: a multisampled scope neither
  ///       loads nor stores it (@ref RenderTarget::begin), so clear it every
  ///       frame (`load_op = CLEAR`, the default).
  VkFormat depth_format = VK_FORMAT_UNDEFINED;
};

/// @brief Classify a status from acquire / present / @ref FrameLoop as
///        "swapchain stale": the swapchain merely needs recreation
///        (`VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR`), as opposed to an
///        error to abort on (device loss, invalid usage).
/// @param status  A status returned by @ref Swapchain::acquire_next_image,
///                @ref Swapchain::present, or the @ref FrameLoop frame calls.
/// @return `true` when the right response is to recreate and continue.
inline bool swapchain_stale(const core::Status& status) noexcept {
  const std::optional<VkResult> result = core::vk_result(status);
  return result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR;
}

/// @brief Owns a `VkSwapchainKHR`, an image view per swapchain image, one
///        multisampled colour (+ depth) set, and a @ref RenderTarget over each
///        image — the windowing-tier sibling of @ref OffscreenTarget, so a
///        pass renders into either.
///
/// Every window renders at @ref kWindowSamples. Each image's
/// @ref render_target draws into the one shared multisampled set and, when
/// its scope ends, resolves the colour into that image, so the caller
/// transitions only the swapchain image, as @ref FrameLoop does. One set
/// serves every image because @ref RenderTarget::begin orders each scope
/// after the previous one on the same queue. The set is `TRANSIENT`.
///
/// Format/color-space/present-mode are chosen once at @ref create and held
/// stable across @ref recreate, so a pipeline built for @ref layout stays
/// compatible after a resize; the multisampled set is rebuilt at the new
/// extent. @ref acquire_next_image / @ref present surface
/// `VK_ERROR_OUT_OF_DATE_KHR` as a non-OK `core::Status` whose
/// `core::Status::code` is that result, so the caller knows to @ref recreate.
/// A default-constructed `Swapchain` is empty (`valid()` is false).
///
/// @warning The @p device, surface and @p allocator passed to @ref create must
///          outlive the swapchain (it borrows all three). Destroy the
///          swapchain before its surface, allocator, and device.
///
/// @code
/// auto sc = windowing::Swapchain::create(
///     device, surface.handle(),
///     {.extent = {1280, 720}, .depth_format = VK_FORMAT_D32_SFLOAT},
///     allocator);
/// if (!sc) return sc.status();
/// // ... build a pipeline for sc.value().layout(), then drive it via FrameLoop
/// @endcode
class VG_WINDOWING_API Swapchain {
 public:
  /// @brief Construct an empty swapchain (owns nothing; `valid()` is false).
  Swapchain() = default;

  /// @brief Create a swapchain on @p surface.
  /// @param device     A device created with present support
  ///                   (`DeviceRequirements::needs_present`) that enabled the
  ///                   renderer's requirements (@ref device_requirements).
  /// @param surface    The surface to present to.
  /// @param config     Size, format/present-mode preferences, and the optional
  ///                   depth attachment format.
  /// @param allocator  Allocates the multisampled set (and reallocates it on
  ///                   every @ref recreate). Borrowed; must outlive the
  ///                   swapchain.
  /// @return The swapchain on success, or a non-OK `core::Status`:
  ///         a null @p surface, a non-present @p device, or a non-depth
  ///         `config.depth_format` returns
  ///         `core::Status::Code::InvalidArgument`; a surface with no formats
  ///         / present modes, a `config.depth_format` the device cannot render
  ///         depth through (stencil aspect, or no optimal-tiling
  ///         depth-attachment support), a @p device whose
  ///         `framebufferColorSampleCounts` (or, with a depth format,
  ///         `framebufferDepthSampleCounts`) lacks @ref kWindowSamples, or a
  ///         @p device that did not enable the renderer's requirements,
  ///         returns `core::Status::Code::Unsupported`; a failed Vulkan call
  ///         or allocation propagates its status.
  static core::Result<Swapchain> create(const core::Device& device,
                                        VkSurfaceKHR surface,
                                        const SwapchainConfig& config,
                                        core::Allocator& allocator);

  ~Swapchain();
  // TODO: an aggregate (DECISIONS.md, 2026-10-10): delete the move pair and
  // hand it out by std::unique_ptr, as FrameLoop is.
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
  ///         non-OK `core::Status` carrying that code — @ref recreate and
  ///         retry. An empty swapchain returns
  ///         `core::Status::Code::InvalidArgument`.
  core::Result<uint32_t> acquire_next_image(VkSemaphore image_available,
                                            uint64_t timeout_ns = UINT64_MAX);

  /// @brief Present image @p image_index on the device's present queue.
  /// @param image_index     An index from @ref acquire_next_image.
  /// @param render_finished  Waited before presenting (signaled by the render
  ///                        submission).
  /// @return OK on success; a non-OK `core::Status` carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR` when the
  ///         swapchain should be recreated, or another failed `VkResult`.
  ///         An empty swapchain returns `core::Status::Code::InvalidArgument`.
  core::Status present(uint32_t image_index, VkSemaphore render_finished);

  /// @brief Rebuild the swapchain for @p extent (resize / out-of-date), keeping
  ///        the format and present mode. Idles the device, then hands the old
  ///        chain to `vkCreateSwapchainKHR` as `oldSwapchain` so the driver can
  ///        carry resources across the resize.
  /// @param extent  The new desired size (clamped/overridden as in @ref
  /// create).
  /// @return OK on success. A zero @p extent (minimized window) fails with
  ///         `core::Status::Code::InvalidArgument` *without touching the
  ///         current swapchain*, which stays valid for a later retry. A failed
  ///         rebuild (`vkCreateSwapchainKHR`, or the image-view creation or
  ///         multisampled-set allocation that follows) empties the object
  ///         (`valid()` false) but keeps its device / surface / format, so a
  ///         later @ref recreate can retry it once the transient failure
  ///         clears. Called on a moved-from or
  ///         default-constructed swapchain (no device to rebuild on), fails
  ///         with `core::Status::Code::InvalidArgument`.
  core::Status recreate(VkExtent2D extent);

  /// @return The render target for swapchain image @p image_index.
  /// @pre @p image_index < @ref image_count.
  const RenderTarget& render_target(uint32_t image_index) const;

  /// @return The `VkImage` for swapchain image @p image_index — for the
  ///         layout-transition barriers dynamic rendering leaves to the caller
  ///         (e.g. @ref FrameLoop's present/attachment transitions).
  /// @pre @p image_index < @ref image_count.
  VkImage image(uint32_t image_index) const;

  /// @return The color `VkImageView` for swapchain image @p image_index — the
  ///         view @ref render_target resolves into. A @ref RenderTarget a
  ///         caller assembles over it alone is single-sample, so it does not
  ///         match @ref layout: build its pipelines for its own layout.
  /// @pre @p image_index < @ref image_count.
  VkImageView image_view(uint32_t image_index) const;

  /// @return The format + sample signature shared by every image's
  ///         @ref render_target (`samples` is @ref kWindowSamples), for
  ///         building a compatible pipeline.
  RenderTargetLayout layout() const noexcept;

  /// @return The current image extent in texels (as clamped/overridden by the
  ///         surface).
  VkExtent2D extent() const noexcept { return extent_; }
  /// @return The extent last *requested* of @ref create / @ref recreate, before
  ///         the surface clamped or overrode it (which @ref extent reflects).
  ///         A resize check against this (not @ref extent) does not rebuild
  ///         every frame when the surface pins a request to a different size.
  VkExtent2D requested_extent() const noexcept { return requested_extent_; }
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
  core::Status select_surface_properties(const SwapchainConfig& config);
  core::Status build(VkExtent2D extent);
  // Image views + the multisampled set + render targets.
  core::Status create_image_resources(VkExtent2D extent);
  void destroy_resources() noexcept;  // targets + set + views + swapchain
  void reset_state() noexcept;        // null handles + zero metadata to empty
  void destroy() noexcept;

  const core::Device* device_ = nullptr;   // borrowed; outlives this
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;  // borrowed; outlives this
  core::Allocator* allocator_ = nullptr;   // borrowed; outlives this
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;  // owned by the swapchain, not freed by us
  std::vector<VkImageView> views_;
  // The multisampled set every target draws into: the swapchain's format,
  // COLOR_ATTACHMENT | TRANSIENT; and DEPTH_STENCIL | TRANSIENT depth, empty
  // when depth_format_ is not set.
  core::Image msaa_color_;
  core::Image msaa_depth_;
  std::vector<RenderTarget> targets_;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
  VkColorSpaceKHR color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  VkExtent2D extent_{};            // current image size, surface-clamped
  VkExtent2D requested_extent_{};  // last requested size, pre-clamp
  uint32_t requested_min_image_count_ = 0;
};

}  // namespace windowing
}  // namespace volumetric_kit::gfx

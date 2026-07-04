// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file swapchain.hpp
/// @brief A `VkSwapchainKHR` plus a @ref RenderTarget per image — the
///        window-backed producer of render targets.

#include <vector>

#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/windowing/export.hpp"

namespace volumetric_kit::gfx {

class Allocator;
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
  /// Depth attachment format, or `VK_FORMAT_UNDEFINED` (the default) for a
  /// color-only swapchain. When set, the swapchain owns one depth attachment
  /// *per swapchain image* — sized with the chain and rebuilt on every
  /// @ref Swapchain::recreate — so each @ref Swapchain::render_target is
  /// depth-capable and depth is safe at any frames-in-flight count (no
  /// cross-frame sharing of one depth image). Requires the @p allocator
  /// argument of @ref Swapchain::create.
  ///
  /// @note The depth image is transitioned once into its attachment layout and
  ///       carries no per-frame barrier; cross-frame correctness relies on each
  ///       frame *clearing* depth (`load_op = CLEAR`, the default). A consumer
  ///       that instead loads a prior frame's depth (temporal reuse) must
  ///       insert its own barrier — the swapchain does not synchronize a
  ///       cross-frame depth read.
  VkFormat depth_format = VK_FORMAT_UNDEFINED;
};

/// @brief Classify a status from acquire / present / @ref FrameLoop as
///        "swapchain stale": the swapchain merely needs recreation
///        (`VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR`), as opposed to an
///        error to abort on (device loss, invalid usage).
/// @param status  A status returned by @ref Swapchain::acquire_next_image,
///                @ref Swapchain::present, or the @ref FrameLoop frame calls.
/// @return `true` when the right response is to recreate and continue.
inline bool swapchain_stale(const Status& status) noexcept {
  return status.domain() == Status::Code::Vulkan &&
         (status.code() == VK_ERROR_OUT_OF_DATE_KHR ||
          status.code() == VK_SUBOPTIMAL_KHR);
}

/// @brief Owns a `VkSwapchainKHR`, an image view per swapchain image, an
///        optional depth attachment per image, and a @ref RenderTarget over
///        each — the windowing-tier sibling of @ref OffscreenTarget, so a pass
///        renders into either.
///
/// Format/color-space/present-mode are chosen once at @ref create and held
/// stable across @ref recreate, so a pipeline built for @ref layout stays
/// compatible after a resize. With @ref SwapchainConfig::depth_format set, the
/// per-image depth attachments are likewise rebuilt (and re-transitioned into
/// their attachment layout) on every @ref recreate. @ref acquire_next_image /
/// @ref present surface `VK_ERROR_OUT_OF_DATE_KHR` as a non-OK @ref Status
/// whose @ref Status::code is that result, so the caller knows to @ref
/// recreate. A default-constructed `Swapchain` is empty (`valid()` is false).
///
/// @warning The @p device and surface passed to @ref create must outlive the
///          swapchain (it borrows both), as must the @p allocator when
///          @ref SwapchainConfig::depth_format is set. Destroy the swapchain
///          before its surface, allocator, and device.
///
/// @code
/// auto sc = windowing::Swapchain::create(
///     device, surface.handle(),
///     {.extent = {1280, 720}, .depth_format = VK_FORMAT_D32_SFLOAT},
///     &allocator);
/// if (!sc) return sc.status();
/// // ... build a pipeline for sc.value().layout(), then drive it via FrameLoop
/// @endcode
class VG_WINDOWING_API Swapchain {
 public:
  /// @brief Construct an empty swapchain (owns nothing; `valid()` is false).
  Swapchain() = default;

  /// @brief Create a swapchain on @p surface.
  /// @param device     A device created with present support
  ///                   (`DeviceConfig::needs_present`).
  /// @param surface    The surface to present to.
  /// @param config     Size, format/present-mode preferences, and the optional
  ///                   depth attachment format.
  /// @param allocator  Required when `config.depth_format` is set: it allocates
  ///                   the per-image depth attachments (and reallocates them on
  ///                   every @ref recreate). Borrowed; must outlive the
  ///                   swapchain. Unused for a color-only swapchain.
  /// @return The swapchain on success, or a non-OK @ref Status:
  ///         a null @p surface, a non-present @p device, a non-depth
  ///         `config.depth_format`, or a set `config.depth_format` without an
  ///         @p allocator returns @ref Status::Code::InvalidArgument; a surface
  ///         with no formats / present modes, or a `config.depth_format` the
  ///         device cannot render depth through (stencil aspect, or no
  ///         optimal-tiling depth-attachment support), returns
  ///         @ref Status::Code::Unsupported; a failed Vulkan call carries its
  ///         `VkResult`.
  static Result<Swapchain> create(const Device& device, VkSurfaceKHR surface,
                                  const SwapchainConfig& config,
                                  Allocator* allocator = nullptr);

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
  ///         swapchain*, which stays valid for a later retry. A failed rebuild
  ///         (`vkCreateSwapchainKHR`, or the image-view creation that follows)
  ///         empties the object (`valid()` false) but keeps its device /
  ///         surface / format, so a later @ref recreate can retry it once the
  ///         transient failure clears. Called on a moved-from or
  ///         default-constructed swapchain (no device to rebuild on), fails
  ///         with
  ///         @ref Status::Code::InvalidArgument.
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
  ///         it with externally-owned attachments) instead of using
  ///         @ref render_target.
  /// @pre @p image_index < @ref image_count.
  VkImageView image_view(uint32_t image_index) const;

  /// @return The format + sample signature shared by every image, for building
  /// a
  ///         compatible pipeline.
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
  Status select_surface_properties(const SwapchainConfig& config);
  Status build(VkExtent2D extent);
  // Image views + optional per-image depth attachments + render targets.
  Status create_image_resources(VkExtent2D extent);
  void destroy_resources() noexcept;  // views + depth + swapchain
  void reset_state() noexcept;        // null handles + zero metadata to empty
  void destroy() noexcept;

  const Device* device_ = nullptr;         // borrowed; outlives this
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;  // borrowed; outlives this
  // Borrowed; outlives this. Non-null only when depth_format_ is set.
  Allocator* allocator_ = nullptr;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;  // owned by the swapchain, not freed by us
  std::vector<VkImageView> views_;
  std::vector<Texture> depth_textures_;  // one per image when depth_format_ set
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

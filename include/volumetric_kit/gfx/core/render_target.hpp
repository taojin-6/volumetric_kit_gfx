// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file render_target.hpp
/// @brief What a draw renders into: an attachment set (@ref RenderTarget) and
///        the format/sample signature (@ref RenderTargetLayout) a pipeline is
///        built against.

#include <array>
#include <cstdint>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief The format + sample signature of a render target: the compatibility
///        currency between a @ref GraphicsPipeline and a @ref RenderTarget.
///
/// A pipeline is built for a layout (its color/depth formats and sample count
/// are baked into `VkPipelineRenderingCreateInfo`); any target with a @ref
/// compatible_with layout accepts it. This is what lets one pipeline draw into
/// an offscreen image, a swapchain image, and an OpenXR view buffer without
/// rebuilding — they share a layout, not a render-pass object.
///
/// @code
/// RenderTargetLayout layout;
/// layout.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
/// layout.color_count = 1;
/// // ... GraphicsPipelineDesc{.layout = layout} ...
/// @endcode
struct RenderTargetLayout {
  /// Upper bound on simultaneous color attachments this kit supports. The
  /// Vulkan-guaranteed minimum `maxColorAttachments` is 4; 8 covers real
  /// hardware. A pipeline rejects a `color_count` above this cap; the driver
  /// enforces the device's actual `maxColorAttachments` at pipeline creation.
  static constexpr uint32_t kMaxColorAttachments = 8;

  /// Color attachment formats; only the first @ref color_count are significant.
  std::array<VkFormat, kMaxColorAttachments> color_formats{};
  /// Number of color attachments (0..@ref kMaxColorAttachments).
  uint32_t color_count = 0;
  /// Depth(/stencil) attachment format, or `VK_FORMAT_UNDEFINED` for none.
  VkFormat depth_format = VK_FORMAT_UNDEFINED;
  /// Sample count shared by every attachment.
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

  /// @brief Whether a pipeline built for this layout can draw into a target of
  ///        @p other: same color count + formats, depth format, and samples.
  /// @param other  The layout to compare against.
  /// @return `true` if the two are render-compatible.
  bool compatible_with(const RenderTargetLayout& other) const noexcept;
};

/// @brief One attachment of a @ref RenderTarget: the image, the view a draw
///        renders through, and its format.
///
/// The @ref image is carried alongside the @ref view so the caller can issue
/// the layout-transition barriers dynamic rendering leaves to the application
/// (e.g. `UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL` before a draw); @ref
/// RenderTarget::begin uses only the @ref view.
struct RenderTargetAttachment {
  VkImage image = VK_NULL_HANDLE;         ///< For the caller's layout barriers.
  VkImageView view = VK_NULL_HANDLE;      ///< Fed to `vkCmdBeginRendering`.
  VkFormat format = VK_FORMAT_UNDEFINED;  ///< The attachment's format.
};

/// @brief Per-begin load/store ops and clear values passed to @ref
///        RenderTarget::begin.
///
/// The ops apply uniformly to every attachment; @ref clear_color is used by
/// each color attachment, and @ref clear_depth by the depth attachment, when
/// @ref load_op is `VK_ATTACHMENT_LOAD_OP_CLEAR`. Per-attachment clears arrive
/// with multi-attachment (MRT) support.
struct RenderTargetBeginInfo {
  VkAttachmentLoadOp load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
  VkAttachmentStoreOp store_op = VK_ATTACHMENT_STORE_OP_STORE;
  VkClearColorValue clear_color{};  ///< When @ref load_op clears a color view.
  /// When @ref load_op clears a depth attachment; `depth = 1.0` pairs with a
  /// `VK_COMPARE_OP_LESS` test (clear to the far plane).
  VkClearDepthStencilValue clear_depth{1.0f, 0};
};

/// @brief A non-owning view of the attachments a draw renders into, plus the
///        `vkCmdBeginRendering` / `vkCmdEndRendering` scope around it.
///
/// Unlike the handle-owning core types, `RenderTarget` owns nothing — it is a
/// value bundle of attachment handles (like a `std::span` over images held
/// elsewhere), so it is freely copyable and has no deleter. Producers own the
/// images and hand out a target per frame: @ref OffscreenTarget (headless)
/// today; a swapchain (windowing) and an OpenXR swapchain (xr) later.
///
/// Layout transitions are the caller's responsibility (dynamic rendering does
/// not perform them): transition each attachment image into its attachment
/// layout before @ref begin, and out of it (to present / transfer / sampled)
/// after @ref end. @ref RenderTargetAttachment::image is exposed for exactly
/// those barriers — record them with @ref cmd_image_barrier
/// (core/image_barrier.hpp), or let the producing target do it (e.g.
/// @ref OffscreenTarget::prepare).
///
/// @warning The images/views this references are owned by the producing target,
///          which must outlive every `RenderTarget` it hands out; the views
///          must stay valid across the @ref begin / @ref end scope.
///
/// @code
/// RenderTarget rt = offscreen.target();
/// // ... barrier the color image to COLOR_ATTACHMENT_OPTIMAL ...
/// RenderTargetBeginInfo begin;
/// begin.clear_color.float32[3] = 1.0f;  // opaque black
/// rt.begin(cmd, begin);
/// // ... bind pipeline, set viewport/scissor, draw ...
/// rt.end(cmd);
/// @endcode
class VG_CORE_API RenderTarget {
 public:
  /// @brief Construct an empty target (no attachments; `valid()` is false).
  RenderTarget() = default;

  /// @brief Bundle @p color attachments and an optional @p depth attachment.
  /// @param extent       Render-area size in texels (every attachment's size).
  /// @param color        Pointer to @p color_count color attachments.
  /// @param color_count  Number of color attachments
  ///                     (1..@ref RenderTargetLayout::kMaxColorAttachments).
  /// @param samples      Sample count shared by every attachment.
  /// @param depth        Optional depth attachment, or `nullptr` for a
  ///                     color-only target.
  RenderTarget(VkExtent2D extent, const RenderTargetAttachment* color,
               uint32_t color_count, VkSampleCountFlagBits samples,
               const RenderTargetAttachment* depth = nullptr);

  /// @brief Begin a dynamic-rendering scope over these attachments.
  /// @param cmd   A command buffer in the recording state.
  /// @param info  Load/store ops and clear values for this scope.
  /// @pre `valid()`; every attachment image is already in its attachment
  /// layout.
  void begin(VkCommandBuffer cmd, const RenderTargetBeginInfo& info) const;

  /// @brief End the dynamic-rendering scope opened by @ref begin.
  /// @param cmd  The same command buffer passed to @ref begin.
  void end(VkCommandBuffer cmd) const;

  /// @return The render-area extent in texels.
  VkExtent2D extent() const noexcept { return extent_; }

  /// @return The format + sample signature, for building a compatible pipeline.
  RenderTargetLayout layout() const noexcept;

  /// @return `true` if this has at least one color attachment.
  bool valid() const noexcept { return color_count_ > 0; }

 private:
  /// @return Whether a depth attachment is bundled (its view is non-null).
  bool has_depth() const noexcept { return depth_.view != VK_NULL_HANDLE; }

  VkExtent2D extent_{};
  std::array<RenderTargetAttachment, RenderTargetLayout::kMaxColorAttachments>
      color_{};
  uint32_t color_count_ = 0;
  RenderTargetAttachment depth_{};  ///< Null view for a color-only target.
  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
};

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file render_target.hpp
/// @brief What a draw renders into: an attachment set (@ref RenderTarget) and
///        the format/sample signature (@ref RenderTargetLayout) a pipeline is
///        built against.

#include <array>
#include <cstdint>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/export.hpp"

namespace volumetric_kit::gfx {

/// The sample count gfx multisamples at: the one count besides 1 an
/// @ref OffscreenTarget takes. Vulkan guarantees it for depth and for every
/// colour format that averages when resolved (UNORM, sRGB and float), on
/// every device, so a target of this count needs no device query.
inline constexpr VkSampleCountFlagBits kWindowSamples = VK_SAMPLE_COUNT_4_BIT;

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
/// (e.g. `UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL` before a draw). At one sample
/// @ref RenderTarget::begin uses only the @ref view; above one it also
/// transitions the multisampled attachments' @ref image itself.
struct RenderTargetAttachment {
  VkImage image = VK_NULL_HANDLE;         ///< For the layout barriers.
  VkImageView view = VK_NULL_HANDLE;      ///< Fed to `vkCmdBeginRendering`.
  VkFormat format = VK_FORMAT_UNDEFINED;  ///< The attachment's format.
};

/// @brief Per-begin load/store ops and clear values passed to @ref
///        RenderTarget::begin.
///
/// @ref load_op applies to every attachment; @ref clear_color is used by each
/// color attachment, and @ref clear_depth by the depth attachment, when @ref
/// load_op is `VK_ATTACHMENT_LOAD_OP_CLEAR`. Per-attachment clears arrive with
/// multi-attachment (MRT) support.
///
/// On a multisampled target the samples never outlive the scope: the colours
/// resolve into the target's resolve attachments, which always store, so
/// @ref store_op is ignored, and a @ref load_op of `LOAD` or a
/// @ref depth_store_op of `STORE` is a programmer error (`VKC_CHECK`).
struct RenderTargetBeginInfo {
  VkAttachmentLoadOp load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
  /// The color attachments' store op.
  VkAttachmentStoreOp store_op = VK_ATTACHMENT_STORE_OP_STORE;
  /// The depth attachment's store op. `DONT_CARE` drops depth when the scope
  /// ends, so a tile-based GPU does not write it back to memory; set `STORE`
  /// when a later scope loads it.
  VkAttachmentStoreOp depth_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
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
/// not perform them): at one sample, transition each attachment image into
/// its attachment layout before @ref begin, and out of it (to present /
/// transfer / sampled) after @ref end. @ref RenderTargetAttachment::image is
/// exposed for exactly those barriers — record them with @ref cmd_image_barrier
/// (core/image_barrier.hpp), or let the producing target do it (e.g.
/// @ref OffscreenTarget::prepare).
///
/// A multisampled target (samples above one) also bundles one single-sample
/// resolve attachment per colour. Its scope keeps no samples: @ref begin
/// transitions the multisampled colour and depth images itself, and @ref end
/// resolves each colour (`AVERAGE`) into its resolve attachment. So the caller
/// transitions only the resolve images, exactly as it transitions a
/// single-sample target's images; one multisampled set may then serve every
/// target that resolves into a different image, such as each image of a
/// swapchain.
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
  /// @param resolve      Required when @p samples is above one, and `nullptr`
  ///                     otherwise: @p color_count single-sample attachments
  ///                     the colours resolve into, each with a view and of
  ///                     its colour's format (all checked with `VKC_CHECK`).
  RenderTarget(VkExtent2D extent, const RenderTargetAttachment* color,
               uint32_t color_count, VkSampleCountFlagBits samples,
               const RenderTargetAttachment* depth = nullptr,
               const RenderTargetAttachment* resolve = nullptr);

  /// @brief Begin a dynamic-rendering scope over these attachments.
  ///
  /// Above one sample it first records one barrier that moves the
  /// multisampled colour and depth images from `UNDEFINED` to their
  /// attachment layouts, after any earlier scope's attachment work on them,
  /// and the scope stores them `DONT_CARE` and resolves each colour into its
  /// resolve attachment.
  /// @param cmd   A command buffer in the recording state.
  /// @param info  Load/store ops and clear values for this scope.
  /// @pre `valid()`; at one sample every attachment image, and above one every
  ///      resolve image, is already in its attachment layout
  ///      (`COLOR_ATTACHMENT_OPTIMAL` for a resolve image); above one,
  ///      @p info's `load_op` is not `LOAD` and its `depth_store_op` not
  ///      `STORE` (checked with `VKC_CHECK`).
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
  /// The colours' resolve attachments; only above one sample.
  std::array<RenderTargetAttachment, RenderTargetLayout::kMaxColorAttachments>
      resolve_{};
  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
};

}  // namespace volumetric_kit::gfx

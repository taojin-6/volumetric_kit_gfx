// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file offscreen_target.hpp
/// @brief A headless render target: a device-local color image (with an
///        optional depth image) and a host-visible readback buffer.

#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

class Allocator;

/// @brief Parameters for @ref OffscreenTarget::create.
struct OffscreenTargetDesc {
  /// Width/height of every attachment, in texels. Must be non-zero.
  VkExtent2D extent{};
  /// Color attachment format. Must not be `VK_FORMAT_UNDEFINED`.
  VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
  /// Optional depth attachment format, or `VK_FORMAT_UNDEFINED` for a
  /// color-only target. Must be a depth-only format (e.g.
  /// `VK_FORMAT_D32_SFLOAT`) to render with a depth-testing pipeline; combined
  /// depth/stencil formats are not yet supported. The chosen format is reported
  /// by @ref layout.
  VkFormat depth_format = VK_FORMAT_UNDEFINED;
  /// Allocate a host-visible buffer sized to the color attachment so the render
  /// can be copied back to the CPU (see @ref OffscreenTarget::record_readback).
  bool readback = true;
};

/// @brief Owns the images a headless render draws into, plus the staging buffer
///        it is read back through — the no-window path for golden-image tests
///        and offscreen capture.
///
/// The color attachment is created `COLOR_ATTACHMENT | TRANSFER_SRC` so it can
/// be both rendered into and copied out; an optional depth attachment is added
/// when @ref OffscreenTargetDesc::depth_format is set. @ref target hands out a
/// non-owning @ref RenderTarget over the attachments, @ref layout gives the
/// matching pipeline signature, @ref prepare records the transitions into the
/// attachment layouts, and @ref record_readback records the copy-to-buffer the
/// readback path needs. This is the headless sibling of the windowing tier's
/// swapchain; both produce a @ref RenderTarget, so a pass renders into either
/// unchanged.
///
/// @warning The @p allocator passed to @ref create must outlive the target: the
///          owned @ref Texture / @ref Buffer free through it. Retire the target
///          ahead of allocator teardown.
///
/// @code
/// Result<OffscreenTarget> rt = OffscreenTarget::create(
///     allocator, {.extent = {1280, 720}, .color_format = kFormat});
/// if (!rt) return rt.status();
/// rt.value().prepare(cmd);              // attachments -> attachment layouts
/// rt.value().target().begin(cmd, {});   // draw ...   .end(cmd);
/// rt.value().record_readback(cmd);      // copy color -> host buffer
/// // ... submit + fence-wait ...
/// const auto* pixels = static_cast<const uint8_t*>(rt.value().pixels());
/// @endcode
class VG_CORE_API OffscreenTarget {
 public:
  /// @brief Construct an empty target (owns nothing; `valid()` is false).
  OffscreenTarget() = default;

  /// @brief Allocate the attachment image(s) and readback buffer for @p desc.
  /// @param allocator  The allocator that backs (and outlives) the images.
  /// @param desc       Extent, formats, and whether to allocate readback.
  /// @return The target on success, or a non-OK @ref Status:
  ///         - a zero @ref OffscreenTargetDesc::extent or a
  ///         `VK_FORMAT_UNDEFINED`
  ///           @ref OffscreenTargetDesc::color_format returns
  ///           @ref Status::Code::InvalidArgument;
  ///         - a @ref OffscreenTargetDesc::readback request on a color format
  ///           whose texel size this kit does not yet know returns
  ///           @ref Status::Code::Unsupported;
  ///         - a non-depth @ref OffscreenTargetDesc::depth_format returns
  ///           @ref Status::Code::InvalidArgument, and a combined depth/stencil
  ///           one @ref Status::Code::Unsupported (depth-only for now);
  ///         - a failed image/buffer allocation propagates its @ref Status.
  static Result<OffscreenTarget> create(Allocator& allocator,
                                        const OffscreenTargetDesc& desc);

  ~OffscreenTarget() = default;
  OffscreenTarget(OffscreenTarget&&) noexcept = default;
  OffscreenTarget& operator=(OffscreenTarget&&) noexcept = default;
  OffscreenTarget(const OffscreenTarget&) = delete;
  OffscreenTarget& operator=(const OffscreenTarget&) = delete;

  /// @return A non-owning @ref RenderTarget over the owned attachments.
  RenderTarget target() const;

  /// @return The format + sample signature, for building a compatible pipeline.
  RenderTargetLayout layout() const;

  /// @return The attachment extent in texels.
  VkExtent2D extent() const noexcept { return color_.extent(); }

  /// @return The color `VkImage`, for the caller's pre/post-render barriers.
  VkImage color_image() const noexcept { return color_.image(); }

  /// @return The depth `VkImage` for the caller's barriers, or `VK_NULL_HANDLE`
  ///         when the target has no depth attachment.
  VkImage depth_image() const noexcept { return depth_.image(); }

  /// @brief Record the transitions into the attachment layouts (dynamic
  ///        rendering performs none itself): the color image
  ///        `UNDEFINED → COLOR_ATTACHMENT_OPTIMAL` and, when the target has a
  ///        depth attachment, the depth image
  ///        `UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL`.
  ///
  /// Call before @ref RenderTarget::begin. The `UNDEFINED` source layout
  /// discards any previous contents (a load-op clear rewrites them), so this
  /// also re-prepares the target for another render after a
  /// @ref record_readback left the color image in `TRANSFER_SRC_OPTIMAL`.
  /// Consumers sequencing their own layouts (e.g. preserving the previous
  /// render) record @ref cmd_image_barrier directly instead.
  /// @param cmd  A command buffer in the recording state.
  /// @pre `valid()`.
  void prepare(VkCommandBuffer cmd) const;

  /// @brief Record the color-attachment → host-buffer copy-out.
  ///
  /// Transitions the color image `COLOR_ATTACHMENT_OPTIMAL → TRANSFER_SRC`,
  /// copies it into the readback buffer, and makes the write available to a
  /// host read. After the submitted command buffer's fence signals, @ref pixels
  /// holds the rendered image.
  /// @param cmd  A command buffer in the recording state.
  /// @pre `valid()`, the target was created with readback, and the color image
  ///      is currently in `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` (the state
  ///      a stored dynamic-rendering scope leaves it in).
  void record_readback(VkCommandBuffer cmd) const;

  /// @return The mapped readback buffer (the rendered pixels once the
  /// readback's
  ///         fence has signaled), or `nullptr` if created without readback.
  const void* pixels() const noexcept { return readback_.mapped(); }

  /// @return `true` if this owns a color attachment.
  bool valid() const noexcept { return color_.valid(); }

 private:
  Texture color_;
  Texture depth_;    // empty when created without a depth_format
  Buffer readback_;  // empty when created without readback
};

}  // namespace volumetric_kit::gfx

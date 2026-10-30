// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/depth_attachment.hpp
/// Shared attachment helpers for the two render-target owners that grow an
/// optional depth image and a multisampled colour image — @ref
/// OffscreenTarget (core) and @ref windowing::Swapchain — so the depth-only
/// format validation (and its combined-depth/stencil TODO) and the depth and
/// multisampled colour `Image` builds live in one place instead of being
/// copied per owner. Not a public header.

#include <string>
#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/format.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Reject a depth attachment format that is not depth-only.
/// @param format   The requested depth format (never `VK_FORMAT_UNDEFINED` —
///                 the caller gates on that first).
/// @param context  Caller name prefixed onto the message (e.g.
///                 `"Swapchain::create"`).
/// @return OK for a depth-only format; `core::Status::Code::InvalidArgument` if
///         it carries no depth aspect; `core::Status::Code::Unsupported` if it
///         carries a stencil aspect.
///
/// RenderTarget renders depth through `DEPTH_ATTACHMENT_OPTIMAL`, valid for a
/// stencil-bearing image only with the `separateDepthStencilLayouts` feature.
/// TODO: support combined depth/stencil — needs a stencil attachment wired
/// through RenderTarget and that device feature; today only depth-only formats
/// render correctly.
inline core::Status validate_depth_only_format(VkFormat format,
                                               std::string_view context) {
  if (!core::format_has_depth(format)) {
    return core::Status::invalid_argument(
        std::string(context) +
        ": depth_format must be a depth format (or VK_FORMAT_UNDEFINED for a "
        "color-only target)");
  }
  if (core::format_has_stencil(format)) {
    return core::Status::unsupported(
        std::string(context) +
        ": combined depth/stencil depth_format is not yet supported; use a "
        "depth-only format such as VK_FORMAT_D32_SFLOAT");
  }
  return core::Status{};
}

/// @brief Allocate a device-local depth attachment at @p extent.
/// @param allocator  Allocates the image + its DEPTH-aspect view.
/// @param extent     Attachment size in texels.
/// @param format     A depth-only format (validate with @ref
///                   validate_depth_only_format first).
/// @param samples    Its sample count. A multisampled one lives for a
///                   rendering scope alone, so it is also `TRANSIENT`.
/// @return The depth `core::Image`, or a propagated allocation failure.
inline core::Result<core::Image> make_depth_attachment(
    core::Allocator& allocator, VkExtent2D extent, VkFormat format,
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
  core::ImageDesc desc;
  desc.extent = extent;
  desc.format = format;
  desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (samples != VK_SAMPLE_COUNT_1_BIT) {
    desc.usage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  }
  desc.samples = samples;
  return allocator.create_image(desc);
}

/// @brief Allocate the multisampled colour image a target draws into and
///        resolves from. Its samples live for a rendering scope alone, so it
///        is `TRANSIENT`.
/// @param allocator  Allocates the image + its view.
/// @param extent     Attachment size in texels.
/// @param format     The colour format, its resolve attachment's.
/// @param samples    Its sample count.
/// @pre @p samples is above `VK_SAMPLE_COUNT_1_BIT`.
/// @return The colour `core::Image`, or a propagated allocation failure.
inline core::Result<core::Image> make_multisampled_color(
    core::Allocator& allocator, VkExtent2D extent, VkFormat format,
    VkSampleCountFlagBits samples) {
  core::ImageDesc desc;
  desc.extent = extent;
  desc.format = format;
  desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  desc.samples = samples;
  return allocator.create_image(desc);
}

}  // namespace volumetric_kit::gfx

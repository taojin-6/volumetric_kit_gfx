// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/depth_attachment.hpp
/// Shared depth-attachment helpers for the two render-target owners that grow
/// an optional depth image — @ref OffscreenTarget (core) and @ref
/// windowing::Swapchain — so the depth-only format validation (and its
/// combined-depth/stencil TODO) and the depth `Texture` build live in one
/// place instead of being copied per owner. Not a public header.

#include <string>
#include <string_view>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/impl/vk_format.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Reject a depth attachment format that is not depth-only.
/// @param format   The requested depth format (never `VK_FORMAT_UNDEFINED` —
///                 the caller gates on that first).
/// @param context  Caller name prefixed onto the message (e.g.
///                 `"Swapchain::create"`).
/// @return OK for a depth-only format; @ref Status::Code::InvalidArgument if it
///         carries no depth aspect; @ref Status::Code::Unsupported if it
///         carries a stencil aspect.
///
/// RenderTarget renders depth through `DEPTH_ATTACHMENT_OPTIMAL`, valid for a
/// stencil-bearing image only with the `separateDepthStencilLayouts` feature.
/// TODO: support combined depth/stencil — needs a stencil attachment wired
/// through RenderTarget and that device feature; today only depth-only formats
/// render correctly.
inline Status validate_depth_only_format(VkFormat format,
                                         std::string_view context) {
  if (!format_has_depth(format)) {
    return Status::invalid_argument(
        std::string(context) +
        ": depth_format must be a depth format (or VK_FORMAT_UNDEFINED for a "
        "color-only target)");
  }
  if (format_has_stencil(format)) {
    return Status::unsupported(
        std::string(context) +
        ": combined depth/stencil depth_format is not yet supported; use a "
        "depth-only format such as VK_FORMAT_D32_SFLOAT");
  }
  return Status{};
}

/// @brief Allocate a device-local depth attachment at @p extent.
/// @param allocator  Allocates the image + its DEPTH-aspect view.
/// @param extent     Attachment size in texels.
/// @param format     A depth-only format (validate with @ref
///                   validate_depth_only_format first).
/// @return The depth @ref Texture, or a propagated allocation failure.
inline Result<Texture> make_depth_attachment(Allocator& allocator,
                                             VkExtent2D extent,
                                             VkFormat format) {
  TextureDesc desc;
  desc.extent = extent;
  desc.format = format;
  desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  return allocator.create_image(desc);
}

}  // namespace volumetric_kit::gfx

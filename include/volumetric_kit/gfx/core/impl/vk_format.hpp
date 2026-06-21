// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/vk_format.hpp
/// Internal `VkFormat` lookups shared by the resource and render-target code:
/// the view aspect a format implies and the byte size of one texel. Both defer
/// to Khronos' Vulkan-Utility-Libraries (`vkuFormat*`) -- an authoritative
/// source of truth regenerated each Vulkan release -- rather than a
/// hand-maintained table, so covering new formats and colorspaces is a single
/// pinned-tag bump (see third_party/CMakeLists.txt), not a per-format edit
/// here. The library vendors these headers (some platforms' system Vulkan ships
/// neither them nor recent format enums). Not a public header.

#include <cstdint>

#include <vulkan/utility/vk_format_utils.h>

#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief The image-view aspect a format implies.
/// @return `VK_IMAGE_ASPECT_DEPTH_BIT` for depth and combined depth/stencil
///         formats, `VK_IMAGE_ASPECT_STENCIL_BIT` for stencil-only, otherwise
///         `VK_IMAGE_ASPECT_COLOR_BIT`.
inline VkImageAspectFlags aspect_mask_for(VkFormat format) {
  // Combined depth/stencil resolves to a depth-only view: a single VkImageView
  // may not mix depth and stencil for sampling, and depth is the dominant
  // attachment/sample use. A stencil view is requested explicitly once
  // TextureDesc carries an aspect.
  if (vkuFormatHasDepth(format)) {
    return VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  if (vkuFormatHasStencil(format)) {
    return VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  return VK_IMAGE_ASPECT_COLOR_BIT;
}

/// @brief Whether @p format carries a depth aspect (depth-only or combined
///        depth/stencil).
inline bool format_has_depth(VkFormat format) {
  return vkuFormatHasDepth(format);
}

/// @brief Whether @p format carries a stencil aspect (stencil-only or combined
///        depth/stencil).
inline bool format_has_stencil(VkFormat format) {
  return vkuFormatHasStencil(format);
}

/// @brief Size in bytes of one texel of an uncompressed, single-plane color
///        format -- the per-texel stride a tightly packed image<->buffer copy
///        uses (e.g. to size an offscreen readback buffer).
/// @return The texel size, or 0 for a format a flat per-texel copy cannot size
///         (compressed, multi-planar, subsampled, or depth/stencil), so a
///         caller can reject the request rather than under-allocate.
inline uint32_t texel_size(VkFormat format) {
  // Only uncompressed, single-plane color formats have a 1:1 texel-to-block
  // mapping where the block size is the per-texel stride. vkuFormatIsColor
  // excludes depth/stencil, multi-planar, and undefined;
  // vkuFormatIsBlockedImage excludes the rest a flat copy cannot size --
  // compressed and single-plane 4:2:2, whose blocks span several texels yet
  // report texels-per-block == 1 (so that field alone would let PVRTC and 4:2:2
  // slip through and return a bogus non-zero stride).
  if (!vkuFormatIsColor(format) || vkuFormatIsBlockedImage(format)) {
    return 0;
  }
  return vkuFormatTexelBlockSize(format);
}

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/vk_format.hpp
/// Internal `VkFormat` lookups shared by the resource and render-target code:
/// the view aspect a format implies and the byte size of one texel. Both are
/// small hand-maintained tables, kept in one place so they cannot drift apart
/// (Vulkan ships an authoritative `vkuFormatTexelBlockSize`, but only via the
/// Vulkan-Utility-Libraries headers, which the Ubuntu CI's `libvulkan-dev` does
/// not carry). Not a public header.

#include <cstdint>

#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief The image-view aspect a format implies.
/// @return `VK_IMAGE_ASPECT_DEPTH_BIT` for depth and combined depth/stencil
///         formats, `VK_IMAGE_ASPECT_STENCIL_BIT` for stencil-only, otherwise
///         `VK_IMAGE_ASPECT_COLOR_BIT`.
inline VkImageAspectFlags aspect_mask_for(VkFormat format) {
  switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
      return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_S8_UINT:
      return VK_IMAGE_ASPECT_STENCIL_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      // DEPTH only: a single VkImageView may not mix depth and stencil for
      // sampling, and depth is the dominant attachment/sample use. A stencil
      // view (or a depth-peel pass needing stencil) requests its aspect once
      // TextureDesc carries one.
      return VK_IMAGE_ASPECT_DEPTH_BIT;
    default:
      return VK_IMAGE_ASPECT_COLOR_BIT;
  }
}

/// @brief Size in bytes of one texel of an uncompressed, single-plane color
///        format -- the per-texel stride a tightly packed image<->buffer copy
///        uses (e.g. to size an offscreen readback buffer).
/// @return The texel size, or 0 for a format this kit does not size --
///         compressed, multi-planar, depth/stencil, or simply not listed yet --
///         so a caller can reject the request rather than under-allocate.
inline uint32_t texel_size(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_SRGB:
      return 1;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UINT:
      return 2;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
      return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32G32_SFLOAT:
      return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
      return 16;
    default:
      return 0;
  }
}

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/color_attachment.hpp
/// What a color-attachment format does with the values a fragment shader
/// writes, for the tiers whose shaders write color -- pipelines and ui -- so
/// each picks its encoding by one rule, for windowing's choice of a
/// swapchain format, and for the sample counts an offscreen target takes.
/// Not a public header.

#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Whether a color-attachment format encodes sRGB on write.
///
/// Only the 8-bit sRGB formats can be color attachments; the compressed ones
/// cannot, so they are not listed.
/// @param format  A color-attachment format.
/// @return `true` for an `_SRGB` attachment format.
inline bool is_srgb_attachment(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_R8G8_SRGB:
    case VK_FORMAT_R8G8B8_SRGB:
    case VK_FORMAT_B8G8R8_SRGB:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
      return true;
    default:
      return false;
  }
}

/// @brief Whether a color-attachment format stores floating-point values: a
///        target that holds linear light, as an HDR or offscreen one does.
///
/// Lists the float formats a color attachment can have; the 64-bit ones and
/// `E5B9G9R9` cannot be rendered to, so they are not listed.
/// @param format  A color-attachment format.
/// @return `true` for an `_SFLOAT` or `_UFLOAT` attachment format.
inline bool is_float_attachment(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      return true;
    default:
      return false;
  }
}

/// @brief Whether a color-attachment format stores integers: one whose resolve
///        cannot average, and whose multisampling a device need not support
///        (its `framebufferIntegerColorSampleCounts` may be 1 alone).
///
/// Lists the `_UINT` / `_SINT` formats a color attachment can have; the
/// 3-component and 64-bit ones cannot be rendered to, so they are not listed.
/// gfx enables no extension that adds one.
/// @param format  A color-attachment format.
/// @return `true` for a `_UINT` or `_SINT` attachment format.
inline bool is_integer_attachment(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_B8G8R8A8_UINT:
    case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_A8B8G8R8_UINT_PACK32:
    case VK_FORMAT_A8B8G8R8_SINT_PACK32:
    case VK_FORMAT_A2R10G10B10_UINT_PACK32:
    case VK_FORMAT_A2R10G10B10_SINT_PACK32:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_A2B10G10R10_SINT_PACK32:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
      return true;
    default:
      return false;
  }
}

/// @brief Whether a fragment shader writes linear light to a color attachment
///        of @p format: an `_SRGB` one encodes it on write and a float one
///        stores it. Any other, such as a `_UNORM` one, stores the shader's
///        values as given, so the shader writes them encoded.
/// @param format  A color-attachment format.
/// @return `true` for an `_SRGB` or float attachment format.
inline bool takes_linear_light(VkFormat format) {
  return is_srgb_attachment(format) || is_float_attachment(format);
}

}  // namespace volumetric_kit::gfx

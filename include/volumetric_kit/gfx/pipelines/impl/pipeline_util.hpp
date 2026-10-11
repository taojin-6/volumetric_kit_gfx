// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pipeline_util.hpp
/// @brief Small command-recording helpers shared by the pipelines tier's
///        `submit()` paths. Internal -- not part of the public technique API.

#include <algorithm>
#include <cstdint>

#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::gfx::pipelines {

/// @brief Record a dynamic viewport of @p viewport and a scissor of
///        @p viewport clipped to @p target.
///
/// Every graphics pipeline in this tier declares viewport + scissor as dynamic
/// state (see @ref GraphicsPipeline), so each `submit()` sets its rectangles
/// through these helpers. This centralizes that one convention (depth
/// `[0, 1]`, no Y flip) so a change to it lands in a single place. The
/// scissor keeps a draw inside its viewport, which a point wider than a pixel
/// may otherwise leave, and inside the target, as Vulkan refuses a negative
/// scissor offset.
/// @param cmd       A recording-state command buffer.
/// @param viewport  The rectangle drawn into, in target pixels; it may extend
///                  past the target.
/// @param target    The target's pixel dimensions.
/// @return `false`, recording nothing, when @p viewport and the target do not
///         overlap.
[[nodiscard]] inline bool set_viewport_scissor(VkCommandBuffer cmd,
                                               const VkRect2D& viewport,
                                               VkExtent2D target) {
  // 64-bit, so an offset plus an extent cannot wrap.
  const int64_t x0 = std::max<int64_t>(viewport.offset.x, 0);
  const int64_t y0 = std::max<int64_t>(viewport.offset.y, 0);
  const int64_t x1 = std::min<int64_t>(
      int64_t{viewport.offset.x} + viewport.extent.width, target.width);
  const int64_t y1 = std::min<int64_t>(
      int64_t{viewport.offset.y} + viewport.extent.height, target.height);
  if (x1 <= x0 || y1 <= y0) {
    return false;
  }

  VkViewport vp{};
  vp.x = static_cast<float>(viewport.offset.x);
  vp.y = static_cast<float>(viewport.offset.y);
  vp.width = static_cast<float>(viewport.extent.width);
  vp.height = static_cast<float>(viewport.extent.height);
  vp.minDepth = 0.0f;
  vp.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp);

  VkRect2D scissor{};
  scissor.offset = {static_cast<int32_t>(x0), static_cast<int32_t>(y0)};
  scissor.extent = {static_cast<uint32_t>(x1 - x0),
                    static_cast<uint32_t>(y1 - y0)};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  return true;
}

/// @brief Record a dynamic viewport and scissor covering the whole target, by
///        @ref set_viewport_scissor.
/// @param cmd     A recording-state command buffer.
/// @param extent  The target's pixel dimensions; an empty one records nothing.
inline void set_full_viewport_scissor(VkCommandBuffer cmd, VkExtent2D extent) {
  static_cast<void>(set_viewport_scissor(cmd, {{0, 0}, extent}, extent));
}

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

/// @brief Whether a fragment shader writes linear light to a color attachment
///        of @p format: an `_SRGB` one encodes it on write and a float one
///        stores it. Any other, such as a `_UNORM` one, stores the shader's
///        values as given, so the shader writes them encoded.
/// @param format  A color-attachment format.
/// @return `true` for an `_SRGB` or float attachment format.
inline bool takes_linear_light(VkFormat format) {
  return is_srgb_attachment(format) || is_float_attachment(format);
}

}  // namespace volumetric_kit::gfx::pipelines

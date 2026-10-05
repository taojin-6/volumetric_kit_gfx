// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pipeline_util.hpp
/// @brief Small command-recording helpers shared by the pipelines tier's
///        `submit()` paths. Internal -- not part of the public technique API.

#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::gfx::pipelines {

/// @brief Record a full-target dynamic viewport + scissor covering @p extent.
///
/// Every graphics pipeline in this tier declares viewport + scissor as dynamic
/// state (see @ref GraphicsPipeline), so each `submit()` sets a full-target
/// rectangle at record time. This centralizes that one convention so a change
/// to it (e.g. a Y-flipped viewport) lands in a single place.
/// @param cmd     A recording-state command buffer.
/// @param extent  The target's pixel dimensions.
inline void set_full_viewport_scissor(VkCommandBuffer cmd, VkExtent2D extent) {
  VkViewport viewport{};
  viewport.width = static_cast<float>(extent.width);
  viewport.height = static_cast<float>(extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.extent = extent;
  vkCmdSetScissor(cmd, 0, 1, &scissor);
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

}  // namespace volumetric_kit::gfx::pipelines

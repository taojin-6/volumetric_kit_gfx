// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/command.hpp
/// Internal command-recording helpers shared across tiers. Dynamic rendering
/// leaves every image layout transition to the application, so each target
/// producer (offscreen readback, swapchain present) ends up recording the same
/// single-subresource `VkImageMemoryBarrier`; @ref cmd_image_barrier is the one
/// place that boilerplate lives. Not a public header.

#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Record a one-subresource image layout-transition barrier.
/// @param cmd         A command buffer in the recording state.
/// @param image       The image to transition.
/// @param src_stage   Pipeline stages that must complete before the transition.
/// @param dst_stage   Pipeline stages that wait on the transition.
/// @param src_access  Access made available before the transition.
/// @param dst_access  Access made visible after the transition.
/// @param old_layout  Current layout (`VK_IMAGE_LAYOUT_UNDEFINED` discards
///                    contents).
/// @param new_layout  Layout to transition into.
/// @param aspect      Image aspect (color by default); always mip 0, layer 0.
inline void cmd_image_barrier(
    VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags src_stage,
    VkPipelineStageFlags dst_stage, VkAccessFlags src_access,
    VkAccessFlags dst_access, VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = {aspect, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);
}

}  // namespace volumetric_kit::gfx

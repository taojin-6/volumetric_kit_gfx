// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_barrier.hpp
/// @brief Record an image layout-transition barrier — the transitions dynamic
///        rendering leaves to the application.

#include <cstdint>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Parameters for @ref cmd_image_barrier: one image layout transition
///        with full subresource-range control.
///
/// Dynamic rendering performs no layout transitions, so every producer and
/// consumer of an image (attachment prep, present, upload, readback) records
/// one of these. The defaults describe the common whole-image color case: the
/// range spans every mip level and array layer
/// (`VK_REMAINING_MIP_LEVELS` / `VK_REMAINING_ARRAY_LAYERS`); narrow @ref
/// base_mip / @ref mip_count / @ref base_layer / @ref layer_count to transition
/// a sub-range (e.g. the single level a mip-chain blit reads). Queue-family
/// ownership is never transferred (both families are
/// `VK_QUEUE_FAMILY_IGNORED`).
///
/// @code
/// // Prepare a freshly created cubemap for its staging copy: every face
/// // (and mip) UNDEFINED -> TRANSFER_DST in one barrier.
/// ImageBarrierDesc to_dst;
/// to_dst.image = cube.image();
/// to_dst.src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
/// to_dst.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
/// to_dst.dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
/// to_dst.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
/// to_dst.new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
/// cmd_image_barrier(cmd, to_dst);
/// @endcode
struct ImageBarrierDesc {
  /// The image to transition.
  VkImage image = VK_NULL_HANDLE;
  /// Pipeline stages that must complete before the transition.
  VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  /// Pipeline stages that wait on the transition.
  VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  /// Access made available before the transition.
  VkAccessFlags src_access = 0;
  /// Access made visible after the transition.
  VkAccessFlags dst_access = 0;
  /// Current layout (`VK_IMAGE_LAYOUT_UNDEFINED` discards contents).
  VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  /// Layout to transition into.
  VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  /// Image aspect(s) the barrier covers (color by default).
  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  /// First mip level covered.
  uint32_t base_mip = 0;
  /// Mip levels covered from @ref base_mip (all remaining by default).
  uint32_t mip_count = VK_REMAINING_MIP_LEVELS;
  /// First array layer covered.
  uint32_t base_layer = 0;
  /// Array layers covered from @ref base_layer (all remaining by default).
  uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;
};

/// @brief Record the image layout-transition barrier described by @p desc.
/// @param cmd   A command buffer in the recording state.
/// @param desc  The image, execution/memory scopes, layouts, and subresource
///              range to transition.
/// @pre `desc.image != VK_NULL_HANDLE`, and `desc.old_layout` matches the
///      subresource range's current layout (or is `VK_IMAGE_LAYOUT_UNDEFINED`).
VG_CORE_API void cmd_image_barrier(VkCommandBuffer cmd,
                                   const ImageBarrierDesc& desc);

}  // namespace volumetric_kit::gfx

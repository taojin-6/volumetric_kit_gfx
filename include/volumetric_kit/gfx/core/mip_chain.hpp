// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mip_chain.hpp
/// @brief Build a 2D image's mip chain on the GPU: halving linear blits down
///        from level 0, recorded into the caller's command buffer.

#include <cstdint>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/export.hpp"

namespace volumetric_kit::gfx {

/// @brief The length of a full mip chain for @p extent:
///        `floor(log2(max(width, height))) + 1`.
/// @param extent  Level 0's size in texels.
/// @return The number of levels down to and including 1 x 1 (1 for a zero or
///         1 x 1 extent).
VG_CORE_API uint32_t mip_level_count(VkExtent2D extent) noexcept;

/// @brief The size of mip level @p level of an image whose level 0 is
///        @p extent: each dimension halved per level, floored at 1.
/// @param extent  Level 0's size in texels.
/// @param level   The level.
/// @return The level's size in texels.
VG_CORE_API VkExtent2D mip_level_extent(VkExtent2D extent,
                                        uint32_t level) noexcept;

/// @brief What @ref cmd_generate_mips needs to know about an image: its
///        chain, and the layouts and last accesses its levels enter with.
///
/// The defaults describe an upload that just copied level 0 and left every
/// level in `TRANSFER_DST_OPTIMAL` (@ref UploadBatch). An image rebuilt every
/// frame instead enters with level 0 just rendered (@ref base_layout
/// `COLOR_ATTACHMENT_OPTIMAL`) and its upper levels discarded
/// (@ref upper_layout `UNDEFINED`).
///
/// @code
/// MipChainDesc mips;  // level 0 was just rendered; the rest is stale
/// mips.image = display.handle();
/// mips.extent = {display.width(), display.height()};
/// mips.mip_levels = display.mip_levels();
/// mips.base_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
/// mips.base_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
/// mips.base_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
/// mips.upper_layout = VK_IMAGE_LAYOUT_UNDEFINED;
/// cmd_generate_mips(cmd, mips);
/// @endcode
struct MipChainDesc {
  /// The image: 2D, single-layer, color, created with `TRANSFER_SRC` and
  /// `TRANSFER_DST` usage, in a format that supports linear blits and linear
  /// filtering.
  VkImage image = VK_NULL_HANDLE;
  /// Level 0's size in texels.
  VkExtent2D extent{};
  /// The levels to fill: at least 1, and at most @ref mip_level_count
  /// `(extent)`.
  uint32_t mip_levels = 1;
  /// Level 0's layout on entry; it holds the pixels the chain is built from.
  VkImageLayout base_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  /// The stage that wrote level 0.
  VkPipelineStageFlags base_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  /// The access that wrote level 0.
  VkAccessFlags base_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  /// Levels 1 and up on entry: `TRANSFER_DST_OPTIMAL`, or `UNDEFINED` to
  /// discard what they held, which a chain rebuilt over a sampled image does.
  VkImageLayout upper_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  /// The stages that may still read levels 1 and up from earlier work, which
  /// their blits must wait for; consulted only when @ref upper_layout is not
  /// `TRANSFER_DST_OPTIMAL`.
  VkPipelineStageFlags upper_read_stages =
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  /// The stages that will sample the finished chain.
  VkPipelineStageFlags dst_stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
};

/// @brief Record the blits that fill levels 1 and up of an image from level
///        0, and the barriers around them; every level ends in
///        `SHADER_READ_ONLY_OPTIMAL`.
///
/// Each level is a linear blit of the one above at half the size, floored at
/// 1, so an odd dimension drops its last row or column's share. A blit
/// samples and writes as a shader would: an `_SRGB` format is decoded before
/// filtering and encoded after it, so its chain averages in linear light
/// rather than over the encoded values -- why a display image is kept
/// `_SRGB`.
/// @param cmd   A command buffer in the recording state, outside a
///              dynamic-rendering scope.
/// @param desc  The image and the state its levels enter with.
/// @pre `desc.image != VK_NULL_HANDLE` and
///      `1 <= desc.mip_levels <= mip_level_count(desc.extent)` (checked with
///      `VKC_CHECK`); the format supports linear blits and filtering.
VG_CORE_API void cmd_generate_mips(VkCommandBuffer cmd,
                                   const MipChainDesc& desc);

}  // namespace volumetric_kit::gfx

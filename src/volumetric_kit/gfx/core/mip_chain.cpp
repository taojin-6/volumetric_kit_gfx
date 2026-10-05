// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/mip_chain.hpp"

#include <algorithm>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"

namespace volumetric_kit::gfx {

uint32_t mip_level_count(VkExtent2D extent) noexcept {
  uint32_t max_dim = std::max(extent.width, extent.height);
  uint32_t levels = 1;
  while (max_dim > 1) {
    max_dim >>= 1;
    ++levels;
  }
  return levels;
}

VkExtent2D mip_level_extent(VkExtent2D extent, uint32_t level) noexcept {
  if (level >= 32) {
    return {1, 1};
  }
  return {std::max(extent.width >> level, 1u),
          std::max(extent.height >> level, 1u)};
}

void cmd_generate_mips(VkCommandBuffer cmd, const MipChainDesc& desc) {
  VKC_CHECK(desc.image != VK_NULL_HANDLE, "cmd_generate_mips: null image");
  VKC_CHECK(
      desc.mip_levels >= 1 && desc.mip_levels <= mip_level_count(desc.extent),
      "cmd_generate_mips: mip_levels outside 1..mip_level_count(extent)");

  // Every barrier here is a single-level transition on the image; a local
  // helper stands in for the designated initializers C++17 lacks, so each is
  // one call instead of a nine-line struct.
  auto barrier = [&](uint32_t level, uint32_t count, VkImageLayout old_layout,
                     VkImageLayout new_layout, VkPipelineStageFlags src_stage,
                     VkPipelineStageFlags dst_stage, VkAccessFlags src_access,
                     VkAccessFlags dst_access) {
    ImageBarrierDesc b;
    b.image = desc.image;
    b.src_stage = src_stage;
    b.dst_stage = dst_stage;
    b.src_access = src_access;
    b.dst_access = dst_access;
    b.old_layout = old_layout;
    b.new_layout = new_layout;
    b.base_mip = level;
    b.mip_count = count;
    cmd_image_barrier(cmd, b);
  };

  // A single-level image has no chain: hand level 0 to the samplers.
  if (desc.mip_levels == 1) {
    barrier(0, 1, desc.base_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            desc.base_stage, desc.dst_stages, desc.base_access,
            VK_ACCESS_SHADER_READ_BIT);
    return;
  }

  // Levels 1 and up become blit destinations. From UNDEFINED their contents
  // are discarded, but earlier reads of them -- the previous frame sampling
  // the previous chain -- must finish before the blits overwrite them: a
  // write-after-read hazard, which an execution dependency alone resolves.
  if (desc.upper_layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier(1, desc.mip_levels - 1, desc.upper_layout,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, desc.upper_read_stages,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
  }

  int32_t src_w = static_cast<int32_t>(desc.extent.width);
  int32_t src_h = static_cast<int32_t>(desc.extent.height);
  for (uint32_t level = 1; level < desc.mip_levels; ++level) {
    // The level above becomes the blit source. Level 0 arrives in whatever
    // layout its writer left; every later one was the previous blit's
    // destination.
    if (level == 1) {
      barrier(0, 1, desc.base_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              desc.base_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, desc.base_access,
              VK_ACCESS_TRANSFER_READ_BIT);
    } else {
      barrier(level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    }

    const int32_t dst_w = src_w > 1 ? src_w / 2 : 1;
    const int32_t dst_h = src_h > 1 ? src_h / 2 : 1;
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
    blit.srcOffsets[1] = {src_w, src_h, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
    blit.dstOffsets[1] = {dst_w, dst_h, 1};
    vkCmdBlitImage(cmd, desc.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   desc.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_LINEAR);

    // The source level is done being read: hand it to the samplers.
    barrier(level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, desc.dst_stages,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);

    src_w = dst_w;
    src_h = dst_h;
  }

  // The last level was only ever a blit destination, never a source, so it is
  // still TRANSFER_DST.
  barrier(desc.mip_levels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_PIPELINE_STAGE_TRANSFER_BIT, desc.dst_stages,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

}  // namespace volumetric_kit::gfx

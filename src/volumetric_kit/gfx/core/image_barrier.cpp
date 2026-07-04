// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/image_barrier.hpp"

namespace volumetric_kit::gfx {

void cmd_image_barrier(VkCommandBuffer cmd, const ImageBarrierDesc& desc) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = desc.src_access;
  barrier.dstAccessMask = desc.dst_access;
  barrier.oldLayout = desc.old_layout;
  barrier.newLayout = desc.new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = desc.image;
  barrier.subresourceRange = {desc.aspect, desc.base_mip, desc.mip_count,
                              desc.base_layer, desc.layer_count};
  vkCmdPipelineBarrier(cmd, desc.src_stage, desc.dst_stage, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

}  // namespace volumetric_kit::gfx

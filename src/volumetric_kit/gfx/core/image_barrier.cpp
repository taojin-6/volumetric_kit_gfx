// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/image_barrier.hpp"

#include <array>
#include <vector>

#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::gfx {

namespace {

VkImageMemoryBarrier to_vk(const ImageBarrierDesc& desc) {
  // new_layout has no valid default: VK_IMAGE_LAYOUT_UNDEFINED is never a legal
  // barrier target, so catch a caller who left it unset here -- otherwise it
  // records a silent no-op transition only a validation layer would flag.
  VKC_CHECK(desc.new_layout != VK_IMAGE_LAYOUT_UNDEFINED,
            "cmd_image_barrier: new_layout must be set (not UNDEFINED)");
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
  return barrier;
}

}  // namespace

void cmd_image_barrier(VkCommandBuffer cmd, const ImageBarrierDesc& desc) {
  const VkImageMemoryBarrier barrier = to_vk(desc);
  vkCmdPipelineBarrier(cmd, desc.src_stage, desc.dst_stage, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

void cmd_image_barriers(VkCommandBuffer cmd, const ImageBarrierDesc* descs,
                        uint32_t count) {
  if (count == 0) {
    return;
  }
  // A few transitions -- the common case, recorded every frame -- stay on the
  // stack; more spill to the heap.
  constexpr uint32_t kInline = 8;
  std::array<VkImageMemoryBarrier, kInline> inline_barriers{};
  std::vector<VkImageMemoryBarrier> heap_barriers;
  VkImageMemoryBarrier* barriers = inline_barriers.data();
  if (count > kInline) {
    heap_barriers.resize(count);
    barriers = heap_barriers.data();
  }
  VkPipelineStageFlags src_stage = 0;
  VkPipelineStageFlags dst_stage = 0;
  for (uint32_t i = 0; i < count; ++i) {
    barriers[i] = to_vk(descs[i]);
    src_stage |= descs[i].src_stage;
    dst_stage |= descs[i].dst_stage;
  }
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr,
                       count, barriers);
}

}  // namespace volumetric_kit::gfx

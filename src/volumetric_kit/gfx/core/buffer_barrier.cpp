// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/buffer_barrier.hpp"

#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::gfx {

void cmd_buffer_barrier(VkCommandBuffer cmd, const BufferBarrierDesc& desc) {
  VKC_CHECK(desc.buffer != VK_NULL_HANDLE,
            "cmd_buffer_barrier: buffer must be set");
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = desc.src_access;
  barrier.dstAccessMask = desc.dst_access;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = desc.buffer;
  barrier.offset = desc.offset;
  barrier.size = desc.size;
  vkCmdPipelineBarrier(cmd, desc.src_stage, desc.dst_stage, 0, 0, nullptr, 1,
                       &barrier, 0, nullptr);
}

}  // namespace volumetric_kit::gfx

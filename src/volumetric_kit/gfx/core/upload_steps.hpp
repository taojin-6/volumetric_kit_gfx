// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The steps gfx's uploads share: the blocking UploadBatch (upload_texture,
// upload_buffer) and the in-frame record_image_update / record_image_upload.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"

namespace volumetric_kit::gfx::detail {

// A staging buffer -- host memory, mapped, written once front to back
// (write-combined where the device has it) -- holding `size` bytes copied
// from `src`.
inline core::Result<core::Buffer> make_staging(core::Allocator& allocator,
                                               const void* src,
                                               VkDeviceSize size) {
  core::BufferDesc desc;
  desc.size = size;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = core::MemoryUsage::Staging;
  desc.host_access = core::HostAccess::SequentialWrite;
  VKC_ASSIGN(core::Buffer staging, allocator.create_buffer(desc));
  std::memcpy(staging.mapped(), src, static_cast<std::size_t>(size));
  return staging;
}

// Records the transition that discards `image`'s contents -- every level and
// layer, UNDEFINED to TRANSFER_DST_OPTIMAL, once `src_stages` are done (none
// when 0) -- then the copies from `source`. The image is left in
// TRANSFER_DST_OPTIMAL for the caller's own end.
inline void record_copies_to_image(VkCommandBuffer cmd, VkBuffer source,
                                   VkImage image,
                                   VkPipelineStageFlags src_stages,
                                   const VkBufferImageCopy* regions,
                                   std::uint32_t region_count) {
  ImageBarrierDesc to_copy;
  to_copy.image = image;
  to_copy.src_stage =
      src_stages != 0 ? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  to_copy.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  to_copy.dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_copy.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_copy.new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  cmd_image_barrier(cmd, to_copy);
  vkCmdCopyBufferToImage(cmd, source, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region_count,
                         regions);
}

// Records `image`'s transition -- every level and layer -- from
// TRANSFER_DST_OPTIMAL to SHADER_READ_ONLY_OPTIMAL, making the copies into it
// visible to `dst_stages`.
inline void record_copied_to_shader_read(VkCommandBuffer cmd, VkImage image,
                                         VkPipelineStageFlags dst_stages) {
  ImageBarrierDesc to_read;
  to_read.image = image;
  to_read.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  to_read.dst_stage = dst_stages;
  to_read.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_read.dst_access = VK_ACCESS_SHADER_READ_BIT;
  to_read.old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_read.new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  cmd_image_barrier(cmd, to_read);
}

}  // namespace volumetric_kit::gfx::detail

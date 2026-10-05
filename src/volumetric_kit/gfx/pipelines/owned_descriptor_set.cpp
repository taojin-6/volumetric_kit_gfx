// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/impl/owned_descriptor_set.hpp"

#include <cstring>
#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/check.hpp"

namespace volumetric_kit::gfx::pipelines {

namespace {

// vkCmdUpdateBuffer's bounds (VUID-vkCmdUpdateBuffer-dataSize-00037/00038).
constexpr VkDeviceSize kMaxUpdateBytes = 65536;

}  // namespace

Result<Buffer> make_frame_uniform_buffer(Allocator& allocator,
                                         VkDeviceSize size,
                                         FrameUniformMemory memory) {
  if (size == 0 || size % 4 != 0 || size > kMaxUpdateBytes) {
    return Status::invalid_argument(
        "make_frame_uniform_buffer: size must be a non-zero multiple of 4, at "
        "most 65536 bytes");
  }
  BufferDesc desc;
  desc.size = size;
  if (memory == FrameUniformMemory::Prefer) {
    desc.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    desc.memory = MemoryUsage::DeviceMapped;
    desc.host_access = HostAccess::SequentialWrite;  // written, never read
    Result<Buffer> mapped = allocator.create_buffer(desc);
    if (mapped.ok()) {
      return mapped;
    }
    // No device-mapped memory suits it (Unsupported), or the window is full:
    // take device-only memory, which the update writes. Anything else -- a
    // malformed desc -- is a real failure.
    const std::optional<VkResult> result = vk_result(mapped.status());
    if (mapped.status().domain() != Status::Code::Unsupported &&
        result != VK_ERROR_OUT_OF_DEVICE_MEMORY) {
      return mapped.status();
    }
  }
  desc.usage =
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  desc.memory = MemoryUsage::DeviceOnly;
  return allocator.create_buffer(desc);
}

Result<OwnedDescriptorSet> OwnedDescriptorSet::create(
    VkDevice device, Buffer ubo, VkDescriptorSetLayout layout,
    uint32_t sampler_count) {
  // One-set pool: the UBO + the owner's combined-image-samplers.
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampler_count}};
  VG_ASSIGN(DescriptorPool pool, DescriptorPool::create(device, sizes, 2, 1));
  VG_ASSIGN(DescriptorSet set, pool.allocate(layout));
  set.write_uniform_buffer(0, ubo.handle(), 0, ubo.size());

  OwnedDescriptorSet owned;
  owned.pool_ = std::move(pool);
  owned.set_ = set;
  owned.ubo_ = std::move(ubo);
  return owned;
}

void OwnedDescriptorSet::write_uniform(VkCommandBuffer cmd, const void* data,
                                       VkDeviceSize size) const {
  VG_CHECK(valid() && size % 4 == 0 && size <= ubo_.size() &&
               size <= kMaxUpdateBytes,
           "OwnedDescriptorSet::write_uniform: an empty set, or a size that "
           "is not a multiple of 4 within the buffer");
  if (ubo_.mapped() != nullptr) {
    // Coherent: the submit that follows makes the write visible.
    std::memcpy(ubo_.mapped(), data, size);
    return;
  }
  vkCmdUpdateBuffer(cmd, ubo_.handle(), 0, size, data);
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = ubo_.handle();
  barrier.offset = 0;
  barrier.size = size;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       0, 0, nullptr, 1, &barrier, 0, nullptr);
}

}  // namespace volumetric_kit::gfx::pipelines

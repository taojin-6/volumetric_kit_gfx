// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/impl/owned_descriptor_set.hpp"

#include <utility>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer_barrier.hpp"
#include "volumetric_kit/gfx/core/check.hpp"

namespace volumetric_kit::gfx::pipelines {

namespace {

// vkCmdUpdateBuffer's bounds (VUID-vkCmdUpdateBuffer-dataSize-00037/00038).
constexpr VkDeviceSize kMaxUpdateBytes = 65536;

// The stages whose uniform reads the PBR shaders make.
constexpr VkPipelineStageFlags kUniformStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

}  // namespace

Result<core::Buffer> make_frame_uniform_buffer(core::Allocator& allocator,
                                               VkDeviceSize size) {
  if (size == 0 || size % 4 != 0 || size > kMaxUpdateBytes) {
    return Status::invalid_argument(
        "make_frame_uniform_buffer: size must be a non-zero multiple of 4, at "
        "most 65536 bytes");
  }
  core::BufferDesc desc;
  desc.size = size;
  desc.usage =
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  desc.memory = core::MemoryUsage::DeviceOnly;
  return allocator.create_buffer(desc);
}

Result<OwnedDescriptorSet> OwnedDescriptorSet::create(
    VkDevice device, VkDescriptorSetLayout layout, uint32_t sampler_count) {
  // One-set pool: the UBO + the owner's combined-image-samplers.
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampler_count}};
  VG_ASSIGN(DescriptorPool pool, DescriptorPool::create(device, sizes, 2, 1));
  VG_ASSIGN(DescriptorSet set, pool.allocate(layout));

  OwnedDescriptorSet owned;
  owned.pool_ = std::move(pool);
  owned.set_ = set;
  return owned;
}

void OwnedDescriptorSet::bind_uniform(std::shared_ptr<const core::Buffer> ubo,
                                      VkDeviceSize offset, VkDeviceSize range) {
  VG_CHECK(valid() && ubo != nullptr && ubo->valid() && range != 0 &&
               offset <= ubo->size() && range <= ubo->size() - offset,
           "OwnedDescriptorSet::bind_uniform: an empty set, or a range "
           "outside the buffer");
  set_.write_uniform_buffer(0, ubo->handle(), offset, range);
  ubo_ = std::move(ubo);
  offset_ = offset;
  range_ = range;
}

void OwnedDescriptorSet::write_uniform(VkCommandBuffer cmd, const void* data,
                                       VkDeviceSize size) const {
  VG_CHECK(cmd != VK_NULL_HANDLE,
           "OwnedDescriptorSet::write_uniform: the write is recorded, so it "
           "needs the command buffer");
  VG_CHECK(valid() && ubo_ != nullptr && size % 4 == 0 && size <= range_ &&
               size <= kMaxUpdateBytes,
           "OwnedDescriptorSet::write_uniform: an empty set, or a size that "
           "is not a multiple of 4 within the bound range");
  // Earlier work in the queue may still read the range -- a pass recorded
  // after a previous write -- or still be writing it: order the update after
  // both.
  BufferBarrierDesc before;
  before.buffer = ubo_->handle();
  before.src_stage = kUniformStages | VK_PIPELINE_STAGE_TRANSFER_BIT;
  before.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  before.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  before.dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  before.offset = offset_;
  before.size = size;
  cmd_buffer_barrier(cmd, before);

  vkCmdUpdateBuffer(cmd, ubo_->handle(), offset_, size, data);

  BufferBarrierDesc after;
  after.buffer = ubo_->handle();
  after.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  after.dst_stage = kUniformStages;
  after.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  after.dst_access = VK_ACCESS_UNIFORM_READ_BIT;
  after.offset = offset_;
  after.size = size;
  cmd_buffer_barrier(cmd, after);
}

}  // namespace volumetric_kit::gfx::pipelines

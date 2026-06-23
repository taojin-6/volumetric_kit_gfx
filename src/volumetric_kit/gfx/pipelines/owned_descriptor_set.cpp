// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/impl/owned_descriptor_set.hpp"

#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"

namespace volumetric_kit::gfx::pipelines {

Result<OwnedDescriptorSet> OwnedDescriptorSet::create(
    VkDevice device, Allocator& allocator, VkDescriptorSetLayout layout,
    VkDeviceSize ubo_size, uint32_t sampler_count) {
  // Host-mapped uniform buffer at binding 0; the owner fills it (once for a
  // material's factors, per frame for the scene camera).
  BufferDesc bd;
  bd.size = ubo_size;
  bd.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  bd.memory = MemoryUsage::HostVisible;
  bd.mapped = true;
  VG_ASSIGN(Buffer ubo, allocator.create_buffer(bd));

  // One-set pool: the UBO + the owner's combined-image-samplers.
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampler_count}};
  VG_ASSIGN(DescriptorPool pool, DescriptorPool::create(device, sizes, 2, 1));
  VG_ASSIGN(DescriptorSet set, pool.allocate(layout));
  set.write_uniform_buffer(0, ubo.handle(), 0, ubo_size);

  OwnedDescriptorSet owned;
  owned.pool_ = std::move(pool);
  owned.set_ = set;
  owned.ubo_ = std::move(ubo);
  return owned;
}

}  // namespace volumetric_kit::gfx::pipelines

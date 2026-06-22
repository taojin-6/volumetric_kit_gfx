// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/descriptor.hpp"

#include "volumetric_kit/gfx/core/check.hpp"

namespace volumetric_kit::gfx {

Result<DescriptorSetLayout> DescriptorSetLayout::create(
    VkDevice device, const VkDescriptorSetLayoutBinding* bindings,
    uint32_t count) {
  // Argument checks first (no device needed), device last -- keeps the
  // no-device validation tests meaningful.
  if (count > 0 && bindings == nullptr) {
    return Status::invalid_argument(
        "DescriptorSetLayout::create: bindings must be non-null when count is "
        "non-zero");
  }
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "DescriptorSetLayout::create: device must be non-null");
  }

  VkDescriptorSetLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  info.bindingCount = count;
  info.pBindings = bindings;

  VkDescriptorSetLayout handle = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateDescriptorSetLayout(device, &info, nullptr, &handle));

  DescriptorSetLayout layout;
  layout.layout_ =
      UniqueHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>(device,
                                                                        handle);
  return layout;
}

Result<DescriptorPool> DescriptorPool::create(VkDevice device,
                                              const VkDescriptorPoolSize* sizes,
                                              uint32_t size_count,
                                              uint32_t max_sets) {
  // Argument checks first (no device needed), device last -- keeps the
  // no-device validation tests meaningful.
  if (sizes == nullptr || size_count == 0) {
    return Status::invalid_argument(
        "DescriptorPool::create: sizes must be non-null with a non-zero count");
  }
  if (max_sets == 0) {
    return Status::invalid_argument(
        "DescriptorPool::create: max_sets must be non-zero");
  }
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "DescriptorPool::create: device must be non-null");
  }

  VkDescriptorPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  info.maxSets = max_sets;
  info.poolSizeCount = size_count;
  info.pPoolSizes = sizes;

  VkDescriptorPool handle = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateDescriptorPool(device, &info, nullptr, &handle));

  DescriptorPool pool;
  pool.device_ = device;
  pool.pool_ =
      UniqueHandle<VkDescriptorPool, vkDestroyDescriptorPool>(device, handle);
  return pool;
}

Result<DescriptorSet> DescriptorPool::allocate(VkDescriptorSetLayout layout) {
  VG_CHECK(valid(), "DescriptorPool::allocate on an empty pool");
  VG_CHECK(layout != VK_NULL_HANDLE,
           "DescriptorPool::allocate with a null layout");

  VkDescriptorSetAllocateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  info.descriptorPool = pool_.get();
  info.descriptorSetCount = 1;
  info.pSetLayouts = &layout;

  VkDescriptorSet set = VK_NULL_HANDLE;
  VG_VK_TRY(vkAllocateDescriptorSets(device_, &info, &set));
  return DescriptorSet(device_, set);
}

void DescriptorSet::write_uniform_buffer(uint32_t binding, VkBuffer buffer,
                                         VkDeviceSize offset,
                                         VkDeviceSize range) const {
  VG_CHECK(valid(), "DescriptorSet::write_uniform_buffer on an empty set");

  VkDescriptorBufferInfo buffer_info{};
  buffer_info.buffer = buffer;
  buffer_info.offset = offset;
  buffer_info.range = range;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = set_;
  write.dstBinding = binding;
  write.dstArrayElement = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write.pBufferInfo = &buffer_info;

  vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void DescriptorSet::write_combined_image_sampler(uint32_t binding,
                                                 VkImageView view,
                                                 VkSampler sampler,
                                                 VkImageLayout layout) const {
  VG_CHECK(valid(),
           "DescriptorSet::write_combined_image_sampler on an empty set");

  VkDescriptorImageInfo image_info{};
  image_info.sampler = sampler;
  image_info.imageView = view;
  image_info.imageLayout = layout;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = set_;
  write.dstBinding = binding;
  write.dstArrayElement = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &image_info;

  vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

}  // namespace volumetric_kit::gfx

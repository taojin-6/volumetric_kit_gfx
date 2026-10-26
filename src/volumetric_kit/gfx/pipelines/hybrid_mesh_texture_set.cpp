// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_set.hpp"

#include <memory>
#include <utility>

#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"

namespace volumetric_kit::gfx::pipelines {

namespace {

// The core address modes; MIRROR_CLAMP_TO_EDGE needs a feature gfx does not
// enable.
bool core_address_mode(VkSamplerAddressMode address) {
  switch (address) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT:
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
      return true;
    default:
      return false;
  }
}

}  // namespace

HybridMeshTextureSet::HybridMeshTextureSet(Sampler sampler,
                                           core::DescriptorPool pool,
                                           core::DescriptorSet set) noexcept
    : sampler_(std::move(sampler)),
      pool_(std::move(pool)),
      set_(std::move(set)) {}

HybridMeshTextureSet::~HybridMeshTextureSet() = default;

core::Result<std::unique_ptr<HybridMeshTextureSet>>
HybridMeshTextureSet::create(const core::Device& device,
                             const HybridMeshPipeline& pipeline,
                             VkImageView view, const Options& options) try {
  if (!pipeline.valid()) {
    return core::Status::invalid_argument(
        "HybridMeshTextureSet::create: empty pipeline");
  }
  if (view == VK_NULL_HANDLE) {
    return core::Status::invalid_argument(
        "HybridMeshTextureSet::create: the view is null");
  }
  if (options.filter != VK_FILTER_NEAREST &&
      options.filter != VK_FILTER_LINEAR) {
    return core::Status::invalid_argument(
        "HybridMeshTextureSet::create: the filter is not NEAREST or LINEAR");
  }
  if (!core_address_mode(options.address)) {
    return core::Status::invalid_argument(
        "HybridMeshTextureSet::create: the address mode is not REPEAT, "
        "MIRRORED_REPEAT, CLAMP_TO_EDGE or CLAMP_TO_BORDER");
  }

  // One filter for both directions and between levels, one address mode on
  // every axis. Sampler::create refuses a max_lod below 0 or NaN.
  SamplerDesc sampler_desc;
  sampler_desc.mag_filter = options.filter;
  sampler_desc.min_filter = options.filter;
  sampler_desc.mipmap_mode = options.filter == VK_FILTER_LINEAR
                                 ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                 : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampler_desc.address_mode_u = options.address;
  sampler_desc.address_mode_v = options.address;
  sampler_desc.address_mode_w = options.address;
  sampler_desc.max_lod = options.max_lod;
  const VkDevice vk = device.handle();
  VKC_ASSIGN(Sampler sampler, Sampler::create(vk, sampler_desc));

  const VkDescriptorPoolSize pool_size{
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  VKC_ASSIGN(core::DescriptorPool pool,
             core::DescriptorPool::create(vk, &pool_size, 1, 1));
  VKC_ASSIGN(core::DescriptorSet set,
             pool.allocate(pipeline.descriptor_set_layout(0)));

  // The constructor is private, so no std::make_unique.
  std::unique_ptr<HybridMeshTextureSet> texture(new HybridMeshTextureSet(
      std::move(sampler), std::move(pool), std::move(set)));
  texture->rebind(view);
  return texture;
} catch (...) {
  return core::exception_status("HybridMeshTextureSet::create",
                                core::Status::Code::OutOfMemory);
}

void HybridMeshTextureSet::rebind(VkImageView view) {
  // Binding 0 of set 0: hybrid_mesh.frag's sampler2D.
  set_.write_combined_image_sampler(0, view, sampler_.handle(),
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

}  // namespace volumetric_kit::gfx::pipelines

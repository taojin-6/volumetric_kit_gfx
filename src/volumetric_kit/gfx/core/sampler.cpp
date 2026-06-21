// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/sampler.hpp"

namespace volumetric_kit::gfx {

// The move/destroy lifecycle lives in UniqueHandle (see unique_handle.hpp);
// here we only validate, create the handle, and adopt it.

Result<Sampler> Sampler::create(VkDevice device, const SamplerDesc& desc) {
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument("Sampler::create: device is null");
  }
  if (desc.max_lod < desc.min_lod) {
    // VUID-VkSamplerCreateInfo-maxLod-01973: an empty LOD range is invalid use.
    // Reject it as a domain error rather than passing the contradiction to
    // vkCreateSampler.
    return Status::invalid_argument(
        "Sampler::create: max_lod must be >= min_lod");
  }

  VkSamplerCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.magFilter = desc.mag_filter;
  info.minFilter = desc.min_filter;
  info.mipmapMode = desc.mipmap_mode;
  info.addressModeU = desc.address_mode_u;
  info.addressModeV = desc.address_mode_v;
  info.addressModeW = desc.address_mode_w;
  info.minLod = desc.min_lod;
  info.maxLod = desc.max_lod;
  info.borderColor = desc.border_color;
  // The zero-initialized remainder disables anisotropy and the compare op,
  // keeps normalized coordinates, and leaves mipLodBias at 0.

  VkSampler handle = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateSampler(device, &info, nullptr, &handle));

  Sampler sampler;
  sampler.handle_ = UniqueHandle<VkSampler, vkDestroySampler>(device, handle);
  return sampler;
}

}  // namespace volumetric_kit::gfx

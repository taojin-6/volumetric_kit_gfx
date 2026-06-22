// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/sampler.hpp"

#include <cmath>

namespace volumetric_kit::gfx {

// The move/destroy lifecycle lives in UniqueHandle (see unique_handle.hpp);
// here we only validate, create the handle, and adopt it.

Result<Sampler> Sampler::create(VkDevice device, const SamplerDesc& desc) {
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument("Sampler::create: device is null");
  }
  // Reject an empty or NaN LOD range up front as a domain error rather than
  // passing the contradiction to vkCreateSampler (VUID-VkSamplerCreateInfo-
  // maxLod-01973). NaN compares false against everything, so a NaN bound would
  // slip past a bare `max_lod < min_lod` test and must be checked explicitly.
  if (std::isnan(desc.min_lod) || std::isnan(desc.max_lod) ||
      desc.max_lod < desc.min_lod) {
    return Status::invalid_argument(
        "Sampler::create: min_lod/max_lod must be numbers with "
        "max_lod >= min_lod");
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

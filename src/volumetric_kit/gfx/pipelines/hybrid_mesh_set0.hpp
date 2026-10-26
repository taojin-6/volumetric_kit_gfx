// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// HybridMeshPipeline's set 0, as hybrid_mesh.frag declares it: what the
// pipeline's fallback, StreamedAtlas and HybridMeshTextureSet write. Binding 0
// is an RGBA image or a camera picture's luma, binding 1 its chroma (an RGBA
// set binds a placeholder the shader never reads), and binding 2 a
// PictureUniform saying which.

#include <array>
#include <cstddef>
#include <cstdint>

#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::gfx::pipelines::detail {

// hybrid_mesh.frag's `Picture` block (std140, 32 bytes). Plain floats rather
// than glm types, so no glm alignment setting can move a member.
struct PictureUniform {
  uint32_t two_plane;   // 0: binding 0 is RGBA; 1: Y'CbCr 4:2:0 planes
  uint32_t full_range;  // the rest is a two-plane picture's coding
  float kr;
  float kb;
  float extent[2];         // the picture's size in luma texels
  uint32_t chroma_siting;  // ChromaSiting
  uint32_t padding;        // to std140's 16-byte multiple
};
static_assert(sizeof(PictureUniform) == 32, "PictureUniform must match");
static_assert(offsetof(PictureUniform, extent) == 16,
              "PictureUniform layout drift");
static_assert(offsetof(PictureUniform, chroma_siting) == 24,
              "PictureUniform layout drift");

// The pool sizes that hold `sets` set-0 sets.
inline std::array<VkDescriptorPoolSize, 2> set0_pool_sizes(uint32_t sets) {
  return {
      VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * sets},
      VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sets}};
}

// One plane binding: a view, sampled in `layout`.
struct PlaneBinding {
  VkImageView view = VK_NULL_HANDLE;
  VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
};

// Writes `set`, allocated against set 0's layout: both planes through
// `sampler`, and `picture`, a buffer holding a PictureUniform.
inline void write_set0(const core::DescriptorSet& set, VkSampler sampler,
                       PlaneBinding plane0, PlaneBinding plane1,
                       VkBuffer picture) {
  set.write_combined_image_sampler(0, plane0.view, sampler, plane0.layout);
  set.write_combined_image_sampler(1, plane1.view, sampler, plane1.layout);
  set.write_uniform_buffer(2, picture, 0, sizeof(PictureUniform));
}

}  // namespace volumetric_kit::gfx::pipelines::detail

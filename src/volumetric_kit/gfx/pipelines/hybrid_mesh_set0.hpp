// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// HybridMeshPipeline's set 0, as hybrid_mesh.frag declares it: what the
// pipeline's fallback, StreamedAtlas and HybridMeshTextureSet write. Binding 0
// is an RGBA image or picture 0's luma, binding 1 picture 0's chroma, binding
// 2 a PictureUniform saying which, and bindings 3 and 4 picture 1's luma and
// chroma. A binding the set's form does not read holds a placeholder: an RGBA
// set's chroma and picture 1, and a one-picture set's picture 1.

#include <array>
#include <cstddef>
#include <cstdint>

#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::gfx::pipelines::detail {

// The most Y'CbCr pictures set 0 binds: HybridMeshTextureSet::kMaxPictures.
constexpr uint32_t kSet0Pictures = 2;

// One picture's coding in hybrid_mesh.frag's `Picture` block (std140, 32
// bytes). Plain floats rather than glm types, so no glm alignment setting can
// move a member.
struct PictureCoding {
  uint32_t full_range;
  float kr;
  float kb;
  uint32_t chroma_siting;  // ChromaSiting
  float extent[2];         // the picture's size in luma texels
  uint32_t padding[2];     // to std140's 16-byte array stride
};
static_assert(sizeof(PictureCoding) == 32, "PictureCoding must match");
static_assert(offsetof(PictureCoding, extent) == 16,
              "PictureCoding layout drift");

// hybrid_mesh.frag's `Picture` block (std140, 80 bytes).
struct PictureUniform {
  uint32_t pictures;    // 0: binding 0 is RGBA; n: n Y'CbCr 4:2:0 pictures
  uint32_t padding[3];  // to the array's 16-byte alignment
  PictureCoding coding[kSet0Pictures];
};
static_assert(sizeof(PictureUniform) == 80, "PictureUniform must match");
static_assert(offsetof(PictureUniform, coding) == 16,
              "PictureUniform layout drift");

// The pool sizes that hold `sets` set-0 sets.
inline std::array<VkDescriptorPoolSize, 2> set0_pool_sizes(uint32_t sets) {
  return {VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               2 * kSet0Pictures * sets},
          VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sets}};
}

// One plane binding: a view, sampled in `layout`.
struct PlaneBinding {
  VkImageView view = VK_NULL_HANDLE;
  VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
};

// The planes set 0 binds: picture p's luma (or an RGBA image) and chroma at
// planes[2p] and planes[2p + 1].
using Set0Planes = std::array<PlaneBinding, 2 * kSet0Pictures>;

// Writes `set`, allocated against set 0's layout: every plane through
// `sampler`, and `picture`, a buffer holding a PictureUniform.
inline void write_set0(const core::DescriptorSet& set, VkSampler sampler,
                       const Set0Planes& planes, VkBuffer picture) {
  // Bindings 0 and 1, then 3 and 4: the uniform sits between the pictures.
  constexpr std::array<uint32_t, 2 * kSet0Pictures> kBindings = {0, 1, 3, 4};
  for (size_t i = 0; i < planes.size(); ++i) {
    set.write_combined_image_sampler(kBindings[i], planes[i].view, sampler,
                                     planes[i].layout);
  }
  set.write_uniform_buffer(2, picture, 0, sizeof(PictureUniform));
}

}  // namespace volumetric_kit::gfx::pipelines::detail

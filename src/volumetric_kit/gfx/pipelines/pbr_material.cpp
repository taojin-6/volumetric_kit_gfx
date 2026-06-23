// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"

#include <cstring>
#include <utility>

#include <glm/vec4.hpp>

namespace volumetric_kit::gfx::pipelines {

namespace {

// std140 per-material parameters; mirrors the Material block in the embedded
// model.frag (16-byte aligned vec4s, then four tightly-packed floats = 48 B).
struct MaterialUbo {
  glm::vec4 base_color_factor;
  glm::vec4 emissive_factor;  // .rgb used
  float metallic_factor;
  float roughness_factor;
  float normal_scale;
  float occlusion_strength;
};
static_assert(sizeof(MaterialUbo) == 48,
              "MaterialUbo must match the std140 Material block");

}  // namespace

Result<PbrMaterial> PbrMaterial::create(VkDevice device, Allocator& allocator,
                                        VkDescriptorSetLayout material_layout,
                                        const PbrMaterialDesc& desc) {
  if (device == VK_NULL_HANDLE || material_layout == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "PbrMaterial::create: device and material_layout must be non-null");
  }
  if (desc.base_color == VK_NULL_HANDLE ||
      desc.metallic_roughness == VK_NULL_HANDLE ||
      desc.normal == VK_NULL_HANDLE || desc.occlusion == VK_NULL_HANDLE ||
      desc.emissive == VK_NULL_HANDLE || desc.sampler == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "PbrMaterial::create: all five map views and the sampler must be "
        "non-null");
  }

  VG_ASSIGN(OwnedDescriptorSet resources,
            OwnedDescriptorSet::create(device, allocator, material_layout,
                                       sizeof(MaterialUbo), 5));

  // Factor UBO (binding 0): written once -- the factors do not change per
  // frame.
  MaterialUbo block{};
  block.base_color_factor = desc.base_color_factor;
  block.emissive_factor = glm::vec4(desc.emissive_factor, 0.0f);
  block.metallic_factor = desc.metallic_factor;
  block.roughness_factor = desc.roughness_factor;
  block.normal_scale = desc.normal_scale;
  block.occlusion_strength = desc.occlusion_strength;
  std::memcpy(resources.mapped(), &block, sizeof(block));

  // The five maps follow the factor UBO at bindings 1-5, matching model.frag.
  const VkImageView maps[5] = {desc.base_color, desc.metallic_roughness,
                               desc.normal, desc.occlusion, desc.emissive};
  for (uint32_t b = 0; b < 5; ++b) {
    resources.set().write_combined_image_sampler(
        b + 1, maps[b], desc.sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  PbrMaterial material;
  material.resources_ = std::move(resources);
  return material;
}

}  // namespace volumetric_kit::gfx::pipelines

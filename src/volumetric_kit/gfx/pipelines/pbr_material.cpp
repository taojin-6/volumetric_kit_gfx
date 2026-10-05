// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"

#include <utility>

#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/core/texture_upload.hpp"

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

Result<PbrMaterial> PbrMaterial::create(VkDevice device, UploadBatch& batch,
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

  // Factor UBO (binding 0): uploaded once into device-only memory -- the
  // factors do not change per frame. add_buffer copies the block into staging
  // now, so it need not outlive this call.
  MaterialUbo block{};
  block.base_color_factor = desc.base_color_factor;
  block.emissive_factor = glm::vec4(desc.emissive_factor, 0.0f);
  block.metallic_factor = desc.metallic_factor;
  block.roughness_factor = desc.roughness_factor;
  block.normal_scale = desc.normal_scale;
  block.occlusion_strength = desc.occlusion_strength;
  BufferUploadDesc upload;
  upload.data = &block;
  upload.size = sizeof(block);
  upload.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  VG_ASSIGN(Buffer ubo, batch.add_buffer(upload));

  Result<OwnedDescriptorSet> made =
      OwnedDescriptorSet::create(device, std::move(ubo), material_layout, 5);
  if (!made.ok()) {
    // The batch queued a copy into the UBO, which failing here freed.
    batch.poison();
    return made.status();
  }
  OwnedDescriptorSet resources = std::move(made).value();

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

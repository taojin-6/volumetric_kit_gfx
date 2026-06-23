// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"

#include <cstring>
#include <utility>

#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/core/allocator.hpp"

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

  // Factor UBO: host-mapped, written once (the factors do not change per
  // frame).
  BufferDesc bd;
  bd.size = sizeof(MaterialUbo);
  bd.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  bd.memory = MemoryUsage::HostVisible;
  bd.mapped = true;
  VG_ASSIGN(Buffer ubo, allocator.create_buffer(bd));

  MaterialUbo block{};
  block.base_color_factor = desc.base_color_factor;
  block.emissive_factor = glm::vec4(desc.emissive_factor, 0.0f);
  block.metallic_factor = desc.metallic_factor;
  block.roughness_factor = desc.roughness_factor;
  block.normal_scale = desc.normal_scale;
  block.occlusion_strength = desc.occlusion_strength;
  std::memcpy(ubo.mapped(), &block, sizeof(block));

  // One-set pool: this material's factor UBO + its five sampled maps.
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5}};
  VG_ASSIGN(DescriptorPool pool, DescriptorPool::create(device, sizes, 2, 1));
  VG_ASSIGN(DescriptorSet set, pool.allocate(material_layout));

  set.write_uniform_buffer(0, ubo.handle(), 0, sizeof(MaterialUbo));
  // The five maps follow the factor UBO at bindings 1-5, matching model.frag.
  const VkImageView maps[5] = {desc.base_color, desc.metallic_roughness,
                               desc.normal, desc.occlusion, desc.emissive};
  for (uint32_t b = 0; b < 5; ++b) {
    set.write_combined_image_sampler(b + 1, maps[b], desc.sampler,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  PbrMaterial material;
  material.pool_ = std::move(pool);
  material.set_ = set;
  material.ubo_ = std::move(ubo);
  return material;
}

// The pool + UBO move themselves; null the set value too (it is a borrowed
// handle, freed with the pool) so a moved-from material is fully empty and its
// accessors stay consistent with valid().
PbrMaterial::PbrMaterial(PbrMaterial&& other) noexcept
    : pool_(std::move(other.pool_)),
      set_(other.set_),
      ubo_(std::move(other.ubo_)) {
  other.set_ = DescriptorSet{};
}

PbrMaterial& PbrMaterial::operator=(PbrMaterial&& other) noexcept {
  if (this != &other) {
    pool_ = std::move(other.pool_);
    set_ = other.set_;
    ubo_ = std::move(other.ubo_);
    other.set_ = DescriptorSet{};
  }
  return *this;
}

}  // namespace volumetric_kit::gfx::pipelines

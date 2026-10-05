// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
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

// Distance between two materials' blocks in the shared factor UBO: 256 bytes,
// the largest minUniformBufferOffsetAlignment Vulkan allows, so every offset
// suits any device without asking it.
constexpr VkDeviceSize kFactorStride = 256;
static_assert(sizeof(MaterialUbo) <= kFactorStride,
              "a material's block must fit its stride");

MaterialUbo factors_of(const PbrMaterialDesc& desc) {
  MaterialUbo block{};
  block.base_color_factor = desc.base_color_factor;
  block.emissive_factor = glm::vec4(desc.emissive_factor, 0.0f);
  block.metallic_factor = desc.metallic_factor;
  block.roughness_factor = desc.roughness_factor;
  block.normal_scale = desc.normal_scale;
  block.occlusion_strength = desc.occlusion_strength;
  return block;
}

}  // namespace

core::Result<std::vector<PbrMaterial>> PbrMaterial::create_all(
    VkDevice device, UploadBatch& batch, VkDescriptorSetLayout material_layout,
    const std::vector<PbrMaterialDesc>& descs) {
  if (device == VK_NULL_HANDLE || material_layout == VK_NULL_HANDLE) {
    return core::Status::invalid_argument(
        "PbrMaterial::create_all: device and material_layout must be "
        "non-null");
  }
  if (descs.empty()) {
    return core::Status::invalid_argument(
        "PbrMaterial::create_all: descs must name at least one material");
  }
  for (const PbrMaterialDesc& desc : descs) {
    if (desc.base_color == VK_NULL_HANDLE ||
        desc.metallic_roughness == VK_NULL_HANDLE ||
        desc.normal == VK_NULL_HANDLE || desc.occlusion == VK_NULL_HANDLE ||
        desc.emissive == VK_NULL_HANDLE || desc.sampler == VK_NULL_HANDLE) {
      return core::Status::invalid_argument(
          "PbrMaterial::create_all: all five map views and the sampler must "
          "be non-null");
    }
  }

  // Everything that can fail comes before the factors' upload is queued, so a
  // failed call leaves the batch as it found it: the pools and sets, and every
  // host allocation the binding below needs, so nothing after the add throws
  // and drops the buffer its queued copy writes.
  std::vector<OwnedDescriptorSet> sets;
  sets.reserve(descs.size());
  for (size_t i = 0; i < descs.size(); ++i) {
    VKC_ASSIGN(OwnedDescriptorSet set,
               OwnedDescriptorSet::create(device, material_layout, 5));
    sets.push_back(std::move(set));
  }
  std::vector<uint8_t> packed(descs.size() * kFactorStride, 0);
  for (size_t i = 0; i < descs.size(); ++i) {
    const MaterialUbo block = factors_of(descs[i]);
    std::memcpy(packed.data() + i * kFactorStride, &block, sizeof(block));
  }
  auto factors = std::make_shared<core::Buffer>();
  std::vector<PbrMaterial> materials;
  materials.reserve(descs.size());

  // Factor UBO (binding 0): uploaded once into device-only memory -- the
  // factors do not change per frame. add_buffer copies the blocks into
  // staging now, so they need not outlive this call, and a failed add queues
  // nothing.
  BufferUploadDesc upload;
  upload.data = packed.data();
  upload.size = packed.size();
  upload.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  VKC_ASSIGN(core::Buffer uploaded, batch.add_buffer(upload));
  *factors = std::move(uploaded);

  for (size_t i = 0; i < descs.size(); ++i) {
    OwnedDescriptorSet& set = sets[i];
    set.bind_uniform(factors, i * kFactorStride, sizeof(MaterialUbo));
    // The five maps follow the factor UBO at bindings 1-5, matching
    // model.frag.
    const PbrMaterialDesc& desc = descs[i];
    const VkImageView maps[5] = {desc.base_color, desc.metallic_roughness,
                                 desc.normal, desc.occlusion, desc.emissive};
    for (uint32_t b = 0; b < 5; ++b) {
      set.set().write_combined_image_sampler(
          b + 1, maps[b], desc.sampler,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    PbrMaterial material;
    material.resources_ = std::move(set);
    materials.push_back(std::move(material));
  }
  return materials;
}

core::Result<PbrMaterial> PbrMaterial::create(
    VkDevice device, UploadBatch& batch, VkDescriptorSetLayout material_layout,
    const PbrMaterialDesc& desc) {
  VKC_ASSIGN(std::vector<PbrMaterial> materials,
             create_all(device, batch, material_layout, {desc}));
  return std::move(materials.front());
}

}  // namespace volumetric_kit::gfx::pipelines

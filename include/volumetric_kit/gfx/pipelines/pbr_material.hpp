// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pbr_material.hpp
/// @brief One material's set-1 binding for @ref PbrPipeline: a factor UBO plus
///        the five glTF metallic-roughness maps.

#include <vector>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/impl/owned_descriptor_set.hpp"

namespace volumetric_kit::gfx {
class UploadBatch;
}  // namespace volumetric_kit::gfx

namespace volumetric_kit::gfx::pipelines {

/// @brief The factors and the five maps that describe one PBR material.
///
/// The factors mirror @ref PbrPipeline's embedded `model.frag` Material block;
/// every map is a sampled image **view** the caller owns (uploaded once and
/// shared between materials in practice). All five views and the @ref sampler
/// must be non-null: the shader samples each unconditionally, so a material
/// with no texture for a slot binds a fallback (e.g. a 1x1 white image, or a
/// flat
/// `(0.5,0.5,1)` normal) rather than leaving the slot empty.
struct PbrMaterialDesc {
  glm::vec4 base_color_factor{1.0f};  ///< Multiplies the base-color texture.
  glm::vec3 emissive_factor{0.0f};    ///< Multiplies the emissive texture.
  float metallic_factor = 1.0f;       ///< Multiplies the metallic channel.
  float roughness_factor = 1.0f;      ///< Multiplies the roughness channel.
  float normal_scale = 1.0f;          ///< Scales the tangent-space normal.
  float occlusion_strength = 1.0f;    ///< Lerps occlusion toward 1.0.

  VkImageView base_color =
      VK_NULL_HANDLE;  ///< sRGB base-color map (binding 1).
  VkImageView metallic_roughness =
      VK_NULL_HANDLE;                   ///< Linear metal-rough (binding 2).
  VkImageView normal = VK_NULL_HANDLE;  ///< Linear tangent normal (binding 3).
  VkImageView occlusion = VK_NULL_HANDLE;  ///< Linear occlusion (binding 4).
  VkImageView emissive = VK_NULL_HANDLE;   ///< sRGB emissive map (binding 5).
  VkSampler sampler = VK_NULL_HANDLE;      ///< Filters all five maps.
};

/// @brief Owns the set-1 descriptor resources for one @ref PbrPipeline
///        material: a factor uniform buffer plus the five sampled maps.
///
/// Self-contained, like @ref GpuMesh: it owns its own one-set
/// `core::DescriptorPool` and the `core::DescriptorSet` allocated from it, and
/// shares the factor UBO its factors live in. The factors never change, so the
/// UBO is device-only memory, uploaded through an @ref UploadBatch as a mesh's
/// buffers are: build the materials with @ref create_all (or one with
/// @ref create) against `PbrPipeline::descriptor_set_layout(1)`, finish the
/// batch, then name them in @ref PbrDraw "PbrDraws". @ref create_all packs
/// every material's factors into one UBO, uploaded by one copy. A
/// default-constructed `PbrMaterial` is empty (`valid()` is false) and safe
/// to move-assign into.
///
/// @warning The device, plus every image @ref PbrMaterialDesc names, must
///          outlive the material — its descriptor set points at them. Keep
///          the material alive until the batch it was created on has
///          finished: the copy the batch queued writes its factor UBO, so
///          destroying it earlier would submit against a freed buffer.
///
/// @code
/// std::vector<PbrMaterialDesc> descs(model.materials.size());
/// descs[0].base_color_factor = model.materials[0].base_color_factor;
/// descs[0].base_color = base_view;  // ... the other four maps + sampler
/// VG_ASSIGN(UploadBatch batch, UploadBatch::begin(device, allocator));
/// VG_ASSIGN(std::vector<pipelines::PbrMaterial> materials,
///           pipelines::PbrMaterial::create_all(
///               device.handle(), batch, pbr.descriptor_set_layout(1), descs));
/// VG_TRY(batch.finish());  // the factors are in place; draw the materials
/// @endcode
class VG_PIPELINES_API PbrMaterial {
 public:
  /// @brief Construct an empty material (owns nothing; `valid()` is false).
  PbrMaterial() = default;

  /// @brief Build the set-1 bindings for many materials, queuing one upload
  ///        of all their factors on @p batch.
  ///
  /// Every material's factors go into one device-only UBO the materials
  /// share, each at its own 256-byte-aligned offset (a multiple of any
  /// device's `minUniformBufferOffsetAlignment`): one staging buffer, one
  /// destination and one copy for the lot. The pools and sets are allocated
  /// before the upload is queued, so a failed call leaves @p batch unchanged
  /// and usable.
  /// @param device          The logical device that owns the pools + sets.
  /// @param batch           An open batch, on @p device, that uploads the
  ///                        factor UBO; draw the materials only once it has
  ///                        finished.
  /// @param material_layout The reflected set-1 layout, from
  ///                        @ref PbrPipeline::descriptor_set_layout(1).
  /// @param descs           One desc per material: factors plus the five map
  ///                        views + sampler.
  /// @pre @p device and @p material_layout are non-`VK_NULL_HANDLE`, @p descs
  ///      is non-empty, and every view in each desc plus its sampler is
  ///      non-`VK_NULL_HANDLE` — validated before Vulkan is touched,
  ///      otherwise a non-OK @ref Status with domain
  ///      @ref Status::Code::InvalidArgument.
  /// @return The materials, parallel to @p descs, on success; or a non-OK
  ///         @ref Status: a backend Status from pool / set allocation, or
  ///         what @ref UploadBatch::add_buffer returns.
  /// @warning Keep the materials alive until @p batch's
  ///          @ref UploadBatch::finish returns (see the class warning).
  static Result<std::vector<PbrMaterial>> create_all(
      VkDevice device, UploadBatch& batch,
      VkDescriptorSetLayout material_layout,
      const std::vector<PbrMaterialDesc>& descs);

  /// @brief Build the set-1 binding for one material, queuing its factors'
  ///        upload on @p batch: @ref create_all for one desc.
  /// @param device          The logical device that owns the pool + set.
  /// @param batch           An open batch, on @p device, that uploads the
  ///                        factor UBO; draw the material only once it has
  ///                        finished.
  /// @param material_layout The reflected set-1 layout, from
  ///                        @ref PbrPipeline::descriptor_set_layout(1).
  /// @param desc            Factors plus the five map views + sampler.
  /// @pre As for @ref create_all.
  /// @return The material on success, or a non-OK @ref Status as
  ///         @ref create_all returns; a failed call leaves @p batch
  ///         unchanged.
  /// @warning Keep the material alive until @p batch's
  ///          @ref UploadBatch::finish returns (see the class warning).
  static Result<PbrMaterial> create(VkDevice device, UploadBatch& batch,
                                    VkDescriptorSetLayout material_layout,
                                    const PbrMaterialDesc& desc);

  ~PbrMaterial() = default;
  PbrMaterial(PbrMaterial&&) noexcept = default;
  PbrMaterial& operator=(PbrMaterial&&) noexcept = default;
  PbrMaterial(const PbrMaterial&) = delete;
  PbrMaterial& operator=(const PbrMaterial&) = delete;

  /// @return The set-1 `VkDescriptorSet` to bind (`VK_NULL_HANDLE` when empty).
  VkDescriptorSet descriptor_set() const noexcept {
    return resources_.descriptor_set();
  }

  /// @return `true` if this owns a built material set.
  bool valid() const noexcept { return resources_.valid(); }

 private:
  OwnedDescriptorSet resources_;  // set 1: pool + set + shared factor UBO
};

}  // namespace volumetric_kit::gfx::pipelines

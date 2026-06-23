// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pbr_material.hpp
/// @brief One material's set-1 binding for @ref PbrPipeline: a factor UBO plus
///        the five glTF metallic-roughness maps.

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::gfx {
class Allocator;
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
/// Self-contained, like @ref GpuMesh: it owns its own one-set @ref
/// DescriptorPool, the @ref DescriptorSet allocated from it, and the factor
/// UBO. Build one per material with @ref create against
/// `PbrPipeline::descriptor_set_layout(1)`, then name it in a @ref PbrDraw. A
/// default-constructed `PbrMaterial` is empty (`valid()` is false) and safe to
/// move-assign into.
///
/// @warning The @p allocator passed to @ref create (and the device that backs
///          it), plus every image @ref PbrMaterialDesc names, must outlive the
///          material — its descriptor set points at them.
///
/// @code
/// PbrMaterialDesc d;
/// d.base_color_factor = m.base_color_factor;
/// d.base_color = base_view;  // ... and the other four maps + sampler
/// Result<pipelines::PbrMaterial> mat = pipelines::PbrMaterial::create(
///     device, allocator, pbr.descriptor_set_layout(1), d);
/// @endcode
class VG_PIPELINES_API PbrMaterial {
 public:
  /// @brief Construct an empty material (owns nothing; `valid()` is false).
  PbrMaterial() = default;

  /// @brief Build the set-1 binding for one material.
  /// @param device          The logical device that owns the pool + set.
  /// @param allocator       Allocates the factor UBO; must outlive the
  /// material.
  /// @param material_layout The reflected set-1 layout, from
  ///                        @ref PbrPipeline::descriptor_set_layout(1).
  /// @param desc            Factors plus the five map views + sampler.
  /// @pre @p device and @p material_layout are non-`VK_NULL_HANDLE`, and every
  ///      view in @p desc plus @p desc.sampler is non-`VK_NULL_HANDLE` —
  ///      validated before Vulkan is touched, otherwise a non-OK @ref Status
  ///      with domain @ref Status::Code::InvalidArgument.
  /// @return The material on success, or a non-OK @ref Status (a Vulkan-domain
  ///         Status from buffer / pool / set allocation).
  static Result<PbrMaterial> create(VkDevice device, Allocator& allocator,
                                    VkDescriptorSetLayout material_layout,
                                    const PbrMaterialDesc& desc);

  ~PbrMaterial() = default;
  PbrMaterial(PbrMaterial&& other) noexcept;
  PbrMaterial& operator=(PbrMaterial&& other) noexcept;
  PbrMaterial(const PbrMaterial&) = delete;
  PbrMaterial& operator=(const PbrMaterial&) = delete;

  /// @return The set-1 `VkDescriptorSet` to bind (`VK_NULL_HANDLE` when empty).
  VkDescriptorSet descriptor_set() const noexcept { return set_.handle(); }

  /// @return `true` if this owns a built material set.
  bool valid() const noexcept { return pool_.valid(); }

 private:
  DescriptorPool pool_;  // one-set pool that owns set_'s lifetime
  DescriptorSet set_;    // set 1: factor UBO + the five maps
  Buffer ubo_;           // host-mapped factor UBO the set points at
};

}  // namespace volumetric_kit::gfx::pipelines

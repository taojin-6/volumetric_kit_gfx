// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pbr_scene.hpp
/// @brief The per-frame set-0 binding for @ref PbrPipeline: the camera plus the
///        image-based-lighting environment maps.

#include <glm/vec3.hpp>

#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/impl/owned_descriptor_set.hpp"

namespace volumetric_kit::gfx {
class Allocator;
}  // namespace volumetric_kit::gfx

namespace volumetric_kit::gfx::pipelines {

/// @brief The image-based-lighting maps bound into a @ref PbrScene (set 0,
///        bindings 1-3): a diffuse irradiance cube, a roughness-prefiltered
///        specular cube, and the BRDF integration LUT.
///
/// Every view and the @ref sampler must be non-null — the shader samples all
/// three unconditionally. The caller owns the images (they outlive the scene).
struct PbrSceneDesc {
  VkImageView irradiance = VK_NULL_HANDLE;  ///< Diffuse irradiance cube.
  VkImageView prefilter = VK_NULL_HANDLE;   ///< Prefiltered specular cube.
  VkImageView brdf_lut = VK_NULL_HANDLE;    ///< 2D BRDF integration LUT.
  VkSampler sampler = VK_NULL_HANDLE;  ///< Filters all three (mipped cube).
};

/// @brief Owns the set-0 descriptor resources @ref PbrPipeline binds once per
///        frame: a host-mapped camera uniform buffer plus the IBL maps.
///
/// This is the per-frame "view + environment lighting" half of the technique
/// (set 1, the material, is @ref PbrMaterial). It is **not** a scene graph —
/// the library keeps none; it is just the descriptor set for frame-constant
/// data. Self-contained like @ref GpuMesh: it owns its own one-set @ref
/// DescriptorPool, the @ref DescriptorSet, and the camera UBO. Refresh the
/// camera each frame with @ref set_camera, then name it in a @ref PbrFrame. A
/// default-constructed `PbrScene` is empty (`valid()` is false) and safe to
/// move-assign into.
///
/// @warning The @p allocator passed to @ref create (and the device that backs
///          it), plus the IBL images @ref PbrSceneDesc names, must outlive the
///          scene. The camera UBO is single-buffered, so @ref set_camera is
///          safe only with one frame in flight (the previous frame's fence has
///          been waited); more frames in flight need one scene per slot.
///
/// @code
/// Result<pipelines::PbrScene> scene = pipelines::PbrScene::create(
///     device, allocator, pbr.descriptor_set_layout(0), ibl_desc);
/// // each frame:
/// scene.value().set_camera(eye, prefilter_max_lod);
/// @endcode
class VG_PIPELINES_API PbrScene {
 public:
  /// @brief Construct an empty scene (owns nothing; `valid()` is false).
  PbrScene() = default;

  /// @brief Build the set-0 binding from the IBL environment.
  /// @param device       The logical device that owns the pool + set.
  /// @param allocator    Allocates the camera UBO; must outlive the scene.
  /// @param scene_layout The reflected set-0 layout, from
  ///                     @ref PbrPipeline::descriptor_set_layout(0).
  /// @param desc         The three IBL map views + sampler.
  /// @pre @p device and @p scene_layout are non-`VK_NULL_HANDLE`, and every
  /// view
  ///      in @p desc plus @p desc.sampler is non-`VK_NULL_HANDLE` — validated
  ///      before Vulkan is touched, otherwise a non-OK @ref Status with domain
  ///      @ref Status::Code::InvalidArgument.
  /// @return The scene on success, or a non-OK @ref Status (a Vulkan-domain
  ///         Status from buffer / pool / set allocation).
  static Result<PbrScene> create(VkDevice device, Allocator& allocator,
                                 VkDescriptorSetLayout scene_layout,
                                 const PbrSceneDesc& desc);

  ~PbrScene() = default;
  PbrScene(PbrScene&&) noexcept = default;
  PbrScene& operator=(PbrScene&&) noexcept = default;
  PbrScene(const PbrScene&) = delete;
  PbrScene& operator=(const PbrScene&) = delete;

  /// @brief Write the per-frame camera into the mapped UBO.
  /// @param eye               World-space camera position.
  /// @param prefilter_max_lod The specular prefilter's highest mip index
  ///                          (mip count - 1); the shader clamps roughness LOD
  ///                          to it.
  /// @pre `valid()`.
  void set_camera(const glm::vec3& eye, float prefilter_max_lod) noexcept;

  /// @return The set-0 `VkDescriptorSet` to bind (`VK_NULL_HANDLE` when empty).
  VkDescriptorSet descriptor_set() const noexcept {
    return resources_.descriptor_set();
  }

  /// @return `true` if this owns a built scene set.
  bool valid() const noexcept { return resources_.valid(); }

 private:
  OwnedDescriptorSet resources_;  // set 0: pool + set + camera UBO
};

}  // namespace volumetric_kit::gfx::pipelines

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pbr_model.hpp
/// @brief The @ref assets::Model -> GPU bridge for @ref PbrPipeline: upload
///        every mesh and material map, build the materials, and flatten the
///        scene into a ready-to-submit draw list.

#include <cstdint>
#include <optional>
#include <vector>

#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_pipeline.hpp"

namespace volumetric_kit::gfx {
class Allocator;
class Device;
namespace assets {
struct Material;
struct Model;
}  // namespace assets
}  // namespace volumetric_kit::gfx

namespace volumetric_kit::gfx::pipelines {

/// @brief Map an @ref assets::Material's factors into a @ref PbrMaterialDesc.
///
/// The single home of the factor mapping (base color, metallic, roughness,
/// normal scale, occlusion strength, emissive): @ref PbrModel runs every
/// material through it, and a consumer wiring @ref PbrMaterial by hand starts
/// from it instead of copying fields. The map views and the sampler are left
/// `VK_NULL_HANDLE` -- resolve those against the images the material's texture
/// indices name (@ref PbrMaterial::create requires all of them non-null).
///
/// @note @ref assets::Material::alpha_mode, @ref assets::Material::alpha_cutoff
///       and @ref assets::Material::double_sided have no @ref PbrPipeline
///       counterpart yet and are not mapped.
/// @param material  The CPU-side material whose factors to map.
/// @return A @ref PbrMaterialDesc with every factor filled and every view plus
///         the sampler left `VK_NULL_HANDLE`.
VG_PIPELINES_API PbrMaterialDesc
pbr_material_desc(const assets::Material& material);

/// @brief Everything GPU-side for one @ref assets::Model against a
///        @ref PbrPipeline: device-local meshes, uploaded material maps, built
///        set-1 materials, and the scene flattened into a @ref PbrDraw list.
///
/// The bridge from the CPU asset (io/assets tiers) to the draw call: @ref
/// create walks the model's scene tree (guarded against cycles and
/// out-of-range indices, so a malformed file cannot recurse forever), composes
/// each node's transform down to world space, uploads every non-empty mesh and
/// every referenced image through **one** @ref UploadBatch (a single submit),
/// and builds one @ref PbrMaterial per material plus a matte-white fallback
/// for meshes with no material. Each image is uploaded once, in the color
/// space its material slots dictate: base-color and emissive maps are sRGB,
/// the rest linear -- and when one image serves both kinds, sRGB wins. Slots
/// with no texture bind a generated 1x1 white (or flat-normal) fallback, so
/// every material satisfies @ref PbrMaterial's all-views-bound contract.
///
/// A model with no meshes (or no drawable ones) is a legal no-op: @ref create
/// succeeds and @ref draws is empty. Feed @ref draws straight into a
/// @ref PbrFrame; the per-frame set 0 (@ref PbrScene) stays with the caller.
///
/// @warning The @p device and @p allocator passed to @ref create must outlive
///          the model -- every buffer, texture, and descriptor set here is
///          allocated from them. The @ref PbrDraw list borrows the model's own
///          meshes and materials, so it is valid exactly as long as this
///          object.
///
/// @code
/// std::optional<assets::Model> model = io::load_gltf("helmet.glb", &err);
/// if (!model) return fail(err);
/// Result<pipelines::PbrModel> gpu =
///     pipelines::PbrModel::create(device, allocator, pbr, *model);
/// if (!gpu) return fail(gpu.status().message());
/// pipelines::PbrFrame frame;  // + extent / view_proj / scene / slot
/// frame.draws = gpu.value().draws().data();
/// frame.draw_count = static_cast<uint32_t>(gpu.value().draws().size());
/// pbr.submit(cmd, frame);
/// @endcode
class VG_PIPELINES_API PbrModel {
 public:
  /// @brief Construct an empty model (owns nothing; `valid()` is false).
  PbrModel() = default;

  /// @brief Upload @p model and build its materials + draw list for
  ///        @p pipeline.
  /// @param device     Runs the upload submit and owns the descriptor pools;
  ///                   must outlive the model.
  /// @param allocator  Allocates every buffer and texture; must outlive the
  ///                   model.
  /// @param pipeline   Supplies the reflected set-1 material layout; the model
  ///                   is drawable through any @ref PbrPipeline with the same
  ///                   layout.
  /// @param model      The CPU asset to upload. Not referenced after create
  ///                   returns. Out-of-range node/mesh/material indices are
  ///                   skipped; empty meshes stay as skipped draws; a model
  ///                   with no meshes yields a valid model with zero draws.
  /// @pre @p device holds a live `VkDevice` and @p pipeline is `valid()` --
  ///      validated before Vulkan is touched, otherwise a non-OK @ref Status
  ///      with domain @ref Status::Code::InvalidArgument.
  /// @return The model on success, or a non-OK @ref Status (a Vulkan-domain
  ///         Status from the upload, sampler, or material step).
  static Result<PbrModel> create(const Device& device, Allocator& allocator,
                                 const PbrPipeline& pipeline,
                                 const assets::Model& model);

  ~PbrModel() = default;
  PbrModel(PbrModel&&) noexcept = default;
  // Hand-written to guard self-move (a defaulted move-assign hands each vector
  // a self-move, which is valid-but-unspecified). Members are adopted in
  // reverse declaration order so the destination's old materials release their
  // descriptor sets before the textures those sets reference are freed.
  PbrModel& operator=(PbrModel&& other) noexcept {
    if (this != &other) {
      draws_ = std::move(other.draws_);
      materials_ = std::move(other.materials_);
      sampler_ = std::move(other.sampler_);
      textures_ = std::move(other.textures_);
      meshes_ = std::move(other.meshes_);
    }
    return *this;
  }
  PbrModel(const PbrModel&) = delete;
  PbrModel& operator=(const PbrModel&) = delete;

  /// @return The flattened draw list -- one @ref PbrDraw per (instanced) mesh,
  ///         borrowing this model's meshes and materials. Feed it to
  ///         @ref PbrFrame::draws. Empty when the source model drew nothing.
  const std::vector<PbrDraw>& draws() const noexcept { return draws_; }

  /// @return One GPU mesh slot per source @ref assets::Model::meshes entry
  ///         (empty source meshes keep a default, skipped slot); 0 when empty.
  uint32_t mesh_count() const noexcept {
    return static_cast<uint32_t>(meshes_.size());
  }

  /// @return One material per source @ref assets::Model::materials entry plus
  ///         the shared fallback -- so >= 1 on any created model; 0 when empty.
  uint32_t material_count() const noexcept {
    return static_cast<uint32_t>(materials_.size());
  }

  /// @return `true` if this owns a built model (a created model always owns at
  ///         least the fallback material, even with zero draws).
  bool valid() const noexcept { return !materials_.empty(); }

 private:
  // Destruction runs bottom-up: draws_ (borrowed pointers) first, then the
  // materials -- whose descriptor sets reference textures_ and sampler_ -- and
  // only then the textures, sampler, and meshes they point at. Keep textures_
  // and sampler_ declared before materials_.
  std::vector<GpuMesh> meshes_;     // parallel to Model::meshes
  std::vector<Texture> textures_;   // every uploaded map + the two fallbacks
  std::optional<Sampler> sampler_;  // filters every material map
  std::vector<PbrMaterial> materials_;  // parallel to Model::materials, then
                                        // the fallback material last
  std::vector<PbrDraw> draws_;  // borrows meshes_ + materials_ (stable: vector
                                // moves keep element addresses)
};

}  // namespace volumetric_kit::gfx::pipelines

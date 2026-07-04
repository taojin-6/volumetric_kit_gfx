// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pbr_pipeline.hpp
/// @brief The glTF metallic-roughness PBR graphics pipeline, shaders embedded.

#include <cstdint>

#include <glm/mat4x4.hpp>

#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::gfx::pipelines {

class GpuMesh;
class PbrMaterial;
class PbrScene;
struct PbrFrame;

/// @brief The metallic-roughness PBR graphics pipeline: an interleaved
///        @ref assets::Vertex mesh, depth-tested, shaded by the library's own
///        embedded GLSL (no shader files for the consumer to manage).
///
/// Wraps a @ref GraphicsPipeline built from SPIR-V compiled into the library,
/// so a consumer gets the technique from @ref create alone. The reflected
/// layout exposes two descriptor sets, built by the matching helper types:
/// **set 0** (the per-frame @ref PbrScene -- camera + IBL) and **set 1** (one
/// @ref PbrMaterial per material -- a factor UBO + the five glTF maps). Record
/// a frame's draws with @ref submit. A default-constructed `PbrPipeline` is
/// empty
/// (`valid()` is false) and safe to move-assign into.
///
/// @warning The @p device passed to @ref create must outlive the pipeline.
///
/// @code
/// Result<pipelines::PbrPipeline> pbr =
///     pipelines::PbrPipeline::create(device, target.layout());
/// if (!pbr) return pbr.status();
/// // Build set 0 (PbrScene) + set 1 (PbrMaterial) against its reflected
/// // layouts, then each frame (f.slot = the in-flight slot):
/// scene.set_camera(f.slot, eye, prefilter_max_lod);
/// pbr.value().submit(cmd, frame);  // frame names the scene, slot, and draws
/// @endcode
class VG_PIPELINES_API PbrPipeline {
 public:
  /// @brief Construct an empty pipeline (owns nothing; `valid()` is false).
  PbrPipeline() = default;

  /// @brief Build the metallic-roughness pipeline for a render-target layout.
  /// @param device  The logical device that owns the pipeline.
  /// @param layout  The target's format/sample signature; must carry a depth
  ///                format (the pipeline is depth-tested).
  /// @pre @p device is non-`VK_NULL_HANDLE`; @p layout carries a depth format
  ///      and at least one color attachment with a defined format. These are
  ///      validated before Vulkan is touched and otherwise yield a non-OK
  ///      @ref Status with domain @ref Status::Code::InvalidArgument.
  /// @return The pipeline on success, or a non-OK @ref Status (e.g. @ref
  ///         Status::Code::InvalidArgument when @p layout has no depth format,
  ///         or a Vulkan-domain Status from shader-module / pipeline creation).
  static Result<PbrPipeline> create(VkDevice device,
                                    const RenderTargetLayout& layout);

  ~PbrPipeline() = default;
  PbrPipeline(PbrPipeline&&) noexcept = default;
  PbrPipeline& operator=(PbrPipeline&&) noexcept = default;
  PbrPipeline(const PbrPipeline&) = delete;
  PbrPipeline& operator=(const PbrPipeline&) = delete;

  /// @return The underlying `VkPipeline` (`VK_NULL_HANDLE` when empty).
  VkPipeline handle() const noexcept { return pipeline_.handle(); }

  /// @return The `VkPipelineLayout` (for push constants + descriptor binding).
  VkPipelineLayout layout() const noexcept { return pipeline_.layout(); }

  /// @return `true` if this owns a pipeline.
  bool valid() const noexcept { return pipeline_.valid(); }

  /// @return The number of descriptor sets the embedded shaders declare
  ///         (reflected -- set 0 scene + set 1 material, so 2).
  uint32_t descriptor_set_count() const noexcept {
    return pipeline_.descriptor_set_count();
  }

  /// @brief The reflected layout for descriptor @p set (0 = scene, 1 =
  /// material)
  ///        -- pass to @ref DescriptorPool::allocate.
  /// @param set  The set index.
  /// @return The set's `VkDescriptorSetLayout`, or `VK_NULL_HANDLE` if @p set
  /// is
  ///         beyond what the shaders declare.
  VkDescriptorSetLayout descriptor_set_layout(uint32_t set) const noexcept {
    return pipeline_.descriptor_set_layout(set);
  }

  /// @brief Record one frame's draws into @p cmd: bind the pipeline + a
  ///        full-target viewport, bind the scene set once (set 0), then for
  ///        each draw push its transform, bind its material set (set 1), and
  ///        draw the mesh.
  /// @param cmd    A recording-state command buffer, inside a dynamic-rendering
  ///               scope whose target matches the layout @ref create was given.
  /// @param frame  The scene, the draw list, and the view-projection (see
  ///               @ref PbrFrame).
  /// @pre `valid()`; `frame.scene` and each `frame.draws[i].mesh` / `.material`
  ///      are non-null and built against this pipeline's reflected layouts.
  ///      Draws whose mesh is null/empty or whose material is null are skipped.
  void submit(VkCommandBuffer cmd, const PbrFrame& frame) const;

 private:
  GraphicsPipeline pipeline_;
};

/// @brief One thing to draw: a @ref GpuMesh under a world transform, shaded by
/// a
///        @ref PbrMaterial.
///
/// A glTF mesh may be instanced by several nodes, so the transform lives on the
/// draw, not the mesh. The mesh and material are borrowed (they outlive the
/// @ref PbrFrame that names them).
struct PbrDraw {
  const GpuMesh* mesh = nullptr;  ///< The geometry to draw.
  glm::mat4 world{1.0f};          ///< World transform for this instance.
  const PbrMaterial* material = nullptr;  ///< Its set-1 material.
};

/// @brief Everything @ref PbrPipeline::submit records for one frame: the target
///        extent, the camera's view-projection, the scene set, and the draws.
///
/// The arrays/pointers are borrowed for the duration of the @ref
/// PbrPipeline::submit call.
struct PbrFrame {
  VkExtent2D extent{};        ///< Target size (sets the dynamic viewport).
  glm::mat4 view_proj{1.0f};  ///< projection * view; per draw `* world`.
  const PbrScene* scene = nullptr;  ///< Set 0, bound once for the frame.
  /// The frame's in-flight slot (e.g. the windowing `Frame::slot`): selects
  /// which of the scene's per-slot camera UBOs / descriptor sets to bind.
  uint32_t slot = 0;
  const PbrDraw* draws = nullptr;  ///< The draw list.
  uint32_t draw_count = 0;         ///< Number of @ref draws.
};

}  // namespace volumetric_kit::gfx::pipelines

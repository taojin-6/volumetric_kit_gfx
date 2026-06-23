// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pbr_pipeline.hpp
/// @brief The glTF metallic-roughness PBR graphics pipeline, shaders embedded.

#include <cstdint>

#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::gfx::pipelines {

/// @brief The metallic-roughness PBR graphics pipeline: an interleaved
///        @ref assets::Vertex mesh, depth-tested, shaded by the library's own
///        embedded GLSL (no shader files for the consumer to manage).
///
/// Wraps a @ref GraphicsPipeline built from SPIR-V compiled into the library,
/// so a consumer gets the technique from @ref create alone. The reflected
/// layout exposes two descriptor sets the consumer fills: **set 0** (scene --
/// camera + the IBL textures) and **set 1** (material -- a factor UBO + the
/// five glTF maps). A default-constructed `PbrPipeline` is empty (`valid()` is
/// false) and safe to move-assign into.
///
/// @warning The @p device passed to @ref create must outlive the pipeline.
///
/// @code
/// Result<pipelines::PbrPipeline> pbr =
///     pipelines::PbrPipeline::create(device, target.layout());
/// if (!pbr) return pbr.status();
/// // ... allocate set 0/1 from pbr.value().descriptor_set_layout(0/1),
/// //     bind pbr.value().handle(), then draw each GpuMesh ...
/// @endcode
class VG_PIPELINES_API PbrPipeline {
 public:
  /// @brief Construct an empty pipeline (owns nothing; `valid()` is false).
  PbrPipeline() = default;

  /// @brief Build the metallic-roughness pipeline for a render-target layout.
  /// @param device  The logical device that owns the pipeline.
  /// @param layout  The target's format/sample signature; must carry a depth
  ///                format (the pipeline is depth-tested).
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

 private:
  GraphicsPipeline pipeline_;
};

}  // namespace volumetric_kit::gfx::pipelines

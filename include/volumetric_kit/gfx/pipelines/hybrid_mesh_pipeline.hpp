// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file hybrid_mesh_pipeline.hpp
/// @brief The reconstruction "hybrid mesh" graphics pipeline: a world-space
///        vertex mesh shaded from a projective-texturing atlas where available
///        and from per-vertex color otherwise, shaders embedded.

#include <cstdint>
#include <variant>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/live_mesh.hpp"

namespace volumetric_kit::gfx::pipelines {

class GpuMesh;
struct HybridMeshFrame;

/// @brief Shading flags for @ref HybridMeshFrame::flags.
enum HybridMeshFlags : uint32_t {
  /// Apply single-directional-light diffuse + a constant ambient term. Clear
  /// for flat, unlit albedo (the raw projected/vertex color).
  kHybridMeshLit = 1u << 0,
};

/// @brief The reconstruction hybrid-mesh pipeline: an interleaved
///        @ref assets::Vertex mesh in **world space**, depth-tested, shaded by
///        the library's own embedded GLSL.
///
/// This is the renderer side of the `volumetric_kit_recon` handoff (gfx's
/// "hybrid mesh pipeline" milestone). Albedo is chosen per fragment: where a
/// triangle carries a valid atlas coordinate in `uv0` (projective texturing won
/// it a camera) the atlas texel is sampled; where `uv0` is **negative** the
/// interpolated per-vertex `color` is used (the TSDF vertex-color fallback).
/// The test is the sign, not the exact `(-1, -1)`, and a negative `uv0` still
/// carries a usable coordinate -- decoded as `-uv0 - 1`, of which `(-1, -1)`
/// is the `(0, 0)` case. See @ref LiveMesh. Lighting is toggled by
/// @ref kHybridMeshLit.
///
/// Wraps a @ref GraphicsPipeline built from SPIR-V compiled into the library,
/// so a consumer gets the technique from @ref create alone. The reflected
/// layout exposes **one** descriptor set -- **set 0, binding 0**: the atlas as
/// a combined image sampler. The fragment shader samples it unconditionally, so
/// a valid set is **always required** -- a purely vertex-colored mesh binds a
/// 1x1 image (whose texels are simply never selected). Per-frame constants (the
/// view-projection, light direction, and flags) ride a push constant, so there
/// is no per-frame UBO to manage. Record a frame's draws with @ref submit. A
/// default-constructed pipeline is empty (`valid()` is false) and safe to
/// move-assign into.
///
/// Vertices are consumed in world space -- the reconstruction mesh tier already
/// emits world-space positions and normals -- so the pipeline applies only the
/// camera's view-projection and carries no per-draw model matrix. That
/// world-space contract is also what keeps the push block within the 128-byte
/// guaranteed `maxPushConstantsSize`: a per-draw model plus a normal matrix,
/// alongside the light, would not fit -- so instancing one mesh under several
/// transforms is deliberately out of scope for this technique.
///
/// @warning The @p device passed to @ref create must outlive the pipeline.
///
/// @code
/// Result<pipelines::HybridMeshPipeline> pipe =
///     pipelines::HybridMeshPipeline::create(device, target.layout());
/// if (!pipe) return pipe.status();
/// // Allocate an atlas set against descriptor_set_layout(0), write the atlas
/// // combined-image-sampler into binding 0, then each frame:
/// pipe.value().submit(cmd, frame);  // frame names the atlas set, draws,
/// camera
/// @endcode
class VG_PIPELINES_API HybridMeshPipeline {
 public:
  /// @brief Construct an empty pipeline (owns nothing; `valid()` is false).
  HybridMeshPipeline() = default;

  /// @brief Build the hybrid-mesh pipeline for a render-target layout.
  /// @param device  The logical device that owns the pipeline.
  /// @param layout  The target's format/sample signature; must carry a depth
  ///                format (the pipeline is depth-tested).
  /// @pre @p device is non-`VK_NULL_HANDLE`; @p layout carries a depth format
  ///      and at least one color attachment with a defined format.
  /// @return The pipeline on success, or a non-OK @ref Status (e.g. @ref
  ///         Status::Code::InvalidArgument when @p layout has no depth format,
  ///         or a Vulkan-domain Status from shader-module / pipeline creation).
  static Result<HybridMeshPipeline> create(VkDevice device,
                                           const RenderTargetLayout& layout);

  ~HybridMeshPipeline() = default;
  HybridMeshPipeline(HybridMeshPipeline&&) noexcept = default;
  HybridMeshPipeline& operator=(HybridMeshPipeline&&) noexcept = default;
  HybridMeshPipeline(const HybridMeshPipeline&) = delete;
  HybridMeshPipeline& operator=(const HybridMeshPipeline&) = delete;

  /// @return The underlying `VkPipeline` (`VK_NULL_HANDLE` when empty).
  VkPipeline handle() const noexcept { return pipeline_.handle(); }

  /// @return The `VkPipelineLayout` (for push constants + descriptor binding).
  VkPipelineLayout layout() const noexcept { return pipeline_.layout(); }

  /// @return `true` if this owns a pipeline.
  bool valid() const noexcept { return pipeline_.valid(); }

  /// @return The number of descriptor sets the embedded shaders declare
  ///         (reflected -- set 0 atlas, so 1).
  uint32_t descriptor_set_count() const noexcept {
    return pipeline_.descriptor_set_count();
  }

  /// @brief The reflected layout for descriptor @p set (0 = the atlas sampler)
  ///        -- pass to @ref DescriptorPool::allocate.
  /// @param set  The set index.
  /// @return The set's `VkDescriptorSetLayout`, or `VK_NULL_HANDLE` if @p set
  /// is
  ///         beyond what the shaders declare.
  VkDescriptorSetLayout descriptor_set_layout(uint32_t set) const noexcept {
    return pipeline_.descriptor_set_layout(set);
  }

  /// @brief Record one frame's draws into @p cmd: bind the pipeline + a
  ///        full-target viewport, bind the atlas set once (set 0), push the
  ///        camera + lighting constants, then draw each mesh.
  /// @param cmd    A recording-state command buffer, inside a dynamic-rendering
  ///               scope whose target matches the layout @ref create was given.
  /// @param frame  The atlas set, the draw list, the view-projection, and the
  ///               lighting (see @ref HybridMeshFrame).
  /// @pre `valid()`; `frame.atlas` is a non-null set built against
  ///      @ref descriptor_set_layout `(0)` -- the shader samples it
  ///      unconditionally. A `VK_NULL_HANDLE` atlas records **nothing** (the
  ///      whole frame is dropped rather than draw against an unbound set).
  ///      Draws whose geometry is empty -- a null static @ref GpuMesh, or a
  ///      live @ref LiveMesh with any of its three handles unbound -- are
  ///      skipped. A *bound* @ref LiveMesh is always drawn: its index count
  ///      lives in the producer's indirect command, which this pipeline never
  ///      reads (see @ref LiveMesh on saying "nothing this frame").
  void submit(VkCommandBuffer cmd, const HybridMeshFrame& frame) const;

 private:
  GraphicsPipeline pipeline_;
};

/// @brief One thing to draw: either a static @ref GpuMesh or a live @ref
///        LiveMesh (world-space interleaved vertices, either way).
///
/// The geometry is a `std::variant`, so exactly one source is named -- a static
/// @ref GpuMesh (borrowed by pointer; owns device-local buffers with a fixed
/// index count) or a @ref LiveMesh (a value that borrows the producer's buffers
/// and draws them indirectly with a GPU-driven count -- the recon handoff).
/// The draw *list* is borrowed for the duration of the @ref
/// HybridMeshPipeline::submit call, but the geometry it names must outlive the
/// recorded **frame**: `submit` only records, and the GPU reads the buffers
/// when that frame executes (see @ref LiveMesh's lifetime `@warning` -- for a
/// live mesh this is what keeps a producer from recycling a slot that is still
/// in flight). A default-constructed draw holds a null @ref GpuMesh pointer and
/// records nothing. There is no per-draw transform -- reconstruction geometry
/// is already in world space.
struct HybridMeshDraw {
  /// The geometry source. Aggregate-initializes from either alternative:
  /// `HybridMeshDraw{&gpu_mesh}` or `HybridMeshDraw{live_mesh}`.
  std::variant<const GpuMesh*, LiveMesh> geometry;
};

/// @brief Everything @ref HybridMeshPipeline::submit records for one frame: the
///        target extent, the camera's view-projection, the light, the atlas
///        set, and the draws.
///
/// The arrays/pointers are borrowed for the duration of the @ref
/// HybridMeshPipeline::submit call.
struct HybridMeshFrame {
  VkExtent2D extent{};        ///< Target size (dynamic viewport).
  glm::mat4 view_proj{1.0f};  ///< projection * view (world-space).
  /// World-space direction **to** the light (need not be unit -- @ref submit
  /// normalizes it once per frame); the diffuse term is
  /// `max(dot(normal, light_dir), 0)`.
  glm::vec3 light_dir{0.5f, 0.8f, 0.6f};
  uint32_t flags = kHybridMeshLit;  ///< @ref HybridMeshFlags bitmask.
  /// Set 0: the atlas combined-image-sampler. Required -- a `VK_NULL_HANDLE`
  /// records nothing (see @ref submit); a vertex-color-only mesh binds a 1x1
  /// image.
  VkDescriptorSet atlas = VK_NULL_HANDLE;
  const HybridMeshDraw* draws = nullptr;  ///< The draw list.
  uint32_t draw_count = 0;                ///< Number of @ref draws.
};

}  // namespace volumetric_kit::gfx::pipelines

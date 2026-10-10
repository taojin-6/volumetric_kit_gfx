// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file hybrid_mesh_pipeline.hpp
/// @brief The reconstruction "hybrid mesh" graphics pipeline: a world-space
///        vertex mesh shaded from a projective-texturing atlas where available
///        and from per-vertex color otherwise, shaders embedded.

#include <cstdint>
#include <optional>
#include <variant>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/live_mesh.hpp"

namespace volumetric_kit::core {
class Allocator;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

class GpuMesh;
class StreamedAtlas;
struct HybridMeshFrame;

/// @brief Shading flags for @ref HybridMeshFrame::flags.
///
/// Bits not named here are reserved: leave them clear. The shader tests only
/// the named bits, so a reserved bit has no effect today, but a later flag may
/// claim it.
enum HybridMeshFlags : uint32_t {
  /// Apply single-directional-light diffuse + a constant ambient term. Clear
  /// for flat, unlit albedo (the raw projected/vertex color).
  kHybridMeshLit = 1u << 0,
  /// Debug view: color each fragment by its world-space normal,
  /// `normalize(n) * 0.5 + 0.5` (+X red, +Y green, +Z blue), in place of the
  /// albedo. Takes precedence over @ref kHybridMeshLit, which it ignores. The
  /// normal is the mesh's own -- *not* flipped on back faces the way lit
  /// shading flips it -- so a region whose normals point the wrong way shows
  /// as a color jump. Winding does not reach this view: a region wound the
  /// wrong way but carrying correct normals looks like its neighbours. Lit
  /// shading, which flips the normal on back faces, is what exposes that --
  /// and it hides the case where normal and winding are both inverted, which
  /// this view shows. A zero normal shows mid-grey. The stored color is the
  /// encoding itself (to 8-bit rounding: 0.5 lands on 127 or 128) on UNORM and
  /// sRGB targets alike -- for an sRGB target the shader writes the linear
  /// value the hardware encodes back to it -- so it matches other normal-map
  /// tools on the default swapchain too.
  kHybridMeshNormals = 1u << 1,
  /// Shade every fragment from its interpolated vertex `color`, as if every
  /// `uv0` were negative: the atlas is not shown. Ignored by
  /// @ref kHybridMeshNormals. @ref HybridMeshPipeline::submit sets it for a
  /// frame with no atlas.
  kHybridMeshVertexColor = 1u << 2,
  // TODO: an on-screen toggle for kHybridMeshNormals once an example draws
  // with HybridMeshPipeline.
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
/// @ref kHybridMeshLit; @ref kHybridMeshNormals swaps the albedo for a
/// normal-as-color debug view.
///
/// Wraps a @ref GraphicsPipeline built from SPIR-V compiled into the library,
/// so a consumer gets the technique from @ref create alone. The reflected
/// layout exposes **one** descriptor set -- **set 0, binding 0**: the atlas as
/// a combined image sampler. A @ref StreamedAtlas made for the pipeline streams
/// one; a consumer may also bind a set of its own. The fragment shader samples
/// the set unconditionally, so the pipeline owns a fallback -- a 1x1 image and
/// its set -- which @ref submit binds for a frame with no atlas, drawing every
/// triangle in its vertex color. Per-frame constants (the view-projection,
/// light direction, and flags) ride a push constant, so there is no per-frame
/// UBO to manage. Record a frame's draws with @ref submit. A
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
/// @warning The @p device passed to @ref create must outlive the pipeline, and
///          the pipeline every @ref StreamedAtlas made for it.
///
/// @code
/// VKC_ASSIGN(pipelines::HybridMeshPipeline pipe,
///            pipelines::HybridMeshPipeline::create(device, allocator,
///                                                  target.layout()));
/// // each frame, inside the rendering scope:
/// pipelines::HybridMeshFrame frame;
/// frame.extent = extent;
/// frame.view_proj = camera.view_proj();
/// frame.atlas = atlas.use(f.number);  // a StreamedAtlas, or VK_NULL_HANDLE
/// frame.draws = &draw;
/// frame.draw_count = 1;
/// pipe.submit(f.cmd, frame);
/// @endcode
class VG_PIPELINES_API HybridMeshPipeline {
 public:
  /// @brief Construct an empty pipeline (owns nothing; `valid()` is false).
  HybridMeshPipeline() = default;

  /// @brief Build the hybrid-mesh pipeline for a render-target layout, and its
  ///        fallback atlas.
  ///
  /// Uploads the fallback's one texel with @ref upload_texture, which blocks
  /// on the device's queue: a setup-time call.
  /// @param device     The device; it must have enabled the renderer's
  ///                   requirements (@ref device_requirements).
  /// @param allocator  Allocates the fallback image; it may be destroyed
  ///                   before the pipeline.
  /// @param layout     The target's format/sample signature; must carry a
  ///                   depth format (the pipeline is depth-tested).
  /// @pre @p layout carries at least one color attachment with a defined
  ///      format.
  /// @return The pipeline on success, or a non-OK `core::Status`:
  ///         `core::Status::Code::InvalidArgument` when @p layout has no depth
  ///         format; `core::Status::Code::Unsupported` for a device without
  ///         the renderer's requirements; or a backend Status from
  ///         shader-module, pipeline, sampler or descriptor creation or the
  ///         fallback's upload.
  static core::Result<HybridMeshPipeline> create(
      const core::Device& device, core::Allocator& allocator,
      const RenderTargetLayout& layout);

  ~HybridMeshPipeline() = default;
  HybridMeshPipeline(HybridMeshPipeline&& other) noexcept;
  HybridMeshPipeline& operator=(HybridMeshPipeline&& other) noexcept;
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
  ///        -- pass to `core::DescriptorPool::allocate`.
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
  /// @pre `valid()`; an empty pipeline records nothing. `frame.atlas` is a set
  ///      built against @ref descriptor_set_layout `(0)`, or `VK_NULL_HANDLE`:
  ///      the pipeline then binds its fallback and sets
  ///      @ref kHybridMeshVertexColor, so every triangle draws in its vertex
  ///      color. Draws whose geometry is empty -- a null static @ref GpuMesh,
  ///      or a live @ref LiveMesh with any of its three handles unbound -- are
  ///      skipped. A *bound* @ref LiveMesh is always drawn: its index count
  ///      lives in the producer's indirect command, which this pipeline never
  ///      reads (see @ref LiveMesh on saying "nothing this frame").
  void submit(VkCommandBuffer cmd, const HybridMeshFrame& frame) const;

 private:
  friend class StreamedAtlas;  // reads device_ and sampler_

  const core::Device* device_ = nullptr;  // borrowed; outlives this
  GraphicsPipeline pipeline_;
  // The atlas sampler: bilinear, clamped to the edge, one level. Shared by
  // the fallback and every StreamedAtlas made for the pipeline.
  std::optional<Sampler> sampler_;
  // The fallback: one white texel, its pool, and the set binding it.
  core::Image fallback_;
  core::DescriptorPool pool_;
  core::DescriptorSet fallback_set_;
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
  /// Set 0: the atlas combined-image-sampler -- @ref StreamedAtlas::use, or a
  /// set of the caller's. `VK_NULL_HANDLE` draws every triangle in its vertex
  /// color (see @ref HybridMeshPipeline::submit).
  VkDescriptorSet atlas = VK_NULL_HANDLE;
  const HybridMeshDraw* draws = nullptr;  ///< The draw list.
  uint32_t draw_count = 0;                ///< Number of @ref draws.
};

}  // namespace volumetric_kit::gfx::pipelines

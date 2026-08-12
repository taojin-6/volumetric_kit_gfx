// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pipelines/patch_mesh_pipeline.hpp
/// @brief Draws a reconstruction mesh shaded from a per-TRIANGLE patch atlas,
///        addressed per primitive rather than through a per-vertex uv.

#include <cstdint>
#include <variant>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/live_mesh.hpp"

namespace volumetric_kit::gfx::pipelines {

class GpuMesh;
struct PatchMeshFrame;

/// @brief Shading flags for @ref PatchMeshFrame::flags.
enum PatchMeshFlags : std::uint32_t {
  /// Apply the directional + ambient term. Clear it to pass albedo through.
  kPatchMeshLit = 1u << 0,
};

/// @brief Draws world-space reconstruction geometry whose colour lives in a
///        progressive per-triangle **patch atlas** rather than in a texture
///        the mesh has uv coordinates into.
///
/// A producer that accumulates colour into one small patch per triangle -- so
/// surface colour resolves at the camera's resolution rather than the voxel's,
/// with no UV unwrapping -- cannot express that atlas through a per-vertex
/// coordinate. A patch belongs to a *triangle*, so its three corners need three
/// distinct coordinates, and a vertex shared between up to six triangles has
/// only one slot to put them in. @ref HybridMeshPipeline, whose `uv0` is per
/// vertex, therefore cannot draw one; this pipeline exists for exactly that
/// gap and changes nothing about that one.
///
/// **The atlas offset is `gl_PrimitiveID`.** Under an indexed indirect draw
/// with `firstIndex == 0` that is precisely the producer's arena triangle slot,
/// so no side table maps one to the other. The position *within* a patch comes
/// from the fragment's barycentric coordinate, which the fragment stage
/// recovers from the interpolated world position and the triangle's three
/// corners -- fetched from the same index and vertex buffers the draw is
/// already reading. `VK_KHR_fragment_shader_barycentric` would supply those
/// directly, and is supported on the Apple targets, but it is a **device
/// feature**: taking it would thread a requirement through every embedder's
/// device creation, including a neutral bootstrap merging two libraries'
/// needs, to save a 2x2 solve. Recovering them needs nothing.
///
/// **Set 0 is three storage buffers, not a sampler**: the patch atlas, the
/// index run, and the vertices. All three are the producer's own buffers, so a
/// consumer at the zero-copy seam writes the set once per slot and rebinds
/// nothing per draw. They are declared as flat scalar arrays in the shader
/// rather than as struct mirrors, so no `scalarBlockLayout` is required either.
///
/// **A texel no frame observed falls back to the per-vertex colour**, decided
/// per *fragment* on the stored weight rather than per triangle on a sentinel.
/// So a partially observed surface fades between the two along real patch
/// boundaries instead of switching whole triangles, and a producer needs no
/// convention for "untextured" beyond leaving the patch at zero.
///
/// **The atlas is decoded in the shader.** It holds canonical encoded 8-bit
/// colour and this pipeline shades in linear, and a storage buffer has no
/// `_SRGB` format to do that in hardware -- so the fragment stage applies the
/// exact piecewise curve itself. That is the one thing a buffer atlas gives up
/// against an image, along with filtering and mips.
///
/// Usage mirrors @ref HybridMeshPipeline:
///
/// ```
/// Result<pipelines::PatchMeshPipeline> pipe =
///     pipelines::PatchMeshPipeline::create(device, target_layout);
/// // Allocate a set against descriptor_set_layout(0) and write the producer's
/// // patch / index / vertex buffers into bindings 0, 1, 2.
/// pipe.value().submit(cmd, frame);
/// ```
class VG_PIPELINES_API PatchMeshPipeline {
 public:
  /// @brief Construct an empty pipeline (owns nothing; `valid()` is false).
  PatchMeshPipeline() = default;

  /// @brief Build the pipeline for a render target's attachment layout.
  /// @param device  The logical device (must outlive the pipeline).
  /// @param layout  The target's colour/depth attachment layout.
  /// @return The pipeline, or a non-OK @ref Status from shader-module or
  ///         pipeline creation.
  static Result<PatchMeshPipeline> create(VkDevice device,
                                          const RenderTargetLayout& layout);

  /// @return `true` if this owns a live pipeline.
  bool valid() const noexcept { return pipeline_.valid(); }

  /// @return Descriptor sets the layout declares (reflected -- set 0, so 1).
  std::uint32_t descriptor_set_count() const noexcept {
    return pipeline_.descriptor_set_count();
  }

  /// @brief The reflected layout for descriptor @p set (0 = the atlas set).
  /// @return The set's `VkDescriptorSetLayout`, or `VK_NULL_HANDLE` if @p set
  ///         is out of range or this pipeline is empty.
  VkDescriptorSetLayout descriptor_set_layout(
      std::uint32_t set) const noexcept {
    return pipeline_.descriptor_set_layout(set);
  }

  /// @brief Record the frame: bind the pipeline, set a full-target viewport,
  ///        bind the atlas set once, push the camera + light + patch shape,
  ///        then record each draw.
  ///
  /// @pre `valid()`; `frame.atlas` is a non-null set built against
  ///      @ref descriptor_set_layout(0), and `frame.patch_leg` is at least 2 --
  ///      the fragment stage divides by `patch_leg - 1`. A frame failing any of
  ///      these records **nothing**, rather than drawing against an unbound set
  ///      or dividing by zero.
  void submit(VkCommandBuffer cmd, const PatchMeshFrame& frame) const;

 private:
  GraphicsPipeline pipeline_;
};

/// @brief One thing to draw, exactly as @ref HybridMeshDraw: a static
///        @ref GpuMesh or a @ref LiveMesh, world-space either way.
///
/// The draw *list* is borrowed for the duration of @ref
/// PatchMeshPipeline::submit, but the geometry it names must outlive the
/// recorded **frame** -- `submit` only records, and the GPU reads the buffers
/// when that frame executes.
struct PatchMeshDraw {
  std::variant<const GpuMesh*, LiveMesh> geometry;
};

/// @brief Everything @ref PatchMeshPipeline::submit records for one frame.
///
/// The arrays/pointers are borrowed for the duration of the call.
struct PatchMeshFrame {
  VkExtent2D extent{};        ///< Target size (dynamic viewport).
  glm::mat4 view_proj{1.0f};  ///< projection * view (world-space geometry).
  /// World-space direction **to** the light (need not be unit -- @ref
  /// PatchMeshPipeline::submit normalizes it once per frame).
  glm::vec3 light_dir{0.5f, 0.8f, 0.6f};
  std::uint32_t flags = kPatchMeshLit;  ///< @ref PatchMeshFlags bitmask.
  /// Set 0: the producer's patch atlas (binding 0), index run (1) and vertices
  /// (2). Required -- a `VK_NULL_HANDLE` records nothing.
  VkDescriptorSet atlas = VK_NULL_HANDLE;
  /// @brief Texels along each leg of a patch.
  ///
  /// The producer's `patch_leg`, and it must be the value the atlas was
  /// accumulated with: it is what turns a barycentric coordinate into a texel
  /// and what strides between rows of one patch, so a mismatch does not fail --
  /// it samples a different patch's texels and renders plausible nonsense.
  /// Must be at least 2.
  std::uint32_t patch_leg = 0;
  /// @brief Texels one patch holds -- the producer's `texels_per_patch`, which
  ///        is `patch_leg * (patch_leg + 1) / 2` for a right-triangle patch.
  ///
  /// Passed rather than derived from @ref patch_leg so the producer's packing
  /// stays the single source of truth: a producer that changed its patch shape
  /// would otherwise be silently disagreed with by this arithmetic.
  std::uint32_t texels_per_patch = 0;
  const PatchMeshDraw* draws = nullptr;  ///< The draw list.
  std::uint32_t draw_count = 0;          ///< Number of @ref draws.
};

}  // namespace volumetric_kit::gfx::pipelines

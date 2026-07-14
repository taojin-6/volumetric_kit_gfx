// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file live_mesh.hpp
/// @brief A borrowed, per-frame-variable triangle mesh drawn indirectly -- the
///        `volumetric_kit_recon` live zero-copy handoff.

#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::gfx::pipelines {

/// @brief A live triangle mesh drawn with `vkCmdDrawIndexedIndirect`: the
///        producer's vertex, index, and one-element indirect-command buffers
///        are all **borrowed**, and the index count is read GPU-side from the
///        command -- never known to the CPU.
///
/// This is the moving half of the `volumetric_kit_recon` handoff. Where @ref
/// GpuMesh owns device-local buffers with an index count fixed at upload, a
/// `LiveMesh` owns **nothing**: it names three buffers the producer writes each
/// frame (e.g. a reconstruction compute pass emitting marching-cubes geometry)
/// and records an *indirect* indexed draw against them. The draw count lives in
/// the buffer, so a mesh that grew or shrank this frame draws correctly with no
/// CPU round trip. The buffers may live on a @ref Device "device" the renderer
/// **adopted** from the producer, making the handoff zero-copy.
///
/// Because it borrows, a `LiveMesh` is a plain value (copy it freely) -- there
/// is no ownership to move. A default-constructed one is empty (`valid()` is
/// false) and records nothing.
///
/// **Buffer contract (what the producer must provide):**
/// - @ref vertices -- interleaved @ref assets::Vertex, created with
///   `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`. Only position/normal/uv0/color are
///   read (the hybrid technique does not bind tangent).
/// - @ref indices -- **32-bit** indices (`VK_INDEX_TYPE_UINT32`), created with
///   `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`.
/// - @ref indirect -- one `VkDrawIndexedIndirectCommand` at @ref
///   indirect_offset, created with `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT`. Fill
///   it with `instanceCount = 1` and `firstInstance = 0` (a non-zero
///   `firstInstance` needs the `drawIndirectFirstInstance` device feature);
///   `indexCount`, `firstIndex`, and `vertexOffset` are the producer's to set.
///
/// A single indirect draw (`drawCount = 1`) is core Vulkan 1.0 -- it adds
/// **no** device-feature requirement to a shared/adopted device.
///
/// **Sub-allocation.** A producer packing many meshes into shared pools can
/// offset either way: the byte-granular bind offsets on this struct (@ref
/// vertex_offset and friends), or the element-granular `firstIndex` /
/// `vertexOffset` inside the command. The two compose **additively** on the
/// same buffer -- the bind offset is added to `firstIndex * 4` (indices) and to
/// `vertexOffset * stride` (vertices) -- so set at most one per axis; using
/// both double-counts and fetches past the mesh.
///
/// **Offset alignment.** All three byte offsets must be 4-byte aligned: @ref
/// index_offset and @ref indirect_offset per core Vulkan (the
/// `VK_INDEX_TYPE_UINT32` index size, and the indirect-command offset rule),
/// and @ref vertex_offset per MoltenVK/Metal (core Vulkan is laxer). @ref
/// record_draw does not check this -- an unaligned offset is a validation
/// error, and a Metal fault on Apple.
///
/// @warning **Synchronization is the caller's.** @ref record_draw records only
///          the binds + the indirect draw; it inserts no barrier. The
///          producer's writes to all three buffers must already be visible to
///          `VK_PIPELINE_STAGE_VERTEX_INPUT_BIT` (vertex + index reads) and
///          `VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT` (the command read) before the
///          draw executes -- via a `vkCmdPipelineBarrier` when producer and
///          renderer share a queue, or a semaphore across queues. The
///          `LiveMesh`'s buffers must also outlive the frame that draws them
///          (release them only once its fence signals).
///
/// @code
/// // recon writes vertices, 32-bit indices, and a VkDrawIndexedIndirectCommand
/// // into device buffers it owns, then hands the handles over -- no copy:
/// pipelines::LiveMesh live;
/// live.vertices = recon_vertices;
/// live.indices = recon_indices;
/// live.indirect = recon_indirect;
/// // ... after a barrier/semaphore makes those writes visible to the draw ...
/// const pipelines::HybridMeshDraw draw{live};
/// frame.draws = &draw;
/// frame.draw_count = 1;
/// pipeline.submit(cmd, frame);  // records vkCmdDrawIndexedIndirect
/// @endcode
struct VG_PIPELINES_API LiveMesh {
  /// Interleaved @ref assets::Vertex buffer (`VERTEX_BUFFER` usage).
  VkBuffer vertices = VK_NULL_HANDLE;
  /// Byte offset of the first vertex within @ref vertices (4-byte aligned on
  /// MoltenVK; see "Offset alignment").
  VkDeviceSize vertex_offset = 0;
  /// 32-bit index buffer (`INDEX_BUFFER` usage).
  VkBuffer indices = VK_NULL_HANDLE;
  /// Byte offset of the first index within @ref indices (a multiple of 4).
  VkDeviceSize index_offset = 0;
  /// One `VkDrawIndexedIndirectCommand` (`INDIRECT_BUFFER` usage).
  VkBuffer indirect = VK_NULL_HANDLE;
  /// Byte offset of the command within @ref indirect (a multiple of 4).
  VkDeviceSize indirect_offset = 0;

  /// @return `true` if all three buffers are bound (drawable).
  bool valid() const noexcept {
    return vertices != VK_NULL_HANDLE && indices != VK_NULL_HANDLE &&
           indirect != VK_NULL_HANDLE;
  }

  /// @brief Bind the vertex buffer (binding 0) + the 32-bit index buffer and
  ///        record `vkCmdDrawIndexedIndirect` (one command, GPU-driven count).
  /// @param cmd  A recording-state command buffer, inside a render scope.
  /// @pre `valid()`; the caller has bound a compatible pipeline (its vertex
  ///      input must match the @ref assets::Vertex layout) plus any descriptor
  ///      sets / push constants the draw needs, and has made the producer's
  ///      buffer writes visible to the vertex-input + draw-indirect stages (see
  ///      the class `@warning`).
  void record_draw(VkCommandBuffer cmd) const;
};

}  // namespace volumetric_kit::gfx::pipelines

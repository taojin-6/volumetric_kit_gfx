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
/// - @ref vertices -- interleaved @ref assets::Vertex, laid out byte-for-byte
///   as that struct declares it (the pipeline derives the stride and every
///   attribute offset from it, so a repacked or reordered vertex is read at the
///   wrong offsets with no size mismatch to catch it). Only position, normal,
///   uv0 and color are read -- the hybrid technique does not bind tangent, but
///   the field still occupies its slot. `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`.
/// - @ref indices -- **32-bit** indices (`VK_INDEX_TYPE_UINT32`), created with
///   `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`.
/// - @ref indirect -- one `VkDrawIndexedIndirectCommand` at @ref
///   indirect_offset, created with `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT`. Fill
///   it with `instanceCount = 1` and `firstInstance = 0` (a non-zero
///   `firstInstance` needs the `drawIndirectFirstInstance` device feature);
///   `indexCount`, `firstIndex`, and `vertexOffset` are the producer's to set.
///
/// Those are the usages this **draw** needs. Usage flags are a union, not a
/// choice: a producer that writes the buffers from its own compute pass must
/// additionally create them with what that write path requires (typically
/// `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`, or
/// `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` for a buffer-device-address
/// write) -- a buffer created with the draw flags alone cannot be bound to the
/// producer's own descriptors.
///
/// A single indirect draw (`drawCount = 1`) is core Vulkan 1.0 -- it adds
/// **no** device-feature requirement to a shared/adopted device.
///
/// **Albedo class is per triangle.** The technique picks the atlas texel or the
/// per-vertex color from `uv0`, but resolves the choice per *vertex* and
/// forwards it `flat`: a triangle is shaded entirely under its **provoking
/// vertex's** class. The test is `uv0.x < 0` -- recon's `(-1, -1)` sentinel is
/// one such value, but so is any other negative x, including an atlas
/// coordinate that lands marginally outside the map and was meant to clamp. So
/// the producer must keep the two classes triangle-aligned (a triangle
/// straddling a coverage boundary samples a meaningless uv for its whole area)
/// and must never emit a negative atlas coordinate it expects the sampler's
/// clamp to absorb.
///
/// **Sub-allocation.** A producer packing many meshes into shared pools can
/// offset either way: the byte-granular bind offsets on this struct (@ref
/// vertex_offset and friends), or the element-granular `firstIndex` /
/// `vertexOffset` inside the command. The two compose **additively** on the
/// same buffer -- the bind offset is added to `firstIndex * 4` (indices) and to
/// `vertexOffset * stride` (vertices) -- so set at most one per axis; using
/// both double-counts and fetches past the mesh.
///
/// **Offset rules.** All three byte offsets must be 4-byte aligned: @ref
/// index_offset and @ref indirect_offset per core Vulkan (the
/// `VK_INDEX_TYPE_UINT32` index size, and the indirect-command offset rule),
/// and @ref vertex_offset per MoltenVK/Metal (core Vulkan is laxer). Each must
/// also leave room for what the draw reads: @ref indirect_offset `+
/// sizeof(VkDrawIndexedIndirectCommand)` must not exceed the indirect buffer's
/// size (the last command in a ring is at `size - sizeof(command)`, not `size -
/// 16`), and the vertex/index offsets must fall strictly inside their buffers.
/// @ref record_draw checks none of this and @ref valid() cannot -- a `LiveMesh`
/// borrows handles and never learns a size -- so a violation is a validation
/// error, and a Metal fault on Apple.
///
/// **Saying "nothing this frame".** @ref valid() gates on the three handles
/// alone, so a bound `LiveMesh` always draws whatever its command currently
/// holds. A stale or never-written `indexCount` fetches indices outside the
/// producer's arena, which `robustBufferAccess` does **not** cover (index
/// fetches are unchecked: a device loss, or a Metal fault on Apple). A producer
/// whose slot carries no geometry this frame must therefore either pass a
/// default-constructed `LiveMesh` (unbound -- skipped) or write `indexCount =
/// 0`, rather than hand over a slot whose command has not been written yet.
///
/// @warning **Synchronization is the caller's.** @ref record_draw records only
///          the binds + the indirect draw; it inserts no barrier. Before the
///          draw executes, the producer's writes to all three buffers must be
///          visible to `VK_PIPELINE_STAGE_VERTEX_INPUT_BIT` (vertex + index
///          reads) and `VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT` (the command
///          read). On a shared queue family that is a `vkCmdPipelineBarrier`.
///          Across queue *families* a semaphore is **not** sufficient on its
///          own: it carries execution and memory dependencies but not queue
///          ownership, so a `VK_SHARING_MODE_EXCLUSIVE` buffer additionally
///          needs a release barrier on the producer's family and a matching
///          acquire on the renderer's -- or the buffers must be created
///          `VK_SHARING_MODE_CONCURRENT` over both families. This is the
///          ordinary case on Apple, where a bootstrap typically hands the
///          producer and the renderer queues from different families.
/// @warning **The buffers must outlive the frame, not the record call.** @ref
///          record_draw only *records*; the GPU reads all three buffers when
///          that frame executes, long after @ref
///          HybridMeshPipeline::submit returns. Release or recycle a slot only
///          once the frame's fence has signalled -- doing it on return from
///          `submit` recycles geometry that is still in flight.
///
/// @code
/// // recon writes vertices, 32-bit indices, and a VkDrawIndexedIndirectCommand
/// // into device buffers it owns, then hands the handles over -- no copy:
/// pipelines::LiveMesh live;
/// live.vertices = recon_vertices;
/// live.indices = recon_indices;
/// live.indirect = recon_indirect;
///
/// const pipelines::HybridMeshDraw draw{live};
/// pipelines::HybridMeshFrame frame;
/// frame.extent = target_extent;
/// frame.view_proj = camera_view_proj;
/// frame.atlas = atlas_set;  // required: a null set records NOTHING
/// frame.draws = &draw;
/// frame.draw_count = 1;
/// // ... after a barrier/semaphore makes recon's writes visible to the draw
/// ... pipeline.submit(cmd, frame);  // records vkCmdDrawIndexedIndirect
/// // ... and recon keeps the slot until this frame's fence signals.
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

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file gpu_mesh.hpp
/// @brief A triangle mesh's vertex + index buffers on the GPU, ready to draw.

#include <cstdint>

#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::gfx {
class Allocator;
class Device;
class UploadBatch;
namespace assets {
struct Mesh;
}  // namespace assets
}  // namespace volumetric_kit::gfx

namespace volumetric_kit::gfx::pipelines {

/// @brief Owns a triangle mesh's device-local interleaved vertex buffer and
///        32-bit index buffer, and records an indexed draw of them.
///
/// Produced by @ref upload_mesh from an @ref assets::Mesh. A
/// default-constructed `GpuMesh` is empty (`valid()` is false) and safe to
/// move-assign into. The buffers' producing @ref Allocator must outlive the
/// mesh (see @ref Buffer).
///
/// @code
/// // Many meshes: record every upload into one batch, one submit total.
/// Result<UploadBatch> batch = UploadBatch::begin(device, allocator);
/// if (!batch) return batch.status();
/// std::vector<pipelines::GpuMesh> meshes;
/// for (const assets::Mesh& m : model.meshes) {
///   Result<pipelines::GpuMesh> mesh =
///       pipelines::upload_mesh(batch.value(), m);
///   if (!mesh) return mesh.status();
///   meshes.push_back(std::move(mesh).value());
/// }
/// VG_TRY(batch.value().finish());  // meshes now draw-ready
/// // ... bind a pipeline + descriptor sets, then per mesh:
/// meshes[0].record_draw(cmd);
/// @endcode
class VG_PIPELINES_API GpuMesh {
 public:
  /// @brief Construct an empty mesh (owns nothing; `valid()` is false).
  GpuMesh() noexcept = default;

  /// @brief Adopt the @p vertices + @p indices buffers and the index count.
  ///        Produced by @ref upload_mesh; rarely constructed directly.
  /// @param vertices     Interleaved vertex buffer (`VERTEX_BUFFER` usage).
  /// @param indices      32-bit index buffer (`INDEX_BUFFER` usage).
  /// @param index_count  Number of indices to draw.
  GpuMesh(Buffer vertices, Buffer indices, uint32_t index_count) noexcept;

  ~GpuMesh() = default;
  GpuMesh(GpuMesh&& other) noexcept;
  GpuMesh& operator=(GpuMesh&& other) noexcept;
  GpuMesh(const GpuMesh&) = delete;
  GpuMesh& operator=(const GpuMesh&) = delete;

  /// @return The number of indices drawn (`0` when empty).
  uint32_t index_count() const noexcept { return index_count_; }

  /// @return `true` if this owns drawable vertex + index buffers.
  bool valid() const noexcept {
    return vertices_.valid() && indices_.valid() && index_count_ > 0;
  }

  /// @brief Bind the vertex buffer (binding 0) + index buffer and record an
  ///        indexed draw.
  /// @param cmd  A recording-state command buffer, inside a render scope.
  /// @pre `valid()`; the caller has already bound a compatible pipeline (its
  ///      vertex input must match the mesh's @ref assets::Vertex layout) plus
  ///      any descriptor sets / push constants the draw needs.
  void record_draw(VkCommandBuffer cmd) const;

 private:
  Buffer vertices_;
  Buffer indices_;
  uint32_t index_count_ = 0;
};

/// @brief Record @p mesh's vertex + index uploads into @p batch, returning
///        the device-local GPU mesh they fill.
///
/// Two @ref UploadBatch::add_buffer calls (`VERTEX_BUFFER`, then
/// `INDEX_BUFFER`), so the mesh shares the caller's one submit with everything
/// else the batch carries.
/// @param batch  An open batch; the mesh is drawable only after the caller's
///               @ref UploadBatch::finish returns OK.
/// @param mesh   The CPU mesh; its `vertices` and `indices` must be non-empty.
/// @return The GPU mesh on success, or a non-OK @ref Status: @ref
///         Status::Code::InvalidArgument when @p mesh has no vertices or
///         indices (checked before anything records, leaving @p batch
///         unchanged) or when @p batch is empty; otherwise a Vulkan-domain
///         Status from buffer allocation.
/// @warning Keep the returned mesh alive at least until the batch's finish
///          returns (see @ref UploadBatch::add_buffer). A Vulkan-domain failure
///          here can land between the two recorded uploads; it then
///          @ref UploadBatch::poison "poisons" the batch, so a later
///          @ref UploadBatch::finish safely discards (returns @ref
///          Status::Code::InvalidArgument) instead of submitting the dropped
///          vertex buffer's now-dangling copy. Discarding the batch directly
///          (destroy without finishing) is equally fine.
VG_PIPELINES_API Result<GpuMesh> upload_mesh(UploadBatch& batch,
                                             const assets::Mesh& mesh);

/// @brief Upload @p mesh's interleaved vertices + 32-bit indices into
///        device-local vertex/index buffers in one blocking submit.
///
/// A one-mesh batch (begin + record + finish). Uploading many meshes? Share
/// one @ref UploadBatch via the other overload instead of paying a CPU-GPU
/// round trip each.
/// @param device     The device whose graphics queue runs the one-time
///                   transfer.
/// @param allocator  Allocates the buffers; must outlive the returned mesh.
/// @param mesh       The CPU mesh; its `vertices` and `indices` must be
///                   non-empty.
/// @return The GPU mesh -- draw-ready -- on success, or a non-OK @ref Status:
///         @ref Status::Code::InvalidArgument when @p mesh has no vertices or
///         indices; otherwise a Vulkan-domain Status from the buffer or
///         submit step.
VG_PIPELINES_API Result<GpuMesh> upload_mesh(const Device& device,
                                             Allocator& allocator,
                                             const assets::Mesh& mesh);

}  // namespace volumetric_kit::gfx::pipelines

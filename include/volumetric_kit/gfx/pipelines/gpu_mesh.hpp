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
namespace assets {
struct Mesh;
}  // namespace assets
}  // namespace volumetric_kit::gfx

namespace volumetric_kit::gfx::pipelines {

/// @brief Owns a triangle mesh's interleaved vertex buffer and 32-bit index
///        buffer, and records an indexed draw of them.
///
/// Produced by @ref upload_mesh from an @ref assets::Mesh. A
/// default-constructed `GpuMesh` is empty (`valid()` is false) and safe to
/// move-assign into. The buffers' producing @ref Allocator must outlive the
/// mesh (see @ref Buffer).
///
/// @code
/// Result<pipelines::GpuMesh> mesh = pipelines::upload_mesh(allocator, m);
/// if (!mesh) return mesh.status();
/// // ... bind a pipeline + descriptor sets, then:
/// mesh.value().record_draw(cmd);
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
  /// @param cmd  A command buffer in the recording state, inside a render
  /// scope.
  /// @pre `valid()`; the caller has already bound a compatible pipeline (its
  ///      vertex input must match the mesh's @ref assets::Vertex layout) plus
  ///      any descriptor sets / push constants the draw needs.
  void record_draw(VkCommandBuffer cmd) const;

 private:
  Buffer vertices_;
  Buffer indices_;
  uint32_t index_count_ = 0;
};

/// @brief Upload @p mesh's interleaved vertices + 32-bit indices into GPU
///        vertex/index buffers.
/// @param allocator  Allocates the buffers; must outlive the returned mesh.
/// @param mesh       The CPU mesh; its `vertices` and `indices` must be
///                   non-empty.
/// @return The GPU mesh on success, or a non-OK @ref Status: @ref
///         Status::Code::InvalidArgument when @p mesh has no vertices or
///         indices; otherwise a Vulkan-domain Status from buffer allocation.
VG_PIPELINES_API Result<GpuMesh> upload_mesh(Allocator& allocator,
                                             const assets::Mesh& mesh);

}  // namespace volumetric_kit::gfx::pipelines

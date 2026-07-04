// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"

#include <utility>

#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"

namespace volumetric_kit::gfx::pipelines {

GpuMesh::GpuMesh(Buffer vertices, Buffer indices, uint32_t index_count) noexcept
    : vertices_(std::move(vertices)),
      indices_(std::move(indices)),
      index_count_(index_count) {}

GpuMesh::GpuMesh(GpuMesh&& other) noexcept
    : vertices_(std::move(other.vertices_)),
      indices_(std::move(other.indices_)),
      index_count_(other.index_count_) {
  other.index_count_ = 0;
}

GpuMesh& GpuMesh::operator=(GpuMesh&& other) noexcept {
  if (this != &other) {
    vertices_ = std::move(other.vertices_);
    indices_ = std::move(other.indices_);
    index_count_ = other.index_count_;
    other.index_count_ = 0;
  }
  return *this;
}

void GpuMesh::record_draw(VkCommandBuffer cmd) const {
  const VkDeviceSize offset = 0;
  const VkBuffer vertex_buffer = vertices_.handle();
  vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
  vkCmdBindIndexBuffer(cmd, indices_.handle(), 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, index_count_, 1, 0, 0, 0);
}

Result<GpuMesh> upload_mesh(UploadBatch& batch, const assets::Mesh& mesh) {
  // Mesh-level validation before either add_buffer, so an invalid mesh leaves
  // the batch untouched (add_buffer rejects an empty batch itself).
  if (mesh.vertices.empty() || mesh.indices.empty()) {
    return Status::invalid_argument(
        "upload_mesh: mesh has no vertices or indices");
  }
  BufferUploadDesc vertex_desc;
  vertex_desc.data = mesh.vertices.data();
  vertex_desc.size = mesh.vertices.size() * sizeof(assets::Vertex);
  vertex_desc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VG_ASSIGN(Buffer vertices, batch.add_buffer(vertex_desc));
  BufferUploadDesc index_desc;
  index_desc.data = mesh.indices.data();
  index_desc.size = mesh.indices.size() * sizeof(uint32_t);
  index_desc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  VG_ASSIGN(Buffer indices, batch.add_buffer(index_desc));
  return GpuMesh(std::move(vertices), std::move(indices),
                 static_cast<uint32_t>(mesh.indices.size()));
}

Result<GpuMesh> upload_mesh(const Device& device, Allocator& allocator,
                            const assets::Mesh& mesh) {
  // The one-mesh batch: record both buffers, one submit, one fence wait. A
  // failed record leaves the batch to its destructor, which discards the
  // never-submitted command buffer.
  VG_ASSIGN(UploadBatch batch, UploadBatch::begin(device, allocator));
  VG_ASSIGN(GpuMesh gpu, upload_mesh(batch, mesh));
  VG_TRY(batch.finish());
  return gpu;
}

}  // namespace volumetric_kit::gfx::pipelines

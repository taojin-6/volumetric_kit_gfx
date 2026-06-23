// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"

#include <cstring>
#include <utility>

#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"

namespace volumetric_kit::gfx::pipelines {
namespace {

// Upload `size` bytes of `data` into a host-visible, mapped buffer of `usage`.
// TODO: stage through a DeviceLocal buffer (as upload_texture does) so the
// vertex/index data lives in GPU-preferred memory rather than host-visible.
Result<Buffer> upload_buffer(Allocator& allocator, const void* data,
                             VkDeviceSize size, VkBufferUsageFlags usage) {
  BufferDesc desc;
  desc.size = size;
  desc.usage = usage;
  desc.memory = MemoryUsage::HostVisible;
  desc.mapped = true;
  desc.host_access = HostAccess::SequentialWrite;
  VG_ASSIGN(Buffer buffer, allocator.create_buffer(desc));
  std::memcpy(buffer.mapped(), data, size);
  return buffer;
}

}  // namespace

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

Result<GpuMesh> upload_mesh(Allocator& allocator, const assets::Mesh& mesh) {
  if (mesh.vertices.empty() || mesh.indices.empty()) {
    return Status::invalid_argument(
        "upload_mesh: mesh has no vertices or indices");
  }
  VG_ASSIGN(Buffer vertices,
            upload_buffer(allocator, mesh.vertices.data(),
                          mesh.vertices.size() * sizeof(assets::Vertex),
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
  VG_ASSIGN(Buffer indices,
            upload_buffer(allocator, mesh.indices.data(),
                          mesh.indices.size() * sizeof(uint32_t),
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
  return GpuMesh(std::move(vertices), std::move(indices),
                 static_cast<uint32_t>(mesh.indices.size()));
}

}  // namespace volumetric_kit::gfx::pipelines

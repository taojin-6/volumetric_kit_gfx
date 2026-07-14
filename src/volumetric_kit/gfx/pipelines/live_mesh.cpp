// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/live_mesh.hpp"

namespace volumetric_kit::gfx::pipelines {

void LiveMesh::record_draw(VkCommandBuffer cmd) const {
  vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &vertex_offset);
  vkCmdBindIndexBuffer(cmd, indices, index_offset, VK_INDEX_TYPE_UINT32);
  // drawCount = 1: the index count rides the command, so it is read GPU-side
  // and never crosses the CPU. Vulkan ignores the stride for a single draw
  // (only drawCount > 1 constrains it); pass the command size as the
  // conventional value.
  vkCmdDrawIndexedIndirect(cmd, indirect, indirect_offset, 1,
                           sizeof(VkDrawIndexedIndirectCommand));
}

}  // namespace volumetric_kit::gfx::pipelines

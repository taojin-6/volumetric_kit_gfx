// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file buffer_barrier.hpp
/// @brief Record a buffer memory barrier — the buffer counterpart of
///        @ref cmd_image_barrier.

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Parameters for @ref cmd_buffer_barrier: one buffer memory barrier
///        over a byte range.
///
/// The defaults cover the whole buffer (`offset` 0, `VK_WHOLE_SIZE`); narrow
/// @ref offset / @ref size to order a sub-range, such as one block of a
/// shared uniform buffer. Queue-family ownership is never transferred (both
/// families are `VK_QUEUE_FAMILY_IGNORED`).
///
/// @code
/// // Make a recorded copy visible to the vertex stage's attribute fetches.
/// BufferBarrierDesc to_vertex;
/// to_vertex.buffer = vertices.handle();
/// to_vertex.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
/// to_vertex.dst_stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
/// to_vertex.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
/// to_vertex.dst_access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
/// cmd_buffer_barrier(cmd, to_vertex);
/// @endcode
struct BufferBarrierDesc {
  /// The buffer to order access to.
  VkBuffer buffer = VK_NULL_HANDLE;
  /// Pipeline stages that must complete before the barrier.
  VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  /// Pipeline stages that wait on the barrier.
  VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  /// Access made available before the barrier.
  VkAccessFlags src_access = 0;
  /// Access made visible after the barrier.
  VkAccessFlags dst_access = 0;
  /// First byte covered.
  VkDeviceSize offset = 0;
  /// Bytes covered from @ref offset (the rest of the buffer by default).
  VkDeviceSize size = VK_WHOLE_SIZE;
};

/// @brief Record the buffer memory barrier described by @p desc.
/// @param cmd   A command buffer in the recording state.
/// @param desc  The buffer, execution/memory scopes, and byte range to order.
/// @pre `desc.buffer != VK_NULL_HANDLE` (checked with `VG_CHECK`), and the
///      range lies within the buffer.
VG_CORE_API void cmd_buffer_barrier(VkCommandBuffer cmd,
                                    const BufferBarrierDesc& desc);

}  // namespace volumetric_kit::gfx

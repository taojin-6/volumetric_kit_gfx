// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/impl/image_copy_layout.hpp
/// Shared validation for uncompressed copies into images: where a region
/// lies, and its buffer source's layout. Not a public header.

#include <cstdint>
#include <limits>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"

namespace volumetric_kit::gfx::detail {

// Whether the texels [offset, offset + extent) of one axis lie in [0, size).
inline bool span_inside(std::int32_t offset, std::uint32_t extent,
                        std::uint32_t size) {
  return offset >= 0 && extent <= size &&
         static_cast<std::uint32_t>(offset) <= size - extent;
}

// Whether the texels [a_offset, a_offset + a_extent) and
// [b_offset, b_offset + b_extent) of one axis have one in common; spans that
// only touch do not. Two regions write a texel in common when they overlap on
// every axis.
inline bool spans_overlap(std::int32_t a_offset, std::uint32_t a_extent,
                          std::int32_t b_offset, std::uint32_t b_extent) {
  return std::int64_t{a_offset} + a_extent > b_offset &&
         std::int64_t{b_offset} + b_extent > a_offset;
}

// Check the copy's last texel against the source capacity by subtracting and
// dividing that capacity. Multiplying caller-provided strides first can wrap
// even when the image itself is tiny. The caller has established non-zero
// extents, layer counts and texel size, and strides at least as large as the
// copied extents.
inline bool image_copy_fits_source(VkDeviceSize size, VkDeviceSize texel,
                                   const VkBufferImageCopy& region) {
  if (region.bufferOffset > size) {
    return false;
  }
  const VkDeviceSize available = (size - region.bufferOffset) / texel;
  if (region.imageExtent.width > available) {
    return false;
  }
  const VkDeviceSize row = region.bufferRowLength != 0
                               ? region.bufferRowLength
                               : region.imageExtent.width;
  const VkDeviceSize rows = region.bufferImageHeight != 0
                                ? region.bufferImageHeight
                                : region.imageExtent.height;
  // Both factors are uint32_t, so this product fits in VkDeviceSize.
  const VkDeviceSize slices = VkDeviceSize{region.imageSubresource.layerCount} *
                              region.imageExtent.depth;
  const VkDeviceSize preceding_rows =
      (available - region.imageExtent.width) / row;
  const VkDeviceSize last_slice_rows = region.imageExtent.height - 1;
  return last_slice_rows <= preceding_rows &&
         slices - 1 <= (preceding_rows - last_slice_rows) / rows;
}

// The caller validates the destination first, establishing non-zero extents,
// layer count and texel size. Validate everything about the source before
// recording any commands or changing tracked image state.
inline core::Status check_buffer_image_copy(const std::string& which,
                                            const core::Buffer* source,
                                            const VkBufferImageCopy& region,
                                            VkDeviceSize texel) {
  if (source == nullptr || !source->valid() ||
      (source->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
    return core::Status::invalid_argument(
        which + "'s source buffer is null, empty or lacks TRANSFER_SRC usage");
  }
  if ((region.bufferRowLength != 0 &&
       region.bufferRowLength < region.imageExtent.width) ||
      (region.bufferImageHeight != 0 &&
       region.bufferImageHeight < region.imageExtent.height)) {
    return core::Status::invalid_argument(
        which + "'s row length or image height is shorter than the region");
  }
  // VUID-vkCmdCopyBufferToImage-bufferRowLength-09108 bounds the byte pitch
  // even when only one row is copied and source bounds do not constrain it.
  if (region.bufferRowLength >
      std::numeric_limits<std::int32_t>::max() / texel) {
    return core::Status::invalid_argument(
        which + "'s row pitch exceeds 2^31 - 1 bytes");
  }
  if (region.bufferOffset % texel != 0) {
    return core::Status::invalid_argument(
        which + "'s buffer offset is not a multiple of the texel size");
  }
  if (!image_copy_fits_source(source->size(), texel, region)) {
    return core::Status::invalid_argument(
        which + " reads past the end of its source buffer");
  }
  return core::Status{};
}

}  // namespace volumetric_kit::gfx::detail

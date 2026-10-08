// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/image_update.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/format.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"

namespace volumetric_kit::gfx {
namespace {

// The stages that sample an image: what SHADER_READ access, and the layout the
// update leaves, are for.
constexpr VkPipelineStageFlags kShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

// The image, the scope and the regions' placement in the image -- everything
// but the source buffer, so an upload checks them before it allocates. Returns
// the texel size.
core::Result<VkDeviceSize> check_target(const char* call, VkCommandBuffer cmd,
                                        const core::Image& image,
                                        const VkBufferImageCopy* regions,
                                        std::uint32_t region_count,
                                        const ImageUpdateScope& scope) {
  const std::string name = call;
  if (cmd == VK_NULL_HANDLE) {
    return core::Status::invalid_argument(name + ": null command buffer");
  }
  if (scope.dst_stages == 0 || (scope.dst_stages & ~kShaderStages) != 0 ||
      (scope.src_stages & ~kShaderStages) != 0) {
    return core::Status::invalid_argument(
        name +
        ": the scope's stages must be shader stages, and dst_stages non-empty");
  }
  if (!image.valid()) {
    return core::Status::invalid_argument(name + ": empty image");
  }
  constexpr VkImageUsageFlags kUsage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  if ((image.usage() & kUsage) != kUsage) {
    return core::Status::invalid_argument(
        name + ": the image lacks TRANSFER_DST or SAMPLED usage");
  }
  if (image.samples() != VK_SAMPLE_COUNT_1_BIT) {
    return core::Status::invalid_argument(name + ": the image is multisampled");
  }
  const VkDeviceSize texel = core::texel_bytes(image.format());
  if (texel == 0) {
    return core::Status::unsupported(
        name +
        ": the image's format is not an uncompressed single-plane color "
        "format");
  }
  if (regions == nullptr || region_count == 0) {
    return core::Status::invalid_argument(name + ": no regions");
  }
  for (std::uint32_t i = 0; i < region_count; ++i) {
    const VkBufferImageCopy& r = regions[i];
    const std::string which = name + ": region " + std::to_string(i);
    const VkImageSubresourceLayers& sub = r.imageSubresource;
    if (sub.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
        sub.mipLevel >= image.mip_levels() || sub.layerCount == 0 ||
        sub.baseArrayLayer >= image.array_layers() ||
        sub.layerCount > image.array_layers() - sub.baseArrayLayer) {
      return core::Status::invalid_argument(
          which + " names an aspect, level or layer the image lacks");
    }
    const std::uint32_t level_width =
        std::max<std::uint32_t>(1, image.width() >> sub.mipLevel);
    const std::uint32_t level_height =
        std::max<std::uint32_t>(1, image.height() >> sub.mipLevel);
    const std::uint32_t level_depth =
        std::max<std::uint32_t>(1, image.depth() >> sub.mipLevel);
    const VkOffset3D& o = r.imageOffset;
    const VkExtent3D& e = r.imageExtent;
    if (o.x < 0 || o.y < 0 || o.z < 0 || e.width == 0 || e.height == 0 ||
        e.depth == 0 || e.width > level_width ||
        static_cast<std::uint32_t>(o.x) > level_width - e.width ||
        e.height > level_height ||
        static_cast<std::uint32_t>(o.y) > level_height - e.height ||
        e.depth > level_depth ||
        static_cast<std::uint32_t>(o.z) > level_depth - e.depth) {
      return core::Status::invalid_argument(
          which + " is empty or outside the image level");
    }
    if ((r.bufferRowLength != 0 && r.bufferRowLength < e.width) ||
        (r.bufferImageHeight != 0 && r.bufferImageHeight < e.height)) {
      return core::Status::invalid_argument(
          which + "'s row length or image height is shorter than the region");
    }
    if (r.bufferOffset % texel != 0) {
      return core::Status::invalid_argument(
          which + "'s buffer offset is not a multiple of the texel size");
    }
  }
  return texel;
}

}  // namespace

core::Status record_image_update(VkCommandBuffer cmd,
                                 const core::Buffer& source, core::Image& image,
                                 const VkBufferImageCopy* regions,
                                 std::uint32_t region_count,
                                 const ImageUpdateScope& scope) {
  constexpr const char* kCall = "record_image_update";
  VKC_ASSIGN(const VkDeviceSize texel,
             check_target(kCall, cmd, image, regions, region_count, scope));
  if (!source.valid() ||
      (source.usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
    return core::Status::invalid_argument(
        "record_image_update: the source buffer is empty or lacks "
        "TRANSFER_SRC usage");
  }
  for (std::uint32_t i = 0; i < region_count; ++i) {
    // The last texel the copy reads: rows of `row` texels, `rows` rows to a
    // slice, a slice per layer (or per depth step of a 3D image).
    const VkBufferImageCopy& r = regions[i];
    const VkDeviceSize row =
        r.bufferRowLength != 0 ? r.bufferRowLength : r.imageExtent.width;
    const VkDeviceSize rows =
        r.bufferImageHeight != 0 ? r.bufferImageHeight : r.imageExtent.height;
    const VkDeviceSize slices =
        VkDeviceSize{r.imageSubresource.layerCount} * r.imageExtent.depth;
    const VkDeviceSize texels = (slices - 1) * rows * row +
                                (r.imageExtent.height - 1) * row +
                                r.imageExtent.width;
    if (r.bufferOffset > source.size() ||
        texels * texel > source.size() - r.bufferOffset) {
      return core::Status::invalid_argument(
          "record_image_update: region " + std::to_string(i) +
          " reads past the end of the source buffer");
    }
  }

  // The contents are discarded, so the transition starts from UNDEFINED; it
  // waits for the earlier reads the scope names, and for nothing when none can
  // still be running.
  ImageBarrierDesc to_copy;
  to_copy.image = image.handle();
  to_copy.src_stage = scope.src_stages != 0 ? scope.src_stages
                                            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  to_copy.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  to_copy.dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_copy.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_copy.new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  cmd_image_barrier(cmd, to_copy);

  vkCmdCopyBufferToImage(cmd, source.handle(), image.handle(),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region_count,
                         regions);

  ImageBarrierDesc to_read;
  to_read.image = image.handle();
  to_read.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  to_read.dst_stage = scope.dst_stages;
  to_read.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_read.dst_access = VK_ACCESS_SHADER_READ_BIT;
  to_read.old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_read.new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  cmd_image_barrier(cmd, to_read);
  image.set_layout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  return core::Status{};
}

core::Status record_image_upload(VkCommandBuffer cmd,
                                 core::Allocator& allocator,
                                 RetireQueue& retire, std::uint64_t retire_at,
                                 core::Image& image, const void* pixels,
                                 VkDeviceSize size,
                                 const ImageUpdateScope& scope) {
  constexpr const char* kCall = "record_image_upload";
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = image.extent();
  VKC_ASSIGN(const VkDeviceSize texel,
             check_target(kCall, cmd, image, &region, 1, scope));
  if (pixels == nullptr) {
    return core::Status::invalid_argument("record_image_upload: null pixels");
  }
  const VkDeviceSize level =
      VkDeviceSize{image.width()} * image.height() * image.depth() * texel;
  if (size != level) {
    return core::Status::invalid_argument(
        "record_image_upload: size must be the first level's, tightly "
        "packed");
  }

  core::BufferDesc desc;
  desc.size = size;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = core::MemoryUsage::Staging;
  desc.host_access = core::HostAccess::SequentialWrite;
  VKC_ASSIGN(core::Buffer buffer, allocator.create_buffer(desc));
  std::memcpy(buffer.mapped(), pixels, static_cast<std::size_t>(size));
  // Shared, as the RetireQueue's deleter must be copyable.
  auto staging = std::make_shared<core::Buffer>(std::move(buffer));

  VKC_TRY(record_image_update(cmd, *staging, image, &region, 1, scope));
  retire.push(retire_at, [staging]() mutable { staging.reset(); });
  return core::Status{};
}

}  // namespace volumetric_kit::gfx

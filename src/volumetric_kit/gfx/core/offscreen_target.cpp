// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/offscreen_target.hpp"

#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/check.hpp"

namespace volumetric_kit::gfx {
namespace {

// Bytes per texel for the color formats a readback target can size a staging
// buffer for. Returns 0 for a format whose layout this kit does not yet handle,
// so create() can reject a readback request rather than under-allocate.
uint32_t color_texel_size(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_SRGB:
      return 1;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UINT:
      return 2;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
      return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32G32_SFLOAT:
      return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
      return 16;
    default:
      return 0;
  }
}

}  // namespace

Result<OffscreenTarget> OffscreenTarget::create(
    Allocator& allocator, const OffscreenTargetDesc& desc) {
  if (desc.extent.width == 0 || desc.extent.height == 0) {
    return Status::invalid_argument(
        "OffscreenTarget::create: extent must be non-zero");
  }
  if (desc.color_format == VK_FORMAT_UNDEFINED) {
    return Status::invalid_argument(
        "OffscreenTarget::create: color_format must not be "
        "VK_FORMAT_UNDEFINED");
  }

  VkDeviceSize readback_size = 0;
  if (desc.readback) {
    const uint32_t texel = color_texel_size(desc.color_format);
    if (texel == 0) {
      return Status::unsupported(
          "OffscreenTarget::create: readback unsupported for this color "
          "format");
    }
    readback_size =
        VkDeviceSize{desc.extent.width} * desc.extent.height * texel;
  }

  // Color target: COLOR_ATTACHMENT to render into, TRANSFER_SRC to copy out.
  TextureDesc color_desc;
  color_desc.extent = desc.extent;
  color_desc.format = desc.color_format;
  color_desc.usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  VG_ASSIGN(Texture color, allocator.create_image(color_desc));

  OffscreenTarget target;
  target.color_ = std::move(color);

  if (desc.depth_format != VK_FORMAT_UNDEFINED) {
    TextureDesc depth_desc;
    depth_desc.extent = desc.extent;
    depth_desc.format = desc.depth_format;
    depth_desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VG_ASSIGN(Texture depth, allocator.create_image(depth_desc));
    target.depth_ = std::move(depth);
  }

  if (desc.readback) {
    BufferDesc readback_desc;
    readback_desc.size = readback_size;
    readback_desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    readback_desc.memory = MemoryUsage::HostVisible;
    readback_desc.mapped = true;
    VG_ASSIGN(Buffer readback, allocator.create_buffer(readback_desc));
    target.readback_ = std::move(readback);
  }

  return target;
}

RenderTarget OffscreenTarget::target() const {
  const RenderTargetAttachment color{color_.image(), color_.view(),
                                     color_.format()};
  RenderTargetAttachment depth{};
  const bool has_depth = depth_.valid();
  if (has_depth) {
    depth = {depth_.image(), depth_.view(), depth_.format()};
  }
  return RenderTarget(color_.extent(), &color, 1, has_depth ? &depth : nullptr,
                      VK_SAMPLE_COUNT_1_BIT);
}

RenderTargetLayout OffscreenTarget::layout() const { return target().layout(); }

void OffscreenTarget::record_readback(VkCommandBuffer cmd) const {
  VG_CHECK(valid(), "OffscreenTarget::record_readback on an empty target");
  VG_CHECK(readback_.valid(),
           "OffscreenTarget::record_readback without a readback buffer");

  const VkExtent2D ext = color_.extent();
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  // The render left the color image in COLOR_ATTACHMENT_OPTIMAL; move it to
  // TRANSFER_SRC for the copy-out.
  VkImageMemoryBarrier to_src{};
  to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_src.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  to_src.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_src.image = color_.image();
  to_src.subresourceRange = range;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_src);

  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {ext.width, ext.height, 1};
  vkCmdCopyImageToBuffer(cmd, color_.image(),
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         readback_.handle(), 1, &copy);

  // Make the copy available to the host read.
  VkBufferMemoryBarrier to_host{};
  to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_host.buffer = readback_.handle();
  to_host.offset = 0;
  to_host.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &to_host,
                       0, nullptr);
}

}  // namespace volumetric_kit::gfx

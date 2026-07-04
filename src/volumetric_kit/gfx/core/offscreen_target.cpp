// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/offscreen_target.hpp"

#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/check.hpp"
#include "volumetric_kit/gfx/core/impl/command.hpp"
#include "volumetric_kit/gfx/core/impl/depth_attachment.hpp"
#include "volumetric_kit/gfx/core/impl/vk_format.hpp"

namespace volumetric_kit::gfx {

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
  if (desc.depth_format != VK_FORMAT_UNDEFINED) {
    VG_TRY(validate_depth_only_format(desc.depth_format,
                                      "OffscreenTarget::create"));
  }

  VkDeviceSize readback_size = 0;
  if (desc.readback) {
    const uint32_t texel = texel_size(desc.color_format);
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

  // Optional depth attachment: device-local, depth-stencil usage. depth_format
  // is validated depth-only above, so create_image derives a DEPTH-aspect view.
  if (desc.depth_format != VK_FORMAT_UNDEFINED) {
    VG_ASSIGN(Texture depth,
              make_depth_attachment(allocator, desc.extent, desc.depth_format));
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
  if (depth_.valid()) {
    const RenderTargetAttachment depth{depth_.image(), depth_.view(),
                                       depth_.format()};
    return RenderTarget(color_.extent(), &color, 1, VK_SAMPLE_COUNT_1_BIT,
                        &depth);
  }
  return RenderTarget(color_.extent(), &color, 1, VK_SAMPLE_COUNT_1_BIT);
}

RenderTargetLayout OffscreenTarget::layout() const {
  // Built directly from the owned color attachment rather than via a throwaway
  // target(), since the format signature is all a caller needs to build a
  // compatible pipeline.
  RenderTargetLayout layout;
  layout.color_formats[0] = color_.format();
  layout.color_count = 1;
  layout.depth_format = depth_.valid() ? depth_.format() : VK_FORMAT_UNDEFINED;
  layout.samples = VK_SAMPLE_COUNT_1_BIT;
  return layout;
}

void OffscreenTarget::record_readback(VkCommandBuffer cmd) const {
  VG_CHECK(valid(), "OffscreenTarget::record_readback on an empty target");
  VG_CHECK(readback_.valid(),
           "OffscreenTarget::record_readback without a readback buffer");

  const VkExtent2D ext = color_.extent();

  // The render left the color image in COLOR_ATTACHMENT_OPTIMAL; move it to
  // TRANSFER_SRC for the copy-out.
  cmd_image_barrier(
      cmd, color_.image(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

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

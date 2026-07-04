// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/offscreen_target.hpp"

#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/check.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
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

void OffscreenTarget::prepare(VkCommandBuffer cmd) const {
  VG_CHECK(valid(), "OffscreenTarget::prepare on an empty target");

  // UNDEFINED discards the previous contents (a load-op clear rewrites them),
  // so this is valid whatever layout a prior render/readback left them in. The
  // src scope covers those priors so a *same-submit* re-prepare is safe: a
  // prior render's color write (COLOR_ATTACHMENT_OUTPUT) and a prior
  // @ref record_readback's copy-out (a TRANSFER read of the color image) must
  // both complete before the next render's clear reuses the image -- a
  // TOP_OF_PIPE src would not order the readback copy, letting the clear race
  // it into a torn readback. On the first prepare (no prior access) the wider
  // src simply waits on nothing.
  ImageBarrierDesc to_color;
  to_color.image = color_.image();
  to_color.src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_TRANSFER_BIT;
  to_color.dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  to_color.src_access =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  to_color.dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_color.new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  cmd_image_barrier(cmd, to_color);

  if (depth_.valid()) {
    // Depth is never read back; its only prior use is an earlier render's
    // depth write, which the src scope orders before the next clear.
    ImageBarrierDesc to_depth;
    to_depth.image = depth_.image();
    to_depth.src_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    to_depth.dst_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    to_depth.src_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_depth.dst_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    to_depth.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_depth.new_layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    to_depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    cmd_image_barrier(cmd, to_depth);
  }
}

void OffscreenTarget::record_readback(VkCommandBuffer cmd) const {
  VG_CHECK(valid(), "OffscreenTarget::record_readback on an empty target");
  VG_CHECK(readback_.valid(),
           "OffscreenTarget::record_readback without a readback buffer");

  const VkExtent2D ext = color_.extent();

  // The render left the color image in COLOR_ATTACHMENT_OPTIMAL; move it to
  // TRANSFER_SRC for the copy-out.
  ImageBarrierDesc to_src;
  to_src.image = color_.image();
  to_src.src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  to_src.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  to_src.src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_src.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
  to_src.old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_src.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  cmd_image_barrier(cmd, to_src);

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

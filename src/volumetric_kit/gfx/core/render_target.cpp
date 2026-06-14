// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/render_target.hpp"

#include "volumetric_kit/gfx/core/check.hpp"

namespace volumetric_kit::gfx {

bool RenderTargetLayout::compatible_with(
    const RenderTargetLayout& other) const noexcept {
  if (color_count != other.color_count || depth_format != other.depth_format ||
      samples != other.samples) {
    return false;
  }
  for (uint32_t i = 0; i < color_count; ++i) {
    if (color_formats[i] != other.color_formats[i]) {
      return false;
    }
  }
  return true;
}

RenderTarget::RenderTarget(VkExtent2D extent,
                           const RenderTargetAttachment* color,
                           uint32_t color_count,
                           const RenderTargetAttachment* depth,
                           VkSampleCountFlagBits samples)
    : extent_(extent), color_count_(color_count), samples_(samples) {
  VG_CHECK(color != nullptr && color_count > 0,
           "RenderTarget needs at least one color attachment");
  VG_CHECK(color_count <= RenderTargetLayout::kMaxColorAttachments,
           "RenderTarget color count exceeds kMaxColorAttachments");
  for (uint32_t i = 0; i < color_count; ++i) {
    color_[i] = color[i];
  }
  if (depth != nullptr) {
    depth_ = *depth;
    has_depth_ = true;
  }
}

RenderTargetLayout RenderTarget::layout() const noexcept {
  RenderTargetLayout layout;
  layout.color_count = color_count_;
  for (uint32_t i = 0; i < color_count_; ++i) {
    layout.color_formats[i] = color_[i].format;
  }
  layout.depth_format = has_depth_ ? depth_.format : VK_FORMAT_UNDEFINED;
  layout.samples = samples_;
  return layout;
}

void RenderTarget::begin(VkCommandBuffer cmd,
                         const RenderTargetBeginInfo& info) const {
  VG_CHECK(valid(), "RenderTarget::begin on an empty target");

  std::array<VkRenderingAttachmentInfo,
             RenderTargetLayout::kMaxColorAttachments>
      color_attachments{};
  for (uint32_t i = 0; i < color_count_; ++i) {
    color_attachments[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachments[i].imageView = color_[i].view;
    color_attachments[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachments[i].loadOp = info.load_op;
    color_attachments[i].storeOp = info.store_op;
    color_attachments[i].clearValue.color = info.clear_color;
  }

  const bool has_depth = has_depth_;
  VkRenderingAttachmentInfo depth_attachment{};
  if (has_depth) {
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = depth_.view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = info.load_op;
    depth_attachment.storeOp = info.store_op;
    depth_attachment.clearValue.depthStencil = info.clear_depth;
  }

  VkRenderingInfo rendering{};
  rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  rendering.renderArea.extent = extent_;
  rendering.layerCount = 1;
  rendering.colorAttachmentCount = color_count_;
  rendering.pColorAttachments = color_attachments.data();
  rendering.pDepthAttachment = has_depth ? &depth_attachment : nullptr;

  vkCmdBeginRendering(cmd, &rendering);
}

void RenderTarget::end(VkCommandBuffer cmd) const { vkCmdEndRendering(cmd); }

}  // namespace volumetric_kit::gfx

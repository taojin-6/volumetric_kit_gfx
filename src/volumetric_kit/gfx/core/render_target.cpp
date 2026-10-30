// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/render_target.hpp"

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"

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
                           uint32_t color_count, VkSampleCountFlagBits samples,
                           const RenderTargetAttachment* depth,
                           const RenderTargetAttachment* resolve)
    : extent_(extent), color_count_(color_count), samples_(samples) {
  VKC_CHECK(color != nullptr && color_count > 0,
            "RenderTarget needs at least one color attachment");
  VKC_CHECK(color_count <= RenderTargetLayout::kMaxColorAttachments,
            "RenderTarget color count exceeds kMaxColorAttachments");
  const bool multisampled = samples != VK_SAMPLE_COUNT_1_BIT;
  VKC_CHECK(multisampled == (resolve != nullptr),
            "RenderTarget takes resolve attachments exactly when samples > 1");
  for (uint32_t i = 0; i < color_count; ++i) {
    color_[i] = color[i];
    if (multisampled) {
      // VUID-VkRenderingAttachmentInfo-imageView-06865: a resolve keeps its
      // colour's format.
      VKC_CHECK(resolve[i].format == color[i].format,
                "RenderTarget resolve format differs from its color's");
      VKC_CHECK(resolve[i].view != VK_NULL_HANDLE,
                "RenderTarget resolve attachment has no view");
      resolve_[i] = resolve[i];
    }
  }
  if (depth != nullptr && depth->view != VK_NULL_HANDLE) {
    depth_ = *depth;
  }
}

RenderTargetLayout RenderTarget::layout() const noexcept {
  RenderTargetLayout layout;
  layout.color_count = color_count_;
  for (uint32_t i = 0; i < color_count_; ++i) {
    layout.color_formats[i] = color_[i].format;
  }
  layout.depth_format = has_depth() ? depth_.format : VK_FORMAT_UNDEFINED;
  layout.samples = samples_;
  return layout;
}

void RenderTarget::begin(VkCommandBuffer cmd,
                         const RenderTargetBeginInfo& info) const {
  VKC_CHECK(valid(), "RenderTarget::begin on an empty target");
  const bool multisampled = samples_ != VK_SAMPLE_COUNT_1_BIT;

  if (multisampled) {
    // The samples live for this scope alone, so nothing loads them and depth
    // is never stored; the colours reach memory through their resolves.
    VKC_CHECK(info.load_op != VK_ATTACHMENT_LOAD_OP_LOAD,
              "RenderTarget::begin: a multisampled target cannot LOAD");
    VKC_CHECK(info.depth_store_op != VK_ATTACHMENT_STORE_OP_STORE,
              "RenderTarget::begin: a multisampled target cannot STORE depth");

    // UNDEFINED -> attachment layout every scope: the samples are discarded,
    // and the source scope orders the transition after an earlier scope's
    // clear, draws, resolve read and DONT_CARE store on the same images -- the
    // previous frame's, when one set serves several targets.
    std::array<ImageBarrierDesc, RenderTargetLayout::kMaxColorAttachments + 1>
        barriers{};
    uint32_t count = 0;
    for (uint32_t i = 0; i < color_count_; ++i) {
      ImageBarrierDesc& b = barriers[count++];
      b.image = color_[i].image;
      b.src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      b.dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      b.src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      b.dst_access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      b.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
      b.new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (has_depth()) {
      ImageBarrierDesc& b = barriers[count++];
      b.image = depth_.image;
      b.src_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
      b.dst_stage = b.src_stage;
      b.src_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      b.dst_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      b.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
      b.new_layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      b.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    cmd_image_barriers(cmd, barriers.data(), count);
  }

  std::array<VkRenderingAttachmentInfo,
             RenderTargetLayout::kMaxColorAttachments>
      color_attachments{};
  for (uint32_t i = 0; i < color_count_; ++i) {
    VkRenderingAttachmentInfo& a = color_attachments[i];
    a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    a.imageView = color_[i].view;
    a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    a.loadOp = info.load_op;
    a.storeOp = info.store_op;
    a.clearValue.color = info.clear_color;
    if (multisampled) {
      a.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      a.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
      a.resolveImageView = resolve_[i].view;
      a.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
  }

  // The target is depth-only (OffscreenTarget rejects combined depth/stencil
  // formats), so DEPTH_ATTACHMENT_OPTIMAL is valid without the
  // separateDepthStencilLayouts feature.
  VkRenderingAttachmentInfo depth_attachment{};
  if (has_depth()) {
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = depth_.view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = info.load_op;
    depth_attachment.storeOp =
        multisampled ? VK_ATTACHMENT_STORE_OP_DONT_CARE : info.depth_store_op;
    depth_attachment.clearValue.depthStencil = info.clear_depth;
  }

  VkRenderingInfo rendering{};
  rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  rendering.renderArea.extent = extent_;
  rendering.layerCount = 1;
  rendering.colorAttachmentCount = color_count_;
  rendering.pColorAttachments = color_attachments.data();
  if (has_depth()) {
    rendering.pDepthAttachment = &depth_attachment;
  }

  vkCmdBeginRendering(cmd, &rendering);
}

void RenderTarget::end(VkCommandBuffer cmd) const { vkCmdEndRendering(cmd); }

}  // namespace volumetric_kit::gfx

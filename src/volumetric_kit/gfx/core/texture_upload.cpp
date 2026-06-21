// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/texture_upload.hpp"

#include <algorithm>
#include <cstring>

#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/impl/vk_format.hpp"

namespace volumetric_kit::gfx {
namespace {

// Full mip count for an extent: floor(log2(max(w, h))) + 1.
uint32_t mip_levels_for(VkExtent2D extent) {
  uint32_t max_dim = std::max(extent.width, extent.height);
  uint32_t levels = 1;
  while (max_dim > 1) {
    max_dim >>= 1;
    ++levels;
  }
  return levels;
}

// A single-mip-level color image layout-transition barrier. Distinct from
// core/impl/command.hpp's cmd_image_barrier (which is fixed to mip 0) because
// mip generation transitions each level independently as the blit walks down
// the chain.
void mip_barrier(VkCommandBuffer cmd, VkImage image, uint32_t level,
                 VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                 VkAccessFlags src_access, VkAccessFlags dst_access,
                 VkImageLayout old_layout, VkImageLayout new_layout) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);
}

// Copy staging -> mip 0, blit the chain down (if mipped), and leave every level
// in SHADER_READ_ONLY_OPTIMAL ready for sampling.
void record_upload(VkCommandBuffer cmd, VkImage image, VkBuffer staging,
                   VkExtent2D extent, uint32_t mip_levels) {
  // 1. Every level UNDEFINED -> TRANSFER_DST for the copy and blit writes.
  VkImageMemoryBarrier to_dst{};
  to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_dst.srcAccessMask = 0;
  to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.image = image;
  to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_dst);

  // 2. Copy the source pixels into mip 0.
  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {extent.width, extent.height, 1};
  vkCmdCopyBufferToImage(cmd, staging, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

  // 3. Generate mips: blit each level down to the next, moving each finished
  //    source level to SHADER_READ as we pass it.
  int32_t mip_w = static_cast<int32_t>(extent.width);
  int32_t mip_h = static_cast<int32_t>(extent.height);
  for (uint32_t level = 1; level < mip_levels; ++level) {
    // Source (level - 1): TRANSFER_DST -> TRANSFER_SRC for the blit read.
    mip_barrier(cmd, image, level - 1, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    const int32_t dst_w = mip_w > 1 ? mip_w / 2 : 1;
    const int32_t dst_h = mip_h > 1 ? mip_h / 2 : 1;
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
    blit.srcOffsets[1] = {mip_w, mip_h, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
    blit.dstOffsets[1] = {dst_w, dst_h, 1};
    vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_LINEAR);

    // Source level done being read: TRANSFER_SRC -> SHADER_READ.
    mip_barrier(cmd, image, level - 1, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    mip_w = dst_w;
    mip_h = dst_h;
  }

  // 4. The last level was only ever a copy/blit destination (never a source),
  //    so it is still TRANSFER_DST: move it to SHADER_READ too. For a
  //    single-mip image this is the lone mip 0.
  mip_barrier(cmd, image, mip_levels - 1, VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

}  // namespace

Result<Texture> upload_texture(const Device& device, Allocator& allocator,
                               const ImageUploadDesc& desc) {
  if (desc.extent.width == 0 || desc.extent.height == 0) {
    return Status::invalid_argument("upload_texture: extent must be non-zero");
  }
  if (desc.format == VK_FORMAT_UNDEFINED) {
    return Status::invalid_argument(
        "upload_texture: format must not be VK_FORMAT_UNDEFINED");
  }
  if (desc.pixels == nullptr) {
    return Status::invalid_argument("upload_texture: pixels must not be null");
  }
  const uint32_t texel = texel_size(desc.format);
  if (texel == 0) {
    // texel_size returns 0 for formats a flat per-texel copy cannot size:
    // compressed, multi-planar, subsampled, or depth/stencil.
    return Status::unsupported(
        "upload_texture: format must be an uncompressed, single-plane color "
        "format");
  }
  const VkDeviceSize expected =
      VkDeviceSize{desc.extent.width} * desc.extent.height * texel;
  if (desc.size != expected) {
    return Status::invalid_argument(
        "upload_texture: size must equal extent.width * extent.height * "
        "texel_size(format)");
  }

  const uint32_t mip_levels =
      desc.generate_mips ? mip_levels_for(desc.extent) : 1;

  // Mip generation blits with a linear filter, so the format must support both
  // blit endpoints and linear filtering; otherwise the blit is invalid use.
  if (mip_levels > 1) {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device.physical_device(), desc.format,
                                        &props);
    constexpr VkFormatFeatureFlags kNeeded =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if ((props.optimalTilingFeatures & kNeeded) != kNeeded) {
      return Status::unsupported(
          "upload_texture: generate_mips needs a format that supports linear "
          "blit (BLIT_SRC | BLIT_DST | SAMPLED_IMAGE_FILTER_LINEAR)");
    }
  }

  // Staging buffer: host-visible, mapped, written once front-to-back.
  BufferDesc staging_desc;
  staging_desc.size = desc.size;
  staging_desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  staging_desc.memory = MemoryUsage::HostVisible;
  staging_desc.mapped = true;
  staging_desc.host_access = HostAccess::SequentialWrite;
  VG_ASSIGN(Buffer staging, allocator.create_buffer(staging_desc));
  std::memcpy(staging.mapped(), desc.pixels, desc.size);

  // Destination: device-local sampled image. TRANSFER_DST for the copy, plus
  // TRANSFER_SRC when the mip chain blits read earlier levels.
  TextureDesc image_desc;
  image_desc.extent = desc.extent;
  image_desc.format = desc.format;
  image_desc.mip_levels = mip_levels;
  image_desc.usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (mip_levels > 1) {
    image_desc.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  VG_ASSIGN(Texture texture, allocator.create_image(image_desc));

  // submit_single_time blocks on a fence, so `staging` stays alive across the
  // whole transfer and is freed (here, on return) only once the GPU is done.
  const VkImage image = texture.image();
  const VkBuffer staging_handle = staging.handle();
  const VkExtent2D extent = desc.extent;
  VG_TRY(device.submit_single_time(
      [image, staging_handle, extent, mip_levels](VkCommandBuffer cmd) {
        record_upload(cmd, image, staging_handle, extent, mip_levels);
      }));

  return texture;
}

}  // namespace volumetric_kit::gfx

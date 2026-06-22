// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/texture_upload.hpp"

#include <algorithm>
#include <cstring>

#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/impl/command.hpp"
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

// Shader stages that may sample the finished texture. Moving each level to
// SHADER_READ makes the upload visible to vertex-texture fetch as well as
// fragment sampling -- both graphics-queue stages, which submit_single_time
// uses. (Cross-submission readers are additionally ordered by the fence
// submit_single_time waits on, so this dst scope only bites within the
// submission, which records no sampling of its own.)
constexpr VkPipelineStageFlags kSampleStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

// Copy staging -> mip 0, blit the chain down (if mipped), and leave every level
// in SHADER_READ_ONLY_OPTIMAL ready for sampling. Layout transitions go through
// core/impl/command.hpp's cmd_image_barrier, addressing a per-level mip range.
void record_upload(VkCommandBuffer cmd, VkImage image, VkBuffer staging,
                   VkExtent2D extent, uint32_t mip_levels) {
  // 1. Every level UNDEFINED -> TRANSFER_DST for the copy and blit writes.
  cmd_image_barrier(cmd, image, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, /*base_mip=*/0,
                    /*level_count=*/mip_levels);

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
    cmd_image_barrier(cmd, image, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_IMAGE_ASPECT_COLOR_BIT, /*base_mip=*/level - 1);

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
    cmd_image_barrier(cmd, image, VK_PIPELINE_STAGE_TRANSFER_BIT, kSampleStages,
                      VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_IMAGE_ASPECT_COLOR_BIT, /*base_mip=*/level - 1);

    mip_w = dst_w;
    mip_h = dst_h;
  }

  // 4. The last level was only ever a copy/blit destination (never a source),
  //    so it is still TRANSFER_DST: move it to SHADER_READ too. For a
  //    single-mip image this is the lone mip 0.
  cmd_image_barrier(cmd, image, VK_PIPELINE_STAGE_TRANSFER_BIT, kSampleStages,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT, /*base_mip=*/mip_levels - 1);
}

}  // namespace

Result<Texture> upload_texture(const Device& device, Allocator& allocator,
                               const ImageUploadDesc& desc) {
  if (desc.extent.width == 0 || desc.extent.height == 0) {
    return Status::invalid_argument("upload_texture: extent must be non-zero");
  }
  const uint32_t max_dim = device.caps().limits().maxImageDimension2D;
  if (desc.extent.width > max_dim || desc.extent.height > max_dim) {
    // Bounds the image well below INT32_MAX too, so the int32 mip-extent math
    // in record_upload never sees a negative dimension.
    return Status::unsupported(
        "upload_texture: extent exceeds the device's maxImageDimension2D "
        "limit");
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

  // The destination is always created with SAMPLED usage + optimal tiling, so
  // the format must support being sampled there. Reject up front with a clean
  // Unsupported rather than letting create_image trip a validation error.
  if (!device.caps().format_supports(desc.format, VK_IMAGE_TILING_OPTIMAL,
                                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
    return Status::unsupported(
        "upload_texture: format does not support sampling (SAMPLED_IMAGE) with "
        "optimal tiling");
  }

  // TODO: support caller-supplied precomputed mip levels (KTX2 / glTF commonly
  // ship them) by splitting the staging copy from this on-device mip strategy;
  // today generate_mips always downsamples on the GPU with a linear-filter
  // blit.
  const uint32_t mip_levels =
      desc.generate_mips ? mip_levels_for(desc.extent) : 1;

  // Mip generation blits with a linear filter, so the format must additionally
  // support both blit endpoints and linear filtering; otherwise the blit is
  // invalid use.
  if (mip_levels > 1) {
    constexpr VkFormatFeatureFlags kBlitNeeded =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if (!device.caps().format_supports(desc.format, VK_IMAGE_TILING_OPTIMAL,
                                       kBlitNeeded)) {
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

  // Destination: device-local sampled image. SAMPLED to read it in shaders,
  // TRANSFER_DST for the staging copy, and TRANSFER_SRC so the mip-chain blits
  // can read earlier levels -- and, uniformly for the single-mip case too, so
  // the finished texture stays copyable/blittable (readback, screenshots,
  // re-upload) rather than that capability hinging on whether mips were asked
  // for.
  TextureDesc image_desc;
  image_desc.extent = desc.extent;
  image_desc.format = desc.format;
  image_desc.mip_levels = mip_levels;
  image_desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
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

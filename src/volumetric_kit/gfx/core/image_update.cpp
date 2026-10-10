// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/image_update.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "upload_steps.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/format.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/core/impl/image_copy_layout.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"

namespace volumetric_kit::gfx {
namespace {

// The stages that sample an image: what SHADER_READ access, and the layout the
// update leaves, are for.
constexpr VkPipelineStageFlags kShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

// The command buffer, the scope and the image. Returns the texel size.
core::Result<VkDeviceSize> check_image(const std::string& name,
                                       VkCommandBuffer cmd,
                                       const core::Image& image,
                                       const ImageUpdateScope& scope) {
  if (cmd == VK_NULL_HANDLE) {
    return core::Status::invalid_argument(name + ": null command buffer");
  }
  if (scope.src_stages == 0 || (scope.src_stages & ~kShaderStages) != 0 ||
      scope.dst_stages == 0 || (scope.dst_stages & ~kShaderStages) != 0) {
    return core::Status::invalid_argument(
        name + ": the scope's stages must be non-empty shader stages");
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
  return texel;
}

// Region `i`'s placement in the image.
core::Status check_region(const std::string& name, const core::Image& image,
                          const VkBufferImageCopy& r, std::uint32_t i) {
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
  return core::Status{};
}

// Whether regions `a` and `b`, each inside the image, write a texel in
// common: the same level, a layer in common, and boxes that intersect.
bool overlap(const VkBufferImageCopy& a, const VkBufferImageCopy& b) {
  const VkImageSubresourceLayers& sa = a.imageSubresource;
  const VkImageSubresourceLayers& sb = b.imageSubresource;
  if (sa.mipLevel != sb.mipLevel ||
      sa.baseArrayLayer >= sb.baseArrayLayer + sb.layerCount ||
      sb.baseArrayLayer >= sa.baseArrayLayer + sa.layerCount) {
    return false;
  }
  const auto apart = [](std::int32_t a_offset, std::uint32_t a_extent,
                        std::int32_t b_offset, std::uint32_t b_extent) {
    return std::int64_t{a_offset} + a_extent <= b_offset ||
           std::int64_t{b_offset} + b_extent <= a_offset;
  };
  return !apart(a.imageOffset.x, a.imageExtent.width, b.imageOffset.x,
                b.imageExtent.width) &&
         !apart(a.imageOffset.y, a.imageExtent.height, b.imageOffset.y,
                b.imageExtent.height) &&
         !apart(a.imageOffset.z, a.imageExtent.depth, b.imageOffset.z,
                b.imageExtent.depth);
}

}  // namespace

core::Status record_image_update(VkCommandBuffer cmd,
                                 const core::Buffer& source, core::Image& image,
                                 const VkBufferImageCopy* regions,
                                 std::uint32_t region_count,
                                 const ImageUpdateScope& scope) {
  std::vector<ImageCopy> copies;
  if (regions != nullptr) {
    copies.reserve(region_count);
    for (std::uint32_t i = 0; i < region_count; ++i) {
      copies.push_back({&source, regions[i]});
    }
  }
  return record_image_update(cmd, copies.data(),
                             static_cast<std::uint32_t>(copies.size()), image,
                             scope);
}

core::Status record_image_update(VkCommandBuffer cmd, const ImageCopy* copies,
                                 std::uint32_t copy_count, core::Image& image,
                                 const ImageUpdateScope& scope) {
  const std::string name = "record_image_update";
  VKC_ASSIGN(const VkDeviceSize texel, check_image(name, cmd, image, scope));
  if (copies == nullptr || copy_count == 0) {
    return core::Status::invalid_argument(name + ": no regions");
  }
  for (std::uint32_t i = 0; i < copy_count; ++i) {
    VKC_TRY(check_region(name, image, copies[i].region, i));
    VKC_TRY(detail::check_buffer_image_copy(
        name + ": region " + std::to_string(i), copies[i].source,
        copies[i].region, texel));
    // The copies run with no barrier between them, so two that write one
    // texel would race.
    for (std::uint32_t j = 0; j < i; ++j) {
      if (overlap(copies[j].region, copies[i].region)) {
        return core::Status::invalid_argument(
            name + ": region " + std::to_string(i) + " overlaps region " +
            std::to_string(j));
      }
    }
  }

  // The contents are discarded, but not the order: the copy waits for the
  // earlier reads the scope names. Each run of copies from one buffer is one
  // copy command.
  detail::record_discard_for_copy(cmd, image.handle(), scope.src_stages);
  std::vector<VkBufferImageCopy> run;
  for (std::uint32_t i = 0; i < copy_count; ++i) {
    run.push_back(copies[i].region);
    if (i + 1 == copy_count || copies[i + 1].source != copies[i].source) {
      vkCmdCopyBufferToImage(cmd, copies[i].source->handle(), image.handle(),
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             static_cast<std::uint32_t>(run.size()),
                             run.data());
      run.clear();
    }
  }
  detail::record_copied_to_shader_read(cmd, image.handle(), scope.dst_stages);
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
  VKC_ASSIGN(const VkDeviceSize texel, check_image(kCall, cmd, image, scope));
  VKC_TRY(check_region(kCall, image, region, 0));
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

  VKC_ASSIGN(core::Buffer buffer,
             detail::make_staging(allocator, pixels, size));
  // Shared, as the RetireQueue's deleter must be copyable.
  auto staging = std::make_shared<core::Buffer>(std::move(buffer));

  VKC_TRY(record_image_update(cmd, *staging, image, &region, 1, scope));
  retire.push(retire_at, [staging]() mutable { staging.reset(); });
  return core::Status{};
}

}  // namespace volumetric_kit::gfx

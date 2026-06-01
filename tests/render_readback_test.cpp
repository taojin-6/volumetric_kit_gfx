// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Headless offscreen readback: clear an offscreen color image to a known color,
// copy it to a host-visible buffer, fence-wait, and read the pixels back. This
// exercises the offscreen-target + image-layout-transition + image->buffer copy
// + fence-gated readback path that the RGBDA screen-capture feature and the
// future golden-image render tests build on. (A real draw replaces the clear
// once the pipelines tier provides a graphics pipeline.)

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"
#include "vulkan_test_fixture.hpp"

namespace {
using ReadbackTest = VulkanDeviceTest;
}  // namespace

TEST_F(ReadbackTest, ClearedOffscreenImageReadsBackThroughHostBuffer) {
  constexpr uint32_t kWidth = 4;
  constexpr uint32_t kHeight = 4;
  constexpr VkDeviceSize kBytes = kWidth * kHeight * 4;  // R8G8B8A8

  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  // Offscreen color target: COLOR_ATTACHMENT makes its default view valid (and
  // is the real render-target usage); TRANSFER_DST allows the clear and
  // TRANSFER_SRC the copy-out.
  vg::TextureDesc image_desc;
  image_desc.extent = {kWidth, kHeight};
  image_desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  image_desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  auto image = allocator.value().create_image(image_desc);
  ASSERT_TRUE(image.ok()) << image.status().message();

  vg::BufferDesc buffer_desc;
  buffer_desc.size = kBytes;
  buffer_desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_desc.memory = vg::MemoryUsage::HostVisible;
  buffer_desc.mapped = true;
  auto readback = allocator.value().create_buffer(buffer_desc);
  ASSERT_TRUE(readback.ok()) << readback.status().message();

  auto pool = vg::CommandPool::create(device(), device_->graphics_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();

  const VkImage img = image.value().image();
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());

  // UNDEFINED -> TRANSFER_DST for the clear.
  VkImageMemoryBarrier to_dst{};
  to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_dst.srcAccessMask = 0;
  to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.image = img;
  to_dst.subresourceRange = range;
  vkCmdPipelineBarrier(cmd.value().handle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_dst);

  VkClearColorValue clear{};
  clear.float32[0] = 1.0f;  // R
  clear.float32[1] = 0.0f;  // G
  clear.float32[2] = 0.0f;  // B
  clear.float32[3] = 1.0f;  // A
  vkCmdClearColorImage(cmd.value().handle(), img,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);

  // TRANSFER_DST -> TRANSFER_SRC for the copy-out.
  VkImageMemoryBarrier to_src = to_dst;
  to_src.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  to_src.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  vkCmdPipelineBarrier(cmd.value().handle(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_src);

  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {kWidth, kHeight, 1};
  vkCmdCopyImageToBuffer(cmd.value().handle(), img,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         readback.value().handle(), 1, &copy);

  // Make the copy available to the host read.
  VkBufferMemoryBarrier to_host{};
  to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_host.buffer = readback.value().handle();
  to_host.offset = 0;
  to_host.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd.value().handle(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &to_host,
                       0, nullptr);

  ASSERT_TRUE(cmd.value().end().ok());

  auto fence = vg::Fence::create(device());
  ASSERT_TRUE(fence.ok()) << fence.status().message();
  VkCommandBuffer raw = cmd.value().handle();
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &raw;
  ASSERT_EQ(vkQueueSubmit(device_->graphics_queue(), 1, &submit,
                          fence.value().handle()),
            VK_SUCCESS);
  ASSERT_TRUE(fence.value().wait().ok());

  const auto* px = static_cast<const unsigned char*>(readback.value().mapped());
  // The whole image was cleared to opaque red, so the first and last texels
  // match.
  EXPECT_EQ(px[0], 255);  // R
  EXPECT_EQ(px[1], 0);    // G
  EXPECT_EQ(px[2], 0);    // B
  EXPECT_EQ(px[3], 255);  // A
  EXPECT_EQ(px[kBytes - 4], 255);
  EXPECT_EQ(px[kBytes - 1], 255);
}

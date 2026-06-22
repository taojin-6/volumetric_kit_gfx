// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// Adds a VMA allocator on top of the shared device fixture (mirrors
// AllocatorTest): the derived allocator_ is destroyed before the base's
// device_/instance_, and each test's textures/buffers before any of them.
class TextureUploadTest : public VulkanDeviceTest {
 protected:
  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (IsSkipped()) {
      return;  // no Vulkan device; the base already skipped
    }
    auto allocator = vg::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

  std::optional<vg::Allocator> allocator_;
};

}  // namespace

TEST_F(TextureUploadTest, RoundTripsPixelsThroughTheGpu) {
  // A 2x2 RGBA8 image with four distinct texels.
  const std::array<std::uint8_t, 16> src = {0x10, 0x20, 0x30, 0x40,  //
                                            0x50, 0x60, 0x70, 0x80,  //
                                            0x90, 0xA0, 0xB0, 0xC0,  //
                                            0xD0, 0xE0, 0xF0, 0xFF};

  vg::ImageUploadDesc desc;
  desc.extent = {2, 2};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();

  auto texture = vg::upload_texture(*device_, *allocator_, desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_TRUE(texture.value().valid());
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);
  EXPECT_EQ(texture.value().extent().width, 2u);
  EXPECT_EQ(texture.value().extent().height, 2u);
  EXPECT_EQ(texture.value().mip_levels(), 1u);  // no mips requested

  // Copy the uploaded image back into a host-visible buffer and confirm the
  // bytes survived the staging -> image -> readback round trip (proving the
  // copy and the layout transitions landed the data correctly).
  vg::BufferDesc rb;
  rb.size = src.size();
  rb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  rb.memory = vg::MemoryUsage::HostVisible;
  rb.mapped = true;
  auto readback = allocator_->create_buffer(rb);
  ASSERT_TRUE(readback.ok()) << readback.status().message();

  const VkImage image = texture.value().image();
  const VkBuffer dst = readback.value().handle();
  auto recorded =
      device_->submit_single_time([image, dst](VkCommandBuffer cmd) {
        // upload_texture left the image in SHADER_READ_ONLY_OPTIMAL.
        VkImageMemoryBarrier to_src{};
        to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_src.srcAccessMask = 0;
        to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_src.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.image = image;
        to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &to_src);

        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {2, 2, 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               dst, 1, &copy);
      });
  ASSERT_TRUE(recorded.ok()) << recorded.message();

  const auto* got = static_cast<const std::uint8_t*>(readback.value().mapped());
  ASSERT_NE(got, nullptr);
  for (std::size_t i = 0; i < src.size(); ++i) {
    EXPECT_EQ(got[i], src[i]) << "byte " << i;
  }
}

TEST_F(TextureUploadTest, GeneratesMipChain) {
  // 8x8 RGBA8 -> 4 mip levels (8, 4, 2, 1). A successful multi-level blit
  // exercises the per-level barriers and the SHADER_READ end state; the default
  // view spans the whole chain.
  std::array<std::uint8_t, 8 * 8 * 4> src{};
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>(i);
  }

  vg::ImageUploadDesc desc;
  desc.extent = {8, 8};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  desc.generate_mips = true;

  auto texture = vg::upload_texture(*device_, *allocator_, desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_TRUE(texture.value().valid());
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);
  EXPECT_EQ(texture.value().mip_levels(), 4u);  // 8 -> 4 -> 2 -> 1
}

TEST_F(TextureUploadTest, RejectsZeroExtent) {
  std::array<std::uint8_t, 4> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {0, 0};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, RejectsExtentAboveDeviceLimit) {
  std::array<std::uint8_t, 4> src{};
  vg::ImageUploadDesc desc;
  // 1<<20 dwarfs any real maxImageDimension2D (spec minimum 4096; hardware tops
  // out around 16384–32768), so this is rejected up front before allocation.
  desc.extent = {1u << 20, 1u << 20};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::Unsupported);
}

TEST_F(TextureUploadTest, RejectsUndefinedFormat) {
  std::array<std::uint8_t, 4> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {1, 1};
  desc.format = VK_FORMAT_UNDEFINED;
  desc.pixels = src.data();
  desc.size = src.size();
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, RejectsNullPixels) {
  vg::ImageUploadDesc desc;
  desc.extent = {1, 1};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = nullptr;
  desc.size = 4;
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, RejectsSizeMismatch) {
  std::array<std::uint8_t, 8> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {2, 2};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;  // expects 2*2*4 = 16 bytes
  desc.pixels = src.data();
  desc.size = src.size();  // 8 — wrong
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, RejectsCompressedFormat) {
  std::array<std::uint8_t, 8> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {4, 4};
  // Compressed: no flat per-texel size, so a buffer copy cannot describe it.
  desc.format = VK_FORMAT_BC1_RGB_UNORM_BLOCK;
  desc.pixels = src.data();
  desc.size = src.size();
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::Unsupported);
}

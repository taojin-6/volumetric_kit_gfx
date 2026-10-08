// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// record_image_update / record_image_upload: copies into a sampled image,
// recorded between their layout transitions, read back to prove where every
// texel landed. Runs under the validation layer with synchronization
// validation. (That an update waits for earlier draws of the image is
// streamed_atlas_test's, which has a pipeline to draw with.)

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
#include "volumetric_kit/gfx/core/image_update.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr uint32_t kTexel = 4;

using Texels = std::vector<uint32_t>;  // one RGBA8 texel a word

class ImageUpdateTest : public VulkanDeviceTest {
 protected:
  bool wants_validation() const override { return true; }
  bool wants_sync_validation() const override { return true; }

  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

  vkc::Result<vkc::Image> make_image(
      VkExtent2D extent, uint32_t layers = 1,
      VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      VkFormat format = kFormat) {
    vkc::ImageDesc desc;
    desc.extent = extent;
    desc.array_layers = layers;
    desc.format = format;
    desc.usage = usage;
    return allocator_->create_image(desc);
  }

  // A mapped buffer holding `texels`.
  vkc::Result<vkc::Buffer> make_source(
      const Texels& texels,
      VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
    vkc::BufferDesc desc;
    desc.size = texels.size() * sizeof(uint32_t);
    desc.usage = usage;
    desc.memory = vkc::MemoryUsage::Staging;
    VKC_ASSIGN(vkc::Buffer buffer, allocator_->create_buffer(desc));
    std::memcpy(buffer.mapped(), texels.data(), desc.size);
    return buffer;
  }

  // Every layer of `image`'s first level, which an update left in
  // SHADER_READ_ONLY_OPTIMAL for the fragment stage, layer after layer.
  Texels read(const vkc::Image& image) {
    const uint32_t count =
        image.width() * image.height() * image.array_layers();
    vkc::BufferDesc desc;
    desc.size = VkDeviceSize{count} * kTexel;
    desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memory = vkc::MemoryUsage::Staging;
    desc.host_access = vkc::HostAccess::Random;
    auto readback = allocator_->create_buffer(desc);
    EXPECT_TRUE(readback.ok()) << readback.status().message();
    if (!readback.ok()) {
      return {};
    }
    const vkc::Status read =
        device_->submit_single_time([&](VkCommandBuffer cmd) {
          vg::ImageBarrierDesc to_src;
          to_src.image = image.handle();
          to_src.src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
          to_src.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
          to_src.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
          to_src.old_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          to_src.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          vg::cmd_image_barrier(cmd, to_src);
          VkBufferImageCopy copy{};
          copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                                   image.array_layers()};
          copy.imageExtent = {image.width(), image.height(), 1};
          vkCmdCopyImageToBuffer(cmd, image.handle(),
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 readback.value().handle(), 1, &copy);
        });
    EXPECT_TRUE(read.ok()) << read.message();
    Texels texels(count);
    std::memcpy(texels.data(), readback.value().mapped(), desc.size);
    return texels;
  }

  std::optional<vkc::Allocator> allocator_;
};

// Two regions from one buffer, each with a row length and an image height of
// its own and covering both layers of a 2x2 array image: every texel lands
// where its region puts it, and the image records the layout it was left in.
TEST_F(ImageUpdateTest, CopiesRegionsFromRowsOfABuffer) {
  auto image = make_image({2, 2}, 2);
  ASSERT_TRUE(image.ok()) << image.status().message();

  // Region 0: column 0 of both layers, from rows of 3 texels, 3 rows a
  // layer. Region 1: column 1 of both layers, packed tightly after it.
  // Unused texels are 0xEE.
  const Texels source_texels{
      0x01, 0xEE, 0xEE,  // layer 0, row 0
      0x02, 0xEE, 0xEE,  // layer 0, row 1
      0xEE, 0xEE, 0xEE,  // layer 0, the image height's spare row
      0x03, 0xEE, 0xEE,  // layer 1, row 0
      0x04,              // layer 1, row 1: the region's last texel
      0x11, 0x12,        // region 1: layer 0's column 1, then layer 1's
      0x13, 0x14,
  };
  auto source = make_source(source_texels);
  ASSERT_TRUE(source.ok()) << source.status().message();

  std::array<VkBufferImageCopy, 2> regions{};
  regions[0].bufferRowLength = 3;
  regions[0].bufferImageHeight = 3;
  regions[0].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2};
  regions[0].imageExtent = {1, 2, 1};
  regions[1].bufferOffset = 13 * kTexel;
  regions[1].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2};
  regions[1].imageOffset = {1, 0, 0};
  regions[1].imageExtent = {1, 2, 1};

  vkc::Status recorded;
  const vkc::Status submitted =
      device_->submit_single_time([&](VkCommandBuffer cmd) {
        recorded = vg::record_image_update(cmd, source.value(), image.value(),
                                           regions.data(), 2);
      });
  ASSERT_TRUE(recorded.ok()) << recorded.message();
  ASSERT_TRUE(submitted.ok()) << submitted.message();
  EXPECT_EQ(image.value().layout(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  EXPECT_EQ(read(image.value()),
            (Texels{0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 0x04, 0x14}));
}

// One update from several buffers, the sources alternating A, B, A: each
// region reads its own buffer, wherever in the list it is, and the image is
// left ready to sample once, after all of them.
TEST_F(ImageUpdateTest, CopiesEachRegionFromItsOwnBuffer) {
  auto image = make_image({3, 1});
  ASSERT_TRUE(image.ok()) << image.status().message();
  auto a = make_source(Texels{0xA0, 0xA2});
  ASSERT_TRUE(a.ok()) << a.status().message();
  auto b = make_source(Texels{0xB1});
  ASSERT_TRUE(b.ok()) << b.status().message();

  std::array<vg::ImageCopy, 3> copies{};
  for (uint32_t x = 0; x < 3; ++x) {
    copies[x].region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copies[x].region.imageOffset = {static_cast<int32_t>(x), 0, 0};
    copies[x].region.imageExtent = {1, 1, 1};
  }
  copies[0].source = &a.value();
  copies[1].source = &b.value();
  copies[2].source = &a.value();
  copies[2].region.bufferOffset = kTexel;

  vkc::Status recorded;
  const vkc::Status submitted =
      device_->submit_single_time([&](VkCommandBuffer cmd) {
        recorded =
            vg::record_image_update(cmd, copies.data(), 3, image.value());
      });
  ASSERT_TRUE(recorded.ok()) << recorded.message();
  ASSERT_TRUE(submitted.ok()) << submitted.message();
  EXPECT_EQ(image.value().layout(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  EXPECT_EQ(read(image.value()), (Texels{0xA0, 0xB1, 0xA2}));
}

// An upload copies the pixels into a staging buffer at once -- the caller's
// may change straight away -- and queues the buffer on the RetireQueue, which
// frees it once the timeline reaches the frame's value.
TEST_F(ImageUpdateTest, UploadStagesThroughTheRetireQueue) {
  auto image = make_image({2, 1});
  ASSERT_TRUE(image.ok()) << image.status().message();
  auto timeline = vkc::TimelineSemaphore::create(*device_, 0);
  ASSERT_TRUE(timeline.ok()) << timeline.status().message();
  vg::RetireQueue retire(timeline.value());

  Texels pixels{0xAABBCCDD, 0x11223344};
  vkc::Status recorded;
  const vkc::Status submitted =
      device_->submit_single_time([&](VkCommandBuffer cmd) {
        recorded = vg::record_image_upload(cmd, *allocator_, retire, 1,
                                           image.value(), pixels.data(),
                                           pixels.size() * sizeof(uint32_t));
        pixels.assign(pixels.size(), 0);  // already staged
        EXPECT_EQ(retire.pending(), 1u);
        EXPECT_EQ(retire.poll(), 0u) << "frame 1 has not completed";
      });
  ASSERT_TRUE(recorded.ok()) << recorded.message();
  ASSERT_TRUE(submitted.ok()) << submitted.message();
  // The submission has completed; the frame loop would set the value now.
  ASSERT_TRUE(timeline.value().signal(1).ok());
  EXPECT_EQ(retire.poll(), 1u);
  EXPECT_EQ(retire.pending(), 0u);
  EXPECT_EQ(read(image.value()), (Texels{0xAABBCCDD, 0x11223344}));
}

// Everything refused is refused before anything is recorded: the command
// buffer still submits clean, and the image keeps its layout.
TEST_F(ImageUpdateTest, RefusesWhatItCannotRecord) {
  auto image = make_image({2, 2});
  ASSERT_TRUE(image.ok()) << image.status().message();
  auto unsampled = make_image(
      {2, 2}, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
  ASSERT_TRUE(unsampled.ok()) << unsampled.status().message();
  auto depth = make_image(
      {2, 2}, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      VK_FORMAT_D32_SFLOAT);
  ASSERT_TRUE(depth.ok()) << depth.status().message();
  auto source = make_source(Texels(4, 0));
  ASSERT_TRUE(source.ok()) << source.status().message();
  auto not_a_source =
      make_source(Texels(4, 0), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  ASSERT_TRUE(not_a_source.ok()) << not_a_source.status().message();
  auto timeline = vkc::TimelineSemaphore::create(*device_, 0);
  ASSERT_TRUE(timeline.ok()) << timeline.status().message();
  vg::RetireQueue retire(timeline.value());

  auto pool = vkc::CommandPool::create(device(), device_->queue_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto buffer = pool.value().allocate_primary();
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();
  ASSERT_TRUE(
      buffer.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer cmd = buffer.value().handle();

  VkBufferImageCopy whole{};
  whole.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  whole.imageExtent = {2, 2, 1};
  const auto update = [&](const vkc::Buffer& from, vkc::Image& to,
                          const VkBufferImageCopy& region,
                          vg::ImageUpdateScope scope = {}) {
    return vg::record_image_update(cmd, from, to, &region, 1, scope).domain();
  };
  using Code = vkc::Status::Code;

  EXPECT_EQ(vg::record_image_update(VK_NULL_HANDLE, source.value(),
                                    image.value(), &whole, 1)
                .domain(),
            Code::InvalidArgument);
  vkc::Image empty;
  EXPECT_EQ(update(source.value(), empty, whole), Code::InvalidArgument);
  EXPECT_EQ(update(source.value(), unsampled.value(), whole),
            Code::InvalidArgument);
  EXPECT_EQ(update(source.value(), depth.value(), whole), Code::Unsupported);
  EXPECT_EQ(update(not_a_source.value(), image.value(), whole),
            Code::InvalidArgument);
  EXPECT_EQ(
      vg::record_image_update(cmd, source.value(), image.value(), &whole, 0)
          .domain(),
      Code::InvalidArgument);
  EXPECT_EQ(
      update(source.value(), image.value(), whole,
             vg::ImageUpdateScope{VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0}),
      Code::InvalidArgument);
  // No stage to wait for would drop the order against earlier reads.
  EXPECT_EQ(
      update(source.value(), image.value(), whole,
             vg::ImageUpdateScope{0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT}),
      Code::InvalidArgument);
  EXPECT_EQ(update(source.value(), image.value(), whole,
                   vg::ImageUpdateScope{VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT}),
            Code::InvalidArgument);

  VkBufferImageCopy bad = whole;
  bad.imageOffset = {1, 0, 0};  // runs off the right edge
  EXPECT_EQ(update(source.value(), image.value(), bad), Code::InvalidArgument);
  bad = whole;
  bad.imageSubresource.mipLevel = 1;
  EXPECT_EQ(update(source.value(), image.value(), bad), Code::InvalidArgument);
  bad = whole;
  bad.imageSubresource.layerCount = 2;
  EXPECT_EQ(update(source.value(), image.value(), bad), Code::InvalidArgument);
  bad = whole;
  bad.bufferRowLength = 1;  // shorter than a row
  EXPECT_EQ(update(source.value(), image.value(), bad), Code::InvalidArgument);
  bad = whole;
  bad.bufferOffset = 2;  // not a multiple of the texel size
  EXPECT_EQ(update(source.value(), image.value(), bad), Code::InvalidArgument);
  bad = whole;
  bad.bufferRowLength = 3;  // 3 + 2 texels: past the 4-texel source
  EXPECT_EQ(update(source.value(), image.value(), bad), Code::InvalidArgument);

  // The several-buffer form checks each region against its own source.
  auto one_texel = make_source(Texels(1, 0));
  ASSERT_TRUE(one_texel.ok()) << one_texel.status().message();
  std::array<vg::ImageCopy, 2> copies{};
  copies[0] = {&source.value(), whole};
  copies[1] = {&one_texel.value(), whole};  // 4 texels from a 1-texel buffer
  EXPECT_EQ(
      vg::record_image_update(cmd, copies.data(), 2, image.value()).domain(),
      Code::InvalidArgument);
  copies[1] = {nullptr, whole};
  EXPECT_EQ(
      vg::record_image_update(cmd, copies.data(), 2, image.value()).domain(),
      Code::InvalidArgument);
  EXPECT_EQ(
      vg::record_image_update(cmd, copies.data(), 0, image.value()).domain(),
      Code::InvalidArgument);
  EXPECT_EQ(vg::record_image_update(cmd, static_cast<vg::ImageCopy*>(nullptr),
                                    1, image.value())
                .domain(),
            Code::InvalidArgument);
  // A region outside the image, and two regions that write a texel in common,
  // which would race: the copies have no barrier between them.
  VkBufferImageCopy right{};
  right.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  right.imageOffset = {1, 0, 0};
  right.imageExtent = {1, 2, 1};
  copies[0] = {&source.value(), whole};
  copies[1] = {&source.value(), right};
  copies[1].region.imageOffset = {2, 0, 0};  // past the right edge
  EXPECT_EQ(
      vg::record_image_update(cmd, copies.data(), 2, image.value()).domain(),
      Code::InvalidArgument);
  copies[1].region.imageOffset = {1, 0, 0};  // inside `whole`
  EXPECT_EQ(
      vg::record_image_update(cmd, copies.data(), 2, image.value()).domain(),
      Code::InvalidArgument);
  const std::array<VkBufferImageCopy, 2> overlapping{whole, right};
  EXPECT_EQ(vg::record_image_update(cmd, source.value(), image.value(),
                                    overlapping.data(), 2)
                .domain(),
            Code::InvalidArgument);

  const Texels pixels(4, 0);
  EXPECT_EQ(vg::record_image_upload(cmd, *allocator_, retire, 1, image.value(),
                                    nullptr, 16)
                .domain(),
            Code::InvalidArgument);
  EXPECT_EQ(vg::record_image_upload(cmd, *allocator_, retire, 1, image.value(),
                                    pixels.data(), 12)
                .domain(),
            Code::InvalidArgument);
  EXPECT_EQ(retire.pending(), 0u);

  EXPECT_EQ(image.value().layout(), VK_IMAGE_LAYOUT_UNDEFINED);
  ASSERT_TRUE(buffer.value().end().ok());
  submit_and_wait(cmd);
}

}  // namespace

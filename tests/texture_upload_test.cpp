// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// Adds a VMA allocator on top of the shared device fixture (mirrors
// AllocatorTest): the derived allocator_ is destroyed before the base's
// device_/instance_, and each test's textures/buffers before any of them.
class TextureUploadTest : public VulkanDeviceTest {
 protected:
  // Records copies + subresource barriers, so run under the validation layer
  // with teeth: a wrong per-mip/layer copy region or barrier fails the test (on
  // CI, where the layer is present).
  bool wants_validation() const override { return true; }

  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;  // no device, or the base SetUp failed fatally
    }
    auto allocator = vg::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

  // Copy one (mip, layer-range) subresource of `image` -- which the upload left
  // in SHADER_READ_ONLY_OPTIMAL -- into host memory and return its tightly
  // packed bytes (layers contiguous within the mip). Proves the per-mip /
  // per-layer copy offsets landed, not just that a submit succeeded.
  std::vector<std::uint8_t> read_subresource(VkImage image, uint32_t mip,
                                             uint32_t base_layer,
                                             uint32_t layer_count,
                                             VkExtent2D mip_ext,
                                             uint32_t texel_bytes) {
    const VkDeviceSize bytes = VkDeviceSize{mip_ext.width} * mip_ext.height *
                               layer_count * texel_bytes;
    vg::BufferDesc rb;
    rb.size = bytes;
    rb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    rb.memory = vg::MemoryUsage::HostVisible;
    rb.mapped = true;
    auto readback = allocator_->create_buffer(rb);
    EXPECT_TRUE(readback.ok()) << readback.status().message();
    if (!readback.ok()) {
      return {};
    }
    const VkBuffer dst = readback.value().handle();
    const vg::Status recorded =
        device_->submit_single_time([&](VkCommandBuffer cmd) {
          vg::ImageBarrierDesc to_src;
          to_src.image = image;
          to_src.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
          to_src.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
          to_src.old_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          to_src.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          to_src.base_mip = mip;
          to_src.mip_count = 1;
          to_src.base_layer = base_layer;
          to_src.layer_count = layer_count;
          vg::cmd_image_barrier(cmd, to_src);

          VkBufferImageCopy copy{};
          copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, base_layer,
                                   layer_count};
          copy.imageExtent = {mip_ext.width, mip_ext.height, 1};
          vkCmdCopyImageToBuffer(
              cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1, &copy);
        });
    EXPECT_TRUE(recorded.ok()) << recorded.message();
    const auto* mapped =
        static_cast<const std::uint8_t*>(readback.value().mapped());
    return std::vector<std::uint8_t>(mapped, mapped + bytes);
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
  // 8x8 RGBA8 -> 4 mip levels (8, 4, 2, 1). A *solid-color* source makes every
  // generated level that same color (a linear box downsample of a uniform image
  // is that image), so reading a generated level back proves the per-level blit
  // landed content -- not just that four levels exist. Validation-with-teeth
  // covers the per-level barrier correctness alongside.
  constexpr std::uint8_t kR = 0x12, kG = 0x34, kB = 0x56, kA = 0x78;
  std::array<std::uint8_t, 8 * 8 * 4> src{};
  for (std::size_t p = 0; p < src.size(); p += 4) {
    src[p + 0] = kR;
    src[p + 1] = kG;
    src[p + 2] = kB;
    src[p + 3] = kA;
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
  ASSERT_EQ(texture.value().mip_levels(), 4u);  // 8 -> 4 -> 2 -> 1

  // Mip 2 (2x2) is reached by two successive blits down from mip 0; every texel
  // must be the source color.
  const std::vector<std::uint8_t> got =
      read_subresource(texture.value().image(), /*mip=*/2, /*base_layer=*/0,
                       /*layer_count=*/1, {2, 2}, /*texel_bytes=*/4);
  ASSERT_EQ(got.size(), std::size_t{2 * 2 * 4});
  for (std::size_t p = 0; p < got.size(); p += 4) {
    EXPECT_EQ(got[p + 0], kR) << "texel " << p / 4;
    EXPECT_EQ(got[p + 1], kG) << "texel " << p / 4;
    EXPECT_EQ(got[p + 2], kB) << "texel " << p / 4;
    EXPECT_EQ(got[p + 3], kA) << "texel " << p / 4;
  }
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

// --- Cube / array / pre-mipped uploads --------------------------------------

TEST_F(TextureUploadTest, UploadsCubeAndRoutesLayers) {
  // A 4x4 RGBA8 cube: each face filled with a distinct byte, packed layer 0..5
  // (the single-mip case of the mip-major layout).
  constexpr std::uint32_t kSize = 4;
  constexpr std::size_t kFaceBytes = std::size_t{kSize} * kSize * 4;
  std::array<std::uint8_t, kFaceBytes * 6> src{};
  for (std::size_t face = 0; face < 6; ++face) {
    for (std::size_t i = 0; i < kFaceBytes; ++i) {
      src[face * kFaceBytes + i] = static_cast<std::uint8_t>(0x11 * (face + 1));
    }
  }

  vg::ImageUploadDesc desc;
  desc.extent = {kSize, kSize};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  desc.array_layers = 6;
  desc.cube = true;

  auto texture = vg::upload_texture(*device_, *allocator_, desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_TRUE(texture.value().valid());
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);
  EXPECT_EQ(texture.value().mip_levels(), 1u);

  // Read the last face back: proves the per-layer buffer offsets landed each
  // face in its own layer, not just that the submit succeeded.
  vg::BufferDesc rb;
  rb.size = kFaceBytes;
  rb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  rb.memory = vg::MemoryUsage::HostVisible;
  rb.mapped = true;
  auto readback = allocator_->create_buffer(rb);
  ASSERT_TRUE(readback.ok()) << readback.status().message();

  const VkImage image = texture.value().image();
  const VkBuffer dst = readback.value().handle();
  auto recorded =
      device_->submit_single_time([image, dst](VkCommandBuffer cmd) {
        // upload_texture left every face in SHADER_READ_ONLY_OPTIMAL; move
        // just layer 5 to TRANSFER_SRC through the public barrier helper.
        vg::ImageBarrierDesc to_src;
        to_src.image = image;
        to_src.src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        to_src.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        to_src.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
        to_src.old_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_src.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_src.base_layer = 5;
        to_src.layer_count = 1;
        vg::cmd_image_barrier(cmd, to_src);

        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 5, 1};
        copy.imageExtent = {kSize, kSize, 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               dst, 1, &copy);
      });
  ASSERT_TRUE(recorded.ok()) << recorded.message();

  const auto* got = static_cast<const std::uint8_t*>(readback.value().mapped());
  ASSERT_NE(got, nullptr);
  for (std::size_t i = 0; i < kFaceBytes; ++i) {
    ASSERT_EQ(got[i], 0x11 * 6) << "byte " << i;
  }
}

TEST_F(TextureUploadTest, UploadsTwoDArrayAndRoutesLayers) {
  // A non-cube 2x2 RGBA8 array with three layers exercises the
  // VK_IMAGE_VIEW_TYPE_2D_ARRAY create/view path (distinct from cube) and the
  // per-layer copy routing -- no cube test covers it. Packed layer-minor within
  // the single mip.
  constexpr uint32_t kLayers = 3;
  constexpr std::size_t kLayerBytes = 2 * 2 * 4;
  std::array<std::uint8_t, kLayerBytes * kLayers> src{};
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>(i);
  }

  vg::ImageUploadDesc desc;
  desc.extent = {2, 2};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  desc.array_layers = kLayers;  // cube stays false -> a plain 2D array

  auto texture = vg::upload_texture(*device_, *allocator_, desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_TRUE(texture.value().valid());
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);

  // Each layer reads back its own slice of the packed source.
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    const std::vector<std::uint8_t> got =
        read_subresource(texture.value().image(), /*mip=*/0, layer,
                         /*layer_count=*/1, {2, 2}, /*texel_bytes=*/4);
    ASSERT_EQ(got.size(), kLayerBytes);
    for (std::size_t i = 0; i < kLayerBytes; ++i) {
      EXPECT_EQ(got[i], src[layer * kLayerBytes + i])
          << "layer " << layer << " byte " << i;
    }
  }
}

TEST_F(TextureUploadTest, UploadsPreMippedCube) {
  // A 4x4 cube with two supplied mips: [mip0: 6 faces of 4x4][mip1: 6 of 2x2].
  constexpr std::size_t kBytes = ((4 * 4) + (2 * 2)) * 4 * 6;
  std::array<std::uint8_t, kBytes> src{};
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>(i);
  }

  vg::ImageUploadDesc desc;
  desc.extent = {4, 4};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  desc.array_layers = 6;
  desc.cube = true;
  desc.mip_levels = 2;

  auto texture = vg::upload_texture(*device_, *allocator_, desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_TRUE(texture.value().valid());
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);
  EXPECT_EQ(texture.value().mip_levels(), 2u);

  // Read mip 1, face 0 (2x2) back and compare to the packed source at the
  // mip-major offset (all six faces of mip 0, then mip 1's face 0). This is the
  // only check that exercises the per-mip bufferOffset for mip >= 1 -- the size
  // validation uses the same total, so it cannot catch a scrambled offset.
  constexpr std::size_t kMip0Bytes = (4 * 4) * 4 * 6;  // six faces of mip 0
  constexpr std::size_t kMip1FaceBytes = (2 * 2) * 4;
  const std::vector<std::uint8_t> got =
      read_subresource(texture.value().image(), /*mip=*/1, /*base_layer=*/0,
                       /*layer_count=*/1, {2, 2}, /*texel_bytes=*/4);
  ASSERT_EQ(got.size(), kMip1FaceBytes);
  for (std::size_t i = 0; i < kMip1FaceBytes; ++i) {
    EXPECT_EQ(got[i], src[kMip0Bytes + i]) << "mip1 face0 byte " << i;
  }
}

TEST_F(TextureUploadTest, RejectsCubeSizeMismatch) {
  // Correct per the old single-subresource formula but short of the six-layer
  // total, so the packing-contract check must fire.
  std::array<std::uint8_t, 4 * 4 * 4> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {4, 4};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();  // one face, not six
  desc.array_layers = 6;
  desc.cube = true;
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, RejectsCubeWithoutSixLayers) {
  std::array<std::uint8_t, 4 * 4 * 4> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {4, 4};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  desc.cube = true;  // but array_layers stays 1
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, RejectsGenerateMipsOnArrayUpload) {
  std::array<std::uint8_t, 4 * 4 * 4 * 2> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {4, 4};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  desc.array_layers = 2;
  desc.generate_mips = true;  // generation is single-layer only
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, RejectsGenerateMipsWithSuppliedMips) {
  std::array<std::uint8_t, ((4 * 4) + (2 * 2)) * 4> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {4, 4};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  desc.mip_levels = 2;        // pixels carry mips already...
  desc.generate_mips = true;  // ...so generating them too is contradictory
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, RejectsMipLevelsBeyondFullChain) {
  std::array<std::uint8_t, ((4 * 4) + (2 * 2) + 1 + 1) * 4> src{};
  vg::ImageUploadDesc desc;
  desc.extent = {4, 4};  // full chain: 4, 2, 1 -> 3 levels
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = src.data();
  desc.size = src.size();
  desc.mip_levels = 4;
  EXPECT_EQ(vg::upload_texture(*device_, *allocator_, desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

// --- UploadBatch -------------------------------------------------------------

namespace {

// A 2x2 RGBA8 desc over caller-owned pixels, for exercising the batch.
vg::ImageUploadDesc small_desc(const std::array<std::uint8_t, 16>& pixels) {
  vg::ImageUploadDesc desc;
  desc.extent = {2, 2};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.pixels = pixels.data();
  desc.size = pixels.size();
  return desc;
}

// A vertex-buffer upload desc over caller-owned bytes.
vg::BufferUploadDesc buffer_desc(const void* data, VkDeviceSize size) {
  vg::BufferUploadDesc desc;
  desc.data = data;
  desc.size = size;
  desc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  return desc;
}

}  // namespace

TEST_F(TextureUploadTest, BatchUploadsManyTexturesInOneSubmit) {
  // Three textures with *distinct* content in one submit, each read back: a
  // regression that dropped a copy, submitted N-1 of N, or paired staging[i]
  // with the wrong texture fails here -- valid handles alone would not.
  std::array<std::array<std::uint8_t, 16>, 3> src{};
  for (int t = 0; t < 3; ++t) {
    for (std::size_t i = 0; i < src[t].size(); ++i) {
      src[t][i] = static_cast<std::uint8_t>(t * 16 + i);
    }
  }

  auto batch = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  EXPECT_TRUE(batch.value().valid());

  std::vector<vg::Texture> textures;
  for (int t = 0; t < 3; ++t) {
    auto texture = batch.value().add(small_desc(src[t]));
    ASSERT_TRUE(texture.ok()) << texture.status().message();
    textures.push_back(std::move(texture).value());
  }

  const vg::Status finished = batch.value().finish();
  ASSERT_TRUE(finished.ok()) << finished.message();

  // Each texture holds its own pixels -- proving per-texture copy routing.
  for (int t = 0; t < 3; ++t) {
    ASSERT_TRUE(textures[t].valid());
    EXPECT_NE(textures[t].view(), VK_NULL_HANDLE);
    const std::vector<std::uint8_t> got =
        read_subresource(textures[t].image(), /*mip=*/0, /*base_layer=*/0,
                         /*layer_count=*/1, {2, 2}, /*texel_bytes=*/4);
    ASSERT_EQ(got.size(), src[t].size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i], src[t][i]) << "texture " << t << " byte " << i;
    }
  }

  // One-shot: after finish the batch is empty and rejects further use.
  const std::array<std::uint8_t, 16> px{};
  EXPECT_FALSE(batch.value().valid());
  EXPECT_EQ(batch.value().add(small_desc(px)).status().domain(),
            vg::Status::Code::InvalidArgument);
  EXPECT_EQ(batch.value()
                .add_buffer(buffer_desc(px.data(), px.size()))
                .status()
                .domain(),
            vg::Status::Code::InvalidArgument);
  EXPECT_EQ(batch.value().finish().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(TextureUploadTest, BatchFailedAddLeavesBatchUsable) {
  const std::array<std::uint8_t, 16> px{};
  auto batch = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(batch.ok()) << batch.status().message();

  vg::ImageUploadDesc bad = small_desc(px);
  bad.size = 3;  // violates the packing contract
  EXPECT_EQ(batch.value().add(bad).status().domain(),
            vg::Status::Code::InvalidArgument);

  // The failed add recorded nothing; the batch still uploads.
  auto texture = batch.value().add(small_desc(px));
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  const vg::Status finished = batch.value().finish();
  ASSERT_TRUE(finished.ok()) << finished.message();
  EXPECT_TRUE(texture.value().valid());
}

TEST_F(TextureUploadTest, BatchMoveConstructLeavesSourceEmpty) {
  const std::array<std::uint8_t, 16> px{};
  auto batch = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  auto texture = batch.value().add(small_desc(px));
  ASSERT_TRUE(texture.ok()) << texture.status().message();

  vg::UploadBatch moved(std::move(batch).value());
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(batch.value().valid());  // NOLINT(bugprone-use-after-move)

  // The moved-into batch carried the recorded work: it still finishes.
  const vg::Status finished = moved.finish();
  ASSERT_TRUE(finished.ok()) << finished.message();
  EXPECT_TRUE(texture.value().valid());
}

TEST_F(TextureUploadTest, BatchMoveAssignOverLiveDiscardsTheOldBatch) {
  const std::array<std::uint8_t, 16> px{};
  auto dst = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(dst.ok()) << dst.status().message();
  auto discarded = dst.value().add(small_desc(px));  // outlives the discard
  ASSERT_TRUE(discarded.ok()) << discarded.status().message();

  auto src = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(src.ok()) << src.status().message();
  auto texture = src.value().add(small_desc(px));
  ASSERT_TRUE(texture.ok()) << texture.status().message();

  // Frees dst's never-submitted command buffer + staging, then adopts src's
  // (the leak/double-free path ASan checks).
  dst.value() = std::move(src).value();
  EXPECT_TRUE(dst.value().valid());
  EXPECT_FALSE(src.value().valid());  // NOLINT(bugprone-use-after-move)
  ASSERT_TRUE(dst.value().finish().ok());
  EXPECT_TRUE(texture.value().valid());
}

TEST_F(TextureUploadTest, BatchSelfMoveAssignIsSafe) {
  const std::array<std::uint8_t, 16> px{};
  auto batch = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  auto texture = batch.value().add(small_desc(px));  // must outlive finish()
  ASSERT_TRUE(texture.ok()) << texture.status().message();

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the guard
  // must keep the open batch intact.
  vg::UploadBatch* alias = &batch.value();
  batch.value() = std::move(*alias);
  EXPECT_TRUE(batch.value().valid());
  ASSERT_TRUE(batch.value().finish().ok());
  EXPECT_TRUE(texture.value().valid());
}

TEST_F(TextureUploadTest, BatchDestructorWithoutFinishDiscardsCleanly) {
  const std::array<std::uint8_t, 16> px{};
  std::vector<vg::Texture> textures;
  {
    auto batch = vg::UploadBatch::begin(*device_, *allocator_);
    ASSERT_TRUE(batch.ok()) << batch.status().message();
    auto texture = batch.value().add(small_desc(px));
    ASSERT_TRUE(texture.ok()) << texture.status().message();
    textures.push_back(std::move(texture).value());
    // No finish(): the destructor must free the staging buffers and the
    // command buffer without submitting (the sanitizers CI job proves the
    // "no leak" half). The texture stays alive but holds undefined contents.
  }
  EXPECT_TRUE(textures[0].valid());
}

// --- Buffer uploads (add_buffer / upload_buffer) -----------------------------

TEST_F(TextureUploadTest, AddBufferRejectsInvalidDescs) {
  const std::array<std::uint8_t, 4> bytes{};
  auto batch = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(batch.ok()) << batch.status().message();

  vg::BufferUploadDesc null_data = buffer_desc(nullptr, bytes.size());
  EXPECT_EQ(batch.value().add_buffer(null_data).status().domain(),
            vg::Status::Code::InvalidArgument);

  vg::BufferUploadDesc zero_size = buffer_desc(bytes.data(), 0);
  EXPECT_EQ(batch.value().add_buffer(zero_size).status().domain(),
            vg::Status::Code::InvalidArgument);

  vg::BufferUploadDesc zero_usage = buffer_desc(bytes.data(), bytes.size());
  zero_usage.usage = 0;
  EXPECT_EQ(batch.value().add_buffer(zero_usage).status().domain(),
            vg::Status::Code::InvalidArgument);

  // The failed adds recorded nothing; the batch still uploads.
  auto buffer =
      batch.value().add_buffer(buffer_desc(bytes.data(), bytes.size()));
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();
  ASSERT_TRUE(batch.value().finish().ok());
  EXPECT_TRUE(buffer.value().valid());
}

TEST_F(TextureUploadTest, MixedBatchUploadsTextureAndBufferInOneSubmit) {
  const std::array<std::uint8_t, 16> px{};
  const std::array<float, 12> vertices{};
  auto batch = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(batch.ok()) << batch.status().message();

  auto texture = batch.value().add(small_desc(px));
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  auto buffer = batch.value().add_buffer(
      buffer_desc(vertices.data(), vertices.size() * sizeof(float)));
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();

  // Not host-mapped (the staging is internal) -- a necessary but not sufficient
  // proxy for DeviceLocal residency. A positive DEVICE_LOCAL check needs a
  // memory-property accessor Buffer does not expose, and would be moot on the
  // UMA/software CI devices anyway (their single heap is device-local), so the
  // residency rests on the DeviceLocal request in add_buffer + review.
  EXPECT_EQ(buffer.value().mapped(), nullptr);
  EXPECT_EQ(buffer.value().size(), vertices.size() * sizeof(float));

  const vg::Status finished = batch.value().finish();
  ASSERT_TRUE(finished.ok()) << finished.message();
  EXPECT_TRUE(texture.value().valid());
  EXPECT_TRUE(buffer.value().valid());
}

TEST_F(TextureUploadTest, UploadBufferRoundTripsBytesThroughTheGpu) {
  constexpr std::size_t kBytes = 64;
  std::array<std::uint8_t, kBytes> src{};
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>(i * 3 + 1);
  }

  vg::BufferUploadDesc desc = buffer_desc(src.data(), src.size());
  // TRANSFER_SRC on top of the draw usage so the test can copy the
  // device-local result back out.
  desc.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  auto buffer = vg::upload_buffer(*device_, *allocator_, desc);
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();
  EXPECT_TRUE(buffer.value().valid());
  EXPECT_EQ(buffer.value().size(), src.size());
  EXPECT_EQ(buffer.value().mapped(), nullptr);  // device-local, not mapped

  // Copy back into a host-visible buffer and confirm the bytes survived the
  // staging -> device-local -> readback round trip.
  vg::BufferDesc rb;
  rb.size = src.size();
  rb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  rb.memory = vg::MemoryUsage::HostVisible;
  rb.mapped = true;
  auto readback = allocator_->create_buffer(rb);
  ASSERT_TRUE(readback.ok()) << readback.status().message();

  const VkBuffer gpu = buffer.value().handle();
  const VkBuffer dst = readback.value().handle();
  auto recorded = device_->submit_single_time([gpu, dst](VkCommandBuffer cmd) {
    VkBufferCopy region{};
    region.size = kBytes;
    vkCmdCopyBuffer(cmd, gpu, dst, 1, &region);
  });
  ASSERT_TRUE(recorded.ok()) << recorded.message();

  const auto* got = static_cast<const std::uint8_t*>(readback.value().mapped());
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(std::memcmp(got, src.data(), src.size()), 0);
}

// poison() makes finish() discard the recorded work instead of submitting it --
// the safety net for a multi-resource caller (e.g. pipelines::upload_mesh) that
// dropped a resource an earlier add recorded a copy into, where finishing would
// otherwise submit a copy referencing freed memory.
TEST_F(TextureUploadTest, PoisonedBatchFinishDiscardsWithoutSubmitting) {
  const std::array<std::uint8_t, 16> px{};
  auto batch = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  auto buffer = batch.value().add_buffer(buffer_desc(px.data(), px.size()));
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();

  batch.value().poison();
  const vg::Status finished = batch.value().finish();
  EXPECT_FALSE(finished.ok());
  EXPECT_EQ(finished.domain(), vg::Status::Code::InvalidArgument);
  EXPECT_FALSE(batch.value().valid());  // discarded, one-shot
}

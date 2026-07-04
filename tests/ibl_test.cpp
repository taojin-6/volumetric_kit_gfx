// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/ibl.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

// A pure analytic environment (a vertical color gradient): thread-safe, as
// bake_ibl's concurrent convolution requires.
glm::vec3 gradient_env(const glm::vec3& dir) {
  return glm::vec3(0.5f + 0.5f * dir.y, 0.25f, 0.75f - 0.5f * dir.y);
}

// Every texture a few texels and a handful of samples, so a full set bakes in
// milliseconds.
pipelines::IblBakeDesc tiny_desc() {
  pipelines::IblBakeDesc d;
  d.irradiance_size = 4;
  d.irradiance_sample_delta = 0.5f;
  d.prefilter_size = 8;
  d.prefilter_mip_levels = 2;
  d.prefilter_samples = 4;
  d.brdf_lut_size = 8;
  d.brdf_lut_samples = 8;
  return d;
}

// Adds a VMA allocator on top of the shared device fixture, plus a
// whole-texture readback so two bakes can be compared byte for byte.
class IblTest : public VulkanDeviceTest {
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

  // Copy every (mip, layer) of `texture` (left in SHADER_READ_ONLY_OPTIMAL by
  // the bake) into host memory, packed mip-major like ImageUploadDesc. Fails
  // the current test and returns empty on any error.
  std::vector<uint8_t> read_back(const vg::Texture& texture, uint32_t layers,
                                 uint32_t texel_bytes) {
    const uint32_t mips = texture.mip_levels();
    const VkExtent2D extent = texture.extent();
    VkDeviceSize total = 0;
    for (uint32_t m = 0; m < mips; ++m) {
      const uint32_t w = std::max(extent.width >> m, 1u);
      const uint32_t h = std::max(extent.height >> m, 1u);
      total += VkDeviceSize{w} * h * layers * texel_bytes;
    }

    vg::BufferDesc rb;
    rb.size = total;
    rb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    rb.memory = vg::MemoryUsage::HostVisible;
    rb.mapped = true;
    auto readback = allocator_->create_buffer(rb);
    EXPECT_TRUE(readback.ok()) << readback.status().message();
    if (!readback.ok()) {
      return {};
    }

    const VkImage image = texture.image();
    const VkBuffer dst = readback.value().handle();
    const auto recorded = device_->submit_single_time([&](VkCommandBuffer cmd) {
      VkImageMemoryBarrier to_src{};
      to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      to_src.srcAccessMask = 0;
      to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      to_src.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      to_src.image = image;
      to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, layers};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &to_src);

      std::vector<VkBufferImageCopy> copies(mips);
      VkDeviceSize offset = 0;
      for (uint32_t m = 0; m < mips; ++m) {
        const uint32_t w = std::max(extent.width >> m, 1u);
        const uint32_t h = std::max(extent.height >> m, 1u);
        copies[m] = {};
        copies[m].bufferOffset = offset;
        copies[m].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, layers};
        copies[m].imageExtent = {w, h, 1};
        offset += VkDeviceSize{w} * h * layers * texel_bytes;
      }
      vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             dst, mips, copies.data());
    });
    EXPECT_TRUE(recorded.ok()) << recorded.message();
    if (!recorded.ok()) {
      return {};
    }

    std::vector<uint8_t> bytes(total);
    std::memcpy(bytes.data(), readback.value().mapped(), bytes.size());
    return bytes;
  }

  std::optional<vg::Allocator> allocator_;
};

}  // namespace

// The desc defaults are the constants the example's verified bake used; a
// drift here silently changes every consumer's lighting.
TEST(IblBakeDescTest, DefaultsMatchTheVerifiedExampleConstants) {
  const pipelines::IblBakeDesc d;
  EXPECT_EQ(d.irradiance_size, 16u);
  EXPECT_EQ(d.irradiance_sample_delta, 0.1f);
  EXPECT_EQ(d.prefilter_size, 64u);
  EXPECT_EQ(d.prefilter_mip_levels, 5u);
  EXPECT_EQ(d.prefilter_samples, 64u);
  EXPECT_EQ(d.brdf_lut_size, 128u);
  EXPECT_EQ(d.brdf_lut_samples, 256u);
}

TEST(IblMapsTest, DefaultIsEmpty) {
  const pipelines::IblMaps maps;
  EXPECT_FALSE(maps.valid());
  EXPECT_EQ(maps.prefilter_max_lod, 0.0f);
  const pipelines::PbrSceneDesc desc = maps.scene_desc();
  EXPECT_EQ(desc.irradiance, VK_NULL_HANDLE);
  EXPECT_EQ(desc.prefilter, VK_NULL_HANDLE);
  EXPECT_EQ(desc.brdf_lut, VK_NULL_HANDLE);
  EXPECT_EQ(desc.sampler, VK_NULL_HANDLE);
}

TEST_F(IblTest, BakesTinyMapSet) {
  auto baked =
      pipelines::bake_ibl(*device_, *allocator_, gradient_env, tiny_desc());
  ASSERT_TRUE(baked.ok()) << baked.status().message();
  const pipelines::IblMaps maps = std::move(baked).value();

  EXPECT_TRUE(maps.valid());
  EXPECT_TRUE(maps.irradiance.valid());
  EXPECT_EQ(maps.irradiance.extent().width, 4u);
  EXPECT_EQ(maps.irradiance.extent().height, 4u);
  EXPECT_EQ(maps.irradiance.mip_levels(), 1u);
  EXPECT_TRUE(maps.prefilter.valid());
  EXPECT_EQ(maps.prefilter.extent().width, 8u);
  EXPECT_EQ(maps.prefilter.mip_levels(), 2u);
  EXPECT_EQ(maps.prefilter_max_lod, 1.0f);  // mip count - 1
  EXPECT_TRUE(maps.brdf_lut.valid());
  EXPECT_EQ(maps.brdf_lut.extent().width, 8u);
  EXPECT_EQ(maps.brdf_lut.format(), VK_FORMAT_R16G16_SFLOAT);
  ASSERT_TRUE(maps.sampler.has_value());
  EXPECT_TRUE(maps.sampler->valid());
}

TEST_F(IblTest, SceneDescNamesEveryViewAndTheSampler) {
  auto baked =
      pipelines::bake_ibl(*device_, *allocator_, gradient_env, tiny_desc());
  ASSERT_TRUE(baked.ok()) << baked.status().message();
  const pipelines::PbrSceneDesc desc = baked.value().scene_desc();
  EXPECT_EQ(desc.irradiance, baked.value().irradiance.view());
  EXPECT_EQ(desc.prefilter, baked.value().prefilter.view());
  EXPECT_EQ(desc.brdf_lut, baked.value().brdf_lut.view());
  EXPECT_EQ(desc.sampler, baked.value().sampler->handle());
  EXPECT_NE(desc.irradiance, VK_NULL_HANDLE);
  EXPECT_NE(desc.prefilter, VK_NULL_HANDLE);
  EXPECT_NE(desc.brdf_lut, VK_NULL_HANDLE);
  EXPECT_NE(desc.sampler, VK_NULL_HANDLE);
}

TEST_F(IblTest, BakesBrdfLutStandalone) {
  auto lut = pipelines::bake_brdf_lut(*device_, *allocator_, 8, 8);
  ASSERT_TRUE(lut.ok()) << lut.status().message();
  EXPECT_TRUE(lut.value().valid());
  EXPECT_NE(lut.value().view(), VK_NULL_HANDLE);
  EXPECT_EQ(lut.value().extent().width, 8u);
  EXPECT_EQ(lut.value().extent().height, 8u);
  EXPECT_EQ(lut.value().format(), VK_FORMAT_R16G16_SFLOAT);
}

TEST_F(IblTest, BrdfLutRejectsZeroSizeZeroSamplesAndEmptyBatch) {
  EXPECT_EQ(
      pipelines::bake_brdf_lut(*device_, *allocator_, 0).status().domain(),
      vg::Status::Code::InvalidArgument);
  EXPECT_EQ(
      pipelines::bake_brdf_lut(*device_, *allocator_, 8, 0).status().domain(),
      vg::Status::Code::InvalidArgument);
  vg::UploadBatch empty;  // never begun
  EXPECT_EQ(pipelines::bake_brdf_lut(empty, 8, 8).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(IblTest, RejectsInvalidBakeDesc) {
  const auto domain_for = [&](const pipelines::IblBakeDesc& d) {
    return pipelines::bake_ibl(*device_, *allocator_, gradient_env, d)
        .status()
        .domain();
  };

  pipelines::IblBakeDesc d = tiny_desc();
  d.irradiance_size = 0;
  EXPECT_EQ(domain_for(d), vg::Status::Code::InvalidArgument);

  d = tiny_desc();
  d.prefilter_size = 0;
  EXPECT_EQ(domain_for(d), vg::Status::Code::InvalidArgument);

  d = tiny_desc();
  d.brdf_lut_size = 0;
  EXPECT_EQ(domain_for(d), vg::Status::Code::InvalidArgument);

  d = tiny_desc();
  d.irradiance_sample_delta = 0.0f;
  EXPECT_EQ(domain_for(d), vg::Status::Code::InvalidArgument);

  d = tiny_desc();
  d.prefilter_samples = 0;
  EXPECT_EQ(domain_for(d), vg::Status::Code::InvalidArgument);

  d = tiny_desc();
  d.brdf_lut_samples = 0;
  EXPECT_EQ(domain_for(d), vg::Status::Code::InvalidArgument);

  d = tiny_desc();
  d.prefilter_mip_levels = 0;
  EXPECT_EQ(domain_for(d), vg::Status::Code::InvalidArgument);

  d = tiny_desc();  // an 8x8 base carries at most 4 mips (8, 4, 2, 1)
  d.prefilter_mip_levels = 5;
  EXPECT_EQ(domain_for(d), vg::Status::Code::InvalidArgument);
}

TEST_F(IblTest, RejectsNullEnvironment) {
  auto baked = pipelines::bake_ibl(
      *device_, *allocator_, pipelines::EnvironmentSampler{}, tiny_desc());
  ASSERT_FALSE(baked.ok());
  EXPECT_EQ(baked.status().domain(), vg::Status::Code::InvalidArgument);
}

// Two identical bakes must produce identical texels: the concurrent per-face
// convolution may schedule differently between runs, but each block's math and
// the packing order are fixed, so the bytes may not drift.
TEST_F(IblTest, RebakeIsByteIdentical) {
  const pipelines::IblBakeDesc desc = tiny_desc();
  auto first = pipelines::bake_ibl(*device_, *allocator_, gradient_env, desc);
  auto second = pipelines::bake_ibl(*device_, *allocator_, gradient_env, desc);
  ASSERT_TRUE(first.ok()) << first.status().message();
  ASSERT_TRUE(second.ok()) << second.status().message();

  // RGBA16F cubes are 8 bytes/texel over 6 layers; the RG16F LUT is 4.
  EXPECT_EQ(read_back(first.value().irradiance, 6, 8),
            read_back(second.value().irradiance, 6, 8));
  EXPECT_EQ(read_back(first.value().prefilter, 6, 8),
            read_back(second.value().prefilter, 6, 8));
  EXPECT_EQ(read_back(first.value().brdf_lut, 1, 4),
            read_back(second.value().brdf_lut, 1, 4));
  EXPECT_FALSE(read_back(first.value().irradiance, 6, 8).empty());
}

TEST_F(IblTest, MoveLeavesSourceEmpty) {
  auto baked =
      pipelines::bake_ibl(*device_, *allocator_, gradient_env, tiny_desc());
  ASSERT_TRUE(baked.ok()) << baked.status().message();
  pipelines::IblMaps source = std::move(baked).value();
  ASSERT_TRUE(source.valid());

  pipelines::IblMaps moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.prefilter_max_lod, 1.0f);
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(source.irradiance.valid());
  EXPECT_FALSE(source.sampler.has_value());
  EXPECT_EQ(source.prefilter_max_lod, 0.0f);
  EXPECT_EQ(source.scene_desc().sampler, VK_NULL_HANDLE);
}

TEST_F(IblTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto a =
      pipelines::bake_ibl(*device_, *allocator_, gradient_env, tiny_desc());
  auto b =
      pipelines::bake_ibl(*device_, *allocator_, gradient_env, tiny_desc());
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  pipelines::IblMaps dst = std::move(a).value();
  pipelines::IblMaps src = std::move(b).value();

  dst = std::move(src);  // frees dst's maps, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_EQ(dst.prefilter_max_lod, 1.0f);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(src.sampler.has_value());
  EXPECT_EQ(src.prefilter_max_lod, 0.0f);
}

TEST_F(IblTest, SelfMoveAssignIsSafe) {
  auto baked =
      pipelines::bake_ibl(*device_, *allocator_, gradient_env, tiny_desc());
  ASSERT_TRUE(baked.ok()) << baked.status().message();
  pipelines::IblMaps maps = std::move(baked).value();

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must keep the maps intact.
  pipelines::IblMaps* alias = &maps;
  maps = std::move(*alias);
  EXPECT_TRUE(maps.valid());
  EXPECT_EQ(maps.prefilter_max_lod, 1.0f);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// image_barrier: the public header stands alone (included first, before
// anything that could mask a missing include), and the desc-driven
// cmd_image_barrier addresses explicit mip/layer sub-ranges of an array image.

#include "volumetric_kit/gfx/core/image_barrier.hpp"

#include <gtest/gtest.h>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/core/log.hpp"

namespace {

class ImageBarrierTest : public vg_test::RendererDeviceTest {
 protected:
  // Records real barriers, so run under the validation layer: a wrong
  // subresource range, stage or layout fails the test wherever the layer is
  // installed.
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::On;
  }
};

}  // namespace

TEST_F(ImageBarrierTest, TransitionsExplicitLayerRanges) {
  // A four-layer 2D array image, transitioned in per-range steps: layers 1..2
  // first, then the outer layers, then the whole image at once through the
  // remaining-mips/layers defaults.
  vkc::ImageDesc desc;
  desc.extent = {8, 8};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  desc.array_layers = 4;
  auto texture = allocator().create_image(desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();

  const VkImage image = texture.value().handle();
  auto recorded = device().submit_single_time([image](VkCommandBuffer cmd) {
    vg::ImageBarrierDesc to_dst;
    to_dst.image = image;
    to_dst.src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    to_dst.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    to_dst.dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_dst.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // Middle layers first...
    to_dst.base_layer = 1;
    to_dst.layer_count = 2;
    vg::cmd_image_barrier(cmd, to_dst);
    // ...then each outer layer by itself.
    to_dst.base_layer = 0;
    to_dst.layer_count = 1;
    vg::cmd_image_barrier(cmd, to_dst);
    to_dst.base_layer = 3;
    vg::cmd_image_barrier(cmd, to_dst);

    // Every layer now agrees on TRANSFER_DST, so the defaults
    // (VK_REMAINING_MIP_LEVELS / VK_REMAINING_ARRAY_LAYERS) can move the whole
    // image to SHADER_READ in one barrier.
    vg::ImageBarrierDesc to_read;
    to_read.image = image;
    to_read.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    to_read.dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    to_read.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_read.dst_access = VK_ACCESS_SHADER_READ_BIT;
    to_read.old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_read.new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vg::cmd_image_barrier(cmd, to_read);
  });
  ASSERT_TRUE(recorded.ok()) << recorded.message();
}

TEST_F(ImageBarrierTest, BatchesTransitionsIntoOneBarrier) {
  // Two images and two mip ranges of one, as a finished mip chain hands its
  // levels over: each keeps its own range, layouts and access masks, under
  // the union of their stage masks.
  vkc::ImageDesc desc;
  desc.extent = {8, 8};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  desc.mip_levels = 4;
  auto chain = allocator().create_image(desc);
  ASSERT_TRUE(chain.ok()) << chain.status().message();
  desc.mip_levels = 1;
  auto plain = allocator().create_image(desc);
  ASSERT_TRUE(plain.ok()) << plain.status().message();

  const VkImage a = chain.value().handle();
  const VkImage b = plain.value().handle();
  auto recorded = device().submit_single_time([a, b](VkCommandBuffer cmd) {
    vg::ImageBarrierDesc to_dst[2];
    for (vg::ImageBarrierDesc& d : to_dst) {
      d.src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      d.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      d.dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
      d.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
      d.new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    }
    to_dst[0].image = a;
    to_dst[1].image = b;
    vg::cmd_image_barriers(cmd, to_dst, 2);

    // Levels 0..2 of the chain as a blit source, then every level and the
    // other image to SHADER_READ: three ranges, two old layouts, one call.
    vg::ImageBarrierDesc to_src = to_dst[0];
    to_src.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    to_src.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_src.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
    to_src.old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_src.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.mip_count = 3;
    vg::cmd_image_barrier(cmd, to_src);

    vg::ImageBarrierDesc to_read[3];
    for (vg::ImageBarrierDesc& d : to_read) {
      d.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      d.dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      d.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
      d.dst_access = VK_ACCESS_SHADER_READ_BIT;
      d.old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      d.new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    to_read[0].image = a;
    to_read[0].old_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_read[0].src_access = VK_ACCESS_TRANSFER_READ_BIT;
    to_read[0].mip_count = 3;
    to_read[1].image = a;
    to_read[1].base_mip = 3;
    to_read[1].mip_count = 1;
    to_read[2].image = b;
    vg::cmd_image_barriers(cmd, to_read, 3);
    vg::cmd_image_barriers(cmd, to_read, 0);  // records nothing
  });
  ASSERT_TRUE(recorded.ok()) << recorded.message();
}

// new_layout has no valid default: cmd_image_barrier VKC_CHECKs it rather than
// silently recording an UNDEFINED -> UNDEFINED no-op. The check fires before
// any Vulkan call, so no device is needed; the "DeathTest" suffix runs it
// isolated.
TEST(ImageBarrierDeathTest, RejectsUnsetNewLayout) {
  // new_layout defaults to the placeholder UNDEFINED (the invalid target).
  vg::ImageBarrierDesc desc;
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});  // route the abort message to stderr
        vg::cmd_image_barrier(VK_NULL_HANDLE, desc);
      },
      "new_layout must be set");
}

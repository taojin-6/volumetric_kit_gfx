// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// image_barrier: the public header stands alone (included first, before
// anything that could mask a missing include), and the desc-driven
// cmd_image_barrier addresses explicit mip/layer sub-ranges of an array image.

#include "volumetric_kit/gfx/core/image_barrier.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// Adds a VMA allocator on top of the shared device fixture (mirrors
// TextureUploadTest) to create the images the barriers address.
class ImageBarrierTest : public VulkanDeviceTest {
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

TEST_F(ImageBarrierTest, TransitionsExplicitLayerRanges) {
  // A four-layer 2D array image, transitioned in per-range steps: layers 1..2
  // first, then the outer layers, then the whole image at once through the
  // remaining-mips/layers defaults.
  vg::TextureDesc desc;
  desc.extent = {8, 8};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  desc.array_layers = 4;
  auto texture = allocator_->create_image(desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();

  const VkImage image = texture.value().image();
  auto recorded = device_->submit_single_time([image](VkCommandBuffer cmd) {
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

// new_layout has no valid default: cmd_image_barrier VG_CHECKs it rather than
// silently recording an UNDEFINED -> UNDEFINED no-op. The check fires before
// any Vulkan call, so no device is needed; the "DeathTest" suffix runs it
// isolated.
TEST(ImageBarrierDeathTest, RejectsUnsetNewLayout) {
  // new_layout defaults to the placeholder UNDEFINED (the invalid target).
  vg::ImageBarrierDesc desc;
  EXPECT_DEATH(
      {
        vg::set_log_handler({});  // route the abort message to stderr
        vg::cmd_image_barrier(VK_NULL_HANDLE, desc);
      },
      "new_layout must be set");
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// OffscreenTarget: empty/default, argument validation, the move-only lifecycle
// (it owns Texture(s) + a readback Buffer), and an end-to-end
// clear-and-readback through dynamic rendering -- the headless path
// golden-image render tests build on. The clear-and-readback absorbs the former
// render_readback_test, now expressed through the RenderTarget/OffscreenTarget
// abstraction.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

TEST(OffscreenTargetTest, DefaultConstructedIsEmpty) {
  vg::OffscreenTarget target;
  EXPECT_FALSE(target.valid());
  EXPECT_EQ(target.pixels(), nullptr);
}

class OffscreenTargetDeviceTest : public VulkanDeviceTest {
 protected:
  vg::Allocator make_allocator() {
    auto allocator = vg::Allocator::create(instance_->handle(), *device_);
    EXPECT_TRUE(allocator.ok()) << allocator.status().message();
    return std::move(allocator).value();
  }

  vg::OffscreenTarget make_target(vg::Allocator& allocator,
                                  VkExtent2D extent = {32, 32}) {
    vg::OffscreenTargetDesc desc;
    desc.extent = extent;
    desc.color_format = kFormat;
    auto target = vg::OffscreenTarget::create(allocator, desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    return std::move(target).value();
  }
};

// --- Validation: rejected before any allocation ----------------------------

TEST_F(OffscreenTargetDeviceTest, ZeroExtentRejected) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTargetDesc desc;
  desc.extent = {0, 0};
  desc.color_format = kFormat;
  auto target = vg::OffscreenTarget::create(allocator, desc);
  ASSERT_FALSE(target.ok());
  EXPECT_EQ(target.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(OffscreenTargetDeviceTest, UndefinedColorFormatRejected) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTargetDesc desc;
  desc.extent = {32, 32};
  desc.color_format = VK_FORMAT_UNDEFINED;
  auto target = vg::OffscreenTarget::create(allocator, desc);
  ASSERT_FALSE(target.ok());
  EXPECT_EQ(target.status().domain(), vg::Status::Code::InvalidArgument);
}

// --- Move-only lifecycle ---------------------------------------------------

TEST_F(OffscreenTargetDeviceTest, MoveConstructLeavesSourceEmpty) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTarget source = make_target(allocator);
  ASSERT_TRUE(source.valid());

  vg::OffscreenTarget moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.pixels(), nullptr);
}

TEST_F(OffscreenTargetDeviceTest, MoveAssignOverLiveLeavesSourceEmpty) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTarget dst = make_target(allocator);
  vg::OffscreenTarget src = make_target(allocator);

  dst = std::move(src);  // frees dst's images + buffer, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(OffscreenTargetDeviceTest, SelfMoveAssignIsSafe) {
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTarget target = make_target(allocator);

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the
  // composed members' self-move guards must keep the target intact.
  vg::OffscreenTarget* alias = &target;
  target = std::move(*alias);
  EXPECT_TRUE(target.valid());
}

// --- Clear + readback through dynamic rendering ----------------------------

TEST_F(OffscreenTargetDeviceTest, ClearsAndReadsBackThroughDynamicRendering) {
  constexpr uint32_t kSize = 4;
  vg::Allocator allocator = make_allocator();
  vg::OffscreenTarget target = make_target(allocator, {kSize, kSize});

  auto pool = vg::CommandPool::create(device(), device_->graphics_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();

  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();

  // Dynamic rendering does not transition the image; the caller moves it from
  // UNDEFINED to COLOR_ATTACHMENT_OPTIMAL before begin().
  VkImageMemoryBarrier to_color{};
  to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_color.srcAccessMask = 0;
  to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.image = target.color_image();
  to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(raw, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &to_color);

  // loadOp CLEAR fills the attachment with opaque red; no draw needed.
  vg::RenderTargetBeginInfo begin_info;
  begin_info.clear_color.float32[0] = 1.0f;  // R
  begin_info.clear_color.float32[3] = 1.0f;  // A
  const vg::RenderTarget rt = target.target();
  rt.begin(raw, begin_info);
  rt.end(raw);

  target.record_readback(raw);
  ASSERT_TRUE(cmd.value().end().ok());
  submit_and_wait(raw);

  const auto* px = static_cast<const uint8_t*>(target.pixels());
  ASSERT_NE(px, nullptr);
  // The whole image was cleared to opaque red, so the first and last texels
  // match.
  EXPECT_EQ(px[0], 255);  // R
  EXPECT_EQ(px[1], 0);    // G
  EXPECT_EQ(px[2], 0);    // B
  EXPECT_EQ(px[3], 255);  // A
  const size_t last = (static_cast<size_t>(kSize) * kSize - 1) * 4;
  EXPECT_EQ(px[last + 0], 255);
  EXPECT_EQ(px[last + 3], 255);
}

// --- Optional depth attachment ---------------------------------------------

TEST_F(OffscreenTargetDeviceTest, DepthFormatAddsDepthAttachment) {
  vg::Allocator allocator = make_allocator();

  // Color-only (the default): no depth image, and the layout reports no depth.
  vg::OffscreenTarget color_only = make_target(allocator);
  EXPECT_EQ(color_only.depth_image(), VK_NULL_HANDLE);
  EXPECT_EQ(color_only.layout().depth_format, VK_FORMAT_UNDEFINED);

  // With a depth format: a depth image is allocated and surfaced both in the
  // target's layout and in the RenderTarget it hands out.
  vg::OffscreenTargetDesc desc;
  desc.extent = {32, 32};
  desc.color_format = kFormat;
  desc.depth_format = VK_FORMAT_D32_SFLOAT;
  auto target = vg::OffscreenTarget::create(allocator, desc);
  ASSERT_TRUE(target.ok()) << target.status().message();
  EXPECT_NE(target.value().depth_image(), VK_NULL_HANDLE);
  EXPECT_EQ(target.value().layout().depth_format, VK_FORMAT_D32_SFLOAT);
  EXPECT_EQ(target.value().target().layout().depth_format,
            VK_FORMAT_D32_SFLOAT);
}

}  // namespace

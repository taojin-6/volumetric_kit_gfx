// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// OffscreenTarget: empty/default, argument validation, the move-only lifecycle
// (it owns Image(s) + a readback Buffer), and an end-to-end
// clear-and-readback through dynamic rendering -- the headless path
// golden-image render tests build on. The clear-and-readback absorbs the former
// render_readback_test, now expressed through the RenderTarget/OffscreenTarget
// abstraction. A multisampled target is laid out, read back and reused under
// synchronization validation.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"

namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

TEST(OffscreenTargetTest, DefaultConstructedIsEmpty) {
  vg::OffscreenTarget target;
  EXPECT_FALSE(target.valid());
  EXPECT_EQ(target.pixels(), nullptr);
}

class OffscreenTargetDeviceTest : public vg_test::RendererDeviceTest {
 protected:
  // Records attachment/readback barriers, so run under the validation layer:
  // a wrong color/depth transition fails the test.
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::On;
  }

  vg::OffscreenTarget make_target(VkExtent2D extent = {32, 32}) {
    vg::OffscreenTargetDesc desc;
    desc.extent = extent;
    desc.color_format = kFormat;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    // Empty on failure: fail cleanly, never abort.
    return target.ok() ? std::move(target).value() : vg::OffscreenTarget{};
  }

  // A target that also owns a depth attachment, for exercising the depth member
  // through the move-only lifecycle.
  vg::OffscreenTarget make_depth_target(VkExtent2D extent = {32, 32}) {
    vg::OffscreenTargetDesc desc;
    desc.extent = extent;
    desc.color_format = kFormat;
    desc.depth_format = VK_FORMAT_D32_SFLOAT;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    EXPECT_TRUE(target.ok()) << target.status().message();
    // Empty on failure: fail cleanly, never abort.
    return target.ok() ? std::move(target).value() : vg::OffscreenTarget{};
  }
};

// --- Validation: rejected before any allocation ----------------------------

TEST_F(OffscreenTargetDeviceTest, ZeroExtentRejected) {
  vg::OffscreenTargetDesc desc;
  desc.extent = {0, 0};
  desc.color_format = kFormat;
  auto target = vg::OffscreenTarget::create(allocator(), desc);
  ASSERT_FALSE(target.ok());
  EXPECT_EQ(target.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(OffscreenTargetDeviceTest, UndefinedColorFormatRejected) {
  vg::OffscreenTargetDesc desc;
  desc.extent = {32, 32};
  desc.color_format = VK_FORMAT_UNDEFINED;
  auto target = vg::OffscreenTarget::create(allocator(), desc);
  ASSERT_FALSE(target.ok());
  EXPECT_EQ(target.status().domain(), vkc::Status::Code::InvalidArgument);
}

// A vendor extension's format has no texel size in the core's format table
// (core Vulkan and KHR only), so its readback buffer cannot be sized. Passed by
// value: VK_FORMAT_R16G16_SFIXED5_NV, which older headers do not name.
TEST_F(OffscreenTargetDeviceTest, ReadbackOfVendorFormatUnsupported) {
  vg::OffscreenTargetDesc desc;
  desc.extent = {32, 32};
  desc.color_format = static_cast<VkFormat>(1000464000);
  auto target = vg::OffscreenTarget::create(allocator(), desc);
  ASSERT_FALSE(target.ok());
  EXPECT_EQ(target.status().domain(), vkc::Status::Code::Unsupported);
}

// --- Move-only lifecycle ---------------------------------------------------

TEST_F(OffscreenTargetDeviceTest, MoveConstructLeavesSourceEmpty) {
  vg::OffscreenTarget source = make_target();
  ASSERT_TRUE(source.valid());

  vg::OffscreenTarget moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.pixels(), nullptr);
}

TEST_F(OffscreenTargetDeviceTest, MoveAssignOverLiveLeavesSourceEmpty) {
  vg::OffscreenTarget dst = make_target();
  vg::OffscreenTarget src = make_target();

  dst = std::move(src);  // frees dst's images + buffer, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(OffscreenTargetDeviceTest, SelfMoveAssignIsSafe) {
  vg::OffscreenTarget target = make_target();

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the
  // composed members' self-move guards must keep the target intact.
  vg::OffscreenTarget* alias = &target;
  target = std::move(*alias);
  EXPECT_TRUE(target.valid());
}

TEST_F(OffscreenTargetDeviceTest,
       MoveConstructCarriesDepthAndLeavesSourceEmpty) {
  vg::OffscreenTarget source = make_depth_target();
  ASSERT_TRUE(source.valid());
  ASSERT_NE(source.depth_image(), VK_NULL_HANDLE);

  vg::OffscreenTarget moved(std::move(source));
  EXPECT_NE(moved.depth_image(), VK_NULL_HANDLE);  // depth transferred...
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.depth_image(), VK_NULL_HANDLE);  // ...and cleared in source
}

TEST_F(OffscreenTargetDeviceTest,
       MoveAssignOverLiveDepthTargetLeavesSourceEmpty) {
  vg::OffscreenTarget dst = make_depth_target();
  vg::OffscreenTarget src = make_depth_target();

  dst = std::move(src);  // frees dst's depth + color, then adopts src's
  EXPECT_NE(dst.depth_image(), VK_NULL_HANDLE);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(src.depth_image(), VK_NULL_HANDLE);
}

// --- Clear + readback through dynamic rendering ----------------------------

TEST_F(OffscreenTargetDeviceTest, ClearsAndReadsBackThroughDynamicRendering) {
  constexpr uint32_t kSize = 4;
  vg::OffscreenTarget target = make_target({kSize, kSize});

  auto pool =
      vkc::CommandPool::create(device().handle(), device().queue_family());
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
  const vkc::Status submitted = device().submit_and_wait(raw);
  ASSERT_TRUE(submitted.ok()) << submitted.message();

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

// prepare() records the same transitions itself -- including the depth image
// when the target has one -- so the render needs no hand-written barriers.
TEST_F(OffscreenTargetDeviceTest, PrepareReplacesHandWrittenBarriers) {
  constexpr uint32_t kSize = 4;
  vg::OffscreenTarget target = make_depth_target({kSize, kSize});

  auto pool =
      vkc::CommandPool::create(device().handle(), device().queue_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();

  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();

  target.prepare(raw);  // color + depth -> attachment layouts

  // loadOp CLEAR fills the attachments (opaque green here); no draw needed.
  vg::RenderTargetBeginInfo begin_info;
  begin_info.clear_color.float32[1] = 1.0f;  // G
  begin_info.clear_color.float32[3] = 1.0f;  // A
  const vg::RenderTarget rt = target.target();
  rt.begin(raw, begin_info);
  rt.end(raw);

  target.record_readback(raw);
  ASSERT_TRUE(cmd.value().end().ok());
  const vkc::Status submitted = device().submit_and_wait(raw);
  ASSERT_TRUE(submitted.ok()) << submitted.message();

  const auto* px = static_cast<const uint8_t*>(target.pixels());
  ASSERT_NE(px, nullptr);
  EXPECT_EQ(px[0], 0);    // R
  EXPECT_EQ(px[1], 255);  // G
  EXPECT_EQ(px[2], 0);    // B
  EXPECT_EQ(px[3], 255);  // A
  const size_t last = (static_cast<size_t>(kSize) * kSize - 1) * 4;
  EXPECT_EQ(px[last + 1], 255);
  EXPECT_EQ(px[last + 3], 255);
}

// --- Optional depth attachment ---------------------------------------------

TEST_F(OffscreenTargetDeviceTest, DepthFormatAddsDepthAttachment) {
  // Color-only (the default): no depth image, and the layout reports no depth.
  vg::OffscreenTarget color_only = make_target();
  EXPECT_EQ(color_only.depth_image(), VK_NULL_HANDLE);
  EXPECT_EQ(color_only.layout().depth_format, VK_FORMAT_UNDEFINED);

  // With a depth format: a depth image is allocated and surfaced both in the
  // target's layout and in the RenderTarget it hands out.
  vg::OffscreenTargetDesc desc;
  desc.extent = {32, 32};
  desc.color_format = kFormat;
  desc.depth_format = VK_FORMAT_D32_SFLOAT;
  auto target = vg::OffscreenTarget::create(allocator(), desc);
  ASSERT_TRUE(target.ok()) << target.status().message();
  EXPECT_NE(target.value().depth_image(), VK_NULL_HANDLE);
  EXPECT_EQ(target.value().layout().depth_format, VK_FORMAT_D32_SFLOAT);
  EXPECT_EQ(target.value().target().layout().depth_format,
            VK_FORMAT_D32_SFLOAT);
}

// --- Multisampled --------------------------------------------------------

// offscreen_desc lays a target out as a layout, samples included; any other
// field keeps its default.
TEST(OffscreenDescTest, CopiesTheLayoutsFormatsAndSamples) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = VK_FORMAT_B8G8R8A8_SRGB;
  layout.color_count = 1;
  layout.depth_format = VK_FORMAT_D32_SFLOAT;
  layout.samples = vg::kWindowSamples;
  const vg::OffscreenTargetDesc desc = vg::offscreen_desc(layout, {40, 30});
  EXPECT_EQ(desc.extent.width, 40u);
  EXPECT_EQ(desc.extent.height, 30u);
  EXPECT_EQ(desc.color_format, VK_FORMAT_B8G8R8A8_SRGB);
  EXPECT_EQ(desc.depth_format, VK_FORMAT_D32_SFLOAT);
  EXPECT_EQ(desc.samples, vg::kWindowSamples);
  EXPECT_TRUE(desc.readback);
}

class MultisampledOffscreenTargetTest : public vg_test::RendererDeviceTest {
 protected:
  // begin() records the multisampled images' barriers itself, so run under
  // synchronization validation: a scope the barrier fails to order after the
  // previous one fails the test.
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::Sync;
  }

  vkc::Result<vg::OffscreenTarget> create(
      VkSampleCountFlagBits samples, VkFormat color = kFormat,
      VkFormat depth = VK_FORMAT_D32_SFLOAT) {
    vg::OffscreenTargetDesc desc;
    desc.extent = {kSize, kSize};
    desc.color_format = color;
    desc.depth_format = depth;
    desc.samples = samples;
    return vg::OffscreenTarget::create(allocator(), desc);
  }

  // Records one scope per colour in `clears` into `target`, each prepared and
  // cleared to that opaque colour, then the readback; submits and waits.
  void clear_scopes(const vg::OffscreenTarget& target,
                    const std::vector<std::array<float, 3>>& clears) {
    const vkc::Status s = device().submit_single_time([&](VkCommandBuffer cmd) {
      for (const std::array<float, 3>& rgb : clears) {
        target.prepare(cmd);
        vg::RenderTargetBeginInfo begin;
        begin.clear_color.float32[0] = rgb[0];
        begin.clear_color.float32[1] = rgb[1];
        begin.clear_color.float32[2] = rgb[2];
        begin.clear_color.float32[3] = 1.0f;
        const vg::RenderTarget rt = target.target();
        rt.begin(cmd, begin);
        rt.end(cmd);
      }
      target.record_readback(cmd);
    });
    ASSERT_TRUE(s.ok()) << s.message();
  }

  static constexpr uint32_t kSize = 8;
};

TEST_F(MultisampledOffscreenTargetTest, IsLaidOutAtTheWindowsSamples) {
  auto target = create(vg::kWindowSamples);
  ASSERT_TRUE(target.ok()) << target.status().message();
  const vg::RenderTargetLayout layout = target.value().layout();
  EXPECT_EQ(layout.samples, vg::kWindowSamples);
  EXPECT_EQ(layout.color_formats[0], kFormat);
  EXPECT_EQ(layout.depth_format, VK_FORMAT_D32_SFLOAT);
  EXPECT_TRUE(layout.compatible_with(target.value().target().layout()));
  EXPECT_NE(target.value().color_image(), VK_NULL_HANDLE);
  EXPECT_NE(target.value().depth_image(), VK_NULL_HANDLE);

  // offscreen_desc of that layout makes a target of the same layout.
  auto same = vg::OffscreenTarget::create(
      allocator(), vg::offscreen_desc(layout, {kSize, kSize}));
  ASSERT_TRUE(same.ok()) << same.status().message();
  EXPECT_TRUE(same.value().layout().compatible_with(layout));
}

// The scope resolves into color_image(), which readback reads.
TEST_F(MultisampledOffscreenTargetTest, ReadsBackTheClearColour) {
  auto target = create(vg::kWindowSamples);
  ASSERT_TRUE(target.ok()) << target.status().message();
  ASSERT_NO_FATAL_FAILURE(clear_scopes(target.value(), {{1.0f, 0.0f, 0.0f}}));
  const auto* px = static_cast<const uint8_t*>(target.value().pixels());
  ASSERT_NE(px, nullptr);
  for (size_t i = 0; i < size_t{kSize} * kSize; ++i) {
    ASSERT_EQ(px[4 * i + 0], 255) << "texel " << i;
    ASSERT_EQ(px[4 * i + 1], 0) << "texel " << i;
    ASSERT_EQ(px[4 * i + 2], 0) << "texel " << i;
    ASSERT_EQ(px[4 * i + 3], 255) << "texel " << i;
  }
}

// Two scopes on one target in one command buffer reuse its multisampled
// colour and depth: begin()'s barrier orders the second scope's transition and
// clear after the first's resolve and stores, which synchronization
// validation checks. The second clear is what reads back.
TEST_F(MultisampledOffscreenTargetTest, BackToBackScopesAreOrdered) {
  auto target = create(vg::kWindowSamples);
  ASSERT_TRUE(target.ok()) << target.status().message();
  ASSERT_NO_FATAL_FAILURE(
      clear_scopes(target.value(), {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}));
  const auto* px = static_cast<const uint8_t*>(target.value().pixels());
  ASSERT_NE(px, nullptr);
  EXPECT_EQ(px[0], 0);
  EXPECT_EQ(px[1], 255);
  EXPECT_EQ(px[2], 0);
}

// Only the counts Vulkan guarantees: 1 and kWindowSamples.
TEST_F(MultisampledOffscreenTargetTest, OtherSampleCountsAreUnsupported) {
  for (const VkSampleCountFlagBits samples :
       {VK_SAMPLE_COUNT_2_BIT, VK_SAMPLE_COUNT_8_BIT}) {
    auto target = create(samples);
    ASSERT_FALSE(target.ok()) << samples;
    EXPECT_EQ(target.status().domain(), vkc::Status::Code::Unsupported);
  }
}

// An integer colour has no averaging resolve.
TEST_F(MultisampledOffscreenTargetTest, AnIntegerColourIsUnsupported) {
  auto target = create(vg::kWindowSamples, VK_FORMAT_R8G8B8A8_UINT);
  ASSERT_FALSE(target.ok());
  EXPECT_EQ(target.status().domain(), vkc::Status::Code::Unsupported);
}

TEST_F(MultisampledOffscreenTargetTest, MoveCarriesTheSamples) {
  auto created = create(vg::kWindowSamples);
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::OffscreenTarget source = std::move(created).value();
  vg::OffscreenTarget moved(std::move(source));
  EXPECT_EQ(moved.layout().samples, vg::kWindowSamples);
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.depth_image(), VK_NULL_HANDLE);

  auto other = create(vg::kWindowSamples);
  ASSERT_TRUE(other.ok()) << other.status().message();
  vg::OffscreenTarget dst = std::move(other).value();
  dst = std::move(moved);  // frees dst's images, then adopts moved's
  EXPECT_EQ(dst.layout().samples, vg::kWindowSamples);
  EXPECT_FALSE(moved.valid());  // NOLINT(bugprone-use-after-move)
}

}  // namespace

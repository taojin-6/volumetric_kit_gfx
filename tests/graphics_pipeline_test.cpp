// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GraphicsPipeline: validation, move semantics, and an end-to-end offscreen
// draw. The draw test builds the hello-triangle pipeline from the compiled
// triangle shaders for an OffscreenTarget's layout, renders into it through a
// dynamic-rendering scope, reads it back, and asserts that a triangle landed
// where it should -- the first test in the suite that actually rasterizes.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "spirv_test_util.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

// --- Validation rejects: no device needed (checked before the Vulkan call) ---

TEST(GraphicsPipelineTest, DefaultConstructedIsEmpty) {
  vg::GraphicsPipeline pipeline;
  EXPECT_FALSE(pipeline.valid());
  EXPECT_EQ(pipeline.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(pipeline.layout(), VK_NULL_HANDLE);
}

TEST(GraphicsPipelineTest, NullShaderHandlesRejected) {
  // A default desc leaves vertex_shader/fragment_shader null, so create()
  // rejects before touching Vulkan -- no device required.
  vg::GraphicsPipelineDesc desc;
  auto pipeline = vg::GraphicsPipeline::create(VK_NULL_HANDLE, desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

// --- Real pipeline creation + move semantics + draw: needs a device ----------

class GraphicsPipelineDeviceTest : public VulkanDeviceTest {
 protected:
  // A single-color-attachment layout matching the offscreen target the draw
  // test renders into.
  vg::RenderTargetLayout color_layout() {
    vg::RenderTargetLayout layout;
    layout.color_formats[0] = kFormat;
    layout.color_count = 1;
    return layout;
  }

  vg::ShaderModule load_module(const char* spv) {
    std::vector<uint32_t> code = vg_test::load_spirv(vg_test::spirv_path(spv));
    EXPECT_FALSE(code.empty()) << "missing/empty " << spv;
    auto module = vg::ShaderModule::create(device(), code.data(),
                                           code.size() * sizeof(uint32_t));
    EXPECT_TRUE(module.ok()) << module.status().message();
    return std::move(module).value();
  }

  vg::GraphicsPipeline build_pipeline(const vg::RenderTargetLayout& layout,
                                      VkShaderModule vert,
                                      VkShaderModule frag) {
    vg::GraphicsPipelineDesc desc;
    desc.vertex_shader = vert;
    desc.fragment_shader = frag;
    desc.layout = layout;
    auto pipeline = vg::GraphicsPipeline::create(device(), desc);
    EXPECT_TRUE(pipeline.ok()) << pipeline.status().message();
    return std::move(pipeline).value();
  }
};

TEST_F(GraphicsPipelineDeviceTest, BuildsFromTriangleShaders) {
  vg::ShaderModule vert = load_module("triangle.vert.spv");
  vg::ShaderModule frag = load_module("triangle.frag.spv");

  vg::GraphicsPipeline pipeline =
      build_pipeline(color_layout(), vert.handle(), frag.handle());
  EXPECT_TRUE(pipeline.valid());
  EXPECT_NE(pipeline.handle(), VK_NULL_HANDLE);
  EXPECT_NE(pipeline.layout(), VK_NULL_HANDLE);
}

TEST_F(GraphicsPipelineDeviceTest, EmptyLayoutRejected) {
  vg::ShaderModule vert = load_module("triangle.vert.spv");
  vg::ShaderModule frag = load_module("triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  // desc.layout left default: color_count == 0.
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, MoveLeavesSourceEmpty) {
  vg::ShaderModule vert = load_module("triangle.vert.spv");
  vg::ShaderModule frag = load_module("triangle.frag.spv");
  vg::GraphicsPipeline source =
      build_pipeline(color_layout(), vert.handle(), frag.handle());
  ASSERT_TRUE(source.valid());

  vg::GraphicsPipeline moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.layout(), VK_NULL_HANDLE);
}

TEST_F(GraphicsPipelineDeviceTest, MoveAssignOverLiveLeavesSourceEmpty) {
  vg::ShaderModule vert = load_module("triangle.vert.spv");
  vg::ShaderModule frag = load_module("triangle.frag.spv");
  vg::GraphicsPipeline dst =
      build_pipeline(color_layout(), vert.handle(), frag.handle());
  vg::GraphicsPipeline src =
      build_pipeline(color_layout(), vert.handle(), frag.handle());

  dst = std::move(src);  // frees dst's pipeline + layout, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(GraphicsPipelineDeviceTest, SelfMoveAssignIsSafe) {
  vg::ShaderModule vert = load_module("triangle.vert.spv");
  vg::ShaderModule frag = load_module("triangle.frag.spv");
  vg::GraphicsPipeline pipeline =
      build_pipeline(color_layout(), vert.handle(), frag.handle());

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the
  // per-member this != &other guard must keep the pipeline intact.
  vg::GraphicsPipeline* alias = &pipeline;
  pipeline = std::move(*alias);
  EXPECT_TRUE(pipeline.valid());
}

TEST_F(GraphicsPipelineDeviceTest, DrawsTriangleIntoOffscreenTarget) {
  constexpr uint32_t kSize = 32;

  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  vg::OffscreenTargetDesc target_desc;
  target_desc.extent = {kSize, kSize};
  target_desc.color_format = kFormat;
  auto target = vg::OffscreenTarget::create(allocator.value(), target_desc);
  ASSERT_TRUE(target.ok()) << target.status().message();

  vg::ShaderModule vert = load_module("triangle.vert.spv");
  vg::ShaderModule frag = load_module("triangle.frag.spv");
  vg::GraphicsPipeline pipeline =
      build_pipeline(target.value().layout(), vert.handle(), frag.handle());
  ASSERT_TRUE(pipeline.valid());

  auto pool = vg::CommandPool::create(device(), device_->graphics_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();

  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();

  // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL before the dynamic-rendering scope.
  VkImageMemoryBarrier to_color{};
  to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_color.srcAccessMask = 0;
  to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.image = target.value().color_image();
  to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(raw, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &to_color);

  // Opaque-black clear, so the triangle stands out.
  vg::RenderTargetBeginInfo begin_info;
  begin_info.clear_color.float32[3] = 1.0f;
  const vg::RenderTarget rt = target.value().target();
  rt.begin(raw, begin_info);

  vkCmdBindPipeline(raw, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
  VkViewport viewport{};
  viewport.width = static_cast<float>(kSize);
  viewport.height = static_cast<float>(kSize);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(raw, 0, 1, &viewport);
  VkRect2D scissor{};
  scissor.extent = {kSize, kSize};
  vkCmdSetScissor(raw, 0, 1, &scissor);
  vkCmdDraw(raw, 3, 1, 0, 0);

  rt.end(raw);

  target.value().record_readback(raw);
  ASSERT_TRUE(cmd.value().end().ok());
  submit_and_wait(raw);

  const auto* px = static_cast<const uint8_t*>(target.value().pixels());
  ASSERT_NE(px, nullptr);

  // The top-left corner is above the triangle's apex, so the opaque-black
  // clear shows through there untouched.
  EXPECT_EQ(px[0], 0);
  EXPECT_EQ(px[1], 0);
  EXPECT_EQ(px[2], 0);
  EXPECT_EQ(px[3], 255);

  // The image center lies inside the triangle, so a non-black, opaque
  // vertex-interpolated color was rasterized there.
  const size_t center =
      (static_cast<size_t>(kSize / 2) * kSize + kSize / 2) * 4;
  EXPECT_GT(px[center + 0] + px[center + 1] + px[center + 2], 0);
  EXPECT_EQ(px[center + 3], 255);
}

}  // namespace

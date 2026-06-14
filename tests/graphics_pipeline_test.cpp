// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GraphicsPipeline: validation, move semantics, and an end-to-end offscreen
// draw. The draw test builds the hello-triangle pipeline from the compiled
// triangle shaders, renders into an offscreen color target through a render
// pass, copies it to a host-visible buffer, and asserts that a triangle landed
// where it should -- the first test in the suite that actually rasterizes.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "spirv_test_util.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
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
  // A default desc leaves vertex_shader/fragment_shader/render_pass null, so
  // create() rejects before touching Vulkan -- no device required.
  vg::GraphicsPipelineDesc desc;
  auto pipeline = vg::GraphicsPipeline::create(VK_NULL_HANDLE, desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

// --- Real pipeline creation + move semantics + draw: needs a device ----------

class GraphicsPipelineDeviceTest : public VulkanDeviceTest {
 protected:
  // One R8G8B8A8_UNORM color attachment that clears on load, stores, and ends
  // in TRANSFER_SRC_OPTIMAL with a dependency making its writes available to
  // the copy-out the draw test performs. The caller owns the returned handle.
  vg::UniqueHandle<VkRenderPass, vkDestroyRenderPass> make_render_pass() {
    VkAttachmentDescription color{};
    color.format = kFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    // Make the color writes available + visible to the copy-out transfer.
    VkSubpassDependency dep{};
    dep.srcSubpass = 0;
    dep.dstSubpass = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &color;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dep;

    VkRenderPass handle = VK_NULL_HANDLE;
    EXPECT_EQ(vkCreateRenderPass(device(), &info, nullptr, &handle),
              VK_SUCCESS);
    return vg::UniqueHandle<VkRenderPass, vkDestroyRenderPass>(device(),
                                                               handle);
  }

  vg::GraphicsPipeline build_pipeline(VkRenderPass render_pass,
                                      VkShaderModule vert,
                                      VkShaderModule frag) {
    vg::GraphicsPipelineDesc desc;
    desc.vertex_shader = vert;
    desc.fragment_shader = frag;
    desc.render_pass = render_pass;
    auto pipeline = vg::GraphicsPipeline::create(device(), desc);
    EXPECT_TRUE(pipeline.ok()) << pipeline.status().message();
    return std::move(pipeline).value();
  }
};

TEST_F(GraphicsPipelineDeviceTest, BuildsFromTriangleShaders) {
  auto render_pass = make_render_pass();
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipeline pipeline =
      build_pipeline(render_pass.get(), vert.handle(), frag.handle());
  EXPECT_TRUE(pipeline.valid());
  EXPECT_NE(pipeline.handle(), VK_NULL_HANDLE);
  EXPECT_NE(pipeline.layout(), VK_NULL_HANDLE);
}

TEST_F(GraphicsPipelineDeviceTest, NullRenderPassRejected) {
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  // desc.render_pass left VK_NULL_HANDLE.
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, NullEntryPointRejected) {
  auto render_pass = make_render_pass();
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  desc.render_pass = render_pass.get();
  desc.entry_point = nullptr;  // rejected before Vulkan is touched
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, PatchTopologyRejected) {
  auto render_pass = make_render_pass();
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  desc.render_pass = render_pass.get();
  // Patch-list needs tessellation stages this pipeline does not provide.
  desc.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, NullDeviceRejected) {
  // Every desc field is valid, so the null device -- checked last, just before
  // the first Vulkan call -- is what create() rejects.
  auto render_pass = make_render_pass();
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  desc.render_pass = render_pass.get();
  auto pipeline = vg::GraphicsPipeline::create(VK_NULL_HANDLE, desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, MoveLeavesSourceEmpty) {
  auto render_pass = make_render_pass();
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");
  vg::GraphicsPipeline source =
      build_pipeline(render_pass.get(), vert.handle(), frag.handle());
  ASSERT_TRUE(source.valid());

  vg::GraphicsPipeline moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.layout(), VK_NULL_HANDLE);
}

TEST_F(GraphicsPipelineDeviceTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto render_pass = make_render_pass();
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");
  vg::GraphicsPipeline dst =
      build_pipeline(render_pass.get(), vert.handle(), frag.handle());
  vg::GraphicsPipeline src =
      build_pipeline(render_pass.get(), vert.handle(), frag.handle());

  dst = std::move(src);  // frees dst's pipeline + layout, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(GraphicsPipelineDeviceTest, SelfMoveAssignIsSafe) {
  auto render_pass = make_render_pass();
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");
  vg::GraphicsPipeline pipeline =
      build_pipeline(render_pass.get(), vert.handle(), frag.handle());

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the
  // per-member this != &other guard must keep the pipeline intact.
  vg::GraphicsPipeline* alias = &pipeline;
  pipeline = std::move(*alias);
  EXPECT_TRUE(pipeline.valid());
}

TEST_F(GraphicsPipelineDeviceTest, DrawsTriangleIntoOffscreenTarget) {
  constexpr uint32_t kSize = 32;
  constexpr VkDeviceSize kBytes = kSize * kSize * 4;  // R8G8B8A8

  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  vg::TextureDesc image_desc;
  image_desc.extent = {kSize, kSize};
  image_desc.format = kFormat;
  image_desc.usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  auto image = allocator.value().create_image(image_desc);
  ASSERT_TRUE(image.ok()) << image.status().message();

  vg::BufferDesc buffer_desc;
  buffer_desc.size = kBytes;
  buffer_desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_desc.memory = vg::MemoryUsage::HostVisible;
  buffer_desc.mapped = true;
  auto readback = allocator.value().create_buffer(buffer_desc);
  ASSERT_TRUE(readback.ok()) << readback.status().message();

  auto render_pass = make_render_pass();

  VkImageView attachment = image.value().view();
  VkFramebufferCreateInfo fb_info{};
  fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fb_info.renderPass = render_pass.get();
  fb_info.attachmentCount = 1;
  fb_info.pAttachments = &attachment;
  fb_info.width = kSize;
  fb_info.height = kSize;
  fb_info.layers = 1;
  VkFramebuffer fb_handle = VK_NULL_HANDLE;
  ASSERT_EQ(vkCreateFramebuffer(device(), &fb_info, nullptr, &fb_handle),
            VK_SUCCESS);
  vg::UniqueHandle<VkFramebuffer, vkDestroyFramebuffer> framebuffer(device(),
                                                                    fb_handle);

  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");
  vg::GraphicsPipeline pipeline =
      build_pipeline(render_pass.get(), vert.handle(), frag.handle());
  ASSERT_TRUE(pipeline.valid());

  auto pool = vg::CommandPool::create(device(), device_->graphics_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();

  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();

  VkClearValue clear{};
  clear.color.float32[0] = 0.0f;  // opaque black, so the triangle stands out
  clear.color.float32[1] = 0.0f;
  clear.color.float32[2] = 0.0f;
  clear.color.float32[3] = 1.0f;

  VkRenderPassBeginInfo begin_rp{};
  begin_rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  begin_rp.renderPass = render_pass.get();
  begin_rp.framebuffer = framebuffer.get();
  begin_rp.renderArea.extent = {kSize, kSize};
  begin_rp.clearValueCount = 1;
  begin_rp.pClearValues = &clear;
  vkCmdBeginRenderPass(raw, &begin_rp, VK_SUBPASS_CONTENTS_INLINE);

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
  vkCmdEndRenderPass(raw);

  // The render pass left the image in TRANSFER_SRC_OPTIMAL; copy it out and
  // make the result visible to the host read below.
  record_copy_image_to_host(raw, image.value().image(),
                            readback.value().handle(), {kSize, kSize});

  ASSERT_TRUE(cmd.value().end().ok());

  submit_and_wait(raw);

  const auto* px = static_cast<const unsigned char*>(readback.value().mapped());

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

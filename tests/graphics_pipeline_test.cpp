// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GraphicsPipeline: validation, move semantics, and an end-to-end offscreen
// draw. The draw test builds the hello-triangle pipeline from the compiled
// triangle shaders for an OffscreenTarget's layout, renders into it through a
// dynamic-rendering scope, reads it back, and asserts that a triangle landed
// where it should -- the first test in the suite that actually rasterizes.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

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
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipeline pipeline =
      build_pipeline(color_layout(), vert.handle(), frag.handle());
  EXPECT_TRUE(pipeline.valid());
  EXPECT_NE(pipeline.handle(), VK_NULL_HANDLE);
  EXPECT_NE(pipeline.layout(), VK_NULL_HANDLE);
}

TEST_F(GraphicsPipelineDeviceTest, EmptyLayoutRejected) {
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  // desc.layout left default: color_count == 0.
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, NullEntryPointRejected) {
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  desc.layout = color_layout();
  desc.entry_point = nullptr;  // rejected before Vulkan is touched
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, PatchTopologyRejected) {
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  desc.layout = color_layout();
  // Patch-list needs tessellation stages this pipeline does not provide.
  desc.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, NullDeviceRejected) {
  // Every desc field is valid, so the null device -- checked last, just before
  // the first Vulkan call -- is what create() rejects.
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");

  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  desc.layout = color_layout();
  auto pipeline = vg::GraphicsPipeline::create(VK_NULL_HANDLE, desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, MoveLeavesSourceEmpty) {
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");
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
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");
  vg::GraphicsPipeline dst =
      build_pipeline(color_layout(), vert.handle(), frag.handle());
  vg::GraphicsPipeline src =
      build_pipeline(color_layout(), vert.handle(), frag.handle());

  dst = std::move(src);  // frees dst's pipeline + layout, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(GraphicsPipelineDeviceTest, SelfMoveAssignIsSafe) {
  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");
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

  vg::ShaderModule vert = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "triangle.frag.spv");
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

// --- Vertex-buffer input + depth testing -------------------------------------

constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// Interleaved clip-space position + RGB color, matching mesh.vert's two vertex
// attributes (locations 0 and 1).
struct MeshVertex {
  float pos[3];
  float color[3];
};

VkVertexInputBindingDescription mesh_binding() {
  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = sizeof(MeshVertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  return binding;
}

std::array<VkVertexInputAttributeDescription, 2> mesh_attributes() {
  std::array<VkVertexInputAttributeDescription, 2> attrs{};
  attrs[0].location = 0;
  attrs[0].binding = 0;
  attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attrs[0].offset = offsetof(MeshVertex, pos);
  attrs[1].location = 1;
  attrs[1].binding = 0;
  attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attrs[1].offset = offsetof(MeshVertex, color);
  return attrs;
}

vg::Buffer make_host_buffer(vg::Allocator& allocator, const void* data,
                            size_t size, VkBufferUsageFlags usage) {
  vg::BufferDesc desc;
  desc.size = size;
  desc.usage = usage;
  desc.memory = vg::MemoryUsage::HostVisible;
  desc.mapped = true;
  auto buffer = allocator.create_buffer(desc);
  EXPECT_TRUE(buffer.ok()) << buffer.status().message();
  vg::Buffer out = std::move(buffer).value();
  std::memcpy(out.mapped(), data, size);
  return out;
}

vg::GraphicsPipeline build_mesh_pipeline(
    VkDevice device, const vg::RenderTargetLayout& layout, VkShaderModule vert,
    VkShaderModule frag, const VkVertexInputBindingDescription* bindings,
    uint32_t binding_count, const VkVertexInputAttributeDescription* attrs,
    uint32_t attr_count, bool depth_test) {
  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert;
  desc.fragment_shader = frag;
  desc.layout = layout;
  desc.vertex_bindings = bindings;
  desc.vertex_binding_count = binding_count;
  desc.vertex_attributes = attrs;
  desc.vertex_attribute_count = attr_count;
  desc.depth_test = depth_test;
  desc.depth_write = depth_test;
  auto pipeline = vg::GraphicsPipeline::create(device, desc);
  EXPECT_TRUE(pipeline.ok()) << pipeline.status().message();
  return std::move(pipeline).value();
}

void barrier_to_color(VkCommandBuffer cmd, VkImage image) {
  VkImageMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.srcAccessMask = 0;
  b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &b);
}

void barrier_to_depth(VkCommandBuffer cmd, VkImage image) {
  VkImageMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.srcAccessMask = 0;
  b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &b);
}

void set_full_viewport_scissor(VkCommandBuffer cmd, uint32_t size) {
  VkViewport viewport{};
  viewport.width = static_cast<float>(size);
  viewport.height = static_cast<float>(size);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  VkRect2D scissor{};
  scissor.extent = {size, size};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

TEST_F(GraphicsPipelineDeviceTest, DepthTestWithoutDepthFormatRejected) {
  vg::ShaderModule vert = vg_test::load_module(device(), "mesh.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "mesh.frag.spv");
  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  desc.layout = color_layout();  // carries no depth format
  desc.depth_test = true;        // rejected before Vulkan is touched
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, VertexBindingCountWithoutPointerRejected) {
  vg::ShaderModule vert = vg_test::load_module(device(), "mesh.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "mesh.frag.spv");
  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = vert.handle();
  desc.fragment_shader = frag.handle();
  desc.layout = color_layout();
  desc.vertex_binding_count = 1;  // but vertex_bindings stays null
  auto pipeline = vg::GraphicsPipeline::create(device(), desc);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GraphicsPipelineDeviceTest, DrawsFromVertexBuffer) {
  constexpr uint32_t kSize = 32;
  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  vg::OffscreenTargetDesc target_desc;
  target_desc.extent = {kSize, kSize};
  target_desc.color_format = kFormat;
  auto target = vg::OffscreenTarget::create(allocator.value(), target_desc);
  ASSERT_TRUE(target.ok()) << target.status().message();

  // One green triangle covering the image center, fed from a vertex buffer.
  const MeshVertex verts[3] = {
      {{-0.9f, -0.9f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      {{0.9f, -0.9f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      {{0.0f, 0.9f, 0.0f}, {0.0f, 1.0f, 0.0f}},
  };
  vg::Buffer vbuf = make_host_buffer(allocator.value(), verts, sizeof(verts),
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

  vg::ShaderModule vert = vg_test::load_module(device(), "mesh.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "mesh.frag.spv");
  const VkVertexInputBindingDescription binding = mesh_binding();
  const auto attrs = mesh_attributes();
  vg::GraphicsPipeline pipeline = build_mesh_pipeline(
      device(), target.value().layout(), vert.handle(), frag.handle(), &binding,
      1, attrs.data(), static_cast<uint32_t>(attrs.size()), false);
  ASSERT_TRUE(pipeline.valid());

  auto pool = vg::CommandPool::create(device(), device_->graphics_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();
  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();

  barrier_to_color(raw, target.value().color_image());

  vg::RenderTargetBeginInfo begin_info;
  begin_info.clear_color.float32[3] = 1.0f;
  const vg::RenderTarget rt = target.value().target();
  rt.begin(raw, begin_info);

  vkCmdBindPipeline(raw, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
  set_full_viewport_scissor(raw, kSize);
  const VkDeviceSize offset = 0;
  const VkBuffer vb = vbuf.handle();
  vkCmdBindVertexBuffers(raw, 0, 1, &vb, &offset);
  vkCmdDraw(raw, 3, 1, 0, 0);

  rt.end(raw);
  target.value().record_readback(raw);
  ASSERT_TRUE(cmd.value().end().ok());
  submit_and_wait(raw);

  const auto* px = static_cast<const uint8_t*>(target.value().pixels());
  ASSERT_NE(px, nullptr);
  const size_t center =
      (static_cast<size_t>(kSize / 2) * kSize + kSize / 2) * 4;
  // The vertex-buffer triangle rasterized green at the center.
  EXPECT_LT(px[center + 0], 128);
  EXPECT_GT(px[center + 1], 128);
  EXPECT_LT(px[center + 2], 128);
  EXPECT_EQ(px[center + 3], 255);
}

TEST_F(GraphicsPipelineDeviceTest, DepthTestKeepsNearerSurface) {
  constexpr uint32_t kSize = 32;
  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  vg::OffscreenTargetDesc target_desc;
  target_desc.extent = {kSize, kSize};
  target_desc.color_format = kFormat;
  target_desc.depth_format = kDepthFormat;
  auto target = vg::OffscreenTarget::create(allocator.value(), target_desc);
  ASSERT_TRUE(target.ok()) << target.status().message();
  ASSERT_EQ(target.value().layout().depth_format, kDepthFormat);
  ASSERT_NE(target.value().depth_image(), VK_NULL_HANDLE);

  // Two overlapping triangles over the center: a NEAR blue one (z = 0.3) listed
  // first and a FAR red one (z = 0.7) listed last. drawIndexed rasterizes the
  // far triangle after the near, so without depth testing the center would end
  // up red; VK_COMPARE_OP_LESS must reject the far fragments and keep it blue.
  const MeshVertex verts[6] = {
      {{-0.9f, -0.9f, 0.3f}, {0.0f, 0.0f, 1.0f}},
      {{0.9f, -0.9f, 0.3f}, {0.0f, 0.0f, 1.0f}},
      {{0.0f, 0.9f, 0.3f}, {0.0f, 0.0f, 1.0f}},
      {{-0.9f, -0.9f, 0.7f}, {1.0f, 0.0f, 0.0f}},
      {{0.9f, -0.9f, 0.7f}, {1.0f, 0.0f, 0.0f}},
      {{0.0f, 0.9f, 0.7f}, {1.0f, 0.0f, 0.0f}},
  };
  const uint32_t indices[6] = {0, 1, 2, 3, 4, 5};
  vg::Buffer vbuf = make_host_buffer(allocator.value(), verts, sizeof(verts),
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  vg::Buffer ibuf =
      make_host_buffer(allocator.value(), indices, sizeof(indices),
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

  vg::ShaderModule vert = vg_test::load_module(device(), "mesh.vert.spv");
  vg::ShaderModule frag = vg_test::load_module(device(), "mesh.frag.spv");
  const VkVertexInputBindingDescription binding = mesh_binding();
  const auto attrs = mesh_attributes();
  vg::GraphicsPipeline pipeline = build_mesh_pipeline(
      device(), target.value().layout(), vert.handle(), frag.handle(), &binding,
      1, attrs.data(), static_cast<uint32_t>(attrs.size()), true);
  ASSERT_TRUE(pipeline.valid());

  auto pool = vg::CommandPool::create(device(), device_->graphics_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();
  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  const VkCommandBuffer raw = cmd.value().handle();

  barrier_to_color(raw, target.value().color_image());
  barrier_to_depth(raw, target.value().depth_image());

  vg::RenderTargetBeginInfo begin_info;
  begin_info.clear_color.float32[3] = 1.0f;
  const vg::RenderTarget rt = target.value().target();
  rt.begin(raw, begin_info);

  vkCmdBindPipeline(raw, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
  set_full_viewport_scissor(raw, kSize);
  const VkDeviceSize offset = 0;
  const VkBuffer vb = vbuf.handle();
  vkCmdBindVertexBuffers(raw, 0, 1, &vb, &offset);
  vkCmdBindIndexBuffer(raw, ibuf.handle(), 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(raw, 6, 1, 0, 0, 0);

  rt.end(raw);
  target.value().record_readback(raw);
  ASSERT_TRUE(cmd.value().end().ok());
  submit_and_wait(raw);

  const auto* px = static_cast<const uint8_t*>(target.value().pixels());
  ASSERT_NE(px, nullptr);
  const size_t center =
      (static_cast<size_t>(kSize / 2) * kSize + kSize / 2) * 4;
  // Depth kept the nearer (blue) surface and rejected the later far (red) draw.
  EXPECT_GT(px[center + 2], 128) << "near (blue) surface should win the test";
  EXPECT_LT(px[center + 0], 128) << "far (red) surface must be rejected";
  EXPECT_EQ(px[center + 3], 255);
}

}  // namespace

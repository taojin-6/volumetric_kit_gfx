// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// PatchMeshPipeline: the per-primitive atlas path.
//
// What this file covers is the pipeline OBJECT -- that it reflects the three
// storage buffers its fragment stage declares, that it obeys the repo's
// move-only rules, and that it refuses to record a frame it cannot draw
// legally. What it does NOT cover is the shading itself: whether a fragment
// recovers the right barycentric coordinate and lands on the right texel is a
// pixel-level claim, and it is verified where it is visible -- against a real
// accumulated atlas in the reconstruction viewer, not against a synthetic one
// here. That is stated rather than left implied, because a green test file is
// otherwise easy to read as more coverage than it is.

#include "volumetric_kit/gfx/pipelines/patch_mesh_pipeline.hpp"

#include <utility>

#include <gtest/gtest.h>

#include "volumetric_kit/gfx/core/render_target.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

// The signature the depth-tested pipeline is built for.
vg::RenderTargetLayout color_depth_layout() {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
  layout.color_count = 1;
  layout.depth_format = VK_FORMAT_D32_SFLOAT;
  return layout;
}

using PatchMeshPipelineDeviceTest = VulkanDeviceTest;

}  // namespace

TEST(PatchMeshPipelineTest, DefaultConstructedIsEmpty) {
  pipelines::PatchMeshPipeline pipeline;
  EXPECT_FALSE(pipeline.valid());
  EXPECT_EQ(pipeline.descriptor_set_count(), 0u);
  EXPECT_EQ(pipeline.descriptor_set_layout(0), VK_NULL_HANDLE);
}

TEST_F(PatchMeshPipelineDeviceTest, ReflectsOneSetOfThreeStorageBuffers) {
  auto pipeline =
      pipelines::PatchMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
  EXPECT_TRUE(pipeline.value().valid());
  // One set -- the producer's atlas, index run and vertices all live in set 0,
  // so a consumer at the zero-copy seam writes it once per slot.
  EXPECT_EQ(pipeline.value().descriptor_set_count(), 1u);
  EXPECT_NE(pipeline.value().descriptor_set_layout(0), VK_NULL_HANDLE);
  EXPECT_EQ(pipeline.value().descriptor_set_layout(1), VK_NULL_HANDLE);
}

TEST_F(PatchMeshPipelineDeviceTest, MoveLeavesSourceEmpty) {
  auto created =
      pipelines::PatchMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(created.ok());
  pipelines::PatchMeshPipeline source = std::move(created).value();
  ASSERT_TRUE(source.valid());

  pipelines::PatchMeshPipeline moved = std::move(source);
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());
  EXPECT_EQ(source.descriptor_set_count(), 0u);
  EXPECT_EQ(source.descriptor_set_layout(0), VK_NULL_HANDLE);
}

TEST_F(PatchMeshPipelineDeviceTest, MoveAssignOverLiveObjectAdoptsSource) {
  auto first =
      pipelines::PatchMeshPipeline::create(device(), color_depth_layout());
  auto second =
      pipelines::PatchMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  pipelines::PatchMeshPipeline target = std::move(first).value();
  pipelines::PatchMeshPipeline source = std::move(second).value();
  const VkDescriptorSetLayout adopted = source.descriptor_set_layout(0);

  target = std::move(source);
  EXPECT_TRUE(target.valid());
  EXPECT_EQ(target.descriptor_set_layout(0), adopted);
  EXPECT_FALSE(source.valid());
}

TEST_F(PatchMeshPipelineDeviceTest, SelfMoveAssignStaysValid) {
  auto created =
      pipelines::PatchMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(created.ok());
  pipelines::PatchMeshPipeline pipeline = std::move(created).value();
  ASSERT_EQ(pipeline.descriptor_set_count(), 1u);
  const VkDescriptorSetLayout before = pipeline.descriptor_set_layout(0);

  // Laundered through a pointer to dodge -Wself-move under -Werror.
  pipelines::PatchMeshPipeline* alias = &pipeline;
  pipeline = std::move(*alias);

  EXPECT_TRUE(pipeline.valid());
  ASSERT_EQ(pipeline.descriptor_set_count(), 1u);
  EXPECT_EQ(pipeline.descriptor_set_layout(0), before);
}

// The three ways a frame can be undrawable, each of which the pipeline drops
// rather than records. The first is the one the hybrid pipeline also has -- a
// statically-read set that is not bound is illegal. The other two are new and
// are worse than illegal: `patch_leg` is what turns a barycentric coordinate
// into a texel index, so a zero would make the shader divide by minus one and
// address the atlas anywhere at all, and a zero `texels_per_patch` collapses
// every triangle onto patch zero. Both would draw *something*, which is why
// they are refused here rather than left to look like a shading bug.
TEST_F(PatchMeshPipelineDeviceTest, RefusesUndrawableFrames) {
  auto created =
      pipelines::PatchMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(created.ok());
  pipelines::PatchMeshPipeline pipeline = std::move(created).value();

  // A null command buffer is safe precisely because nothing is recorded: each
  // of these returns before the first vkCmd call. Recording into VK_NULL_HANDLE
  // would be undefined, so reaching one is the failure being tested for.
  pipelines::PatchMeshFrame frame{};
  frame.extent = VkExtent2D{64, 64};
  frame.patch_leg = 8;
  frame.texels_per_patch = 36;
  frame.atlas = VK_NULL_HANDLE;  // unbound set
  pipeline.submit(VK_NULL_HANDLE, frame);

  // A leg of 1 has no span between its texels; a leg of 0 underflows.
  frame.atlas = reinterpret_cast<VkDescriptorSet>(uintptr_t{1});
  frame.patch_leg = 1;
  pipeline.submit(VK_NULL_HANDLE, frame);
  frame.patch_leg = 0;
  pipeline.submit(VK_NULL_HANDLE, frame);

  // A patch of no texels puts every triangle on the same one.
  frame.patch_leg = 8;
  frame.texels_per_patch = 0;
  pipeline.submit(VK_NULL_HANDLE, frame);

  // And an empty pipeline records nothing whatever the frame says.
  pipelines::PatchMeshPipeline empty;
  frame.texels_per_patch = 36;
  empty.submit(VK_NULL_HANDLE, frame);
}

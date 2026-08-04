// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_pipeline.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

using PbrPipelineTest = VulkanDeviceTest;

vg::RenderTargetLayout color_depth_layout() {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = VK_FORMAT_R8G8B8A8_SRGB;
  layout.color_count = 1;
  layout.depth_format = VK_FORMAT_D32_SFLOAT;
  return layout;
}

}  // namespace

// This target links gfx_pipelines WITHOUT gfx_camera -- the same link shape the
// volumetric_kit_recon handoff uses -- and PbrFrame/HybridMeshFrame hand the
// consumer a clip-space glm::mat4 to fill. So the Vulkan depth convention has
// to arrive through gfx_pipelines' usage requirements, not through gfx_camera:
// a [-1, 1] projection against the [0, 1] depth attachment these pipelines are
// built for silently clips the near half of the frustum, with no compile error
// and no VUID. Needs no device, so it runs everywhere.
TEST(PipelinesGlmConvention, ProjectionUsesVulkanDepthRange) {
  const glm::mat4 proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  // A point exactly on the near plane maps to z/w == 0 under Vulkan's [0, 1]
  // range; GL's [-1, 1] would put it at -1.
  const glm::vec4 near_point = proj * glm::vec4(0.0f, 0.0f, -0.1f, 1.0f);
  EXPECT_NEAR(near_point.z / near_point.w, 0.0f, 1e-5f);
  // ...and the far plane at 1, not +1 either way -- pinning both ends rules out
  // a reversed-Z mix-up as well.
  const glm::vec4 far_point = proj * glm::vec4(0.0f, 0.0f, -100.0f, 1.0f);
  EXPECT_NEAR(far_point.z / far_point.w, 1.0f, 1e-5f);
}

TEST_F(PbrPipelineTest, CreatesWithReflectedSets) {
  auto pbr = pipelines::PbrPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(pbr.ok()) << pbr.status().message();
  EXPECT_TRUE(pbr.value().valid());
  EXPECT_NE(pbr.value().handle(), VK_NULL_HANDLE);
  EXPECT_NE(pbr.value().layout(), VK_NULL_HANDLE);
  // The embedded shaders declare set 0 (scene) and set 1 (material).
  EXPECT_NE(pbr.value().descriptor_set_layout(0), VK_NULL_HANDLE);
  EXPECT_NE(pbr.value().descriptor_set_layout(1), VK_NULL_HANDLE);
}

TEST_F(PbrPipelineTest, RejectsLayoutWithoutDepth) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = VK_FORMAT_R8G8B8A8_SRGB;
  layout.color_count = 1;  // no depth format -> depth-tested pipeline rejected
  auto pbr = pipelines::PbrPipeline::create(device(), layout);
  ASSERT_FALSE(pbr.ok());
  EXPECT_EQ(pbr.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(PbrPipelineTest, MoveLeavesSourceEmpty) {
  auto created = pipelines::PbrPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(created.ok()) << created.status().message();
  pipelines::PbrPipeline source = std::move(created).value();
  ASSERT_TRUE(source.valid());

  pipelines::PbrPipeline moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(PbrPipelineTest, MoveAssignOverLiveObjectAdoptsSource) {
  auto first = pipelines::PbrPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(first.ok()) << first.status().message();
  auto second = pipelines::PbrPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(second.ok()) << second.status().message();

  pipelines::PbrPipeline dst = std::move(first).value();
  pipelines::PbrPipeline src = std::move(second).value();
  const VkPipeline adopted = src.handle();

  // Move-assign over a live dst: frees dst's pipeline, then adopts src's
  // (the destroy()-then-adopt path where double-free / leak bugs live).
  dst = std::move(src);
  EXPECT_TRUE(dst.valid());
  EXPECT_EQ(dst.handle(), adopted);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(PbrPipelineTest, SelfMoveAssignStaysValid) {
  auto created = pipelines::PbrPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(created.ok()) << created.status().message();
  pipelines::PbrPipeline pbr = std::move(created).value();
  const VkPipeline before = pbr.handle();
  const uint32_t sets_before = pbr.descriptor_set_count();
  ASSERT_EQ(sets_before, 2u);  // set 0 scene, set 1 material
  std::vector<VkDescriptorSetLayout> layouts_before;
  for (uint32_t i = 0; i < sets_before; ++i) {
    layouts_before.push_back(pbr.descriptor_set_layout(i));
  }

  // Launder through a pointer so -Wself-move doesn't fire under -Werror; the
  // guarded move-assign must leave the owned pipeline intact, not free it.
  pipelines::PbrPipeline* p = &pbr;
  pbr = std::move(*p);
  EXPECT_TRUE(pbr.valid());
  EXPECT_EQ(pbr.handle(), before);

  // The embedded GraphicsPipeline also owns the reflected set layouts, and an
  // unguarded move-assign would free them while handle()/valid() stayed intact.
  ASSERT_EQ(pbr.descriptor_set_count(), sets_before);
  for (uint32_t i = 0; i < sets_before; ++i) {
    EXPECT_EQ(pbr.descriptor_set_layout(i), layouts_before[i]) << "set " << i;
  }
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <utility>

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

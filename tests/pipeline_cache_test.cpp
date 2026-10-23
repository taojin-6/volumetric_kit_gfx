// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The built-in pipelines build through their device's pipeline cache
// (core::Device::pipeline_cache), so one built again on the same device -- one
// per view or camera -- reuses its compiled shaders.

#include <gtest/gtest.h>

#include <cstddef>

#include "gfx_test_support.hpp"
#include "spirv_test_util.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/pipelines/depth_cloud_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/image_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/primitive_pipeline.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

using BuiltInPipelineCacheTest = vg_test::RendererTest;

TEST_F(BuiltInPipelineCacheTest, BuildThroughTheDevicesCache) {
  // A device of the test's own, so its cache starts empty.
  const vkc::Result<vkc::Device> device =
      vkc::Device::create(instance(), physical(), requirements());
  ASSERT_TRUE(device.ok()) << device.status().message();
  const vg::ShaderModule vert =
      vg_test::load_module(device->handle(), "image_convert.vert.spv");
  if (!vg_test::driver_keeps_pipelines(device->handle(), vert.handle())) {
    GTEST_SKIP() << "the driver keeps nothing in a pipeline cache";
  }
  vkc::Result<vkc::Allocator> allocator =
      vkc::Allocator::create(instance().handle(), *device);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();
  const vkc::Result<vkc::TimelineSemaphore> timeline =
      vkc::TimelineSemaphore::create(*device);
  ASSERT_TRUE(timeline.ok()) << timeline.status().message();
  vg::RenderTargetLayout color;
  color.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
  color.color_count = 1;
  vg::RenderTargetLayout depth = color;
  depth.depth_format = VK_FORMAT_D32_SFLOAT;

  // Each pipeline adds what it compiles to the cache.
  std::size_t bytes = vg_test::pipeline_cache_bytes(*device);
  const auto grew = [&] {
    const std::size_t now = vg_test::pipeline_cache_bytes(*device);
    const bool more = now > bytes;
    bytes = now;
    return more;
  };
  const auto depth_cloud = pipelines::DepthCloudPipeline::create(
      *device, *allocator, depth, *timeline);
  ASSERT_TRUE(depth_cloud.ok()) << depth_cloud.status().message();
  EXPECT_TRUE(grew()) << "DepthCloudPipeline";
  const auto hybrid_mesh =
      pipelines::HybridMeshPipeline::create(*device, *allocator, depth);
  ASSERT_TRUE(hybrid_mesh.ok()) << hybrid_mesh.status().message();
  EXPECT_TRUE(grew()) << "HybridMeshPipeline";
  const auto image = pipelines::ImagePipeline::create(*device, color);
  ASSERT_TRUE(image.ok()) << image.status().message();
  EXPECT_TRUE(grew()) << "ImagePipeline";
  const auto primitive = pipelines::PrimitivePipeline::create(*device, depth);
  ASSERT_TRUE(primitive.ok()) << primitive.status().message();
  EXPECT_TRUE(grew()) << "PrimitivePipeline";
}

}  // namespace

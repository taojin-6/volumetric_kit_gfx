// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The built-in pipelines build through their device's pipeline cache
// (core::Device::pipeline_cache), so one built again on the same device -- one
// per view or camera -- reuses its compiled shaders.

#include <gtest/gtest.h>

#include <cstddef>
#include <functional>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/pipelines/depth_cloud_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/image_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/primitive_pipeline.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

using Build = std::function<vkc::Status(const vkc::Device&, vkc::Allocator&)>;

class BuiltInPipelineCacheTest : public vg_test::RendererTest {
 protected:
  // Whether `build`, run on a device of the test's own, so that its cache
  // starts empty, leaves anything in that cache.
  bool fills_cache(const Build& build) {
    vkc::Result<vkc::Device> device =
        vkc::Device::create(instance(), physical(), requirements());
    EXPECT_TRUE(device.ok()) << device.status().message();
    if (!device.ok()) {
      return false;
    }
    vkc::Result<vkc::Allocator> allocator =
        vkc::Allocator::create(instance().handle(), *device);
    EXPECT_TRUE(allocator.ok()) << allocator.status().message();
    if (!allocator.ok()) {
      return false;
    }
    const std::size_t empty = vg_test::pipeline_cache_bytes(*device);
    const vkc::Status built = build(*device, *allocator);
    EXPECT_TRUE(built.ok()) << built.message();
    return vg_test::pipeline_cache_bytes(*device) > empty;
  }
};

TEST_F(BuiltInPipelineCacheTest, BuildThroughTheDevicesCache) {
  vg::RenderTargetLayout color;
  color.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
  color.color_count = 1;
  vg::RenderTargetLayout depth = color;
  depth.depth_format = VK_FORMAT_D32_SFLOAT;

  const bool depth_cloud = fills_cache(
      [&](const vkc::Device& device, vkc::Allocator& allocator) -> vkc::Status {
        VKC_ASSIGN(vkc::TimelineSemaphore timeline,
                   vkc::TimelineSemaphore::create(device));
        return pipelines::DepthCloudPipeline::create(device, allocator, depth,
                                                     timeline)
            .status();
      });
  const bool hybrid_mesh =
      fills_cache([&](const vkc::Device& device, vkc::Allocator& allocator) {
        return pipelines::HybridMeshPipeline::create(device, allocator, depth)
            .status();
      });
  const bool image =
      fills_cache([&](const vkc::Device& device, vkc::Allocator&) {
        return pipelines::ImagePipeline::create(device, color).status();
      });
  const bool primitive =
      fills_cache([&](const vkc::Device& device, vkc::Allocator&) {
        return pipelines::PrimitivePipeline::create(device, depth).status();
      });

  if (!depth_cloud && !hybrid_mesh && !image && !primitive) {
    GTEST_SKIP() << "the driver keeps nothing in a pipeline cache";
  }
  EXPECT_TRUE(depth_cloud);
  EXPECT_TRUE(hybrid_mesh);
  EXPECT_TRUE(image);
  EXPECT_TRUE(primitive);
}

}  // namespace

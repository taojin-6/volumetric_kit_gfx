// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file gfx_test_support.hpp
/// What gfx's GPU tests add to the core's test fixtures
/// (`volumetric_kit/core/testing/vulkan_fixture.hpp`), which share the
/// instance and device, skip or fail without them, and fail a test on any
/// validation error: the renderer's device, the instance for a surface the
/// windowing and app tests present to, what a pipeline cache holds, and
/// whether the driver keeps anything in one.

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "volumetric_kit/core/testing/vulkan_fixture.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

namespace vg_test {

// A test with the shared instance and a physical device fit for the
// renderer, for one that makes its own instance or device -- for a surface, or
// inside an app -- under the core fixture's policy.
class RendererTest : public vkc::test::VulkanTest {
 protected:
  vkc::DeviceRequirements requirements() const override {
    return vg::device_requirements();
  }
};

// A test on the shared device made to the renderer's requirements.
class RendererDeviceTest : public vkc::test::VulkanDeviceTest {
 protected:
  vkc::DeviceRequirements requirements() const override {
    return vg::device_requirements();
  }
};

// The test policy's instance, validated as the environment asks or at
// `validation`, with a windowing::SurfaceSource's `instance_extensions`. Above
// the environment's level, create it while a vkc::test::ValidationSession for
// that level lives, as the layer reads its settings then.
inline vkc::InstanceConfig surface_instance_config(
    const std::vector<const char*>& surface_extensions,
    vkc::test::Validation validation = vkc::test::requested_validation()) {
  vkc::InstanceConfig config = vkc::test::instance_config(validation);
  config.extensions = surface_extensions;
  return config;
}

// The bytes `cache` holds: a header, then what the driver keeps.
inline std::size_t pipeline_cache_bytes(VkDevice device,
                                        VkPipelineCache cache) {
  std::size_t bytes = 0;
  EXPECT_EQ(vkGetPipelineCacheData(device, cache, &bytes, nullptr), VK_SUCCESS);
  return bytes;
}

// The bytes `device`'s pipeline cache holds.
inline std::size_t pipeline_cache_bytes(const vkc::Device& device) {
  return pipeline_cache_bytes(device.handle(), device.pipeline_cache());
}

// Whether the driver keeps anything in a pipeline cache (lavapipe keeps
// nothing), asked without the code under test: the smallest graphics pipeline,
// `vertex` (no inputs or resources) with its primitives discarded, built
// straight through Vulkan into a cache of the probe's own.
inline bool driver_keeps_pipelines(VkDevice device, VkShaderModule vertex) {
  VkPipelineCacheCreateInfo cache_info{};
  cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  VkPipelineCache cache = VK_NULL_HANDLE;
  EXPECT_EQ(vkCreatePipelineCache(device, &cache_info, nullptr, &cache),
            VK_SUCCESS);
  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  EXPECT_EQ(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout),
            VK_SUCCESS);

  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
  stage.module = vertex;
  stage.pName = "main";
  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  VkPipelineInputAssemblyStateCreateInfo assembly{};
  assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineRasterizationStateCreateInfo raster{};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.rasterizerDiscardEnable = VK_TRUE;
  raster.lineWidth = 1.0f;
  VkGraphicsPipelineCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  info.stageCount = 1;
  info.pStages = &stage;
  info.pVertexInputState = &vertex_input;
  info.pInputAssemblyState = &assembly;
  info.pRasterizationState = &raster;
  info.layout = layout;

  const std::size_t empty = pipeline_cache_bytes(device, cache);
  VkPipeline pipeline = VK_NULL_HANDLE;
  EXPECT_EQ(
      vkCreateGraphicsPipelines(device, cache, 1, &info, nullptr, &pipeline),
      VK_SUCCESS);
  const bool kept = pipeline_cache_bytes(device, cache) > empty;
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, layout, nullptr);
  vkDestroyPipelineCache(device, cache, nullptr);
  return kept;
}

}  // namespace vg_test

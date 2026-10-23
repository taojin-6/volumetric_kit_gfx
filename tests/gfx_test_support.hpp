// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file gfx_test_support.hpp
/// What gfx's GPU tests add to the core's test fixtures
/// (`volumetric_kit/core/testing/vulkan_fixture.hpp`), which share the
/// instance and device, skip or fail without them, and fail a test on any
/// validation error: the renderer's device, the instance for a surface the
/// windowing and app tests present to, and what a device's pipeline cache
/// holds.

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

// The bytes `device`'s pipeline cache holds: a header, then what the driver
// keeps (lavapipe keeps nothing).
inline std::size_t pipeline_cache_bytes(const vkc::Device& device) {
  std::size_t bytes = 0;
  EXPECT_EQ(vkGetPipelineCacheData(device.handle(), device.pipeline_cache(),
                                   &bytes, nullptr),
            VK_SUCCESS);
  return bytes;
}

}  // namespace vg_test

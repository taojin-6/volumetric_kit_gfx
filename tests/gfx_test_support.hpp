// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file gfx_test_support.hpp
/// What gfx's GPU tests add to the core's test fixtures
/// (`volumetric_kit/core/testing/vulkan_fixture.hpp`), which share the
/// instance and device, skip or fail without them, and fail a test on any
/// validation error: the renderer's device, and the headless surface the
/// windowing and app tests present to.

#include <cstdint>
#include <cstring>
#include <vector>

#include "volumetric_kit/core/testing/vulkan_fixture.hpp"
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

// Whether the loader offers VK_EXT_headless_surface.
inline bool has_headless_surface() {
  std::uint32_t count = 0;
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) !=
      VK_SUCCESS) {
    return false;
  }
  std::vector<VkExtensionProperties> props(count);
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, props.data()) !=
      VK_SUCCESS) {
    return false;
  }
  for (const VkExtensionProperties& p : props) {
    if (std::strcmp(p.extensionName, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) ==
        0) {
      return true;
    }
  }
  return false;
}

// The instance extensions a headless surface needs.
inline std::vector<const char*> headless_surface_extensions() {
  return {VK_KHR_SURFACE_EXTENSION_NAME,
          VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
}

// The test policy's instance, validated as the environment asks or at
// `validation`, with the extensions for a headless surface. Above the
// environment's level, create it while a vkc::test::ValidationSession for that
// level lives, as the layer reads its settings then.
inline vkc::InstanceConfig headless_instance_config(
    vkc::test::Validation validation = vkc::test::requested_validation()) {
  vkc::InstanceConfig config = vkc::test::instance_config(validation);
  config.extensions = headless_surface_extensions();
  return config;
}

}  // namespace vg_test

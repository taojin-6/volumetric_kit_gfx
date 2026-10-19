// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// gfx_test_support.hpp: the fixtures ask for the renderer's requirements, and
// the instance for a surface is validated as the environment asks, or at a
// level a test gives.

#include "gfx_test_support.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <vector>

#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"

namespace {

namespace test = vkc::test;

using RendererFixtureTest = vg_test::RendererTest;

// The core fixture's default is its own floor (Vulkan 1.2, any compute
// queue); gfx's tests need a device every pass can record on.
TEST_F(RendererFixtureTest, SelectsADeviceForTheRenderersRequirements) {
  const vkc::DeviceRequirements asked = requirements();
  const vkc::DeviceRequirements renderer = vg::device_requirements();
  EXPECT_EQ(asked.api_version, renderer.api_version);
  EXPECT_EQ(asked.queue_flags, renderer.queue_flags);
  EXPECT_EQ(asked.dynamic_rendering, renderer.dynamic_rendering);
  EXPECT_EQ(asked.timeline_semaphore, renderer.timeline_semaphore);
  const auto support = vkc::check_device_support(physical(), renderer);
  EXPECT_TRUE(support.ok()) << support.status().message();
}

// The instance for a surface is made, and runs the validation the environment
// asks for. VK_KHR_surface, which every surface source names, is offered
// wherever a driver presents.
TEST_F(RendererFixtureTest, SurfaceInstanceMeetsTheEnvironmentsValidation) {
  auto instance = vkc::Instance::create(
      vg_test::surface_instance_config({VK_KHR_SURFACE_EXTENSION_NAME}));
  ASSERT_TRUE(instance.ok()) << instance.status().message();
  const vkc::Status loaded = test::check_layer_loaded(instance.value());
  EXPECT_TRUE(loaded.ok()) << loaded.message();
}

TEST(SurfaceInstanceConfigTest, ValidatesAsTheEnvironmentAsks) {
  const test::ScopedEnv sync("VKC_TEST_SYNC_VALIDATION", nullptr);
  {
    const test::ScopedEnv on("VKC_TEST_VALIDATION", "1");
    EXPECT_TRUE(vg_test::surface_instance_config({}).enable_validation);
  }
  {
    const test::ScopedEnv off("VKC_TEST_VALIDATION", nullptr);
    EXPECT_FALSE(vg_test::surface_instance_config({}).enable_validation);
  }
}

// A test that validates above the environment asks at its own level.
TEST(SurfaceInstanceConfigTest, ValidatesAtALevelGiven) {
  const test::ScopedEnv sync("VKC_TEST_SYNC_VALIDATION", nullptr);
  const test::ScopedEnv off("VKC_TEST_VALIDATION", nullptr);
  EXPECT_TRUE(vg_test::surface_instance_config({}, test::Validation::Sync)
                  .enable_validation);
  EXPECT_FALSE(vg_test::surface_instance_config({}, test::Validation::Off)
                   .enable_validation);
}

TEST(SurfaceInstanceConfigTest, AsksForTheSurfaceExtensions) {
  const std::vector<const char*> extensions =
      vg_test::surface_instance_config({VK_KHR_SURFACE_EXTENSION_NAME,
                                        VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME})
          .extensions;
  ASSERT_EQ(extensions.size(), 2U);
  EXPECT_EQ(std::string_view(extensions[0]), VK_KHR_SURFACE_EXTENSION_NAME);
  EXPECT_EQ(std::string_view(extensions[1]),
            VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
}

}  // namespace

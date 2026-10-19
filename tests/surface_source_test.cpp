// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// windowing::SurfaceSource's constructors: each one's extensions, and what
// each refuses before and after an instance exists. Making a headless surface
// is windowing_test's; making a Metal one is metal_surface_test's, on Apple.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/gfx/windowing/surface_source.hpp"

namespace win = volumetric_kit::gfx::windowing;

namespace {

// Whether the loader offers `name` as an instance extension, asked apart from
// the code under test.
bool loader_offers(const char* name) {
  std::uint32_t count = 0;
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) !=
      VK_SUCCESS) {
    return false;
  }
  std::vector<VkExtensionProperties> offered(count);
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, offered.data()) !=
      VK_SUCCESS) {
    return false;
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    if (std::strcmp(offered[i].extensionName, name) == 0) {
      return true;
    }
  }
  return false;
}

std::vector<std::string_view> names(const std::vector<const char*>& list) {
  return {list.begin(), list.end()};
}

// Any non-null address: a refusal that comes first never reads it.
const void* placeholder_layer() {
  static const int kLayer = 0;
  return &kLayer;
}

TEST(SurfaceSourceTest, HeadlessIsOfferedExactlyWhereTheLoaderOffersIt) {
  const vkc::Result<win::SurfaceSource> headless = win::headless_surface();
  if (!loader_offers(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME)) {
    ASSERT_FALSE(headless.ok());
    EXPECT_EQ(headless.status().domain(), vkc::Status::Code::Unsupported)
        << headless.status().message();
    return;
  }
  ASSERT_TRUE(headless.ok()) << headless.status().message();
  EXPECT_EQ(
      names(headless.value().instance_extensions),
      (std::vector<std::string_view>{VK_KHR_SURFACE_EXTENSION_NAME,
                                     VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME}));
  EXPECT_TRUE(static_cast<bool>(headless.value().make_surface));
}

// With no driver, the loader offers no headless surface, whatever the
// machine's driver does.
TEST(SurfaceSourceTest, HeadlessIsUnsupportedWithoutADriver) {
  const std::string no_driver =
      ::testing::TempDir() + "vg_surface_source_no_driver.json";
  const vkc::test::ScopedEnv driver_files("VK_DRIVER_FILES", no_driver.c_str());
  const vkc::test::ScopedEnv icd_filenames("VK_ICD_FILENAMES",
                                           no_driver.c_str());
  ASSERT_FALSE(loader_offers(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME));
  const vkc::Result<win::SurfaceSource> headless = win::headless_surface();
  ASSERT_FALSE(headless.ok());
  EXPECT_EQ(headless.status().domain(), vkc::Status::Code::Unsupported)
      << headless.status().message();
}

TEST(SurfaceSourceTest, MetalNamesItsExtensions) {
  const vkc::Result<win::SurfaceSource> metal =
      win::metal_surface(placeholder_layer());
  ASSERT_TRUE(metal.ok()) << metal.status().message();
  // By name: vulkan.h declares VK_EXT_metal_surface's macros only with
  // VK_USE_PLATFORM_METAL_EXT.
  EXPECT_EQ(names(metal.value().instance_extensions),
            (std::vector<std::string_view>{VK_KHR_SURFACE_EXTENSION_NAME,
                                           "VK_EXT_metal_surface"}));
  EXPECT_TRUE(static_cast<bool>(metal.value().make_surface));
}

TEST(SurfaceSourceTest, MetalRefusesANullLayer) {
  const vkc::Result<win::SurfaceSource> metal = win::metal_surface(nullptr);
  ASSERT_FALSE(metal.ok());
  EXPECT_EQ(metal.status().domain(), vkc::Status::Code::InvalidArgument);
}

// An instance with VK_KHR_surface alone, which every surface source names.
class SurfaceSourceInstanceTest : public vg_test::RendererTest {
 protected:
  void SetUp() override {
    RendererTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto instance = vkc::Instance::create(vg_test::surface_instance_config(
        {VK_KHR_SURFACE_EXTENSION_NAME}, active_validation()));
    ASSERT_TRUE(instance.ok()) << instance.status().message();
    instance_.emplace(std::move(instance).value());
  }

  std::optional<vkc::Instance> instance_;
};

// The loader resolves no entry point of an extension the instance did not
// enable, so the factory refuses rather than calls one.
TEST_F(SurfaceSourceInstanceTest,
       MetalFactoryIsUnsupportedWithoutTheExtension) {
  const vkc::Result<win::SurfaceSource> metal =
      win::metal_surface(placeholder_layer());
  ASSERT_TRUE(metal.ok()) << metal.status().message();
  const vkc::Result<VkSurfaceKHR> surface =
      metal.value().make_surface(instance_->handle());
  ASSERT_FALSE(surface.ok());
  EXPECT_EQ(surface.status().domain(), vkc::Status::Code::Unsupported)
      << surface.status().message();
}

TEST_F(SurfaceSourceInstanceTest,
       HeadlessFactoryIsUnsupportedWithoutTheExtension) {
  const vkc::Result<win::SurfaceSource> headless = win::headless_surface();
  if (!headless.ok()) {
    GTEST_SKIP() << headless.status().message();
  }
  const vkc::Result<VkSurfaceKHR> surface =
      headless.value().make_surface(instance_->handle());
  ASSERT_FALSE(surface.ok());
  EXPECT_EQ(surface.status().domain(), vkc::Status::Code::Unsupported)
      << surface.status().message();
}

}  // namespace

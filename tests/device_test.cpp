// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <utility>

#include "vulkan_test_fixture.hpp"

namespace {

// The shared instance + physical-device + headless logical-device fixture.
using DeviceTest = VulkanDeviceTest;

}  // namespace

TEST_F(DeviceTest, ExposesGraphicsQueueAndCommandPool) {
  EXPECT_NE(device_->handle(), VK_NULL_HANDLE);
  EXPECT_NE(device_->graphics_queue(), VK_NULL_HANDLE);
  EXPECT_NE(device_->command_pool(), VK_NULL_HANDLE);
  EXPECT_FALSE(device_->has_present());  // headless config
}

TEST_F(DeviceTest, SingleTimeSubmitRoundTrips) {
  // No-op recording exercises allocate / begin / end / submit / fence-wait.
  vg::Status status = device_->submit_single_time([](VkCommandBuffer) {});
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_F(DeviceTest, NeedsPresentWithoutSurfaceErrors) {
  vg::DeviceConfig config;
  config.needs_present = true;
  auto device =
      vg::Device::create(instance_->handle(), physical_, config);  // no surface
  ASSERT_FALSE(device.ok());
  EXPECT_EQ(device.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(DeviceTest, SelectedDeviceMeetsVulkan12Floor) {
  // Device::create rejects a sub-1.2 device (the timeline-semaphore path uses
  // 1.2 core entry points); the fixture device was created successfully, so the
  // selected physical device must report at least 1.2.
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_, &props);
  EXPECT_GE(props.apiVersion, VK_API_VERSION_1_2);
}

TEST_F(DeviceTest, BogusExtensionFailsUnsupported) {
  vg::DeviceConfig config;
  config.extra_device_extensions = {"VK_VG_definitely_not_a_real_extension"};
  auto device = vg::Device::create(instance_->handle(), physical_, config);
  ASSERT_FALSE(device.ok());
  EXPECT_EQ(device.status().domain(), vg::Status::Code::Unsupported);
}

TEST_F(DeviceTest, KnownExtensionRequestSucceeds) {
  // Request an extension the device actually reports, so create() must accept
  // it.
  auto caps = instance_->query_physical_device(physical_);
  const char* candidate = nullptr;
  if (caps.supports_device_extension("VK_KHR_portability_subset")) {
    candidate = "VK_KHR_portability_subset";
  } else if (caps.supports_device_extension(
                 VK_KHR_MAINTENANCE2_EXTENSION_NAME)) {
    candidate = VK_KHR_MAINTENANCE2_EXTENSION_NAME;
  }
  if (candidate == nullptr) {
    GTEST_SKIP() << "device reports no extension to request";
  }
  vg::DeviceConfig config;
  config.extra_device_extensions = {candidate};
  auto device = vg::Device::create(instance_->handle(), physical_, config);
  EXPECT_TRUE(device.ok()) << device.status().message();
}

TEST_F(DeviceTest, CapsReportsSaneExtensionsAndFormats) {
  const vg::PhysicalDeviceInfo& caps = device_->caps();
  EXPECT_EQ(caps.handle(), physical_);
  EXPECT_FALSE(caps.supports_device_extension("VK_VG_not_real"));
  // R8G8B8A8_UNORM as a sampled OPTIMAL image is required of every conformant
  // implementation — true on MoltenVK and lavapipe.
  EXPECT_TRUE(caps.format_supports(VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_IMAGE_TILING_OPTIMAL,
                                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT));
  EXPECT_FALSE(caps.format_supports(VK_FORMAT_UNDEFINED,
                                    VK_IMAGE_TILING_OPTIMAL,
                                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT));
  EXPECT_GT(caps.limits().maxImageDimension2D, 0u);
}

TEST_F(DeviceTest, InstanceCapsMatchesDeviceCaps) {
  // The same physical device, queried via the instance and via the created
  // device, agrees on extension support.
  auto via_instance = instance_->query_physical_device(physical_);
  EXPECT_EQ(via_instance.supports_device_extension("VK_KHR_swapchain"),
            device_->caps().supports_device_extension("VK_KHR_swapchain"));
}

TEST_F(DeviceTest, MoveConstructTransfersOwnership) {
  auto made =
      vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::Device source = std::move(made).value();
  ASSERT_NE(source.handle(), VK_NULL_HANDLE);

  vg::Device moved(std::move(source));
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  // If the move had not nulled the source, both the moved-from device and
  // `moved` would vkDestroyDevice the same handle at scope exit — a validation
  // error.
  EXPECT_EQ(source.handle(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
  // Metadata is zeroed too, not just the owned handles: a moved-from device
  // reports an empty physical device (the recurring "forgot a scalar" miss).
  EXPECT_EQ(source.physical_device(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(DeviceTest, MoveAssignOverLiveDeviceLeavesSourceEmpty) {
  auto a =
      vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
  auto b =
      vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Device dst = std::move(a).value();
  vg::Device src = std::move(b).value();

  dst = std::move(src);  // frees dst's original VkDevice, then adopts src's
  EXPECT_NE(dst.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(src.handle(), VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(DeviceTest, SelfMoveAssignIsSafe) {
  auto made =
      vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::Device device = std::move(made).value();

  // Pointer-laundered so -Wself-move stays quiet under -Werror.
  vg::Device* alias = &device;
  device = std::move(*alias);
  EXPECT_NE(device.handle(), VK_NULL_HANDLE);
}

// Instance creation needs only the loader, so this runs without a GPU; it still
// skips on a truly Vulkan-less host.
TEST(InstanceTest, MoveConstructLeavesSourceEmpty) {
  auto instance = vg::Instance::create(vg::InstanceConfig{});
  if (!instance.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
  }
  vg::Instance source = std::move(instance).value();
  ASSERT_NE(source.handle(), VK_NULL_HANDLE);

  vg::Instance moved(std::move(source));
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.handle(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST(InstanceTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto a = vg::Instance::create(vg::InstanceConfig{});
  if (!a.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << a.status().message();
  }
  auto b = vg::Instance::create(vg::InstanceConfig{});
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Instance dst = std::move(a).value();
  vg::Instance src = std::move(b).value();

  dst = std::move(src);  // destroys dst's original instance, then adopts src's
  EXPECT_NE(dst.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(src.handle(), VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST(InstanceTest, SelfMoveAssignIsSafe) {
  auto created = vg::Instance::create(vg::InstanceConfig{});
  if (!created.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << created.status().message();
  }
  vg::Instance instance = std::move(created).value();

  vg::Instance* alias = &instance;
  instance = std::move(*alias);
  EXPECT_NE(instance.handle(), VK_NULL_HANDLE);
}

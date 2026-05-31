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

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

// A real instance, a portably-selected physical device, and a headless logical
// device. Skips the whole suite when the runner exposes no Vulkan device (a
// GPU-less CI runner without a software ICD). The instance is declared before
// the device so reverse member-destruction tears the device down first —
// exactly the ordering Device::create's contract requires.
class DeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto instance = vg::Instance::create(vg::InstanceConfig{});
    if (!instance.ok()) {
      GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
    }
    instance_.emplace(std::move(instance).value());

    auto physical = instance_->select_physical_device();
    if (!physical.ok()) {
      GTEST_SKIP() << "no Vulkan device: " << physical.status().message();
    }
    physical_ = physical.value();

    auto device =
        vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
    ASSERT_TRUE(device.ok())
        << "device creation failed: " << device.status().message();
    device_.emplace(std::move(device).value());
  }

  std::optional<vg::Instance> instance_;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  std::optional<vg::Device> device_;
};

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
  EXPECT_EQ(device.status().code(), VK_ERROR_INITIALIZATION_FAILED);
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

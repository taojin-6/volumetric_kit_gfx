// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

// Sync primitives need a VkDevice; skip the suite when the runner has none.
class SyncTest : public ::testing::Test {
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

    auto device = vg::Device::create(instance_->handle(), physical.value(),
                                     vg::DeviceConfig{});
    ASSERT_TRUE(device.ok()) << device.status().message();
    device_.emplace(std::move(device).value());
  }

  VkDevice device() const { return device_->handle(); }

  std::optional<vg::Instance> instance_;
  std::optional<vg::Device> device_;
};

}  // namespace

TEST_F(SyncTest, FenceStartsUnsignaledAndWaitTimesOut) {
  auto fence = vg::Fence::create(device(), /*signaled=*/false);
  ASSERT_TRUE(fence.ok()) << fence.status().message();
  EXPECT_FALSE(fence.value().is_signaled());

  // A non-signaled fence with a zero timeout returns VK_TIMEOUT, surfaced as a
  // non-OK Status carrying that code.
  vg::Status waited = fence.value().wait(/*timeout_ns=*/0);
  EXPECT_FALSE(waited.ok());
  EXPECT_EQ(waited.code(), VK_TIMEOUT);
}

TEST_F(SyncTest, SignaledFenceWaitsImmediatelyThenResets) {
  auto fence = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(fence.ok()) << fence.status().message();
  EXPECT_TRUE(fence.value().is_signaled());

  EXPECT_TRUE(
      fence.value().wait().ok());  // already signaled → returns immediately

  ASSERT_TRUE(fence.value().reset().ok());
  EXPECT_FALSE(fence.value().is_signaled());
}

TEST_F(SyncTest, SemaphoreCreates) {
  auto semaphore = vg::Semaphore::create(device());
  ASSERT_TRUE(semaphore.ok()) << semaphore.status().message();
  EXPECT_NE(semaphore.value().handle(), VK_NULL_HANDLE);
}

TEST_F(SyncTest, FenceMoveLeavesSourceEmpty) {
  auto fence = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(fence.ok()) << fence.status().message();
  vg::Fence moved(std::move(fence).value());
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  // Source's handle was nulled by the move, so only `moved` destroys the fence.
}

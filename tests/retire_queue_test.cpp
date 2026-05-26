// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

// One GPU smoke test that a real VkFence flows through RetireQueue's
// vkGetFenceStatus path. The ordering/run-once/drain logic is covered without a
// device in retire_list_test.cpp.
class RetireQueueTest : public ::testing::Test {
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

TEST_F(RetireQueueTest, SignaledFenceReleasesDeleterOnPoll) {
  auto fence = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(fence.ok()) << fence.status().message();

  int released = 0;
  vg::RetireQueue retire(device());
  retire.push(fence.value().handle(), [&released]() { ++released; });
  EXPECT_EQ(retire.pending(), 1u);

  EXPECT_EQ(retire.poll(), 1u);  // real vkGetFenceStatus reports signaled
  EXPECT_EQ(released, 1);
  EXPECT_EQ(retire.pending(), 0u);
}

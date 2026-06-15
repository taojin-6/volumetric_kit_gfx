// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file vulkan_test_fixture.hpp
/// Shared GoogleTest fixture for the GPU-touching core tests: a real instance,
/// a portably-selected physical device, and a headless logical device. The
/// whole suite skips when the runner exposes no Vulkan device (a GPU-less CI
/// runner without a software ICD). instance_ is declared before device_ so
/// reverse member destruction tears the device down first — the ordering
/// Device::create's contract requires.

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"

namespace vg = volumetric_kit::gfx;

class VulkanDeviceTest : public ::testing::Test {
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
    ASSERT_TRUE(device.ok()) << device.status().message();
    device_.emplace(std::move(device).value());
  }

  VkDevice device() const { return device_->handle(); }

  // Submits `cmd` on the graphics queue gated by a throwaway fence and blocks
  // until it retires. Fails the current test (without aborting it) on a submit
  // or wait error. Shared by the command-buffer and offscreen-readback tests,
  // which all issue a single one-time-submit buffer and read the result back.
  void submit_and_wait(VkCommandBuffer cmd) {
    auto fence = vg::Fence::create(device());
    ASSERT_TRUE(fence.ok()) << fence.status().message();
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    ASSERT_EQ(vkQueueSubmit(device_->graphics_queue(), 1, &submit,
                            fence.value().handle()),
              VK_SUCCESS);
    ASSERT_TRUE(fence.value().wait().ok());
  }

  std::optional<vg::Instance> instance_;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  std::optional<vg::Device> device_;
};

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

  std::optional<vg::Instance> instance_;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  std::optional<vg::Device> device_;
};

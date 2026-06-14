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

  // Records a copy of `image` (which must already be in TRANSFER_SRC_OPTIMAL)
  // into the host-visible `buffer`, followed by a buffer barrier making the
  // copy visible to a host read. Shared by every offscreen-readback test.
  static void record_copy_image_to_host(VkCommandBuffer cmd, VkImage image,
                                        VkBuffer buffer, VkExtent2D extent) {
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buffer, 1, &copy);

    VkBufferMemoryBarrier to_host{};
    to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.buffer = buffer;
    to_host.offset = 0;
    to_host.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &to_host,
                         0, nullptr);
  }

  std::optional<vg::Instance> instance_;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  std::optional<vg::Device> device_;
};

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
///
/// A fixture overrides wants_validation() to return true to run under the
/// validation layer with teeth: an error-recording messenger makes TearDown
/// fail the test on any VUID. Best-effort — when the layer is unavailable
/// (local dev without it, or a manifest whose library fails to load) the test
/// still runs, just without teeth (as it always did); CI (lavapipe + the layer)
/// gets them. The plain device tests leave wants_validation() at its default
/// (false) and are unaffected.

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"

namespace vg = volumetric_kit::gfx;

class VulkanDeviceTest : public ::testing::Test {
 protected:
  // Fixtures exercising raw Vulkan (barriers, copies, uploads) override this to
  // opt into validation-with-teeth. Defaults off so the plain device tests keep
  // running validation-free (and without needing the layer installed).
  virtual bool wants_validation() const { return false; }

  void SetUp() override {
    vg::InstanceConfig icfg;
    icfg.enable_validation = wants_validation();
    auto instance = vg::Instance::create(icfg);
    if (!instance.ok() && wants_validation()) {
      // The layer was requested but the instance failed (e.g. a manifest whose
      // library will not load locally); retry without it so the test still
      // runs, just without teeth.
      instance = vg::Instance::create(vg::InstanceConfig{});
    }
    if (!instance.ok()) {
      GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
    }
    instance_.emplace(std::move(instance).value());

    // Teeth only when the validation layer actually loaded: debug_utils rides
    // on validation here, so its presence is the signal the messenger can route
    // captured VUIDs to us.
    if (wants_validation() && instance_->debug_utils_enabled()) {
      install_validation_capture();
    }

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

  void TearDown() override {
    if (device_) {
      vkDeviceWaitIdle(device_->handle());
    }
    // Destroy the messenger (created on the instance) before the fixture tears
    // the instance down, then surface any captured validation errors.
    if (messenger_ != VK_NULL_HANDLE && destroy_messenger_ != nullptr) {
      destroy_messenger_(instance_->handle(), messenger_, nullptr);
      messenger_ = VK_NULL_HANDLE;
    }
    for (const std::string& msg : validation_errors_) {
      ADD_FAILURE() << "Vulkan validation error: " << msg;
    }
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

 private:
  // Records ERROR-severity validation messages into validation_errors_ (passed
  // via pUserData); returns VK_FALSE so the driver never aborts the call.
  static VKAPI_ATTR VkBool32 VKAPI_CALL record_validation_error(
      VkDebugUtilsMessageSeverityFlagBitsEXT severity,
      VkDebugUtilsMessageTypeFlagsEXT,
      const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0 &&
        user != nullptr) {
      static_cast<std::vector<std::string>*>(user)->emplace_back(
          data != nullptr && data->pMessage != nullptr ? data->pMessage
                                                       : "(validation error)");
    }
    return VK_FALSE;
  }

  // Attach an error-recording messenger to instance_ (a no-op when
  // VK_EXT_debug_utils is unavailable, leaving capture inert).
  void install_validation_capture() {
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_->handle(),
                              "vkCreateDebugUtilsMessengerEXT"));
    destroy_messenger_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_->handle(),
                              "vkDestroyDebugUtilsMessengerEXT"));
    if (create == nullptr || destroy_messenger_ == nullptr) {
      return;  // VK_EXT_debug_utils not enabled; capture stays inert.
    }
    VkDebugUtilsMessengerCreateInfoEXT mcfg{};
    mcfg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    mcfg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    mcfg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    mcfg.pfnUserCallback = &VulkanDeviceTest::record_validation_error;
    mcfg.pUserData = &validation_errors_;
    if (create(instance_->handle(), &mcfg, nullptr, &messenger_) !=
        VK_SUCCESS) {
      messenger_ = VK_NULL_HANDLE;
    }
  }

  std::vector<std::string> validation_errors_;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger_ = nullptr;
};

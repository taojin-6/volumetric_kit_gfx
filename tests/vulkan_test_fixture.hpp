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
/// validation layer with teeth: the core's instance routes the layer's messages
/// to the family's log sink (source "vulkan"), and an error-recording handler
/// there makes TearDown fail the test on any VUID. Best-effort — when the layer
/// is unavailable (local dev without it, or a manifest whose library fails to
/// load) the core's instance continues without it, so the test still runs,
/// just without teeth; CI (lavapipe + the layer) gets them. The plain device
/// tests leave wants_validation() at its default (false) and are unaffected. A
/// fixture that needs more of the device overrides requirements().

#include <gtest/gtest.h>

#include <cstdio>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

class VulkanDeviceTest : public ::testing::Test {
 protected:
  // Fixtures exercising raw Vulkan (barriers, copies, uploads) override this to
  // opt into validation-with-teeth. Defaults off so the plain device tests keep
  // running validation-free (and without needing the layer installed).
  virtual bool wants_validation() const { return false; }
  // What the fixture's device must provide; the renderer's floor by default.
  virtual vg::DeviceRequirements requirements() const {
    return vg::device_requirements();
  }

  void SetUp() override {
    vg::InstanceConfig icfg;
    icfg.enable_validation = wants_validation();
    auto instance = vg::Instance::create(icfg);
    if (!instance.ok()) {
      GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
    }
    instance_.emplace(std::move(instance).value());

    // Teeth only when the layer loaded and its messages reach the log sink.
    if (wants_validation() && instance_->validation_logged()) {
      install_validation_capture();
    }

    const vg::DeviceRequirements reqs = requirements();
    auto physical = instance_->select_physical_device(reqs);
    if (!physical.ok()) {
      GTEST_SKIP() << "no Vulkan device: " << physical.status().message();
    }
    caps_ = physical.value();

    auto device = vg::Device::create(*instance_, caps_, reqs);
    ASSERT_TRUE(device.ok()) << device.status().message();
    device_.emplace(std::move(device).value());
  }

  void TearDown() override {
    if (device_) {
      vkDeviceWaitIdle(device_->handle());
    }
    // The device and instance stay up: a derived fixture's members (an
    // allocator, its resources) are destroyed after this, and before them.
    // Restore the default sink -- the handler captures `this` -- and fail the
    // test on what the layer reported.
    if (capturing_) {
      vg::set_log_handler({});
      capturing_ = false;
    }
    const std::lock_guard<std::mutex> lock(validation_mutex_);
    for (const std::string& msg : validation_errors_) {
      ADD_FAILURE() << "Vulkan validation error: " << msg;
    }
  }

  // True when the base SetUp did not finish, so device_/instance_ must not be
  // touched. A derived SetUp has to check BOTH conditions: ASSERT_ returns only
  // from the function it fires in, so when Device::create fails above, the base
  // SetUp aborts before device_.emplace and control comes straight back here
  // with IsSkipped() false -- a derived fixture guarding on the skip alone
  // would then dereference a disengaged std::optional and segfault, turning a
  // reportable failure into a crash that takes the rest of the suite's output
  // with it.
  bool base_setup_incomplete() const {
    return IsSkipped() || HasFatalFailure();
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
    ASSERT_EQ(device_->queue_submit(1, &submit, fence.value().handle()),
              VK_SUCCESS);
    ASSERT_TRUE(fence.value().wait().ok());
  }

  std::optional<vg::Instance> instance_;
  vg::PhysicalDeviceInfo caps_;
  std::optional<vg::Device> device_;

 private:
  // Route the core's validation messages (source "vulkan", level Error) into
  // validation_errors_; anything else still reaches stderr, as the default
  // sink would print it.
  void install_validation_capture() {
    capturing_ = true;
    vg::set_log_handler([this](vg::LogLevel level, std::string_view source,
                               std::string_view message) {
      if (source == "vulkan" && level == vg::LogLevel::Error) {
        const std::lock_guard<std::mutex> lock(validation_mutex_);
        validation_errors_.emplace_back(message);
      } else if (level >= vg::LogLevel::Warning) {
        std::fprintf(stderr, "[%.*s] %.*s\n", static_cast<int>(source.size()),
                     source.data(), static_cast<int>(message.size()),
                     message.data());
      }
    });
  }

  bool capturing_ = false;
  std::mutex validation_mutex_;
  std::vector<std::string> validation_errors_;
};

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file vulkan_test_fixture.hpp
/// Shared GoogleTest fixture for the GPU-touching tests: a real instance, a
/// portably-selected physical device, and a headless logical device. The
/// whole suite skips when the runner exposes no Vulkan device (a GPU-less CI
/// runner without a software ICD).
///
/// The instance and device are shared: every test in the process that asks
/// for the same instance setup (the wants_*_validation() overrides below)
/// borrows one instance and device, made by the first such test and kept
/// until the process ends. Each test still makes and destroys its own objects
/// on it. On NVIDIA's Linux driver a create/destroy cycle costs ~90 ms that
/// the driver serializes machine-wide, and after 26 of them in one process
/// vkCreateInstance fails (libnvidia-tls runs out of static TLS as the loader
/// reloads the driver), so sharing is what lets many tests run per process.
/// See DECISIONS.md, "GPU tests share a device per process".
///
/// A fixture overrides wants_validation() to return true to run under the
/// validation layer with teeth: the core's instance routes the layer's messages
/// to the family's log sink (source "vulkan"), and an error-recording handler
/// there makes TearDown fail the test on any VUID. Best-effort — when the layer
/// is unavailable (local dev without it, or a manifest whose library fails to
/// load) the core's instance continues without it, so the test still runs,
/// just without teeth; CI (lavapipe + the layer) gets them. The plain device
/// tests leave wants_validation() at its default (false) and are unaffected. A
/// fixture whose tests hinge on the barriers between recorded commands also
/// overrides wants_sync_validation(), and one whose shaders write what its
/// barriers guard overrides wants_shader_access_validation(). A fixture that
/// needs a device made to other requirements overrides custom_requirements(),
/// and gets an instance and device of its own for each test.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/core/log.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

namespace vg_test {

// Sets an environment variable for its lifetime, then restores the value it
// had, or unsets it.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      old_ = old;
    }
    set(name, value);
  }
  ~ScopedEnv() {
    if (old_) {
      set(name_, old_->c_str());
    } else {
#ifdef _WIN32
      _putenv_s(name_, "");
#else
      unsetenv(name_);
#endif
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  static void set(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
  }
  const char* name_;
  std::optional<std::string> old_;
};

// The instance a fixture asks for: which of the validation layer's checks it
// runs under. Each implies the one before it.
struct VulkanSetup {
  bool validation = false;
  bool sync = false;
  bool shader_accesses = false;

  bool operator<(const VulkanSetup& other) const {
    return std::tie(validation, sync, shader_accesses) <
           std::tie(other.validation, other.sync, other.shader_accesses);
  }
};

// An instance and a device on it, or why there are none.
struct VulkanContext {
  std::optional<vkc::Instance> instance;
  vkc::PhysicalDeviceInfo caps;
  std::optional<vkc::Device> device;
  // Set when there is no device: `skip` for a runner without Vulkan or a
  // suitable device (the test skips), otherwise a failure (the test fails).
  bool skip = false;
  std::string error;
};

// Make an instance with `setup`'s checks and a device meeting `reqs` on it.
inline void make_context(const VulkanSetup& setup,
                         const vkc::DeviceRequirements& reqs,
                         VulkanContext& out) {
  vkc::InstanceConfig icfg;
  icfg.enable_validation = setup.validation;
  // The layer reads these settings when the instance is created; keep them
  // from later instances.
  std::optional<ScopedEnv> sync;
  std::optional<ScopedEnv> shader_accesses;
  if (setup.sync) {
    sync.emplace("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "true");
  }
  if (setup.shader_accesses) {
    shader_accesses.emplace(
        "VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC", "true");
  }
  auto instance = vkc::Instance::create(icfg);
  shader_accesses.reset();
  sync.reset();
  if (!instance.ok()) {
    out.skip = true;
    out.error = "no Vulkan instance: " + instance.status().message();
    return;
  }
  out.instance.emplace(std::move(instance).value());

  auto physical = out.instance->select_physical_device(reqs);
  if (!physical.ok()) {
    out.skip = true;
    out.error = "no Vulkan device: " + physical.status().message();
    return;
  }
  out.caps = physical.value();

  auto device = vkc::Device::create(*out.instance, out.caps, reqs);
  if (!device.ok()) {
    out.error = "Device::create: " + device.status().message();
    return;
  }
  out.device.emplace(std::move(device).value());
}

// The process's shared instances and devices, one per VulkanSetup, made to
// the renderer's requirements on first use and kept until the process ends.
class SharedContexts {
 public:
  static SharedContexts& get() {
    static SharedContexts contexts;
    return contexts;
  }

  // The context for `setup`, made now if this is its first use. A setup that
  // could not be made is not retried: every later test reports the same.
  VulkanContext& acquire(const VulkanSetup& setup) {
    Entry& entry = entries_[setup];
    if (entry.lost) {
      // The test that lost it is gone, and with it whatever it made on the
      // device, so it can go now and be made again: the device before the
      // instance it was made from.
      entry.context.device.reset();
      entry.context.instance.reset();
      entry = Entry{};
    }
    if (!entry.made) {
      entry.made = true;
      make_context(setup, vg::device_requirements(), entry.context);
    }
    return entry.context;
  }

  // Replace `setup`'s device at its next acquire(): a test lost it.
  void mark_lost(const VulkanSetup& setup) { entries_[setup].lost = true; }

  // Destroy every shared device and instance, and return the errors the
  // validation layer reported while they went: chiefly an object a test made
  // and never destroyed, which the layer finds only when its device goes.
  std::vector<std::string> release_all() {
    std::vector<std::string> errors;
    vkc::set_log_handler([&errors](vkc::LogLevel level, std::string_view source,
                                   std::string_view message) {
      if (source == "vulkan" && level == vkc::LogLevel::Error) {
        errors.emplace_back(message);
      }
    });
    entries_.clear();
    vkc::set_log_handler({});
    return errors;
  }

 private:
  struct Entry {
    VulkanContext context;
    bool made = false;
    bool lost = false;
  };

  SharedContexts() = default;

  std::map<VulkanSetup, Entry> entries_;
};

// Process-wide setup and teardown around every test.
class VulkanEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // When this process runs more than one test, keep an instance alive
    // throughout, so the loader never unloads the driver between tests that
    // make their own instances: on NVIDIA every reload takes static TLS that
    // is never given back. One test per process (ctest's default here) needs
    // none, and a spare instance would cost it a create and a destroy.
    if (::testing::UnitTest::GetInstance()->test_to_run_count() > 1) {
      auto anchor = vkc::Instance::create(vkc::InstanceConfig{});
      if (anchor.ok()) {
        anchor_.emplace(std::move(anchor).value());
      }
    }
  }

  void TearDown() override {
    for (const std::string& error : SharedContexts::get().release_all()) {
      ADD_FAILURE() << "Vulkan validation error while destroying the shared "
                       "devices (an object some test never destroyed?): "
                    << error;
    }
    anchor_.reset();
  }

 private:
  std::optional<vkc::Instance> anchor_;
};

// Registered once per test binary, before main() runs the tests.
inline ::testing::Environment* const kVulkanEnvironment =
    ::testing::AddGlobalTestEnvironment(new VulkanEnvironment);

}  // namespace vg_test

class VulkanDeviceTest : public ::testing::Test {
 protected:
  // Fixtures exercising raw Vulkan (barriers, copies, uploads) override this to
  // opt into validation-with-teeth. Defaults off so the plain device tests keep
  // running validation-free (and without needing the layer installed).
  virtual bool wants_validation() const { return false; }
  // On top of wants_validation(): turn on the layer's synchronization
  // validation, which reports a missing barrier as a hazard -- core
  // validation does not, and the race it guards is rarely lost in a test.
  // Set through the layer's settings environment variable, which the layer
  // reads when the instance is created; a layer too old to read it runs
  // without it.
  virtual bool wants_sync_validation() const { return false; }
  // On top of wants_sync_validation(): have it track the memory shaders
  // access through their descriptors, from the SPIR-V. Without it the layer
  // (1.4.363) reports no hazard against a compute shader's storage-buffer
  // write. Off by default, as the layer warns it can report false positives.
  virtual bool wants_shader_access_validation() const { return false; }
  // Requirements other than the renderer's floor, for a fixture that needs a
  // device made to them. Such a fixture gets an instance and device of its own
  // for each test, rather than the shared one.
  virtual std::optional<vkc::DeviceRequirements> custom_requirements() const {
    return std::nullopt;
  }

  void SetUp() override {
    const vg_test::VulkanSetup setup = vulkan_setup();
    vg_test::VulkanContext* context = nullptr;
    if (const auto reqs = custom_requirements()) {
      own_ = std::make_unique<vg_test::VulkanContext>();
      vg_test::make_context(setup, *reqs, *own_);
      context = own_.get();
    } else {
      context = &vg_test::SharedContexts::get().acquire(setup);
    }
    if (context->skip) {
      // CI's GPU legs set VG_REQUIRE_VULKAN_DEVICE: there a missing instance
      // or device is a failure, as a shard of tests run in one process would
      // otherwise pass with all of them skipped.
      if (std::getenv("VG_REQUIRE_VULKAN_DEVICE") != nullptr) {
        FAIL() << context->error << " (VG_REQUIRE_VULKAN_DEVICE is set)";
      }
      GTEST_SKIP() << context->error;
    }
    ASSERT_TRUE(context->device.has_value()) << context->error;
    instance_ = &*context->instance;
    device_ = &*context->device;
    caps_ = context->caps;

    // Teeth only when the layer loaded and its messages reach the log sink.
    if (wants_validation() && instance_->validation_logged()) {
      install_validation_capture();
    }
  }

  void TearDown() override {
    if (device_ != nullptr) {
      // The test's work must be done before its objects go, and a shared
      // device must be left idle for the next test.
      const VkResult idle = vkDeviceWaitIdle(device_->handle());
      EXPECT_EQ(idle, VK_SUCCESS) << "vkDeviceWaitIdle after the test";
      if (idle == VK_ERROR_DEVICE_LOST && own_ == nullptr) {
        vg_test::SharedContexts::get().mark_lost(vulkan_setup());
      }
    }
    // The device and instance stay up: a derived fixture's members (an
    // allocator, its resources) are destroyed after this, and before them.
    // Restore the default sink -- the handler captures `this` -- and fail the
    // test on what the layer reported.
    if (capturing_) {
      vkc::set_log_handler({});
      capturing_ = false;
    }
    const std::lock_guard<std::mutex> lock(validation_mutex_);
    for (const std::string& msg : validation_errors_) {
      ADD_FAILURE() << "Vulkan validation error: " << msg;
    }
  }

  // True when the base SetUp did not finish, so device_/instance_ must not be
  // touched. A derived SetUp has to check BOTH conditions: ASSERT_ returns only
  // from the function it fires in, so when the device could not be made the
  // base SetUp aborts before setting device_ and control comes straight back
  // here with IsSkipped() false -- a derived fixture guarding on the skip
  // alone would then dereference a null pointer and segfault, turning a
  // reportable failure into a crash that takes the rest of the suite's output
  // with it.
  bool base_setup_incomplete() const {
    return IsSkipped() || HasFatalFailure();
  }

  VkDevice device() const { return device_->handle(); }

  // Submits `cmd` on the graphics queue gated by a throwaway fence and blocks
  // until it retires. Fails the current test (without aborting it) on a submit
  // or wait error. Shared by the pipeline, offscreen-readback and UI tests,
  // which all issue a single one-time-submit buffer and read the result back.
  void submit_and_wait(VkCommandBuffer cmd) {
    auto fence = vkc::Fence::create(device());
    ASSERT_TRUE(fence.ok()) << fence.status().message();
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    ASSERT_EQ(device_->queue_submit(1, &submit, fence.value().handle()),
              VK_SUCCESS);
    ASSERT_TRUE(fence.value().wait().ok());
  }

  // Borrowed: the shared context's, or own_'s for custom_requirements().
  vkc::Instance* instance_ = nullptr;
  vkc::PhysicalDeviceInfo caps_;
  vkc::Device* device_ = nullptr;

 private:
  vg_test::VulkanSetup vulkan_setup() const {
    vg_test::VulkanSetup setup;
    setup.validation = wants_validation();
    setup.sync = setup.validation && wants_sync_validation();
    setup.shader_accesses = setup.sync && wants_shader_access_validation();
    return setup;
  }

  // Route the core's validation messages (source "vulkan", level Error) into
  // validation_errors_; anything else still reaches stderr, as the default
  // sink would print it.
  void install_validation_capture() {
    capturing_ = true;
    vkc::set_log_handler([this](vkc::LogLevel level, std::string_view source,
                                std::string_view message) {
      if (source == "vulkan" && level == vkc::LogLevel::Error) {
        const std::lock_guard<std::mutex> lock(validation_mutex_);
        validation_errors_.emplace_back(message);
      } else if (level >= vkc::LogLevel::Warning) {
        std::fprintf(stderr, "[%.*s] %.*s\n", static_cast<int>(source.size()),
                     source.data(), static_cast<int>(message.size()),
                     message.data());
      }
    });
  }

  // A custom_requirements() fixture's own instance and device, made per test.
  // Declared after the pointers into it; destroyed with the fixture, after the
  // derived fixture's members.
  std::unique_ptr<vg_test::VulkanContext> own_;
  bool capturing_ = false;
  std::mutex validation_mutex_;
  std::vector<std::string> validation_errors_;
};

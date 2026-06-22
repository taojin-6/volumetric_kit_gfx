// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/debug_label.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"

namespace {

namespace vg = volumetric_kit::gfx;

// A self-contained instance + device with VK_EXT_debug_utils opted in, so the
// resolved table is active wherever the extension is present (and inactive,
// with every label a clean no-op, where it is not). The shared
// VulkanDeviceTest fixture builds its instance with a default config (no
// debug-utils), so this suite stands up its own to exercise the active path —
// while still asserting the no-op behaviour holds when the extension is absent.
class DebugLabelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    vg::InstanceConfig instance_config;
    instance_config.enable_debug_utils = true;  // merged opt-in
    auto instance = vg::Instance::create(instance_config);
    if (!instance.ok()) {
      GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
    }
    instance_.emplace(std::move(instance).value());

    auto physical = instance_->select_physical_device();
    if (!physical.ok()) {
      GTEST_SKIP() << "no Vulkan device: " << physical.status().message();
    }
    physical_ = physical.value();

    vg::DeviceConfig device_config;
    // Thread the instance's decision through: the device-level entry points can
    // only resolve when the instance enabled the extension.
    device_config.enable_debug_utils = instance_->debug_utils_enabled();
    auto device =
        vg::Device::create(instance_->handle(), physical_, device_config);
    ASSERT_TRUE(device.ok()) << device.status().message();
    device_.emplace(std::move(device).value());
  }

  std::optional<vg::Instance> instance_;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  std::optional<vg::Device> device_;
};

}  // namespace

// The device table mirrors the instance flag exactly: active iff the instance
// enabled VK_EXT_debug_utils (which itself depends on the extension being
// present on the loader). This is the single wiring invariant.
TEST_F(DebugLabelTest, TableActiveMatchesInstanceFlag) {
  EXPECT_EQ(device_->debug_utils().active(), instance_->debug_utils_enabled());
}

// End-to-end emit: record a command-buffer label scope and a set_object_name on
// a real handle inside a one-time submit, then open a queue label around the
// submit itself. Whether or not the entry points are live, the submit must
// succeed — labels never turn a working submit into an error.
TEST_F(DebugLabelTest, EmitsLabelsAndObjectNameWithoutError) {
  const vg::DebugUtilsTable& table = device_->debug_utils();

  // A real handle to name: the device's own command pool.
  vg::set_object_name(device_->handle(), table, VK_OBJECT_TYPE_COMMAND_POOL,
                      reinterpret_cast<uint64_t>(device_->command_pool()),
                      "test pool");

  const float color[4] = {0.2f, 0.4f, 0.8f, 1.0f};
  vg::Status record_status =
      device_->submit_single_time([&](VkCommandBuffer cmd) {
        // The scope must open and close while cmd is recording; the inner block
        // ends the region before submit_single_time calls vkEndCommandBuffer.
        {
          vg::DebugLabelScope pass(cmd, table, "pass", color);
          EXPECT_EQ(pass.active(), table.active());
        }
        // A second, sequential region (the first already closed above). This
        // one destructs as the lambda returns, before vkEndCommandBuffer.
        vg::DebugLabelScope upload(cmd, table, "upload");
        EXPECT_EQ(upload.active(), table.active());
      });
  EXPECT_TRUE(record_status.ok()) << record_status.message();

  // Queue label around an empty submit: open a region on the graphics queue,
  // submit a no-op one-time buffer through the device helper, then close it.
  {
    vg::QueueLabelScope queue_scope(device_->graphics_queue(), table, "frame",
                                    color);
    EXPECT_EQ(queue_scope.active(), table.active());
    vg::Status submit_status =
        device_->submit_single_time([](VkCommandBuffer) {});
    EXPECT_TRUE(submit_status.ok()) << submit_status.message();
  }
}

// move-construct: the moved-from source becomes inert, so only ONE End is
// emitted at scope exit (the destination's). A copyable scope, or a move that
// left the source active, would emit a second vkCmdEndDebugUtilsLabelEXT — a
// validation error / double-end. We assert the flag transition; the validation
// layer (when present on the sanitizers job) catches an actual double-end.
TEST_F(DebugLabelTest, DebugLabelMoveConstructLeavesSourceInert) {
  const vg::DebugUtilsTable& table = device_->debug_utils();
  vg::Status status = device_->submit_single_time([&](VkCommandBuffer cmd) {
    vg::DebugLabelScope source(cmd, table, "region");
    const bool was_active = source.active();

    vg::DebugLabelScope moved(std::move(source));
    EXPECT_EQ(moved.active(), was_active);
    EXPECT_FALSE(source.active());  // NOLINT(bugprone-use-after-move)
  });
  EXPECT_TRUE(status.ok()) << status.message();
}

// move-assign over a live scope: the destination ends its own region before
// adopting the source's, and the source is left inert — exercising the
// end-then-adopt path where a double-free/double-end would live. `src` opens
// first and `dst` last, so dst's self-end (top of the label stack) and the
// final destruction stay strictly nested for the validation layer.
TEST_F(DebugLabelTest, DebugLabelMoveAssignOverLiveScope) {
  const vg::DebugUtilsTable& table = device_->debug_utils();
  vg::Status status = device_->submit_single_time([&](VkCommandBuffer cmd) {
    vg::DebugLabelScope src(cmd, table, "src region");
    vg::DebugLabelScope dst(cmd, table, "dst region");
    const bool src_active = src.active();

    dst = std::move(src);  // dst ends its own region, then adopts src's
    EXPECT_EQ(dst.active(), src_active);
    EXPECT_FALSE(src.active());  // NOLINT(bugprone-use-after-move)
  });
  EXPECT_TRUE(status.ok()) << status.message();
}

// self-move: pointer-laundered to dodge -Wself-move under -Werror. The scope
// keeps its state and emits exactly one End at exit.
TEST_F(DebugLabelTest, DebugLabelSelfMoveIsSafe) {
  const vg::DebugUtilsTable& table = device_->debug_utils();
  vg::Status status = device_->submit_single_time([&](VkCommandBuffer cmd) {
    vg::DebugLabelScope scope(cmd, table, "region");
    const bool was_active = scope.active();

    vg::DebugLabelScope* alias = &scope;
    scope = std::move(*alias);
    EXPECT_EQ(scope.active(), was_active);
  });
  EXPECT_TRUE(status.ok()) << status.message();
}

// QueueLabelScope move-construct: same inert-source contract as the command
// scope, on the queue entry points.
TEST_F(DebugLabelTest, QueueLabelMoveConstructLeavesSourceInert) {
  const vg::DebugUtilsTable& table = device_->debug_utils();
  vg::QueueLabelScope source(device_->graphics_queue(), table, "queue region");
  const bool was_active = source.active();

  vg::QueueLabelScope moved(std::move(source));
  EXPECT_EQ(moved.active(), was_active);
  EXPECT_FALSE(source.active());  // NOLINT(bugprone-use-after-move)
}

// Inert (default-constructed) scopes never emit and never end; this holds with
// or without a device, so it documents the branch-to-noop contract directly.
TEST(DebugLabelInertTest, DefaultScopesAreInert) {
  vg::DebugLabelScope cmd_scope;
  vg::QueueLabelScope queue_scope;
  EXPECT_FALSE(cmd_scope.active());
  EXPECT_FALSE(queue_scope.active());

  // Moving an inert scope is a no-op and stays inert.
  vg::DebugLabelScope moved(std::move(cmd_scope));
  EXPECT_FALSE(moved.active());
  EXPECT_FALSE(cmd_scope.active());  // NOLINT(bugprone-use-after-move)
}

// An inactive table makes every entry point a no-op: an all-null table never
// dereferences a pointer, so scopes built from it are inert and set_object_name
// returns without touching the (null) device. Runs without a GPU.
TEST(DebugLabelInertTest, InactiveTableNoOps) {
  vg::DebugUtilsTable table;  // all-null
  EXPECT_FALSE(table.active());

  vg::DebugLabelScope cmd_scope(VK_NULL_HANDLE, table, "x");
  EXPECT_FALSE(cmd_scope.active());
  vg::QueueLabelScope queue_scope(VK_NULL_HANDLE, table, "x");
  EXPECT_FALSE(queue_scope.active());

  // No PFN, so this must return without dereferencing anything.
  vg::set_object_name(VK_NULL_HANDLE, table, VK_OBJECT_TYPE_UNKNOWN, 0, "x");
}

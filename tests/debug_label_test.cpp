// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/debug_label.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <optional>
#include <thread>
#include <utility>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"

namespace {

// Runs under the validation layer, on the instance's debug utils (which every
// fixture instance requests), so the scopes take their active path and any
// label the layer rejects -- an unbalanced or double end, a null name, a queue
// touched by two threads at once -- fails the test. Where the layer is
// unavailable the tests still run, without that backstop.
class DebugLabelTest : public vg_test::RendererDeviceTest {
 protected:
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::On;
  }
};

// A test that makes its own instance, under the core fixture's policy for a
// machine without Vulkan.
class DebugLabelOwnInstanceTest : public vg_test::RendererTest {};

}  // namespace

// A null label name, or a null command buffer, leaves the scope inert even on
// the active path: VkDebugUtilsLabelEXT::pLabelName must be non-null, so the
// scope refuses it rather than emit a label the validation layer rejects.
TEST_F(DebugLabelTest, NullNameOrBufferLeavesScopeInert) {
  vkc::Status status = device().submit_single_time([&](VkCommandBuffer cmd) {
    vg::DebugLabelScope scope(device(), cmd, nullptr);
    EXPECT_FALSE(scope.active());
  });
  EXPECT_TRUE(status.ok()) << status.message();

  vg::DebugLabelScope no_buffer(device(), VK_NULL_HANDLE, "x");
  EXPECT_FALSE(no_buffer.active());
  vg::QueueLabelScope queue_scope(device(), nullptr);
  EXPECT_FALSE(queue_scope.active());
}

// End-to-end emit: record command-buffer label scopes inside a one-time
// submit, then open a queue label around a submit. A named scope is active
// exactly where the device has debug labels, and labels never turn a working
// submit into an error. The validation layer checks that what was emitted is
// well formed and balanced; there is no in-process API to capture a label.
TEST_F(DebugLabelTest, EmitsLabelsWithoutError) {
  const bool labels = device().debug_labels_available();
  vkc::Status record_status =
      device().submit_single_time([&](VkCommandBuffer cmd) {
        // The inner block ends the region before submit_single_time ends the
        // command buffer.
        {
          vg::DebugLabelScope pass(device(), cmd, "pass");
          EXPECT_EQ(pass.active(), labels);
          vg::DebugLabelScope nested(device(), cmd, "nested");
          EXPECT_EQ(nested.active(), labels);
        }
        // A second, sequential region; it ends as the lambda returns.
        vg::DebugLabelScope upload(device(), cmd, "upload");
        EXPECT_EQ(upload.active(), labels);
      });
  EXPECT_TRUE(record_status.ok()) << record_status.message();

  vg::QueueLabelScope frame(device(), "frame");
  EXPECT_EQ(frame.active(), labels);
  vkc::Status submit_status =
      device().submit_single_time([](VkCommandBuffer) {});
  EXPECT_TRUE(submit_status.ok()) << submit_status.message();
}

// Vulkan requires the queue be externally synchronized for the queue-label
// calls, as for vkQueueSubmit. One thread opens and closes queue labels while
// another submits through the device, which holds its submit mutex; the
// validation layer's thread-safety check reports any overlap of the two as an
// error. A scope that labels without that mutex fails here.
TEST_F(DebugLabelTest, QueueLabelsAndSubmitsFromTwoThreadsDoNotRace) {
  constexpr int kIterations = 2000;
  std::atomic<bool> go{false};
  std::thread labeller([&] {
    while (!go.load()) {
    }
    for (int i = 0; i < kIterations; ++i) {
      vg::QueueLabelScope frame(device(), "frame");
    }
  });
  std::thread submitter([&] {
    while (!go.load()) {
    }
    for (int i = 0; i < kIterations; ++i) {
      // An empty batch: the shortest call that writes the queue.
      EXPECT_EQ(device().queue_submit(0, nullptr, VK_NULL_HANDLE), VK_SUCCESS);
    }
  });
  go.store(true);
  labeller.join();
  submitter.join();
}

// move-construct: the moved-from source becomes inert, so only the
// destination's end is emitted at scope exit. A move that left the source
// active would emit a second vkCmdEndDebugUtilsLabelEXT, which the validation
// layer reports as unbalanced.
TEST_F(DebugLabelTest, DebugLabelMoveConstructLeavesSourceInert) {
  vkc::Status status = device().submit_single_time([&](VkCommandBuffer cmd) {
    vg::DebugLabelScope source(device(), cmd, "region");
    const bool was_active = source.active();

    vg::DebugLabelScope moved(std::move(source));
    EXPECT_EQ(moved.active(), was_active);
    EXPECT_FALSE(source.active());  // NOLINT(bugprone-use-after-move)
  });
  EXPECT_TRUE(status.ok()) << status.message();
}

// move-assign over a live scope: the destination ends its own region before
// adopting the source's, and the source is left inert. `src` opens first and
// `dst` last, so dst's own end (the top of the label stack) and the final
// destruction stay strictly nested for the validation layer.
TEST_F(DebugLabelTest, DebugLabelMoveAssignOverLiveScope) {
  vkc::Status status = device().submit_single_time([&](VkCommandBuffer cmd) {
    vg::DebugLabelScope src(device(), cmd, "src region");
    vg::DebugLabelScope dst(device(), cmd, "dst region");
    const bool src_active = src.active();

    dst = std::move(src);  // dst ends its own region, then adopts src's
    EXPECT_EQ(dst.active(), src_active);
    EXPECT_FALSE(src.active());  // NOLINT(bugprone-use-after-move)
  });
  EXPECT_TRUE(status.ok()) << status.message();
}

// self-move: pointer-laundered to dodge -Wself-move under -Werror. The scope
// keeps its state and emits exactly one end at exit.
TEST_F(DebugLabelTest, DebugLabelSelfMoveIsSafe) {
  vkc::Status status = device().submit_single_time([&](VkCommandBuffer cmd) {
    vg::DebugLabelScope scope(device(), cmd, "region");
    const bool was_active = scope.active();

    vg::DebugLabelScope* alias = &scope;
    scope = std::move(*alias);
    EXPECT_EQ(scope.active(), was_active);
  });
  EXPECT_TRUE(status.ok()) << status.message();
}

// The queue scope's move-construct: the same inert-source contract, on the
// queue's label stack.
TEST_F(DebugLabelTest, QueueLabelMoveConstructLeavesSourceInert) {
  vg::QueueLabelScope source(device(), "queue region");
  const bool was_active = source.active();

  vg::QueueLabelScope moved(std::move(source));
  EXPECT_EQ(moved.active(), was_active);
  EXPECT_FALSE(source.active());  // NOLINT(bugprone-use-after-move)
}

// The queue scope's move-assign over a live scope, nested as in the command
// buffer case.
TEST_F(DebugLabelTest, QueueLabelMoveAssignOverLiveScope) {
  vg::QueueLabelScope src(device(), "src region");
  vg::QueueLabelScope dst(device(), "dst region");
  const bool src_active = src.active();

  dst = std::move(src);  // dst ends its own region, then adopts src's
  EXPECT_EQ(dst.active(), src_active);
  EXPECT_FALSE(src.active());  // NOLINT(bugprone-use-after-move)
}

// The queue scope's self-move keeps its state and ends once.
TEST_F(DebugLabelTest, QueueLabelSelfMoveIsSafe) {
  vg::QueueLabelScope scope(device(), "region");
  const bool was_active = scope.active();

  vg::QueueLabelScope* alias = &scope;
  scope = std::move(*alias);
  EXPECT_EQ(scope.active(), was_active);
}

// Inert (default-constructed) scopes never emit and never end; this holds
// without a device, so it documents the no-op contract directly.
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

// On a device whose instance did not enable VK_EXT_debug_utils, named scopes
// are inert and emit nothing: the device resolved no label entry points, and
// the queue scope looks none up. The instance is unvalidated: the validation
// messenger needs VK_EXT_debug_utils and would keep it on.
TEST_F(DebugLabelOwnInstanceTest, ScopesAreInertWithoutDebugUtils) {
  vkc::InstanceConfig instance_config;
  instance_config.request_debug_utils = false;
  auto instance = vkc::Instance::create(instance_config);
  ASSERT_TRUE(instance.ok()) << instance.status().message();
  const vkc::DeviceRequirements reqs = vg::device_requirements();
  auto physical = instance.value().select_physical_device(reqs);
  ASSERT_TRUE(physical.ok()) << physical.status().message();
  std::optional<vkc::Device> device;
  {
    auto made = vkc::Device::create(instance.value(), physical.value(), reqs);
    ASSERT_TRUE(made.ok()) << made.status().message();
    device.emplace(std::move(made).value());
  }
  ASSERT_FALSE(device->debug_labels_available());

  vkc::Status status = device->submit_single_time([&](VkCommandBuffer cmd) {
    vg::DebugLabelScope pass(*device, cmd, "pass");
    EXPECT_FALSE(pass.active());
  });
  EXPECT_TRUE(status.ok()) << status.message();
  vg::QueueLabelScope frame(*device, "frame");
  EXPECT_FALSE(frame.active());
}

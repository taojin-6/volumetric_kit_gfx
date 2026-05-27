// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <utility>

#include "volumetric_kit/gfx/core/retire_queue.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// GPU smoke tests: a real VkFence flowing through RetireQueue's
// vkGetFenceStatus path, plus the move-only queue's hand-written move ops. The
// ordering / run-once / drain logic is covered device-free in
// retire_list_test.cpp.
using RetireQueueTest = VulkanDeviceTest;

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

TEST_F(RetireQueueTest, MoveConstructTransfersPendingDeleters) {
  auto fence = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(fence.ok()) << fence.status().message();

  int ran = 0;
  vg::RetireQueue source(device());
  source.push(fence.value().handle(), [&ran]() { ++ran; });
  ASSERT_EQ(source.pending(), 1u);

  vg::RetireQueue moved(std::move(source));
  EXPECT_EQ(moved.pending(), 1u);
  EXPECT_EQ(source.pending(), 0u);  // NOLINT(bugprone-use-after-move)

  // The moved-to queue still observes the device, so poll releases the deleter.
  EXPECT_EQ(moved.poll(), 1u);
  EXPECT_EQ(ran, 1);
}

TEST_F(RetireQueueTest, MoveAssignOverLiveRunsExistingDeletersThenAdopts) {
  auto fence = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(fence.ok()) << fence.status().message();

  int dst_ran = 0;
  int src_ran = 0;
  vg::RetireQueue dst(device());
  dst.push(fence.value().handle(), [&dst_ran]() { ++dst_ran; });

  {
    vg::RetireQueue src(device());
    src.push(fence.value().handle(), [&src_ran]() { ++src_ran; });
    // Move-assign runs dst's already-queued deleter (device assumed idle), then
    // adopts src's still-pending one.
    dst = std::move(src);
  }
  EXPECT_EQ(dst_ran, 1);  // dst's original deleter ran during the assignment
  EXPECT_EQ(src_ran, 0);  // src's was adopted, not yet run
  EXPECT_EQ(dst.pending(), 1u);

  EXPECT_EQ(dst.poll(), 1u);  // adopted device is valid
  EXPECT_EQ(src_ran, 1);
}

TEST_F(RetireQueueTest, SelfMoveAssignKeepsDeletersPending) {
  auto fence = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(fence.ok()) << fence.status().message();

  int ran = 0;
  vg::RetireQueue queue(device());
  queue.push(fence.value().handle(), [&ran]() { ++ran; });

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must leave the deleter queued and unrun.
  vg::RetireQueue* alias = &queue;
  queue = std::move(*alias);
  EXPECT_EQ(ran, 0);
  EXPECT_EQ(queue.pending(), 1u);

  EXPECT_EQ(queue.poll(), 1u);
  EXPECT_EQ(ran, 1);
}

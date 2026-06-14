// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <utility>

#include "volumetric_kit/gfx/core/sync.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

using SyncTest = VulkanDeviceTest;

}  // namespace

TEST_F(SyncTest, FenceStartsUnsignaledAndWaitTimesOut) {
  auto fence = vg::Fence::create(device(), /*signaled=*/false);
  ASSERT_TRUE(fence.ok()) << fence.status().message();
  EXPECT_FALSE(fence.value().is_signaled());

  // A non-signaled fence with a zero timeout returns VK_TIMEOUT, surfaced as a
  // non-OK Status carrying that code.
  vg::Status waited = fence.value().wait(/*timeout_ns=*/0);
  EXPECT_FALSE(waited.ok());
  EXPECT_EQ(waited.code(), VK_TIMEOUT);
}

TEST_F(SyncTest, SignaledFenceWaitsImmediatelyThenResets) {
  auto fence = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(fence.ok()) << fence.status().message();
  EXPECT_TRUE(fence.value().is_signaled());

  EXPECT_TRUE(
      fence.value().wait().ok());  // already signaled → returns immediately

  ASSERT_TRUE(fence.value().reset().ok());
  EXPECT_FALSE(fence.value().is_signaled());
}

TEST_F(SyncTest, SemaphoreCreates) {
  auto semaphore = vg::Semaphore::create(device());
  ASSERT_TRUE(semaphore.ok()) << semaphore.status().message();
  EXPECT_NE(semaphore.value().handle(), VK_NULL_HANDLE);
  EXPECT_TRUE(semaphore.value().valid());
}

TEST_F(SyncTest, FenceValidReflectsOwnership) {
  auto created = vg::Fence::create(device());
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::Fence fence = std::move(created).value();
  EXPECT_TRUE(fence.valid());

  vg::Fence moved(std::move(fence));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(fence.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(SyncTest, CreateRejectsNullDevice) {
  EXPECT_EQ(vg::Fence::create(VK_NULL_HANDLE).status().domain(),
            vg::Status::Code::InvalidArgument);
  EXPECT_EQ(vg::Semaphore::create(VK_NULL_HANDLE).status().domain(),
            vg::Status::Code::InvalidArgument);
  EXPECT_EQ(vg::TimelineSemaphore::create(VK_NULL_HANDLE).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(SyncTest, FenceMoveLeavesSourceEmpty) {
  auto created = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::Fence source = std::move(created).value();
  ASSERT_NE(source.handle(), VK_NULL_HANDLE);

  vg::Fence moved(std::move(source));
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.handle(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(SyncTest, FenceMoveAssignOverLiveLeavesSourceEmpty) {
  auto a = vg::Fence::create(device(), /*signaled=*/true);
  auto b = vg::Fence::create(device(), /*signaled=*/false);
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Fence dst = std::move(a).value();
  vg::Fence src = std::move(b).value();

  dst = std::move(src);  // frees dst's original fence, then adopts src's
  EXPECT_NE(dst.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(src.handle(), VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(SyncTest, FenceSelfMoveAssignIsSafe) {
  auto created = vg::Fence::create(device(), /*signaled=*/true);
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::Fence fence = std::move(created).value();

  // Launder through a pointer so the compiler can't see the self-move (which
  // trips -Wself-move under -Werror); exercises operator='s this != &other
  // guard.
  vg::Fence* alias = &fence;
  fence = std::move(*alias);
  EXPECT_NE(fence.handle(), VK_NULL_HANDLE);
}

TEST_F(SyncTest, SemaphoreMoveLeavesSourceEmpty) {
  auto created = vg::Semaphore::create(device());
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::Semaphore source = std::move(created).value();
  ASSERT_NE(source.handle(), VK_NULL_HANDLE);

  vg::Semaphore moved(std::move(source));
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.handle(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(SyncTest, TimelineHostSignalAdvancesValue) {
  auto timeline = vg::TimelineSemaphore::create(device(), /*initial_value=*/0);
  if (!timeline.ok()) {
    GTEST_SKIP() << "no timeline-semaphore support: "
                 << timeline.status().message();
  }
  auto initial = timeline.value().value();
  ASSERT_TRUE(initial.ok()) << initial.status().message();
  EXPECT_EQ(initial.value(), 0u);

  ASSERT_TRUE(timeline.value().signal(5).ok());
  auto raised = timeline.value().value();
  ASSERT_TRUE(raised.ok()) << raised.status().message();
  EXPECT_EQ(raised.value(), 5u);
}

TEST_F(SyncTest, TimelineWaitReturnsWhenReached) {
  auto timeline = vg::TimelineSemaphore::create(device(), /*initial_value=*/0);
  if (!timeline.ok()) {
    GTEST_SKIP() << "no timeline-semaphore support: "
                 << timeline.status().message();
  }
  ASSERT_TRUE(timeline.value().signal(3).ok());
  EXPECT_TRUE(
      timeline.value().wait(3).ok());  // already at 3 → returns immediately
}

TEST_F(SyncTest, TimelineWaitTimesOutBeforeSignal) {
  auto timeline = vg::TimelineSemaphore::create(device(), /*initial_value=*/0);
  if (!timeline.ok()) {
    GTEST_SKIP() << "no timeline-semaphore support: "
                 << timeline.status().message();
  }
  // Counter is 0; waiting for 1 with a zero timeout reports VK_TIMEOUT.
  vg::Status waited = timeline.value().wait(/*value=*/1, /*timeout_ns=*/0);
  EXPECT_FALSE(waited.ok());
  EXPECT_EQ(waited.code(), VK_TIMEOUT);
}

TEST_F(SyncTest, TimelineMoveLeavesSourceEmpty) {
  auto created = vg::TimelineSemaphore::create(device());
  if (!created.ok()) {
    GTEST_SKIP() << "no timeline-semaphore support: "
                 << created.status().message();
  }
  vg::TimelineSemaphore source = std::move(created).value();
  ASSERT_NE(source.handle(), VK_NULL_HANDLE);

  vg::TimelineSemaphore moved(std::move(source));
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.handle(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(SyncTest, TimelineMoveAssignOverLiveLeavesSourceEmpty) {
  auto a = vg::TimelineSemaphore::create(device());
  if (!a.ok()) {
    GTEST_SKIP() << "no timeline-semaphore support: " << a.status().message();
  }
  auto b = vg::TimelineSemaphore::create(device());
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::TimelineSemaphore dst = std::move(a).value();
  vg::TimelineSemaphore src = std::move(b).value();

  dst = std::move(src);  // destroys dst's original semaphore, then adopts src's
  EXPECT_NE(dst.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(src.handle(), VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(SyncTest, TimelineSelfMoveAssignIsSafe) {
  auto created = vg::TimelineSemaphore::create(device());
  if (!created.ok()) {
    GTEST_SKIP() << "no timeline-semaphore support: "
                 << created.status().message();
  }
  vg::TimelineSemaphore timeline = std::move(created).value();

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must keep the semaphore intact.
  vg::TimelineSemaphore* alias = &timeline;
  timeline = std::move(*alias);
  EXPECT_NE(timeline.handle(), VK_NULL_HANDLE);
}

TEST_F(SyncTest, TimelineInitialValueIsObserved) {
  auto timeline = vg::TimelineSemaphore::create(device(), /*initial_value=*/7);
  if (!timeline.ok()) {
    GTEST_SKIP() << "no timeline-semaphore support: "
                 << timeline.status().message();
  }
  auto value = timeline.value().value();
  ASSERT_TRUE(value.ok()) << value.status().message();
  EXPECT_EQ(value.value(),
            7u);  // create() must seed the counter at initial_value
}

TEST_F(SyncTest, TimelineSignalRejectsNonIncreasingValue) {
  auto created = vg::TimelineSemaphore::create(device(), /*initial_value=*/5);
  if (!created.ok()) {
    GTEST_SKIP() << "no timeline-semaphore support: "
                 << created.status().message();
  }
  vg::TimelineSemaphore timeline = std::move(created).value();
  EXPECT_TRUE(timeline.valid());

  // Equal-to and below-current host signals must be rejected up front with
  // InvalidArgument, not passed to vkSignalSemaphore (UB in a release build).
  EXPECT_EQ(timeline.signal(5).domain(), vg::Status::Code::InvalidArgument);
  EXPECT_EQ(timeline.signal(3).domain(), vg::Status::Code::InvalidArgument);

  // A strictly-increasing signal succeeds and advances the counter.
  ASSERT_TRUE(timeline.signal(6).ok());
  auto value = timeline.value();
  ASSERT_TRUE(value.ok()) << value.status().message();
  EXPECT_EQ(value.value(), 6u);
}

TEST_F(SyncTest, SemaphoreMoveAssignOverLiveLeavesSourceEmpty) {
  auto a = vg::Semaphore::create(device());
  auto b = vg::Semaphore::create(device());
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Semaphore dst = std::move(a).value();
  vg::Semaphore src = std::move(b).value();

  dst = std::move(src);
  EXPECT_NE(dst.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(src.handle(), VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(SyncTest, SemaphoreSelfMoveAssignIsSafe) {
  auto created = vg::Semaphore::create(device());
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::Semaphore semaphore = std::move(created).value();

  vg::Semaphore* alias = &semaphore;
  semaphore = std::move(*alias);
  EXPECT_NE(semaphore.handle(), VK_NULL_HANDLE);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// GPU tests of RetireQueue on a real timeline semaphore, raised from the host:
// what its value releases, the wait in drain, and the move-only queue's
// hand-written move ops. The ordering / run-once / drain logic is covered
// device-free in retire_list_test.cpp; the frame loop's timeline feeding it is
// in windowing_test.cpp.
class RetireQueueTest : public VulkanDeviceTest {
 protected:
  bool wants_validation() const override { return true; }

  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) return;
    auto timeline = vkc::TimelineSemaphore::create(*device_);
    ASSERT_TRUE(timeline.ok()) << timeline.status().message();
    timeline_ = std::move(timeline).value();
  }

  // Raise the timeline from the host, as a completed frame would.
  void reach(std::uint64_t value) { ASSERT_TRUE(timeline_.signal(value).ok()); }

  vkc::TimelineSemaphore timeline_;
};

}  // namespace

// The defining behavior: a deleter runs once the timeline reaches its value,
// and not one value earlier.
TEST_F(RetireQueueTest, ReleasesExactlyWhenTheTimelineReachesTheValue) {
  int released = 0;
  vg::RetireQueue retire(timeline_);
  retire.push(2, [&released]() { ++released; });

  EXPECT_EQ(retire.poll(), 0u);  // at 0
  reach(1);
  EXPECT_EQ(retire.poll(), 0u);  // one short
  EXPECT_EQ(released, 0);
  EXPECT_EQ(retire.pending(), 1u);

  reach(2);
  EXPECT_EQ(retire.poll(), 1u);
  EXPECT_EQ(released, 1);
  EXPECT_EQ(retire.pending(), 0u);
  EXPECT_EQ(retire.poll(), 0u);  // ran once
  EXPECT_EQ(released, 1);
}

// poll() releases every entry at or below the value reached, in any push
// order, and keeps the rest.
TEST_F(RetireQueueTest, PollReleasesOnlyReachedValues) {
  int first = 0;
  int second = 0;
  int third = 0;
  vg::RetireQueue retire(timeline_);
  retire.push(3, [&third]() { ++third; });
  retire.push(1, [&first]() { ++first; });
  retire.push(2, [&second]() { ++second; });

  reach(2);
  EXPECT_EQ(retire.poll(), 2u);
  EXPECT_EQ(first, 1);
  EXPECT_EQ(second, 1);
  EXPECT_EQ(third, 0);
  EXPECT_EQ(retire.pending(), 1u);

  reach(3);  // so the destructor's drain does not wait forever
}

// drain() waits for the value, set here by a host thread after a delay, then
// runs the deleter.
TEST_F(RetireQueueTest, DrainWaitsForThePendingValue) {
  std::atomic<bool> signalled{false};
  bool ran_after_signal = false;
  vg::RetireQueue retire(timeline_);
  retire.push(1, [&]() { ran_after_signal = signalled.load(); });

  std::thread setter([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    signalled.store(true);
    reach(1);
  });
  retire.drain();
  setter.join();
  EXPECT_TRUE(ran_after_signal);
  EXPECT_EQ(retire.pending(), 0u);
}

TEST_F(RetireQueueTest, ReclaimRunsDeletersWithoutWaiting) {
  // A value nothing sets: poll() defers and drain() would block forever.
  // reclaim() runs the deleter at once -- the no-wait forced reclaim for an
  // idle/lost-device teardown.
  int released = 0;
  vg::RetireQueue retire(timeline_);
  retire.push(5, [&released]() { ++released; });
  EXPECT_EQ(retire.poll(), 0u);

  retire.reclaim();
  EXPECT_EQ(released, 1);
  EXPECT_EQ(retire.pending(), 0u);
}

TEST_F(RetireQueueTest, DestructorDrainsPendingDeleters) {
  reach(1);
  int ran = 0;
  {
    vg::RetireQueue retire(timeline_);
    retire.push(1, [&ran]() { ++ran; });
    // No poll(): the destructor must drain (the value is reached, so the wait
    // returns at once), not leak the deleter.
  }
  EXPECT_EQ(ran, 1);
}

TEST_F(RetireQueueTest, MoveConstructTransfersTimelineAndDeleters) {
  int ran = 0;
  vg::RetireQueue source(timeline_);
  source.push(1, [&ran]() { ++ran; });
  ASSERT_EQ(source.pending(), 1u);

  vg::RetireQueue moved(std::move(source));
  EXPECT_EQ(moved.pending(), 1u);
  EXPECT_EQ(source.pending(), 0u);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.poll(), 0u);     // NOLINT(bugprone-use-after-move)

  // The moved-to queue observes the timeline, so poll releases the deleter.
  reach(1);
  EXPECT_EQ(moved.poll(), 1u);
  EXPECT_EQ(ran, 1);
}

TEST_F(RetireQueueTest, MoveAssignOverLiveRunsExistingDeletersThenAdopts) {
  reach(1);
  int dst_ran = 0;
  int src_ran = 0;
  vg::RetireQueue dst(timeline_);
  dst.push(1, [&dst_ran]() { ++dst_ran; });

  {
    vg::RetireQueue src(timeline_);
    src.push(2, [&src_ran]() { ++src_ran; });
    // Move-assign drains dst's queued deleter (its value is reached), then
    // adopts src's still-pending one.
    dst = std::move(src);
  }
  EXPECT_EQ(dst_ran, 1);  // dst's original deleter ran during the assignment
  EXPECT_EQ(src_ran, 0);  // src's was adopted, not yet run
  EXPECT_EQ(dst.pending(), 1u);

  reach(2);
  EXPECT_EQ(dst.poll(), 1u);  // the adopted timeline is observed
  EXPECT_EQ(src_ran, 1);
}

TEST_F(RetireQueueTest, SelfMoveAssignKeepsDeletersPending) {
  int ran = 0;
  vg::RetireQueue queue(timeline_);
  queue.push(1, [&ran]() { ++ran; });

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must leave the deleter queued and unrun.
  vg::RetireQueue* alias = &queue;
  queue = std::move(*alias);
  EXPECT_EQ(ran, 0);
  EXPECT_EQ(queue.pending(), 1u);

  reach(1);
  EXPECT_EQ(queue.poll(), 1u);
  EXPECT_EQ(ran, 1);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <vector>

#include "volumetric_kit/gfx/core/retire_list.hpp"

namespace vg = volumetric_kit::gfx;

// GPU-free tests of the deferred-deletion bookkeeping. Readiness is driven by a
// plain predicate over int keys, so ordering / run-once / compaction / drain
// are exercised deterministically on every runner — no device, no skip.

TEST(RetireList, PollRunsOnlyReadyEntriesAndRunsEachOnce) {
  vg::RetireList<int> list;
  std::vector<int> ran;
  list.push(1, [&ran]() { ran.push_back(1); });
  list.push(2, [&ran]() { ran.push_back(2); });
  list.push(3, [&ran]() { ran.push_back(3); });

  // Keys 1 and 3 are ready; 2 is not.
  auto ready_1_3 = [](int key) { return key == 1 || key == 3; };
  EXPECT_EQ(list.poll(ready_1_3), 2u);
  EXPECT_EQ(list.pending(), 1u);
  EXPECT_EQ(ran, (std::vector<int>{1, 3}));

  // Re-polling does not re-run the ones already released.
  EXPECT_EQ(list.poll(ready_1_3), 0u);
  EXPECT_EQ(list.pending(), 1u);

  // Key 2 becomes ready.
  EXPECT_EQ(list.poll([](int key) { return key == 2; }), 1u);
  EXPECT_EQ(ran, (std::vector<int>{1, 3, 2}));
  EXPECT_EQ(list.pending(), 0u);
}

TEST(RetireList, RunAllRunsEverythingThenClears) {
  vg::RetireList<int> list;
  int count = 0;
  list.push(10, [&count]() { ++count; });
  list.push(20, [&count]() { ++count; });

  list.run_all();
  EXPECT_EQ(count, 2);
  EXPECT_EQ(list.pending(), 0u);
}

TEST(RetireList, DrainWaitsEachKeyThenRunsAll) {
  vg::RetireList<int> list;
  std::vector<int> waited;
  int ran = 0;
  list.push(7, [&ran]() { ++ran; });
  list.push(8, [&ran]() { ++ran; });

  list.drain([&waited](int key) { waited.push_back(key); });

  EXPECT_EQ(waited, (std::vector<int>{7, 8}));
  EXPECT_EQ(ran, 2);
  EXPECT_EQ(list.pending(), 0u);
}

TEST(RetireList, EmptyPollAndDrainAreNoOps) {
  vg::RetireList<int> list;
  EXPECT_EQ(list.poll([](int) { return true; }), 0u);
  list.drain([](int) {});
  EXPECT_EQ(list.pending(), 0u);
}

TEST(RetireList, EmptyDeleterIsSkippedNotInvoked) {
  vg::RetireList<int> list;
  list.push(1, std::function<void()>{});  // empty target
  EXPECT_NO_THROW(
      list.poll([](int) { return true; }));  // not std::bad_function_call
  EXPECT_EQ(list.pending(), 0u);
}

TEST(RetireList, ThrowingDeleterDoesNotLeaveEntriesToReRun) {
  vg::RetireList<int> list;
  int first = 0;
  list.push(1, [&first]() { ++first; });
  list.push(2, []() { throw std::runtime_error("boom"); });
  list.push(3, [&first]() { ++first; });

  // The throw aborts the poll, but every ready entry was detached before any
  // deleter ran, so nothing already-run survives in the list.
  EXPECT_THROW(list.poll([](int) { return true; }), std::runtime_error);
  EXPECT_EQ(list.pending(), 0u);

  // Destruction / re-poll must not invoke the deleter that already ran.
  list.run_all();
  EXPECT_EQ(first, 1);
}

TEST(RetireList, MoveConstructTransfersPendingAndEmptiesSource) {
  vg::RetireList<int> list;
  int ran = 0;
  list.push(1, [&ran]() { ++ran; });
  list.push(2, [&ran]() { ++ran; });

  vg::RetireList<int> moved(std::move(list));
  EXPECT_EQ(moved.pending(), 2u);
  EXPECT_EQ(list.pending(),
            0u);  // NOLINT(bugprone-use-after-move): source empty

  moved.run_all();
  EXPECT_EQ(ran, 2);
}

TEST(RetireList, MoveAssignOverLiveReleasesDstWithoutRunningThenAdoptsSource) {
  vg::RetireList<int> dst;
  vg::RetireList<int> src;
  int dst_ran = 0;
  int src_ran = 0;
  dst.push(1, [&dst_ran]() { ++dst_ran; });
  src.push(2, [&src_ran]() { ++src_ran; });

  dst = std::move(src);
  // Move-assignment releases dst's pending closures WITHOUT running their
  // bodies (run via drain/run_all first if a body must execute), then adopts
  // src's.
  EXPECT_EQ(dst_ran, 0);
  EXPECT_EQ(dst.pending(), 1u);
  EXPECT_EQ(src.pending(), 0u);  // NOLINT(bugprone-use-after-move)

  dst.run_all();
  EXPECT_EQ(src_ran, 1);  // the adopted deleter ran
  EXPECT_EQ(dst_ran, 0);  // dst's original was released, never run
}

TEST(RetireList, SelfMoveAssignKeepsPending) {
  vg::RetireList<int> list;
  int ran = 0;
  list.push(1, [&ran]() { ++ran; });

  vg::RetireList<int>* alias = &list;  // launder past -Wself-move under -Werror
  list = std::move(*alias);
  EXPECT_EQ(list.pending(), 1u);  // self-move must not drop the pending deleter

  list.run_all();
  EXPECT_EQ(ran, 1);
}

TEST(RetireList, DrainRunsDeleterThatEnqueuesDuringDrain) {
  vg::RetireList<int> list;
  int ran = 0;
  list.push(1, [&list, &ran]() {
    ++ran;
    list.push(2, [&ran]() { ++ran; });  // re-enter push() from within drain()
  });

  // drain() promises to run *every* deleter, including one pushed mid-drain.
  list.drain([](int) {});
  EXPECT_EQ(ran, 2);
  EXPECT_EQ(list.pending(), 0u);
}

TEST(RetireList, RunAllRunsDeleterThatEnqueuesDuringRunAll) {
  vg::RetireList<int> list;
  int ran = 0;
  list.push(1, [&list, &ran]() {
    ++ran;
    list.push(2, [&ran]() { ++ran; });  // re-enter push() from within run_all()
  });

  list.run_all();
  EXPECT_EQ(ran, 2);
  EXPECT_EQ(list.pending(), 0u);
}

TEST(RetireList, DeleterMayEnqueueDuringPoll) {
  vg::RetireList<int> list;
  int ran = 0;
  list.push(1, [&list, &ran]() {
    ++ran;
    list.push(2, [&ran]() { ++ran; });  // re-enter push() from within poll()
  });

  EXPECT_EQ(list.poll([](int key) { return key == 1; }), 1u);
  EXPECT_EQ(ran, 1);
  EXPECT_EQ(list.pending(),
            1u);  // the entry pushed mid-poll survived, not dropped

  EXPECT_EQ(list.poll([](int) { return true; }), 1u);
  EXPECT_EQ(ran, 2);
  EXPECT_EQ(list.pending(), 0u);
}

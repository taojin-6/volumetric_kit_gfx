// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

#include "allocation_failure.hpp"
#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"

namespace vkc = volumetric_kit::core;
namespace vg = volumetric_kit::gfx;

namespace {
thread_local VkResult wait_result = VK_SUCCESS;
thread_local std::uint64_t fail_from = 0;
thread_local std::uint64_t completed = 0;
thread_local int waits = 0;
thread_local bool throw_wait = false;

class RetireQueueFailure : public testing::Test {
 protected:
  void SetUp() override {
    wait_result = VK_SUCCESS;
    fail_from = 0;
    completed = 0;
    waits = 0;
    throw_wait = false;
  }
  vkc::TimelineSemaphore timeline;
};

struct Capture {
  explicit Capture(int& count) : count(count) {}
  ~Capture() { ++count; }
  int& count;
};
}  // namespace

// This binary compiles the production queue with host-only timeline methods.
// No device or driver fault is required to exercise its error policy.
namespace volumetric_kit::core {
Result<std::uint64_t> TimelineSemaphore::value() const { return completed; }
Status TimelineSemaphore::wait(std::uint64_t value, std::uint64_t) const {
  ++waits;
  if (throw_wait) throw std::bad_alloc();
  if (value >= fail_from && wait_result != VK_SUCCESS) {
    return vk_error(wait_result, "wait");
  }
  return {};
}
}  // namespace volumetric_kit::core

TEST_F(RetireQueueFailure, FailedWaitKeepsEntriesAndCapturesForRetry) {
  for (const VkResult failure :
       {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY,
        VK_ERROR_UNKNOWN, VK_TIMEOUT}) {
    SCOPED_TRACE(failure);
    int ran = 0;
    int destroyed = 0;
    vg::RetireQueue queue(timeline);
    auto capture = std::make_shared<Capture>(destroyed);
    queue.push(10, [capture, &ran] { ++ran; });
    capture.reset();
    queue.push(20, [&ran] { ++ran; });
    wait_result = failure;
    const vkc::Status status = queue.drain();
    EXPECT_EQ(vkc::vk_result(status), failure);
    EXPECT_EQ(queue.pending(), 2u);
    EXPECT_EQ(ran, 0);
    EXPECT_EQ(destroyed, 0);
    wait_result = VK_SUCCESS;
    EXPECT_TRUE(queue.drain().ok());
    EXPECT_EQ(queue.pending(), 0u);
    EXPECT_EQ(ran, 2);
    EXPECT_EQ(destroyed, 1);
  }
}

TEST_F(RetireQueueFailure, LaterFailureKeepsOnlyUnresolvedEntries) {
  vg::RetireQueue queue(timeline);
  int ran = 0;
  queue.push(10, [&ran] { ++ran; });
  queue.push(20, [&ran] { ++ran; });
  queue.push(30, [&ran] { ++ran; });
  wait_result = VK_ERROR_OUT_OF_HOST_MEMORY;
  fail_from = 20;
  EXPECT_EQ(vkc::vk_result(queue.drain()), VK_ERROR_OUT_OF_HOST_MEMORY);
  EXPECT_EQ(ran, 1);
  EXPECT_EQ(queue.pending(), 2u);
  wait_result = VK_SUCCESS;
  EXPECT_TRUE(queue.drain().ok());
  EXPECT_EQ(ran, 3);
}

TEST_F(RetireQueueFailure, DeviceLossReclaimsTheRemainingEntries) {
  vg::RetireQueue queue(timeline);
  int ran = 0;
  queue.push(10, [&ran] { ++ran; });
  queue.push(20, [&ran] { ++ran; });
  wait_result = VK_ERROR_DEVICE_LOST;
  EXPECT_EQ(vkc::vk_result(queue.drain()), VK_ERROR_DEVICE_LOST);
  EXPECT_EQ(ran, 2);
  EXPECT_EQ(queue.pending(), 0u);
  EXPECT_EQ(waits, 1);
}

TEST_F(RetireQueueFailure, DestructorPreservesUnresolvedCaptures) {
  EXPECT_EXIT(
      {
        int destroyed = 0;
        wait_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        {
          vg::RetireQueue queue(timeline);
          auto capture = std::make_shared<Capture>(destroyed);
          queue.push(10, [capture] {});
        }
        // Intentional preservation is tested in a child; _Exit avoids reporting
        // these deliberately unreachable GPU owners as process-exit leaks.
        std::_Exit(destroyed == 0 ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

TEST_F(RetireQueueFailure, MoveAssignmentPreservesOldCapturesAndAdoptsSource) {
  EXPECT_EXIT(
      {
        int destroyed = 0;
        int ran = 0;
        vg::RetireQueue destination(timeline);
        auto capture = std::make_shared<Capture>(destroyed);
        destination.push(10, [capture] {});
        capture.reset();
        vg::RetireQueue source(timeline);
        source.push(20, [&ran] { ++ran; });
        wait_result = VK_ERROR_UNKNOWN;
        destination = std::move(source);
        const bool preserved = destroyed == 0 && source.pending() == 0 &&
                               destination.pending() == 1;
        wait_result = VK_SUCCESS;
        const bool drained = destination.drain().ok();
        std::_Exit(preserved && drained && ran == 1 && destroyed == 0 ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

TEST_F(RetireQueueFailure, ThrowingDiagnosticCannotTerminateTeardown) {
  EXPECT_EXIT(
      {
        vkc::set_log_handler([](vkc::LogLevel, std::string_view,
                                std::string_view) { throw std::bad_alloc(); });
        int destroyed = 0;
        wait_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        {
          vg::RetireQueue queue(timeline);
          auto capture = std::make_shared<Capture>(destroyed);
          queue.push(10, [capture] {});
        }
        std::_Exit(destroyed == 0 ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

TEST_F(RetireQueueFailure, WaitAllocationFailureIsReturnedAndRetried) {
  int ran = 0;
  vg::RetireQueue queue(timeline);
  queue.push(10, [&ran] { ++ran; });
  throw_wait = true;
  vg_test::AllocationFailure failure(0, true);
  const vkc::Status result = queue.drain();
  failure.stop();
  EXPECT_EQ(result.domain(), vkc::Status::Code::OutOfMemory);
  EXPECT_EQ(queue.pending(), 1u);
  EXPECT_EQ(ran, 0);
  throw_wait = false;
  EXPECT_TRUE(queue.drain().ok());
  EXPECT_EQ(ran, 1);
}

TEST_F(RetireQueueFailure, PollAllocationFailurePreservesEntriesForRetry) {
  vg::RetireQueue queue(timeline);
  int ready = 0;
  int later = 0;
  queue.push(20, [&later] { ++later; });
  queue.push(10, [&ready] { ++ready; });
  queue.push(30, [&later] { ++later; });
  completed = 10;
  vg_test::AllocationFailure failure;
  const auto ran = queue.poll();
  const bool failed = failure.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(ran, 0u);
  EXPECT_EQ(queue.pending(), 3u);
  EXPECT_EQ(queue.poll(), 1u);
  EXPECT_EQ(ready, 1);
  EXPECT_EQ(later, 0);
  EXPECT_TRUE(queue.drain().ok());
  EXPECT_EQ(later, 2);
}

TEST_F(RetireQueueFailure, TeardownPreservesCapturesWithNoHostAllocation) {
  EXPECT_EXIT(
      {
        int destroyed = 0;
        wait_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        auto queue = std::make_unique<vg::RetireQueue>(timeline);
        auto capture = std::make_shared<Capture>(destroyed);
        queue->push(10, [capture] {});
        capture.reset();
        vg_test::AllocationFailure failure(0, true);
        queue.reset();
        failure.stop();
        std::_Exit(destroyed == 0 ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

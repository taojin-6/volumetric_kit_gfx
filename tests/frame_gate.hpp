// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_gate.hpp
/// A gate that holds a test's frames at the queue: a timeline semaphore the
/// frames wait on for value 1, which the test sets from the host -- at once,
/// at scope exit, or from another thread after a pause, so this one can block
/// on a frame held at it.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"

namespace vg_test {

// Sets the gate, letting every frame held at it run, unless it is set.
inline void open_gate(volumetric_kit::core::TimelineSemaphore& gate) {
  const volumetric_kit::core::Result<std::uint64_t> value = gate.value();
  if (value.ok() && value.value() < 1) {
    EXPECT_TRUE(gate.signal(1).ok());
  }
}

// Opens the gate when it goes out of scope, so a test that stops early never
// leaves frames held: declare it after whatever waits for those frames.
class OpenGateAtExit {
 public:
  explicit OpenGateAtExit(volumetric_kit::core::TimelineSemaphore& gate)
      : gate_(gate) {}
  ~OpenGateAtExit() { open_gate(gate_); }
  OpenGateAtExit(const OpenGateAtExit&) = delete;
  OpenGateAtExit& operator=(const OpenGateAtExit&) = delete;

 private:
  volumetric_kit::core::TimelineSemaphore& gate_;
};

// Opens the gate from another thread after a pause, so this one can block on
// a frame held at it.
class OpenGateLater {
 public:
  explicit OpenGateLater(volumetric_kit::core::TimelineSemaphore& gate)
      : thread_([&gate, this] {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          opened_ = true;
          open_gate(gate);
        }) {}
  ~OpenGateLater() { thread_.join(); }
  OpenGateLater(const OpenGateLater&) = delete;
  OpenGateLater& operator=(const OpenGateLater&) = delete;

  bool opened() const { return opened_; }

 private:
  std::atomic<bool> opened_{false};
  std::thread thread_;
};

}  // namespace vg_test

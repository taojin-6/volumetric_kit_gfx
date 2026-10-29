// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_loop_test_support.hpp
/// What the frame loop's tests share, over a swapchain (windowing_test.cpp)
/// or offscreen targets (offscreen_frame_loop_test.cpp): clear-only frames, a
/// published frame's rows by name, and the timeline semaphore and word
/// buffers the tests order work across submits with.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"

namespace vg_test {

// Record a clear of the frame's target to `color`, and of its depth when it
// has one, as every frame of these tests draws.
inline void record_clear(const vg::windowing::Frame& frame,
                         VkClearColorValue color = {{0.1f, 0.0f, 0.0f, 1.0f}}) {
  vg::RenderTargetBeginInfo begin;
  begin.clear_color = color;
  frame.target->begin(frame.cmd, begin);
  frame.target->end(frame.cmd);
}

// Drive `count` clear-only frames through the raw begin_frame. A fixed
// headless surface's chain never goes out of date, and an offscreen loop has
// none, so any non-OK status is a real failure.
inline vkc::Status run_frames(vg::windowing::FrameLoop& loop, int count) {
  for (int i = 0; i < count; ++i) {
    auto frame = loop.begin_frame();
    if (!frame.ok()) {
      return frame.status();
    }
    record_clear(frame.value());
    vkc::Status end = loop.end_frame(frame.value());
    if (!end.ok()) {
      return end;
    }
  }
  return {};
}

// A published frame's rows by name, in order.
inline std::vector<std::string> row_names(const vg::FrameMetrics& metrics) {
  std::vector<std::string> names;
  for (const vkc::StageRow& row : metrics.sections) {
    names.emplace_back(row.name);
  }
  return names;
}

// A timeline semaphore on `device`, at 0.
inline vkc::TimelineSemaphore make_timeline(const vkc::Device& device) {
  auto timeline = vkc::TimelineSemaphore::create(device);
  EXPECT_TRUE(timeline.ok()) << timeline.status().message();
  return timeline.ok() ? std::move(timeline).value() : vkc::TimelineSemaphore{};
}

// A 4-byte buffer for the tests that order work across submits: device-only
// with transfer usage, or a host-readable readback.
inline vkc::Buffer make_word_buffer(vkc::Allocator& allocator, bool readback) {
  vkc::BufferDesc desc;
  desc.size = sizeof(uint32_t);
  desc.usage = readback ? VK_BUFFER_USAGE_TRANSFER_DST_BIT
                        : VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (readback) {
    desc.memory = vkc::MemoryUsage::Staging;
    desc.host_access = vkc::HostAccess::Random;
  }
  auto buffer = allocator.create_buffer(desc);
  EXPECT_TRUE(buffer.ok()) << buffer.status().message();
  return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
}

}  // namespace vg_test

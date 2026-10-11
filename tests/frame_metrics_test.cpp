// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Pure-CPU tests for the FrameMetrics POD. No Vulkan device is touched, so
// every case runs on every platform (no GTEST_SKIP). Its stages are the core's
// StageRow, and the timestamp math the profiler fills them with is the core's
// (ticks_to_ms, timestamp_delta); both are tested there, so these cover only
// the frame's own default field values.

#include <gtest/gtest.h>

#include "volumetric_kit/gfx/core/frame_metrics.hpp"

namespace vg = volumetric_kit::gfx;

TEST(FrameMetrics, DefaultConstructedIsZeroed) {
  vg::FrameMetrics metrics;
  EXPECT_TRUE(metrics.sections.empty());
  EXPECT_DOUBLE_EQ(metrics.cpu_frame_ms, 0.0);
  EXPECT_DOUBLE_EQ(metrics.fps, 0.0);
  EXPECT_EQ(metrics.memory_used_bytes, 0u);
  EXPECT_EQ(metrics.memory_budget_bytes, 0u);
}

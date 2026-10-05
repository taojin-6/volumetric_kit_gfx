// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Pure-CPU tests for the FrameMetrics POD. No Vulkan device is touched, so
// every case runs on every platform (no GTEST_SKIP). Its stages are the core's
// StageRow, and the timestamp math the profiler fills them with is the core's
// (ticks_to_ms, timestamp_delta); both are tested there, so these cover only
// the frame's own fields and its vector of rows.

#include <gtest/gtest.h>

#include <cstdint>

#include "volumetric_kit/gfx/core/frame_metrics.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

TEST(FrameMetrics, DefaultConstructedIsZeroed) {
  vg::FrameMetrics metrics;
  EXPECT_TRUE(metrics.sections.empty());
  EXPECT_DOUBLE_EQ(metrics.cpu_frame_ms, 0.0);
  EXPECT_DOUBLE_EQ(metrics.fps, 0.0);
  EXPECT_EQ(metrics.memory_used_bytes, 0u);
  EXPECT_EQ(metrics.memory_budget_bytes, 0u);
}

TEST(FrameMetrics, PushingSectionsRecordsThemInOrder) {
  vg::FrameMetrics metrics;
  // StageRow fields are positional (C++17): name, cpu_ms, gpu_ms, has_gpu.
  metrics.sections.push_back({"shadow", 0.8, 1.2, true});
  metrics.sections.push_back({"upload", 0.3, 0.0, false});  // CPU-only
  metrics.cpu_frame_ms = 11.0;
  metrics.fps = 90.0;
  metrics.memory_used_bytes = 256u * 1024u * 1024u;
  metrics.memory_budget_bytes = 1024u * 1024u * 1024u;

  ASSERT_EQ(metrics.sections.size(), 2u);

  const vkc::StageRow& gpu_stage = metrics.sections[0];
  EXPECT_STREQ(gpu_stage.name, "shadow");
  EXPECT_DOUBLE_EQ(gpu_stage.cpu_ms, 0.8);
  EXPECT_DOUBLE_EQ(gpu_stage.gpu_ms, 1.2);
  EXPECT_TRUE(gpu_stage.has_gpu);

  // A CPU-only stage leaves has_gpu false -- that flag is the capability
  // report.
  const vkc::StageRow& cpu_stage = metrics.sections[1];
  EXPECT_STREQ(cpu_stage.name, "upload");
  EXPECT_DOUBLE_EQ(cpu_stage.cpu_ms, 0.3);
  EXPECT_DOUBLE_EQ(cpu_stage.gpu_ms, 0.0);
  EXPECT_FALSE(cpu_stage.has_gpu);

  EXPECT_DOUBLE_EQ(metrics.cpu_frame_ms, 11.0);
  EXPECT_DOUBLE_EQ(metrics.fps, 90.0);
  EXPECT_EQ(metrics.memory_used_bytes, 256u * 1024u * 1024u);
  EXPECT_EQ(metrics.memory_budget_bytes, 1024u * 1024u * 1024u);
}

TEST(FrameMetrics, IsCopyableAndCopyIsIndependent) {
  vg::FrameMetrics original;
  original.sections.push_back({"draw", 2.0, 3.0, true});
  original.fps = 60.0;

  vg::FrameMetrics copy = original;
  ASSERT_EQ(copy.sections.size(), 1u);
  EXPECT_STREQ(copy.sections[0].name, "draw");
  EXPECT_DOUBLE_EQ(copy.fps, 60.0);

  // Mutating the copy's vector must not disturb the original.
  copy.sections.push_back({"post", 1.0, 0.0, false});
  EXPECT_EQ(original.sections.size(), 1u);
  EXPECT_EQ(copy.sections.size(), 2u);
}

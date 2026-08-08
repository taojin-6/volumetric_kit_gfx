// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Pure-CPU tests for the FrameMetrics POD and its ticks_to_ms helper. No Vulkan
// device is touched, so every case runs on every platform (no GTEST_SKIP).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "volumetric_kit/gfx/core/frame_metrics.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

// Builds a Section by field assignment: designated initializers (.name = ...)
// are a C++20 feature and this project compiles as C++17.
vg::FrameMetrics::Section make_section(const char* name, double cpu_ms,
                                       double gpu_ms, bool has_gpu) {
  vg::FrameMetrics::Section section;
  section.name = name;
  section.cpu_ms = cpu_ms;
  section.gpu_ms = gpu_ms;
  section.has_gpu = has_gpu;
  return section;
}

}  // namespace

TEST(TimestampDelta, PlainSpanIsEndMinusBegin) {
  EXPECT_EQ(vg::timestamp_delta(1000, 3000, 64), 2000u);
  EXPECT_EQ(vg::timestamp_delta(1000, 3000, 36), 2000u);
  EXPECT_EQ(vg::timestamp_delta(7, 7, 36), 0u);
}

// The counter is an N-bit ring, so a section straddling a wrap must report the
// short way round -- not 2^64 minus it. 36 valid bits is what Mesa's Intel
// driver reports, where the counter wraps every ~69 s at 1 ns/tick.
TEST(TimestampDelta, WrapsWithinValidBits) {
  constexpr uint32_t kBits = 36;
  constexpr uint64_t kPeriod = uint64_t{1} << kBits;

  // begin near the top of the ring, end just past the wrap: 100 ticks elapsed.
  EXPECT_EQ(vg::timestamp_delta(kPeriod - 50, 50, kBits), 100u);
  // One full lap reads as zero elapsed, not 2^36.
  EXPECT_EQ(vg::timestamp_delta(0, kPeriod, kBits), 0u);
  // High garbage above the valid bits is discarded either way.
  EXPECT_EQ(vg::timestamp_delta(~uint64_t{0} - 9, ~uint64_t{0}, kBits), 9u);

  // Masking the endpoints *before* subtracting -- the bug this replaced --
  // would have produced an astronomically large delta for the wrap case.
  const uint64_t mask = kPeriod - 1;
  const uint64_t wrong = ((50u & mask) - ((kPeriod - 50) & mask));
  EXPECT_NE(wrong, 100u);
  EXPECT_GT(vg::ticks_to_ms(wrong, 1.0f), 1e9);
}

TEST(TimestampDelta, ZeroValidBitsHasNoUsableTiming) {
  EXPECT_EQ(vg::timestamp_delta(1000, 3000, 0), 0u);
}

TEST(TicksToMs, OneMillionTicksAtOneNsPeriodIsOneMillisecond) {
  // 1e6 ticks * 1 ns/tick = 1e6 ns = 1.0 ms.
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(1'000'000, 1.0f), 1.0);
}

TEST(TicksToMs, ScalesLinearlyWithTickDelta) {
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(0, 1.0f), 0.0);
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(500'000, 1.0f), 0.5);
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(2'000'000, 1.0f), 2.0);
}

TEST(TicksToMs, AppliesTimestampPeriod) {
  // A realistic discrete-GPU period (38.4 ns/tick). The period is a float, and
  // 38.4 is not exactly representable, so the result carries float-precision
  // rounding -- compare against the same float-promoted product, not the
  // decimal literal (which would diverge well past EXPECT_DOUBLE_EQ's ULP
  // tolerance).
  constexpr float kPeriod = 38.4f;
  // 1000 ticks -> ~38400 ns -> ~0.0384 ms.
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(1000, kPeriod),
                   1000.0 * static_cast<double>(kPeriod) * 1e-6);
  // 100k ticks -> ~3'840'000 ns -> ~3.84 ms.
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(100'000, kPeriod),
                   100'000.0 * static_cast<double>(kPeriod) * 1e-6);
  // And it lands near the nominal decimal value to float precision.
  EXPECT_NEAR(vg::ticks_to_ms(1000, kPeriod), 0.038'4, 1e-7);
  EXPECT_NEAR(vg::ticks_to_ms(100'000, kPeriod), 3.84, 1e-5);
}

TEST(TicksToMs, FractionalSubMillisecondPeriod) {
  // 1 ns/tick, 12345 ticks = 12345 ns = 0.012345 ms.
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(12'345, 1.0f), 0.012'345);
}

TEST(TicksToMs, NonPositivePeriodYieldsZero) {
  // A device with no usable timestamp timing reports period 0 (or negative);
  // the helper must report 0.0 rather than a bogus duration.
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(1'000'000, 0.0f), 0.0);
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(1'000'000, -1.0f), 0.0);
  // Even an enormous tick delta must collapse to 0.0 when the period is
  // unusable.
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(std::numeric_limits<uint64_t>::max(), 0.0f),
                   0.0);
}

TEST(TicksToMs, LargeTickDeltaDoesNotOverflow) {
  // A 64-bit nanosecond intermediate (tick_delta * period as integers) would
  // overflow here; the double math does not. 1e18 ticks * 1 ns/tick = 1e18 ns =
  // 1e12 ms.
  constexpr uint64_t kHugeTicks = 1'000'000'000'000'000'000ULL;
  EXPECT_DOUBLE_EQ(vg::ticks_to_ms(kHugeTicks, 1.0f), 1.0e12);
  // The result stays finite for the maximum representable tick delta.
  EXPECT_TRUE(std::isfinite(
      vg::ticks_to_ms(std::numeric_limits<uint64_t>::max(), 38.4f)));
}

TEST(TicksToMs, IsConstexpr) {
  // Usable in a constant expression: confirms `constexpr` holds.
  constexpr double kMs = vg::ticks_to_ms(1'000'000, 1.0f);
  static_assert(kMs == 1.0, "ticks_to_ms must be usable at compile time");
  EXPECT_DOUBLE_EQ(kMs, 1.0);
}

TEST(FrameMetrics, DefaultConstructedIsZeroed) {
  vg::FrameMetrics metrics;
  EXPECT_TRUE(metrics.sections.empty());
  EXPECT_DOUBLE_EQ(metrics.cpu_frame_ms, 0.0);
  EXPECT_DOUBLE_EQ(metrics.fps, 0.0);
  EXPECT_EQ(metrics.memory_used_bytes, 0u);
  EXPECT_EQ(metrics.memory_budget_bytes, 0u);
}

TEST(FrameMetrics, DefaultSectionIsZeroedAndCpuOnly) {
  vg::FrameMetrics::Section section;
  EXPECT_EQ(section.name, nullptr);
  EXPECT_DOUBLE_EQ(section.cpu_ms, 0.0);
  EXPECT_DOUBLE_EQ(section.gpu_ms, 0.0);
  EXPECT_FALSE(section.has_gpu);
}

TEST(FrameMetrics, PushingSectionsRecordsThemInOrder) {
  vg::FrameMetrics metrics;
  metrics.sections.push_back(make_section("shadow", 0.8, 1.2, true));
  metrics.sections.push_back(
      make_section("upload", 0.3, 0.0, false));  // CPU-only
  metrics.cpu_frame_ms = 11.0;
  metrics.fps = 90.0;
  metrics.memory_used_bytes = 256u * 1024u * 1024u;
  metrics.memory_budget_bytes = 1024u * 1024u * 1024u;

  ASSERT_EQ(metrics.sections.size(), 2u);

  const vg::FrameMetrics::Section& gpu_stage = metrics.sections[0];
  EXPECT_STREQ(gpu_stage.name, "shadow");
  EXPECT_DOUBLE_EQ(gpu_stage.cpu_ms, 0.8);
  EXPECT_DOUBLE_EQ(gpu_stage.gpu_ms, 1.2);
  EXPECT_TRUE(gpu_stage.has_gpu);

  // A CPU-only stage leaves has_gpu false -- that flag is the capability
  // report.
  const vg::FrameMetrics::Section& cpu_stage = metrics.sections[1];
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
  original.sections.push_back(make_section("draw", 2.0, 3.0, true));
  original.fps = 60.0;

  vg::FrameMetrics copy = original;
  ASSERT_EQ(copy.sections.size(), 1u);
  EXPECT_STREQ(copy.sections[0].name, "draw");
  EXPECT_DOUBLE_EQ(copy.fps, 60.0);

  // Mutating the copy's vector must not disturb the original.
  copy.sections.push_back(make_section("post", 1.0, 0.0, false));
  EXPECT_EQ(original.sections.size(), 1u);
  EXPECT_EQ(copy.sections.size(), 2u);
}

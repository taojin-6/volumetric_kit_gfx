// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_metrics.hpp
/// @brief The backend-free, plain-data programmatic surface of the performance
///        profiler.
///
/// @ref FrameMetrics is the single contract a profiler fills in and a consumer
/// reads. It carries no `Vk*`/`Vma*`/Tracy types, so it lives in `core` and the
/// `ui` tier (which depends only on `core`) -- or a headless CI harness -- can
/// read it without pulling in Vulkan. A producer measures a frame, populates a
/// `FrameMetrics`, and hands it to the overlay; the overlay only reads.

#include <cstdint>
#include <vector>

namespace volumetric_kit::gfx {

/// @brief A timed frame: per-stage CPU/GPU spans plus whole-frame and memory
///        aggregates, with no backend types attached.
///
/// A producer fills one of these per frame from its timers and memory
/// accounting; the overlay (or a headless harness) reads it. Each @ref Section
/// is one labelled stage. @ref cpu_frame_ms / @ref fps describe the frame as a
/// whole, and @ref memory_used_bytes / @ref memory_budget_bytes report
/// aggregate device memory.
///
/// @code
/// FrameMetrics metrics;
/// metrics.sections.push_back(
///     {.name = "shadow", .cpu_ms = 0.8, .gpu_ms = 1.2, .has_gpu = true});
/// metrics.sections.push_back({.name = "upload", .cpu_ms = 0.3});  // CPU-only
/// metrics.cpu_frame_ms = 11.0;
/// metrics.fps = 90.0;
/// for (const FrameMetrics::Section& s : metrics.sections) {
///   // has_gpu IS the capability report: read gpu_ms only when it is set.
///   double gpu = s.has_gpu ? s.gpu_ms : 0.0;
///   draw_row(s.name, s.cpu_ms, gpu, s.has_gpu);
/// }
/// @endcode
struct FrameMetrics {
  /// @brief One labelled stage of the frame: its CPU span, and -- when GPU
  ///        timing is available -- its GPU span.
  struct Section {
    /// Stage label. Assumed to have string-literal lifetime: the section stores
    /// the pointer, not a copy, which is what keeps a @ref Section trivially
    /// copyable (and @ref FrameMetrics cheap to copy -- only the @ref sections
    /// vector allocates). Pass a string literal (or another pointer that
    /// outlives every read of this metrics snapshot); a pointer into a
    /// temporary dangles.
    const char* name = nullptr;
    /// Wall-clock CPU time spent in the stage, in milliseconds. Always
    /// populated.
    double cpu_ms = 0.0;
    /// GPU time spent in the stage, in milliseconds. Meaningful only when
    /// @ref has_gpu is true; otherwise it is `0.0` and carries no information.
    double gpu_ms = 0.0;
    /// Whether @ref gpu_ms holds a real measurement. This flag IS how a
    /// consumer reads GPU-timing availability -- it never branches on a
    /// separate capability field. It is false for a CPU-only stage (e.g. a host
    /// upload) and for every stage when the device offers no usable timestamp
    /// timing (zero `timestampValidBits`, or queries not yet resolved).
    bool has_gpu = false;
  };

  /// The frame's stages, in record order. Empty for a frame with no timed
  /// sections.
  std::vector<Section> sections;
  /// Whole-frame CPU time, in milliseconds.
  double cpu_frame_ms = 0.0;
  /// Frames per second (e.g. a smoothed rate the producer maintains).
  double fps = 0.0;
  /// Aggregate device memory currently in use, in bytes. Summed across heaps:
  /// on a UMA/Apple GPU the heaps are unified, so this is the single pool's
  /// usage. Per-heap detail is available via @ref Allocator::memory_stats.
  uint64_t memory_used_bytes = 0;
  /// Aggregate device memory budget, in bytes, with the same aggregation as
  /// @ref memory_used_bytes (a single unified pool on UMA/Apple GPUs). Per-heap
  /// detail is available via @ref Allocator::memory_stats.
  uint64_t memory_budget_bytes = 0;
};

/// @brief Convert a timestamp-query tick delta to milliseconds.
/// @param tick_delta           End-minus-start ticks from two timestamp
///        queries.
/// @param timestamp_period_ns  Nanoseconds per tick
///        (`VkPhysicalDeviceLimits::timestampPeriod`).
/// @return `tick_delta * timestamp_period_ns * 1e-6` in milliseconds, or `0.0`
///         when @p timestamp_period_ns is not positive (no usable timing).
///
/// The product is computed in `double`, so a large @p tick_delta does not
/// overflow the way a 64-bit-integer nanosecond intermediate could.
constexpr double ticks_to_ms(uint64_t tick_delta,
                             float timestamp_period_ns) noexcept {
  if (timestamp_period_ns <= 0.0f) return 0.0;
  return static_cast<double>(tick_delta) *
         static_cast<double>(timestamp_period_ns) * 1e-6;
}

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_metrics.hpp
/// @brief The backend-free, plain-data programmatic surface of the performance
///        profiler.
///
/// @ref FrameMetrics is the single contract a profiler fills in and a consumer
/// reads. Its stages are the family's `core::StageRow`s (volumetric_kit_core's
/// base tier, the vocabulary recon's stage timings use too), and it carries no
/// `Vk*`/`Vma*`/Tracy types, so the `ui` tier -- or a headless CI harness --
/// can read it without pulling in Vulkan. A producer measures a frame,
/// populates a `FrameMetrics`, and hands it to the overlay; the overlay only
/// reads.

#include <cstdint>
#include <vector>

#include "volumetric_kit/core/base/stage_metrics.hpp"

namespace volumetric_kit::gfx {

/// @brief A timed frame: per-stage host and device spans plus whole-frame and
///        memory aggregates, with no backend types attached.
///
/// A producer fills one of these per frame from its timers and memory
/// accounting; the overlay (or a headless harness) reads it. Each
/// `core::StageRow` in @ref sections is one labelled stage: its host span,
/// the part of it spent blocked (`wait_ms`), and its device half, which
/// `device` says the state of (`core::DeviceTime`). The name is a pointer
/// with string-literal lifetime, not a copy. @ref cpu_frame_ms / @ref fps
/// describe the frame as a whole, and @ref memory_used_bytes / @ref
/// memory_budget_bytes report aggregate device memory.
///
/// @code
/// FrameMetrics metrics;
/// core::StageRow shadow{"shadow", 0.8, 1.2, true};  // name, cpu, gpu, has_gpu
/// shadow.device = core::DeviceTime::Measured;
/// metrics.sections.push_back(shadow);
/// metrics.sections.push_back({"upload", 0.3});  // host-only: device None
/// metrics.cpu_frame_ms = 11.0;
/// metrics.fps = 90.0;
/// for (const core::StageRow& s : metrics.sections) {
///   // device says what gpu_ms holds: read it only when Measured.
///   const bool timed = s.device == core::DeviceTime::Measured;
///   draw_row(s.name, s.cpu_ms - s.wait_ms, s.wait_ms, timed ? s.gpu_ms : 0.0);
/// }
/// @endcode
struct FrameMetrics {
  /// The frame's stages, in record order, then the rows its render loop
  /// added for its own waits, submit and present (@ref Profiler::add_rows).
  /// Empty for a frame with neither.
  std::vector<core::StageRow> sections;
  /// The frame's recording time on the host, in milliseconds: from the
  /// profiler's `begin_frame` to its `end_frame`.
  double cpu_frame_ms = 0.0;
  /// Frames per second (e.g. a smoothed rate the producer maintains); its
  /// inverse is the frame period.
  double fps = 0.0;
  /// Aggregate device memory currently in use, in bytes: each heap's
  /// `HeapStats::usage_bytes`, summed. Where the device enables
  /// `VK_EXT_memory_budget` that is the driver's figure for the whole process
  /// -- every library allocating on the device, not only the renderer --
  /// otherwise the sampled allocator's own. On unified memory the device
  /// typically reports one heap, so this is the one pool's usage. Per-heap
  /// detail, and the allocator's own share, are in
  /// `core::Allocator::memory_stats`.
  uint64_t memory_used_bytes = 0;
  /// Aggregate device memory budget, in bytes, with the same aggregation as
  /// @ref memory_used_bytes. Per-heap detail is available via
  /// `core::Allocator::memory_stats`.
  uint64_t memory_budget_bytes = 0;
};

}  // namespace volumetric_kit::gfx

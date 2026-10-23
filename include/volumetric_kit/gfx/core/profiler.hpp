// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file profiler.hpp
/// @brief The per-frame performance collector: CPU + GPU stage timings, memory,
///        and fps, published as a backend-free @ref FrameMetrics.
///
/// One concrete @ref Profiler unifies the timestamp/CPU collector and the
/// debug-utils marker emitter behind a single surface. A @ref Scope times a CPU
/// span; its GPU form (given a command buffer) additionally marks the frame's
/// device time at its open and close, on the core's `GpuTiles`, and writes a
/// `VK_EXT_debug_utils` region label, so one render pass is a single stage
/// with both a CPU-record time and a GPU-execute time. CPU spans are buffered
/// per in-flight slot and published together with that slot's resolved GPU
/// timings, so each stage's CPU and GPU halves come from the same frame (the
/// whole snapshot lags the in-flight depth). Consumers read the plain
/// @ref FrameMetrics — never a backend type — so the `ui` overlay and a
/// headless harness read it identically.

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/frame_metrics.hpp"

namespace volumetric_kit::core {
class Allocator;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx {

/// @brief Construction parameters for a @ref Profiler.
struct ProfilerConfig {
  /// In-flight depth: how many frames the CPU may run ahead of the GPU. Must
  /// match the render loop's frames-in-flight, since a slot's GPU timings are
  /// resolved only when that slot comes back around (its prior submission
  /// retired). Must be >= 1.
  uint32_t frames_in_flight = 2;
  /// Upper bound on @ref Profiler::gpu_scope calls per frame; sizes the
  /// timestamp pool: a timestamp at each scope's open and close, and one at
  /// each end of the frame, for every in-flight frame. Scopes beyond it in a
  /// frame are still CPU-timed and still labelled, their device half
  /// `core::DeviceTime::NotTimed`. The core's timer holds at most
  /// `2 * core::GpuTimer::kMaxSpans` timestamps, the most a Metal-backed pool
  /// holds, so where the frames in flight need more the bound is lowered to
  /// fit (logged). Must be >= 1.
  uint32_t max_gpu_sections_per_frame = 32;
};

/// @brief Collects per-stage CPU/GPU timings, memory, and fps for a frame and
///        publishes them as a @ref FrameMetrics.
///
/// Built from a `core::Device`'s capabilities once: GPU timing is available
/// only where the graphics queue reports non-zero `timestampValidBits`
/// (MoltenVK may report zero — then GPU scopes degrade to CPU-only timing plus
/// their label, never an error, and their rows read
/// `core::DeviceTime::Unavailable`), and labels emit only where
/// `VK_EXT_debug_utils` is enabled (`core::Device::debug_labels_available`).
/// Drive it once per frame: @ref begin_frame at the top (after the slot's
/// in-flight fence has signalled, not only its frame's timeline value: see
/// @ref windowing::FrameLoop::timeline), @ref cpu_scope / @ref gpu_scope
/// around work, @ref end_frame at the bottom; read the latest resolved
/// snapshot from @ref metrics.
///
/// A GPU scope's device time runs from its open to its close, the scopes
/// nested in it included. Each row's `device` says what its device half
/// holds: measured; not timed (`NotTimed`) past the per-frame bound, for a
/// command buffer other than the frame's, or where the frame's timestamps
/// were not readable when its slot recurred; unavailable where the queue
/// cannot time; none for a CPU scope.
///
/// A scope's host time splits as a `core::StageScope`'s does, as each open
/// scope is one on its thread: the core's blocking waits inside it -- a fence,
/// a timeline semaphore, a blocking submit -- are its `wait_ms`
/// (`core::note_blocked`), so its busy time never holds a wait.
///
/// @warning The @p device passed to @ref create must outlive the profiler (it
///          owns a timestamp pool freed through that device, and borrows the
///          device by address to record labels, so it must not move
///          meanwhile). A @ref Scope borrows the profiler's state: it must not
///          outlive the frame it was opened in, and the profiler must not be
///          moved-assigned-over or destroyed while any Scope from it is still
///          open (that frees the state the Scope finalizes through). A memory
///          source set via @ref set_memory_source must outlive the profiler.
///
/// @code
/// ProfilerConfig cfg;
/// cfg.frames_in_flight = 2;
/// core::Result<Profiler> r = Profiler::create(device, cfg);
/// if (!r) return r.status();
/// Profiler& profiler = r.value();
/// profiler.set_memory_source(&allocator);  // optional memory figures
///
/// // Per frame (slot + cmd from the render loop; cmd outside any render pass):
/// profiler.begin_frame(frame.slot, frame.cmd);
/// {
///   Profiler::Scope pass = profiler.gpu_scope(frame.cmd, "main pass");
///   // ... begin the pass, record its draws into frame.cmd, end it ...
/// }  // pass ends: GPU end-timestamp + label close recorded here
/// profiler.end_frame();
/// const FrameMetrics& m = profiler.metrics();  // resolved snapshot (lags N)
/// @endcode
class VG_CORE_API Profiler {
  struct Impl;  // pImpl: seals the timestamp pool and chrono

 public:
  /// @brief A timed stage: a CPU span, optionally paired with a GPU span and a
  ///        debug-utils label, finalized on destruction.
  ///
  /// Returned by @ref Profiler::cpu_scope / @ref Profiler::gpu_scope.
  /// Destroying it stops the CPU clock and — for a GPU scope — marks the end
  /// of its device time and closes the label, so scope it to exactly the work
  /// it should measure. A default-constructed (or moved-from) `Scope` is inert
  /// and finalizes nothing.
  ///
  /// @warning Close it, or end its frame, on the thread that opened it: it is
  ///          a core scope on that thread's stack (`VKC_CHECK`).
  class Scope {
   public:
    /// @brief An inert scope: times and finalizes nothing.
    Scope() noexcept = default;

    ~Scope();
    Scope(Scope&& other) noexcept;
    Scope& operator=(Scope&& other) noexcept;
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    /// @return Whether this scope still has a stage it will finalize on
    ///         destruction.
    bool active() const noexcept { return impl_ != nullptr; }

   private:
    friend class Profiler;
    Scope(Impl* impl, uint32_t index) noexcept : impl_(impl), index_(index) {}

    Impl* impl_ = nullptr;  // borrowed owner; nullptr == inert / moved-from
    uint32_t index_ = 0;    // index of this scope's stage in the current frame
  };

  /// @brief Create a profiler for @p device.
  /// @param device  The device whose graphics queue is timed; supplies the
  ///                timestamp-validity, period, and debug-utils capabilities.
  ///                It must have enabled the renderer's requirements
  ///                (@ref device_requirements).
  /// @param config  In-flight depth and the per-frame GPU-scope bound.
  /// @return The profiler on success -- with @ref gpu_timing false, logged,
  ///         where the timestamp pool will not allocate -- or a non-OK
  ///         `core::Status`: `core::Status::Code::InvalidArgument` when
  ///         @p config has a zero `frames_in_flight` or
  ///         `max_gpu_sections_per_frame`;
  ///         `core::Status::Code::Unsupported` for a @p device without the
  ///         renderer's requirements.
  static core::Result<Profiler> create(const core::Device& device,
                                       const ProfilerConfig& config = {});

  ~Profiler();
  Profiler(Profiler&& other) noexcept;
  Profiler& operator=(Profiler&& other) noexcept;
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  /// @brief Set the allocator sampled for the aggregate memory figures in
  ///        @ref FrameMetrics.
  /// @param allocator  The allocator to sample at each @ref end_frame, or
  ///                   `nullptr` to report zero memory. Borrowed; must outlive
  ///                   the profiler.
  void set_memory_source(const core::Allocator* allocator) noexcept;

  /// @brief Begin a frame: publish the frame this slot last held, take the
  ///        frame's timestamps from the core's timer, and start the frame's
  ///        CPU clock.
  /// @param slot  The in-flight slot index in `[0, frames_in_flight)`. The
  ///              caller must already have waited this slot's in-flight fence
  ///              (not only its frame's timeline value), so the prior
  ///              submission's timestamps are readable without stalling
  ///              (@ref windowing::FrameLoop::begin_frame does this).
  /// @param cmd   The frame's recording command buffer, outside any render pass
  ///              (where `vkCmdResetQueryPool` is legal). Pass `VK_NULL_HANDLE`
  ///              for a CPU-only frame; GPU scopes then record no timestamps.
  void begin_frame(uint32_t slot, VkCommandBuffer cmd) noexcept;

  /// @brief End the frame: mark the end of its device time, stamp its total
  ///        CPU time and fps and sample memory, buffering them for this slot
  ///        to publish when the slot recurs.
  void end_frame() noexcept;

  /// @brief Add rows of host time the frame spent outside its scopes, shown
  ///        after its own stages: what a render loop adds for its own waits,
  ///        submit and present (@ref windowing::FrameLoop does).
  ///
  /// Between @ref begin_frame and @ref end_frame they join the frame being
  /// recorded; after @ref end_frame, until the next @ref begin_frame, the
  /// frame it ended, which is published only when its slot recurs. Rows of
  /// one name accumulate. Outside both, nothing is added.
  /// @param rows  The rows; their names must outlive every read of the
  ///              snapshot, as string literals do.
  void add_rows(const core::StageMetrics& rows);

  /// @brief Open a CPU-only timed stage.
  /// @param name  Stage label; must have string-literal lifetime (stored by
  ///              pointer, see `core::StageRow::name`). A null name records
  ///              the stage as `"(unnamed)"`, as a row's name is never null.
  /// @return A @ref Scope timing until it is destroyed; inert if called outside
  ///         a @ref begin_frame / @ref end_frame pair.
  Scope cpu_scope(const char* name);

  /// @brief Open a stage that times CPU and — when GPU timing is available and
  ///        the per-frame bound is not exhausted — GPU, and emits a debug-utils
  ///        region label.
  /// @param cmd   The recording command buffer the timestamps and label record
  ///              into; must be the same buffer passed to @ref begin_frame
  ///              (which took the frame's timestamps). A differing cmd —
  ///              including when begin_frame got `VK_NULL_HANDLE` for a
  ///              CPU-only frame — leaves the stage CPU-timed only, with no
  ///              timestamps or label.
  /// @param name  Stage label and label text; string-literal lifetime. A null
  ///              name records and labels the stage as `"(unnamed)"`, as a
  ///              row's name is never null.
  /// @return A @ref Scope timing until it is destroyed; inert if called outside
  ///         a frame or with a null @p cmd.
  /// @note Open it outside a render pass, around the whole pass. On Apple GPUs
  ///       MoltenVK samples GPU time only between Metal encoders, so a scope
  ///       opened and closed inside one dynamic-rendering pass reads about
  ///       0 ms; time the work inside a pass with @ref cpu_scope.
  Scope gpu_scope(VkCommandBuffer cmd, const char* name);

  /// @return The most recently resolved frame's metrics, empty until the first
  ///         slot recurs (the snapshot lags the in-flight depth) and empty for
  ///         a moved-from profiler.
  const FrameMetrics& metrics() const noexcept;

  /// @return Whether GPU timestamp timing is available (the graphics queue
  ///         reports non-zero `timestampValidBits`, and the timestamp pool
  ///         allocated). When false, GPU scopes time CPU only and their rows
  ///         read `core::DeviceTime::Unavailable`.
  bool gpu_timing() const noexcept;

  /// @return `true` if this owns profiler resources (false when moved-from).
  bool valid() const noexcept { return impl_ != nullptr; }

 private:
  Profiler() noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file profiler.hpp
/// @brief The per-frame performance collector: CPU + GPU stage timings, memory,
///        and fps, published as a backend-free @ref FrameMetrics.
///
/// One concrete @ref Profiler unifies the timestamp/CPU collector and the
/// debug-utils marker emitter behind a single surface. A @ref Scope times a CPU
/// span; its GPU form (given a command buffer) additionally writes a timestamp
/// pair and a `VK_EXT_debug_utils` region label, so one render pass is a single
/// stage with both a CPU-record time and a GPU-execute time. CPU spans are
/// buffered per in-flight slot and published together with that slot's resolved
/// GPU timings, so each stage's CPU and GPU halves come from the same frame
/// (the whole snapshot lags the in-flight depth). Consumers read the plain
/// @ref FrameMetrics — never a backend type — so the `ui` overlay and a
/// headless harness read it identically.

#include <cstdint>
#include <memory>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/frame_metrics.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

class Device;
class Allocator;

/// @brief Construction parameters for a @ref Profiler.
struct ProfilerConfig {
  /// In-flight depth: how many frames the CPU may run ahead of the GPU. Must
  /// match the render loop's frames-in-flight, since a slot's GPU timings are
  /// resolved only when that slot comes back around (its prior submission
  /// retired). Must be >= 1.
  uint32_t frames_in_flight = 2;
  /// Upper bound on @ref Profiler::gpu_scope calls per frame; sizes the
  /// timestamp pool (`frames_in_flight * this * 2` queries). Scopes beyond it
  /// in a frame are still CPU-timed and still labelled, but carry no GPU
  /// timing. Must be >= 1.
  uint32_t max_gpu_sections_per_frame = 32;
};

/// @brief Collects per-stage CPU/GPU timings, memory, and fps for a frame and
///        publishes them as a @ref FrameMetrics.
///
/// Built from a @ref Device's capabilities once: GPU timing is available only
/// where the graphics queue reports non-zero `timestampValidBits` (MoltenVK may
/// report zero — then GPU scopes degrade to CPU-only timing plus their label,
/// never an error), and labels emit only where `VK_EXT_debug_utils` is enabled
/// (see @ref Device::debug_utils). Drive it once per frame: @ref begin_frame at
/// the top (after the slot's in-flight fence is waited), @ref cpu_scope /
/// @ref gpu_scope around work, @ref end_frame at the bottom; read the latest
/// resolved snapshot from @ref metrics.
///
/// @warning The @p device passed to @ref create must outlive the profiler (it
///          owns a timestamp pool freed through that device). A @ref Scope must
///          not outlive the frame it was opened in, nor the profiler. A memory
///          source set via @ref set_memory_source must outlive the profiler.
///
/// @code
/// ProfilerConfig cfg;
/// cfg.frames_in_flight = 2;
/// Result<Profiler> r = Profiler::create(device, cfg);
/// if (!r) return r.status();
/// Profiler& profiler = r.value();
/// profiler.set_memory_source(&allocator);  // optional memory figures
///
/// // Per frame (slot + cmd from the render loop; cmd outside any render pass):
/// profiler.begin_frame(frame.slot, frame.cmd);
/// {
///   Profiler::Scope pass = profiler.gpu_scope(frame.cmd, "main pass");
///   // ... record draws into frame.cmd ...
/// }  // pass ends: GPU end-timestamp + label close recorded here
/// profiler.end_frame();
/// const FrameMetrics& m = profiler.metrics();  // resolved snapshot (lags N)
/// @endcode
class VG_CORE_API Profiler {
  struct Impl;  // pImpl: seals the timestamp pool, marker table, and chrono

 public:
  /// @brief A timed stage: a CPU span, optionally paired with a GPU timestamp
  ///        region and a debug-utils label, finalized on destruction.
  ///
  /// Returned by @ref Profiler::cpu_scope / @ref Profiler::gpu_scope.
  /// Destroying it stops the CPU clock and — for a GPU scope — records the end
  /// timestamp and closes the label, so scope it to exactly the work it should
  /// measure. A default-constructed (or moved-from) `Scope` is inert and
  /// finalizes nothing.
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
  /// @param config  In-flight depth and the per-frame GPU-scope bound.
  /// @return The profiler on success, or a non-OK @ref Status:
  ///         @ref Status::Code::InvalidArgument when @p config has a zero
  ///         `frames_in_flight` or `max_gpu_sections_per_frame`, otherwise a
  ///         propagated failure from creating the timestamp pool.
  static Result<Profiler> create(const Device& device,
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
  void set_memory_source(const Allocator* allocator) noexcept;

  /// @brief Begin a frame: publish the frame this slot last held, reset its
  ///        timestamp range, and start the frame's CPU clock.
  /// @param slot  The in-flight slot index in `[0, frames_in_flight)`. The
  ///              caller must already have waited this slot's in-flight fence,
  ///              so the prior submission's timestamps are readable without
  ///              stalling (@ref windowing::FrameLoop::begin_frame does this).
  /// @param cmd   The frame's recording command buffer, outside any render pass
  ///              (where `vkCmdResetQueryPool` is legal). Pass `VK_NULL_HANDLE`
  ///              for a CPU-only frame; GPU scopes then record no timestamps.
  void begin_frame(uint32_t slot, VkCommandBuffer cmd) noexcept;

  /// @brief End the frame: stamp its total CPU time and fps and sample memory,
  ///        buffering them for this slot to publish when the slot recurs.
  void end_frame() noexcept;

  /// @brief Open a CPU-only timed stage.
  /// @param name  Stage label; must have string-literal lifetime (stored by
  ///              pointer, see @ref FrameMetrics::Section::name).
  /// @return A @ref Scope timing until it is destroyed; inert if called outside
  ///         a @ref begin_frame / @ref end_frame pair.
  Scope cpu_scope(const char* name);

  /// @brief Open a stage that times CPU and — when GPU timing is available and
  ///        the per-frame bound is not exhausted — GPU, and emits a debug-utils
  ///        region label.
  /// @param cmd   The recording command buffer the timestamps and label record
  ///              into (the same buffer passed to @ref begin_frame).
  /// @param name  Stage label and label text; string-literal lifetime.
  /// @return A @ref Scope timing until it is destroyed; inert if called outside
  ///         a frame or with a null @p cmd.
  Scope gpu_scope(VkCommandBuffer cmd, const char* name);

  /// @return The most recently resolved frame's metrics, empty until the first
  ///         slot recurs (the snapshot lags the in-flight depth) and empty for
  ///         a moved-from profiler.
  const FrameMetrics& metrics() const noexcept;

  /// @return Whether GPU timestamp timing is available (the graphics queue
  ///         reports non-zero `timestampValidBits`). When false, GPU scopes
  ///         time CPU only and every @ref FrameMetrics::Section reports
  ///         `has_gpu` false.
  bool gpu_timing() const noexcept;

  /// @return `true` if this owns profiler resources (false when moved-from).
  bool valid() const noexcept { return impl_ != nullptr; }

 private:
  Profiler() noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::gfx

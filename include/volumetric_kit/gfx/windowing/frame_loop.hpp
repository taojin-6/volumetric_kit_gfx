// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_loop.hpp
/// @brief The frames-in-flight render loop: per-frame command buffers + sync
///        that drive a @ref Swapchain's acquire → render → present cycle.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/windowing/export.hpp"

namespace volumetric_kit::core {
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx {

class Profiler;

namespace windowing {

class Swapchain;

/// @brief A timeline value a frame's submission waits for, and the stages
///        that wait for it (@ref Frame::waits).
struct FrameWait {
  /// The timeline and the value to reach. Its semaphore is borrowed: it must
  /// outlive the frame's execution.
  core::TimelinePoint point;
  /// The stages that wait; non-zero. Nothing in them starts before the value
  /// is reached, and the producer's writes are visible to them. The default
  /// holds the whole frame; a narrower mask lets the rest start, e.g.
  /// `TRANSFER` before an image update, or `VERTEX_INPUT | DRAW_INDIRECT`
  /// before a live mesh's draw.
  VkPipelineStageFlags stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
};

/// @brief One in-flight frame handed to the caller by @ref
/// FrameLoop::begin_frame.
///
/// Record draws into @ref cmd between `target->begin()` and `target->end()`;
/// the image is already in `COLOR_ATTACHMENT_OPTIMAL`. Add any timeline
/// values the frame waits for or sets, then pass the same `Frame` back to
/// @ref FrameLoop::end_frame to submit + present it.
struct Frame {
  VkCommandBuffer cmd = VK_NULL_HANDLE;  ///< Recording command buffer.
  const RenderTarget* target = nullptr;  ///< The acquired image's target.
  uint32_t image_index = 0;              ///< Swapchain image index.
  uint32_t slot = 0;                     ///< Frame-in-flight slot (internal).
  /// The frame's number: one more than the previous frame's, from 1. The
  /// loop's @ref FrameLoop::timeline reaches it once the frame's GPU work
  /// completes, so it keys what the frame uses (@ref RetireQueue).
  uint64_t number = 0;
  /// Timeline values the frame's submission waits for, as @ref
  /// FrameLoop::end_frame says.
  std::vector<FrameWait> waits;
  /// Timeline values @ref FrameLoop::end_frame sets once the frame's GPU work
  /// completes, one per semaphore. Each semaphore is borrowed: it must
  /// outlive the frame's execution.
  std::vector<core::TimelinePoint> signals;
};

/// @brief Drives a @ref Swapchain with a ring of `N` in-flight frames so the
/// CPU
///        stays at most `N` frames ahead of the GPU.
///
/// Owns, per frame-in-flight slot (`N`), a command buffer + an image-available
/// semaphore + an in-flight fence; per swapchain image (`M`), a
/// render-finished semaphore; and one timeline semaphore that numbers the
/// frames (@ref timeline). The split matters: reusing one render-finished
/// semaphore across slots races the presentation engine, so it is keyed by
/// image (and an image still in flight from an earlier frame is waited for,
/// by number, before reuse).
///
/// The extent-taking @ref begin_frame drives the whole windowed protocol: it
/// rebuilds the swapchain when it went stale or the window resized (running the
/// @ref set_recreate_callback hook after each rebuild), skips the tick while
/// the window is minimized, and hands back a @ref Frame otherwise — @ref
/// end_frame then submits and presents it. The zero-argument @ref begin_frame
/// is the raw building block for callers that own the recreate policy
/// themselves; it and @ref end_frame surface stale results as statuses
/// classified by @ref swapchain_stale.
///
/// A loop is neither copied nor moved: @ref create hands it out behind a
/// `std::unique_ptr`, so it stays where it was made and is never empty.
///
/// @warning The @p device and @p swapchain passed to @ref create must outlive
///          the loop (it borrows both). Destruction drains the renderer's
///          queues (`core::Device::wait_idle`) to finish in-flight frames, so
///          teardown is safe mid-flight; on a shared adopted device that waits
///          only on the renderer's queues, not a sibling library's. A profiler
///          attached via @ref set_profiler and
///          anything captured by the @ref set_recreate_callback hook are
///          likewise borrowed and must outlive the loop, or be detached first.
///
/// @code
/// auto loop = windowing::FrameLoop::create(device, swapchain);
/// RetireQueue retire(loop.value()->timeline());
/// while (running) {
///   auto frame = loop.value()->begin_frame(window_extent());
///   if (!frame) return fail(frame.status());          // hard error only
///   if (!frame.value()) { wait_events(); continue; }  // minimized
///   const Frame& f = *frame.value();
///   retire.poll();  // free what completed frames used
///   f.target->begin(f.cmd, clear);
///   // ... bind pipeline, set viewport/scissor, draw ...
///   f.target->end(f.cmd);
///   core::Status end = loop.value()->end_frame(f);
///   if (!end.ok() && !swapchain_stale(end)) return fail(end);
/// }
/// @endcode
class VG_WINDOWING_API FrameLoop {
 public:
  /// @brief Create a loop driving @p swapchain with @p frames_in_flight slots.
  /// @param device            A device with a graphics (and present) queue that
  ///                          enabled the renderer's requirements
  ///                          (@ref device_requirements).
  /// @param swapchain         The swapchain to acquire from / present to.
  /// @param frames_in_flight  CPU-ahead depth (>= 1; 2 is the common default).
  /// @return The loop on success, or a non-OK `core::Status` (invalid argument
  ///         for a zero count or empty swapchain;
  ///         `core::Status::Code::Unsupported` for a @p device without the
  ///         renderer's requirements; otherwise a propagated failure).
  static core::Result<std::unique_ptr<FrameLoop>> create(
      const core::Device& device, Swapchain& swapchain,
      uint32_t frames_in_flight = 2);

  ~FrameLoop();
  FrameLoop(const FrameLoop&) = delete;
  FrameLoop& operator=(const FrameLoop&) = delete;
  FrameLoop(FrameLoop&&) = delete;
  FrameLoop& operator=(FrameLoop&&) = delete;

  /// @brief Begin the next frame, owning the windowed-loop protocol: rebuilds
  ///        the swapchain when it went stale (a prior out-of-date / suboptimal
  ///        result), a frame failed after its acquire (@ref end_frame), or
  ///        @p current_extent changed, re-runs the @ref set_recreate_callback
  ///        hook after each rebuild, and retries the acquire once.
  /// @param current_extent  The window's current framebuffer extent (e.g. from
  ///                        `glfwGetFramebufferSize`).
  /// @return The @ref Frame to record and pass to @ref end_frame; an *empty*
  ///         optional when nothing can render this tick (minimized window, or
  ///         the surface is still settling after a rebuild) — poll/wait for
  ///         events and call again; a non-OK `core::Status` only for hard
  ///         failures (device loss, a failed rebuild or recreate hook) — do not
  ///         retry those.
  core::Result<std::optional<Frame>> begin_frame(VkExtent2D current_extent);

  /// @brief Register a hook run after every internal swapchain rebuild by the
  ///        extent-taking @ref begin_frame, before the next acquire — rebuild
  ///        swapchain-sized resources here (e.g. a depth attachment).
  /// @param callback  Receives the rebuilt swapchain's extent; a non-OK return
  ///                  aborts the frame and surfaces from @ref begin_frame.
  ///                  Whatever it captures must outlive the loop. Pass an empty
  ///                  function to detach.
  void set_recreate_callback(std::function<core::Status(VkExtent2D)> callback);

  /// @brief Begin the next frame (raw protocol): wait for the frame that last
  ///        used this slot, acquire an image, wait for the frame that last
  ///        drew to it, begin the slot's command buffer, and transition the
  ///        image into `COLOR_ATTACHMENT_OPTIMAL`.
  /// @return The @ref Frame to record into; a non-OK `core::Status` carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` (recreate the swapchain and retry) or
  ///         another failed `VkResult`; `core::Status::Code::InvalidArgument`
  ///         when the borrowed swapchain is empty (after a failed rebuild).
  /// @note A *successful* `begin_frame` must be paired with exactly one @ref
  ///       end_frame for the returned @ref Frame: the acquire signals this
  ///       slot's image-available semaphore, and only @ref end_frame consumes
  ///       it. Dropping a returned @ref Frame leaves that semaphore signalled.
  ///       A failed `begin_frame` returns no @ref Frame and needs no pairing —
  ///       on an internal failure *after* the acquire, an empty submit in the
  ///       frame's place consumes the acquire and sets its number, and the
  ///       acquired image is released as @ref end_frame's note says.
  /// @note Adapts automatically when @ref Swapchain::recreate changes the image
  ///       count: the per-image sync objects are rebuilt to match on entry.
  core::Result<Frame> begin_frame();

  /// @brief End the frame from @ref begin_frame: transition the image to
  ///        `PRESENT_SRC`, end its command buffer and submit it -- waiting for
  ///        `frame.waits` and setting its number on @ref timeline and
  ///        `frame.signals` once it completes -- present it, and advance to
  ///        the next slot.
  ///
  /// A value in `frame.waits` must already be reached, or be set by work
  /// already submitted -- to this queue or another -- that itself waits for
  /// nothing not yet submitted. The present waits for this submit, and Vulkan
  /// requires that of a present's waits
  /// (`VUID-vkQueuePresentKHR-pWaitSemaphores-03268`); and a frame held for a
  /// value nothing has been submitted to set holds every later wait on the
  /// queue, so the next @ref begin_frame, a swapchain rebuild's queue drain
  /// (which holds the submit mutex) and the loop's destruction would block
  /// forever. Gate a producer whose value is not yet submitted on the host
  /// instead. The values are checked as `core::check_timeline_points` checks
  /// them with `core::TimelineWaits::Submitted`, against the core's record of
  /// submitted values, which the core's submits, every frame's submit and
  /// `core::note_timeline_signals` add to: a value waited for must be reached
  /// or recorded, and a value in `frame.signals` must exceed its semaphore's
  /// current value, every recorded value, and any value the frame waits for
  /// on it. What the producer itself waits for is not checked.
  /// @param frame  The frame returned by @ref begin_frame this iteration.
  /// @return OK on success; a non-OK `core::Status` carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR` (classify with
  ///         @ref swapchain_stale; the next extent-taking @ref begin_frame
  ///         rebuilds automatically) or another failed `VkResult`;
  ///         `core::Status::Code::InvalidArgument` when @p frame did not come
  ///         from its @ref begin_frame, a wait has a zero stage mask, a signal
  ///         names @ref timeline, or `core::check_timeline_points` refuses the
  ///         values.
  /// @note On a failure *before* the submit reaches the queue -- a refusal
  ///       included -- the frame's commands never run: an empty submit in its
  ///       place consumes the acquire and sets its number -- and
  ///       `frame.signals`, unless they were refused -- so the *slot* stays
  ///       reusable and nothing waiting for them hangs. The image the frame
  ///       acquired was never presented, and only a present or a swapchain
  ///       rebuild releases an acquired image: the next extent-taking @ref
  ///       begin_frame rebuilds, and a caller of the raw @ref begin_frame
  ///       calls @ref Swapchain::recreate before its next one, or repeated
  ///       failures exhaust the acquirable images. A failed present already
  ///       submitted the frame: the slot advances normally and only the
  ///       presentation is reported.
  core::Status end_frame(const Frame& frame);

  /// @return The timeline the loop's frames set: it reaches a frame's
  ///         @ref Frame::number once that frame's GPU work completes, in
  ///         submission order. It stays at one address for the loop's life,
  ///         so a @ref RetireQueue or a `core::TimelinePoint` may borrow it;
  ///         another queue's work may wait for a frame through it. Only the
  ///         loop sets it.
  /// @note Through MoltenVK a number is reached before the submission's
  ///       completion handler has run, and the frame's query results become
  ///       readable only then: read them once the loop reuses the slot, as
  ///       an attached @ref Profiler does.
  const core::TimelineSemaphore& timeline() const noexcept { return timeline_; }

  /// @return The newest frame number whose GPU work has completed (0 before
  ///         the first), or a backend `core::Status` (device lost).
  core::Result<uint64_t> completed() const;

  /// @return The newest frame number submitted (0 before the first): what
  ///         to retire a resource at between frames, as every frame that may
  ///         use it has this number or a lower one.
  uint64_t submitted() const noexcept { return submitted_; }

  /// @brief Attach a profiler the loop drives automatically, or detach with
  ///        `nullptr`.
  /// @param profiler  Borrowed and nullable. When set, the loop calls the
  ///                  profiler's `begin_frame` (for the slot, after waiting for
  ///                  its previous frame, with the frame's command buffer) and
  ///                  `end_frame` around each frame, so the caller only opens
  ///                  scopes on the frame's `cmd`. Its `frames_in_flight`
  ///                  should match this loop's. Must outlive the loop, or be
  ///                  detached first; `nullptr` restores the unprofiled path.
  void set_profiler(Profiler* profiler) noexcept;

  /// @return The CPU-ahead depth (number of in-flight slots).
  uint32_t frames_in_flight() const noexcept {
    return static_cast<uint32_t>(command_buffers_.size());
  }

 private:
  FrameLoop(const core::Device& device, Swapchain& swapchain);

  // Rebuild the per-image sync state (render_finished_ / image_frames_) when
  // the swapchain handle changed — i.e. after a Swapchain::recreate produced a
  // fresh chain (see last_swapchain_). A no-op (one handle comparison) on the
  // common path.
  core::Status ensure_image_sync();

  // The slot frame `number` records into: frames take the slots in turn.
  uint32_t slot_of(uint64_t number) const noexcept {
    return static_cast<uint32_t>((number - 1) % command_buffers_.size());
  }

  // end_frame's refusals of `frame.waits` / `frame.signals`.
  core::Status check_points(const Frame& frame) const;

  // A submit that sets frame `number` and `signals` reached the queue: count
  // it, and add its values to the core's record (core::note_timeline_signals).
  void record_submit(uint64_t number,
                     const std::vector<core::TimelinePoint>& signals);

  // Stand in for the next frame when it failed between its acquire and its
  // submit: an empty submit waits image_available_[slot], whose pending
  // signal only a queue submit may consume, and sets the frame's number and
  // `signals` as the frame would have, so the slot's next use (ordered after
  // that number like any frame's) finds the semaphore free and nothing
  // waiting for the values hangs. Error path only.
  core::Status recover_slot(uint32_t slot,
                            const std::vector<core::TimelinePoint>& signals);

  // Drain the renderer's own queues (`core::Device::wait_idle`: graphics, and
  // present when distinct) so teardown cannot free command buffers / semaphores
  // the GPU still references. Waiting the queues (not this loop's timeline)
  // also covers a present that reported out-of-date and left a render-finished
  // semaphore that no frame number covers. Never device-wide: on a shared
  // adopted device that would idle a sibling library's queues too. Best-effort:
  // errors are unreportable from the destructor and moot on a lost device.
  void drain() noexcept;

  const core::Device& device_;  // borrowed; outlives this
  Swapchain& swapchain_;        // borrowed; outlives this
  // The frame numbers.
  core::TimelineSemaphore timeline_;
  // The newest frame number submitted; the next frame is one more.
  uint64_t submitted_ = 0;
  // Declared before the buffers it owns so they free back before it is
  // destroyed.
  core::CommandPool pool_;
  std::vector<core::CommandBuffer> command_buffers_;  // per slot (N)
  std::vector<core::Semaphore> image_available_;      // per slot (N)
  std::vector<core::Fence> in_flight_;                // per slot (N)
  std::vector<core::Semaphore> render_finished_;      // per swapchain image (M)
  // Per image (M): the number of the last submitted frame that rendered to
  // it (0 for none), so a re-acquired image still in use is waited on before
  // reuse.
  std::vector<uint64_t> image_frames_;
  // The swapchain handle the per-image sync above was built for. A mismatch in
  // ensure_image_sync means a Swapchain::recreate produced a fresh chain, so
  // render_finished_ / image_frames_ must be rebuilt (a same-count rebuild
  // still retires the old images and can leave a semaphore signaled).
  VkSwapchainKHR last_swapchain_ = VK_NULL_HANDLE;
  Profiler* profiler_ = nullptr;  // borrowed, nullable; optional turnkey driver
  // Managed-protocol state (the extent-taking begin_frame): the rebuild hook
  // and whether a stale acquire/present, a frame that failed after its
  // acquire, or a resize armed a rebuild. Resize is detected against
  // Swapchain::requested_extent() (the pre-clamp requested size), so a request
  // the surface pins does not rebuild every tick.
  std::function<core::Status(VkExtent2D)> recreate_callback_;
  bool needs_recreate_ = false;
};

}  // namespace windowing
}  // namespace volumetric_kit::gfx

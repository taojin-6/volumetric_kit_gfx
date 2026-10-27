// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_loop.hpp
/// @brief The frames-in-flight render loop: per-frame command buffers + sync
///        that drive a @ref Swapchain's acquire → render → present cycle, or
///        a ring of offscreen targets where there is no display.

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/windowing/export.hpp"

namespace volumetric_kit::core {
class Allocator;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx {

class Profiler;

namespace windowing {

class Swapchain;

/// @brief A timeline value a frame's submission waits for, and the stages
///        that wait for it (@ref Frame::waits).
///
/// A live mesh another queue wrote: its producer published it once its
/// writes completed, so the value is reached, as @ref FrameLoop::end_frame
/// requires. The GPU wait then stalls nothing; it makes the writes visible
/// to the draw, and only the draw's stages wait.
///
/// @code
/// // `ready`: the core::TimelinePoint the producer's writes set.
/// frame.waits.push_back({ready, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
///                                   VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT});
/// // ... record the mesh's indirect draw into frame.cmd ...
/// const core::Status ended = loop->end_frame(frame);
/// @endcode
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

/// @brief What an offscreen @ref FrameLoop renders into -- one target per
///        frame-in-flight slot -- and the pace a display would set.
///
/// @code
/// windowing::OffscreenLoopDesc desc;
/// desc.target.extent = {1280, 720};
/// desc.target.color_format = VK_FORMAT_B8G8R8A8_SRGB;
/// desc.target.depth_format = VK_FORMAT_D32_SFLOAT;
/// desc.frame_interval = std::chrono::nanoseconds(1000000000 / 60);
/// auto loop = windowing::FrameLoop::create(device, allocator, desc);
/// @endcode
struct OffscreenLoopDesc {
  /// Every slot's target: its extent, formats, and whether it reads back.
  OffscreenTargetDesc target;
  /// The least time between two frames' @ref FrameLoop::begin_frame, as a
  /// display's refresh paces a FIFO swapchain's presents; not negative. 0
  /// begins a frame as soon as its slot is free.
  std::chrono::nanoseconds frame_interval{0};
};

/// @brief One in-flight frame handed to the caller by @ref
/// FrameLoop::begin_frame.
///
/// Record draws into @ref cmd between `target->begin()` and `target->end()`;
/// the image is already in `COLOR_ATTACHMENT_OPTIMAL` (an offscreen target's
/// depth, in `DEPTH_ATTACHMENT_OPTIMAL`). Add any timeline values the frame
/// waits for or sets, then pass the same `Frame` back to @ref
/// FrameLoop::end_frame to submit + present it (an offscreen loop's, to
/// submit it).
struct Frame {
  VkCommandBuffer cmd = VK_NULL_HANDLE;  ///< Recording command buffer.
  const RenderTarget* target = nullptr;  ///< The frame's image's target.
  /// Swapchain image index; an offscreen loop's frame carries its @ref slot.
  uint32_t image_index = 0;
  /// The frame-in-flight slot, below @ref FrameLoop::frames_in_flight. The
  /// slot's previous frame has completed when @ref FrameLoop::begin_frame
  /// returns it, so per-slot consumers such as recon's `MeshExchange` key on
  /// it.
  uint32_t slot = 0;
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
  /// An offscreen loop's frame: its slot's target, which @ref target
  /// renders into. Record its `record_readback` into @ref cmd after the
  /// render to read the frame back: `pixels()` holds it once the slot recurs
  /// (@ref FrameLoop::begin_frame waited for its fence) or the device has
  /// idled. Null for a swapchain frame.
  const OffscreenTarget* offscreen = nullptr;
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
/// An offscreen loop, made from an @ref OffscreenLoopDesc where there is no
/// display, owns one @ref OffscreenTarget per slot in place of the
/// swapchain, its image-available semaphores and its per-image objects. Its
/// frames have no acquire, present or rebuild: @ref begin_frame waits out the
/// desc's pace where a display's refresh would hold a FIFO acquire, then
/// prepares the slot's target. Every other rule -- the numbering, the waits
/// and signals with their refusals and stand-in submit, the profiler's rows,
/// the timeline, teardown -- is the swapchain loop's, so a renderer draws the
/// same frames with or without a window, and reads chosen ones back
/// (@ref Frame::offscreen).
///
/// @warning The @p device and @p swapchain passed to @ref create must outlive
///          the loop (it borrows both; an offscreen loop borrows only the
///          device). Destruction drains the renderer's
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

  /// @brief Create an offscreen loop: @p frames_in_flight slots, each with an
  ///        @ref OffscreenTarget made to `desc.target`, paced by
  ///        `desc.frame_interval`.
  /// @param device            A device with a graphics queue that enabled the
  ///                          renderer's requirements
  ///                          (@ref device_requirements); it needs no
  ///                          present queue.
  /// @param allocator         Allocates the targets on @p device; it need not
  ///                          outlive the loop.
  /// @param desc              The targets and the pace.
  /// @param frames_in_flight  CPU-ahead depth (>= 1; 2 is the common default).
  /// @return The loop on success, or a non-OK `core::Status` (invalid argument
  ///         for a zero count or a negative interval;
  ///         `core::Status::Code::Unsupported` for a @p device without the
  ///         renderer's requirements; what `OffscreenTarget::create` refuses,
  ///         such as a zero extent or an undefined color format; otherwise a
  ///         propagated failure).
  ///
  /// @code
  /// windowing::OffscreenLoopDesc desc;
  /// desc.target = {extent, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_D32_SFLOAT};
  /// auto loop = windowing::FrameLoop::create(device, allocator, desc);
  /// // ... build pipelines for loop.value()->layout() ...
  /// auto frame = loop.value()->begin_frame();
  /// // ... draw into frame.value().target, then
  /// // frame.value().offscreen->record_readback(frame.value().cmd) ...
  /// core::Status ended = loop.value()->end_frame(frame.value());
  /// // pixels() holds the frame once its slot recurs.
  /// @endcode
  static core::Result<std::unique_ptr<FrameLoop>> create(
      const core::Device& device, core::Allocator& allocator,
      const OffscreenLoopDesc& desc, uint32_t frames_in_flight = 2);

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
  /// @param current_extent  The window's current framebuffer extent (e.g.
  ///                        `glfw::framebuffer_size`).
  /// @return The @ref Frame to record and pass to @ref end_frame; an *empty*
  ///         optional when nothing can render this tick (a 0 x 0
  ///         @p current_extent, as for a minimized window, or the surface is
  ///         still settling after a rebuild) — poll/wait for
  ///         events and call again; a non-OK `core::Status` only for hard
  ///         failures (device loss, a failed rebuild or recreate hook) — do not
  ///         retry those.
  /// @note An offscreen loop has no window to follow: it ignores
  ///       @p current_extent, never skips, and returns the raw
  ///       @ref begin_frame's frame or failure.
  core::Result<std::optional<Frame>> begin_frame(VkExtent2D current_extent);

  /// @brief Register a hook run after every internal swapchain rebuild by the
  ///        extent-taking @ref begin_frame, before the next acquire — rebuild
  ///        swapchain-sized resources here (e.g. a depth attachment). An
  ///        offscreen loop never rebuilds, so never runs it.
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
  ///         when the borrowed swapchain is empty (moved from, or a failed
  ///         rebuild).
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
  /// @note On an offscreen loop: wait for the frame that last used this slot,
  ///       wait out the pace (@ref OffscreenLoopDesc::frame_interval), begin
  ///       the slot's command buffer, and prepare the slot's target
  ///       (`OffscreenTarget::prepare`). It fails only as a Vulkan call does
  ///       (device loss); its frame, too, is ended exactly once.
  core::Result<Frame> begin_frame();

  /// @brief End the frame from @ref begin_frame: transition the image to
  ///        `PRESENT_SRC`, end its command buffer and submit it -- waiting for
  ///        `frame.waits` and setting its number on @ref timeline and
  ///        `frame.signals` once it completes -- present it, and advance to
  ///        the next slot. An offscreen loop's frame is submitted as it was
  ///        recorded, with no transition and no present.
  ///
  /// A value in `frame.waits` must already be reached. A submitted producer
  /// may still depend on an unresolved host signal, directly or through
  /// earlier queue work. Such a dependency would invalidate the present's
  /// waits (`VUID-vkQueuePresentKHR-pWaitSemaphores-03268`) and could hold a
  /// later @ref begin_frame, offscreen or not, or a queue drain forever.
  /// Wait for the producer on the host before ending the frame; the GPU wait
  /// still supplies the memory dependency. `core::check_timeline_points`
  /// checks the values with `core::TimelineWaits::Reached`. A value in
  /// `frame.signals` must exceed its semaphore's current value, every
  /// recorded signal value, and any value the frame waits for on it. The
  /// core's submits, every frame's submit and `core::note_timeline_signals`
  /// record the signal values.
  /// @param frame  The frame returned by @ref begin_frame this iteration.
  /// @return OK on success; a non-OK `core::Status` carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR` (classify with
  ///         @ref swapchain_stale; the next extent-taking @ref begin_frame
  ///         rebuilds automatically) or another failed `VkResult`;
  ///         `core::Status::Code::InvalidArgument` when @p frame is not the
  ///         one @ref begin_frame last handed out (a hand-built, repeated or
  ///         another loop's frame), the swapchain was rebuilt or emptied
  ///         since, a wait has a zero stage mask, a signal names @ref
  ///         timeline, or `core::check_timeline_points` refuses the values.
  /// @note On a failure *before* the submit reaches the queue -- a refusal
  ///       included -- the frame's commands never run: an empty submit in its
  ///       place consumes the acquire (a swapchain frame's) and sets its
  ///       number -- and `frame.signals`, unless they were refused -- so the
  ///       *slot* stays reusable and nothing waiting for them hangs. A
  ///       swapchain frame's image was never presented, and only a present or
  ///       a swapchain rebuild releases an acquired image: the next
  ///       extent-taking @ref
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
  ///
  /// The loop adds its own rows to each frame, after the caller's stages
  /// (`Profiler::add_rows`): `frame wait`, its waits for the slot's previous
  /// frame, the acquire and the image's previous frame, where a loop paced by
  /// the display waits, or an offscreen loop's pace; `submit`, ending and
  /// submitting the command buffer; and `present`, the present call, which an
  /// offscreen loop has none of. `frame wait` and `present` are wait
  /// rows (`cpu_ms == wait_ms`), which reach the scopes the caller has open
  /// around them as blocked time (`core::WaitScope`). Without a profiler the
  /// loop adds no rows, and its waits, the acquire, the pace and the present
  /// included, still reach those scopes (`core::note_blocked`).
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

  /// @return The format + sample signature of what the frames render into --
  ///         the swapchain's (@ref Swapchain::layout) or the offscreen
  ///         targets' -- for building a compatible pipeline either way.
  RenderTargetLayout layout() const;

 private:
  // What a create builds before the loop, so that a failure frees objects no
  // submit has used and only a whole loop's destructor drains the queues
  // (frame_loop.cpp).
  struct Owned;

  // Build the slots' command buffers and fences, a swapchain loop's
  // image-available semaphores (`presents`), and the timeline.
  static core::Result<Owned> own_slots(const core::Device& device,
                                       uint32_t frames_in_flight,
                                       bool presents);

  // `swapchain` is null for an offscreen loop.
  FrameLoop(const core::Device& device, Swapchain* swapchain, Owned owned);

  // An offscreen loop's pace: wait until its next frame may begin, as a
  // blocked span, and set when the one after may.
  void pace();

  // Build the per-image sync state (render_finished_ / image_frames_)
  // for the first chain, and rebuild them when the swapchain handle changed —
  // i.e. after a Swapchain::recreate produced a fresh chain (see
  // last_swapchain_). A no-op (one handle comparison) on the common path.
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

  // Stand in for the next frame when it failed between its acquire (an
  // offscreen loop: its fence wait) and its submit: an empty submit waits
  // image_available_[slot], whose pending signal only a queue submit may
  // consume, and sets the frame's number and `signals` as the frame would
  // have, so the slot's next use (ordered after that number like any
  // frame's) finds the semaphore free and nothing waiting for the values
  // hangs. Error path only.
  core::Status recover_slot(uint32_t slot,
                            const std::vector<core::TimelinePoint>& signals);

  const core::Device& device_;  // borrowed; outlives this
  // Borrowed; outlives this. Null for an offscreen loop.
  Swapchain* swapchain_ = nullptr;
  // The frame numbers.
  core::TimelineSemaphore timeline_;
  // The newest frame number submitted; the next frame is one more.
  uint64_t submitted_ = 0;
  // Declared before the buffers it owns so they free back before it is
  // destroyed.
  core::CommandPool pool_;
  std::vector<core::CommandBuffer> command_buffers_;  // per slot (N)
  std::vector<core::Semaphore> image_available_;  // per slot (N); swapchain's
  std::vector<core::Fence> in_flight_;            // per slot (N)
  // An offscreen loop's targets, per slot (N), and the RenderTarget over each
  // that its frames carry; empty for a swapchain loop.
  std::vector<OffscreenTarget> targets_;
  std::vector<RenderTarget> render_targets_;
  // An offscreen loop's pace (OffscreenLoopDesc::frame_interval), and the
  // time its next frame may begin.
  std::chrono::nanoseconds frame_interval_{0};
  std::chrono::steady_clock::time_point next_begin_{};
  std::vector<core::Semaphore> render_finished_;  // per swapchain image (M)
  // Per image (M): the number of the last submitted frame that rendered to
  // it (0 for none), so a re-acquired image still in use is waited on before
  // reuse.
  std::vector<uint64_t> image_frames_;
  // The swapchain handle the per-image sync above was built for (null until
  // the first begin_frame). A mismatch in ensure_image_sync means a
  // Swapchain::recreate produced a fresh chain, so render_finished_ /
  // image_frames_ must be rebuilt (a same-count rebuild still retires the
  // old images and can leave a semaphore signaled); in end_frame, that the
  // frame's chain is gone.
  VkSwapchainKHR last_swapchain_ = VK_NULL_HANDLE;
  // The image (an offscreen loop: the slot) of the Frame begin_frame last
  // handed out, until end_frame takes it back; end_frame refuses any other
  // Frame.
  std::optional<uint32_t> acquired_image_;
  Profiler* profiler_ = nullptr;  // borrowed, nullable; optional turnkey driver
  // The loop's own rows for the attached profiler, reused frame to frame.
  core::StageMetrics rows_;
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

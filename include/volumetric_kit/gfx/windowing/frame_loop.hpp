// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_loop.hpp
/// @brief The frames-in-flight render loop: per-frame command buffers + sync
///        that drive a @ref Swapchain's acquire → render → present cycle.

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/windowing/export.hpp"

namespace volumetric_kit::gfx {

class Device;
class Profiler;

namespace windowing {

class Swapchain;

/// @brief One in-flight frame handed to the caller by @ref
/// FrameLoop::begin_frame.
///
/// Record draws into @ref cmd between `target->begin()` and `target->end()`;
/// the image is already in `COLOR_ATTACHMENT_OPTIMAL`. Pass the same `Frame`
/// back to @ref FrameLoop::end_frame to submit + present it.
struct Frame {
  VkCommandBuffer cmd = VK_NULL_HANDLE;  ///< Recording command buffer.
  const RenderTarget* target = nullptr;  ///< The acquired image's target.
  uint32_t image_index = 0;              ///< Swapchain image index.
  uint32_t slot = 0;                     ///< Frame-in-flight slot (internal).
};

/// @brief Drives a @ref Swapchain with a ring of `N` in-flight frames so the
/// CPU
///        stays at most `N` frames ahead of the GPU.
///
/// Owns, per frame-in-flight slot (`N`), a command buffer + an image-available
/// semaphore + an in-flight fence; and, per swapchain image (`M`), a
/// render-finished semaphore. The split matters: reusing one render-finished
/// semaphore across slots races the presentation engine, so it is keyed by
/// image (and an image still in flight from an earlier slot is fence-waited
/// before reuse).
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
/// @warning The @p device and @p swapchain passed to @ref create must outlive
///          the loop (it borrows both). Destruction drains the renderer's
///          queues (@ref Device::wait_idle) to finish in-flight frames, so
///          teardown is safe mid-flight; on a shared adopted device that waits
///          only on the renderer's queues, not a sibling library's. A profiler
///          attached via @ref set_profiler and
///          anything captured by the @ref set_recreate_callback hook are
///          likewise borrowed and must outlive the loop, or be detached first.
///
/// @code
/// auto loop = windowing::FrameLoop::create(device, swapchain);
/// while (running) {
///   auto frame = loop.value().begin_frame(window_extent());
///   if (!frame) return fail(frame.status());          // hard error only
///   if (!frame.value()) { wait_events(); continue; }  // minimized
///   const Frame& f = *frame.value();
///   f.target->begin(f.cmd, clear);
///   // ... bind pipeline, set viewport/scissor, draw ...
///   f.target->end(f.cmd);
///   Status end = loop.value().end_frame(f);
///   if (!end.ok() && !swapchain_stale(end)) return fail(end);
/// }
/// @endcode
class VG_WINDOWING_API FrameLoop {
 public:
  /// @brief Construct an empty loop (owns nothing; `valid()` is false).
  FrameLoop() = default;

  /// @brief Create a loop driving @p swapchain with @p frames_in_flight slots.
  /// @param device            A device with a graphics (and present) queue.
  /// @param swapchain         The swapchain to acquire from / present to.
  /// @param frames_in_flight  CPU-ahead depth (>= 1; 2 is the common default).
  /// @return The loop on success, or a non-OK @ref Status (invalid argument for
  ///         a zero count or empty swapchain; otherwise a propagated failure).
  static Result<FrameLoop> create(const Device& device, Swapchain& swapchain,
                                  uint32_t frames_in_flight = 2);

  ~FrameLoop();
  FrameLoop(FrameLoop&& other) noexcept;
  FrameLoop& operator=(FrameLoop&& other) noexcept;
  FrameLoop(const FrameLoop&) = delete;
  FrameLoop& operator=(const FrameLoop&) = delete;

  /// @brief Begin the next frame, owning the windowed-loop protocol: rebuilds
  ///        the swapchain when it went stale (a prior out-of-date / suboptimal
  ///        result) or @p current_extent changed, re-runs the @ref
  ///        set_recreate_callback hook after each rebuild, and retries the
  ///        acquire once.
  /// @param current_extent  The window's current framebuffer extent (e.g. from
  ///                        `glfwGetFramebufferSize`).
  /// @return The @ref Frame to record and pass to @ref end_frame; an *empty*
  ///         optional when nothing can render this tick (minimized window, or
  ///         the surface is still settling after a rebuild) — poll/wait for
  ///         events and call again; a non-OK @ref Status only for hard failures
  ///         (device loss, a failed rebuild or recreate hook) — do not retry
  ///         those.
  Result<std::optional<Frame>> begin_frame(VkExtent2D current_extent);

  /// @brief Register a hook run after every internal swapchain rebuild by the
  ///        extent-taking @ref begin_frame, before the next acquire — rebuild
  ///        swapchain-sized resources here (e.g. a depth attachment).
  /// @param callback  Receives the rebuilt swapchain's extent; a non-OK return
  ///                  aborts the frame and surfaces from @ref begin_frame.
  ///                  Whatever it captures must outlive the loop. Pass an empty
  ///                  function to detach.
  void set_recreate_callback(std::function<Status(VkExtent2D)> callback);

  /// @brief Begin the next frame (raw protocol): wait this slot's fence,
  ///        acquire an image, begin its command buffer, and transition the
  ///        image into `COLOR_ATTACHMENT_OPTIMAL`.
  /// @return The @ref Frame to record into; a non-OK @ref Status carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` (recreate the swapchain and retry) or
  ///         another failed `VkResult`; @ref Status::Code::InvalidArgument when
  ///         the borrowed swapchain is empty (after a failed rebuild).
  /// @note A *successful* `begin_frame` must be paired with exactly one @ref
  ///       end_frame for the returned @ref Frame: the acquire signals this
  ///       slot's image-available semaphore, and only @ref end_frame consumes
  ///       it. Dropping a returned @ref Frame leaves that semaphore signalled.
  ///       A failed `begin_frame` returns no @ref Frame and needs no pairing —
  ///       on an internal failure *after* the acquire, the slot's sync state is
  ///       restored (a brief blocking submit) before the error returns.
  /// @note Adapts automatically when @ref Swapchain::recreate changes the image
  ///       count: the per-image sync objects are rebuilt to match on entry.
  Result<Frame> begin_frame();

  /// @brief End the frame from @ref begin_frame: transition the image to
  ///        `PRESENT_SRC`, end + submit its command buffer, present it, and
  ///        advance to the next slot.
  /// @param frame  The frame returned by @ref begin_frame this iteration.
  /// @return OK on success; a non-OK @ref Status carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR` (classify with
  ///         @ref swapchain_stale; the next extent-taking @ref begin_frame
  ///         rebuilds automatically) or another failed `VkResult`.
  /// @note On a failure *before* the submit reaches the queue, the slot's sync
  ///       state is restored (a brief blocking submit) so the *slot* stays
  ///       reusable — but the image this frame acquired was never presented,
  ///       and an acquired image is only released by a present or a swapchain
  ///       rebuild. So the caller must @ref Swapchain::recreate before
  ///       continuing (the loop below does), not merely retry, or repeated
  ///       failures will exhaust the acquirable images. A failed present
  ///       already submitted the frame: the slot advances normally and only the
  ///       presentation is reported.
  Status end_frame(const Frame& frame);

  /// @brief Attach a profiler the loop drives automatically, or detach with
  ///        `nullptr`.
  /// @param profiler  Borrowed and nullable. When set, the loop calls the
  ///                  profiler's `begin_frame` (for the slot, after waiting its
  ///                  fence, with the frame's command buffer) and `end_frame`
  ///                  around each frame, so the caller only opens scopes on the
  ///                  frame's `cmd`. Its `frames_in_flight` should match this
  ///                  loop's. Must outlive the loop, or be detached first;
  ///                  `nullptr` restores the unprofiled path.
  void set_profiler(Profiler* profiler) noexcept;

  /// @return The CPU-ahead depth (number of in-flight slots).
  uint32_t frames_in_flight() const noexcept {
    return static_cast<uint32_t>(in_flight_.size());
  }

  /// @return `true` if this owns frame resources.
  bool valid() const noexcept { return !in_flight_.empty(); }

 private:
  // Rebuild the per-image sync objects (render_finished_ / images_in_flight_)
  // when the swapchain handle changed — i.e. after a Swapchain::recreate
  // produced a fresh chain (see last_swapchain_). A no-op (one handle
  // comparison) on the common path.
  Status ensure_image_sync();

  // Restore a slot whose acquire signal was never consumed (a failure between
  // acquire and submit): drain image_available_[slot] with an empty submit and
  // re-signal the slot fence. If the drain cannot even be issued (the queue is
  // failing), the fence is instead replaced with a fresh signaled one so the
  // next begin_frame never blocks on it. Blocking; error-path only.
  Status recover_slot(uint32_t slot);

  // Drain the renderer's own queues (@ref Device::wait_idle: graphics, and
  // present when distinct) so teardown cannot free command buffers / semaphores
  // the GPU still references. Waiting the queues (not this loop's fences) still
  // covers a present that reported out-of-date and left a render-finished
  // semaphore with no fence to wait on. Never device-wide: on a shared adopted
  // device that would idle a sibling library's queues too. Best-effort: errors
  // are unreportable from the destructor and moot on a lost device.
  void drain() noexcept;

  const Device* device_ = nullptr;  // borrowed; outlives this
  Swapchain* swapchain_ = nullptr;  // borrowed; outlives this
  // Optional only because CommandPool is create-only (no public default ctor);
  // declared before the buffers it owns so they free back before it is
  // destroyed.
  std::optional<CommandPool> pool_;
  std::vector<CommandBuffer> command_buffers_;  // per slot (N)
  std::vector<Semaphore> image_available_;      // per slot (N)
  std::vector<Fence> in_flight_;                // per slot (N)
  std::vector<Semaphore> render_finished_;      // per swapchain image (M)
  // Per image (M): the in-flight fence of the slot that last rendered to it, so
  // a re-acquired image still in use is waited on before reuse. Non-owning.
  std::vector<VkFence> images_in_flight_;
  // The swapchain handle the per-image sync above was built for. A mismatch in
  // ensure_image_sync means a Swapchain::recreate produced a fresh chain, so
  // render_finished_ / images_in_flight_ must be rebuilt (a same-count rebuild
  // still retires the old images and can leave a semaphore signaled).
  VkSwapchainKHR last_swapchain_ = VK_NULL_HANDLE;
  uint32_t current_slot_ = 0;
  Profiler* profiler_ = nullptr;  // borrowed, nullable; optional turnkey driver
  // Managed-protocol state (the extent-taking begin_frame): the rebuild hook
  // and whether a stale acquire/present or a resize armed a rebuild. Resize is
  // detected against Swapchain::requested_extent() (the pre-clamp requested
  // size), so a request the surface pins does not rebuild every tick.
  std::function<Status(VkExtent2D)> recreate_callback_;
  bool needs_recreate_ = false;
};

}  // namespace windowing
}  // namespace volumetric_kit::gfx

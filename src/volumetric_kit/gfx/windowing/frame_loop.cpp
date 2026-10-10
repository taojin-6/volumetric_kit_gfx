// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/windowing/frame_loop.hpp"

#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace volumetric_kit::gfx::windowing {

namespace {

// The semaphores one of the loop's submits waits for and sets, as the parallel
// arrays VkSubmitInfo and VkTimelineSemaphoreSubmitInfo take. A binary
// semaphore's value is ignored, so it is 0.
class SubmitSync {
 public:
  void wait(VkSemaphore semaphore, VkPipelineStageFlags stages,
            uint64_t value = 0) {
    waits_.push_back(semaphore);
    wait_stages_.push_back(stages);
    wait_values_.push_back(value);
  }
  void signal(VkSemaphore semaphore, uint64_t value = 0) {
    signals_.push_back(semaphore);
    signal_values_.push_back(value);
  }

  // Submit @p cmd (none when null) to the device's queue, signalling @p fence.
  VkResult submit(const core::Device& device, const VkCommandBuffer* cmd,
                  VkFence fence) const {
    VkTimelineSemaphoreSubmitInfo values{};
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values_.size());
    values.pWaitSemaphoreValues = wait_values_.data();
    values.signalSemaphoreValueCount =
        static_cast<uint32_t>(signal_values_.size());
    values.pSignalSemaphoreValues = signal_values_.data();
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &values;
    submit.waitSemaphoreCount = static_cast<uint32_t>(waits_.size());
    submit.pWaitSemaphores = waits_.data();
    submit.pWaitDstStageMask = wait_stages_.data();
    submit.commandBufferCount = cmd != nullptr ? 1 : 0;
    submit.pCommandBuffers = cmd;
    submit.signalSemaphoreCount = static_cast<uint32_t>(signals_.size());
    submit.pSignalSemaphores = signals_.data();
    // Through the device, so a shared (adopted) graphics queue stays
    // externally synchronized under its submit mutex.
    return device.queue_submit(1, &submit, fence);
  }

 private:
  std::vector<VkSemaphore> waits_;
  std::vector<VkPipelineStageFlags> wait_stages_;
  std::vector<uint64_t> wait_values_;
  std::vector<VkSemaphore> signals_;
  std::vector<uint64_t> signal_values_;
};

}  // namespace

FrameLoop::FrameLoop(const core::Device& device, Swapchain& swapchain,
                     core::TimelineSemaphore timeline, core::CommandPool pool,
                     std::vector<core::CommandBuffer> command_buffers,
                     std::vector<core::Semaphore> image_available,
                     std::vector<core::Fence> in_flight)
    : device_(device),
      swapchain_(swapchain),
      timeline_(std::move(timeline)),
      pool_(std::move(pool)),
      command_buffers_(std::move(command_buffers)),
      image_available_(std::move(image_available)),
      in_flight_(std::move(in_flight)) {}

core::Result<std::unique_ptr<FrameLoop>> FrameLoop::create(
    const core::Device& device, Swapchain& swapchain,
    uint32_t frames_in_flight) {
  if (frames_in_flight == 0) {
    return core::Status::invalid_argument(
        "FrameLoop::create: frames_in_flight must be >= 1");
  }
  if (!swapchain.valid()) {
    return core::Status::invalid_argument(
        "FrameLoop::create: swapchain is empty");
  }
  // The loop's per-frame barriers name graphics stages and its frames record
  // dynamic rendering: a device made for another library may lack either.
  VKC_TRY(device.check_enabled(device_requirements())
              .with_context("FrameLoop::create"));

  // Build what the loop owns before the loop, so a failure here frees objects
  // no submit has used, and only a whole loop's destructor drains the queues.
  // Per frame-in-flight slot: a command buffer, an image-available semaphore,
  // and an in-flight fence (created signaled so the first wait does not block).
  // The per-image sync waits for the first begin_frame's ensure_image_sync.
  VKC_ASSIGN(core::CommandPool pool,
             core::CommandPool::create(device.handle(), device.queue_family()));
  std::vector<core::CommandBuffer> command_buffers;
  std::vector<core::Semaphore> image_available;
  std::vector<core::Fence> in_flight;
  for (uint32_t i = 0; i < frames_in_flight; ++i) {
    VKC_ASSIGN(core::CommandBuffer cmd, pool.allocate_primary());
    command_buffers.push_back(std::move(cmd));
    VKC_ASSIGN(core::Semaphore available,
               core::Semaphore::create(device.handle()));
    image_available.push_back(std::move(available));
    VKC_ASSIGN(core::Fence fence,
               core::Fence::create(device.handle(), /*signaled=*/true));
    in_flight.push_back(std::move(fence));
  }

  // The frame numbers: frame n sets n, from 1.
  VKC_ASSIGN(core::TimelineSemaphore timeline,
             core::TimelineSemaphore::create(device));
  // The constructor is private, so no std::make_unique.
  return std::unique_ptr<FrameLoop>(
      new FrameLoop(device, swapchain, std::move(timeline), std::move(pool),
                    std::move(command_buffers), std::move(image_available),
                    std::move(in_flight)));
}

core::Status FrameLoop::ensure_image_sync() {
  // Build the per-image sync objects for the first chain, and rebuild them when
  // the swapchain handle changes — i.e. after a Swapchain::recreate produced a
  // fresh chain. Keying on the handle (not just the image count) also covers a
  // same-count rebuild: that still retires the old images, and a failed present
  // (OUT_OF_DATE) can leave a render-finished semaphore signaled, so reusing it
  // would double-signal on the next submit. recreate() idles the device first,
  // so the old per-image objects are drained and safe to replace, and every
  // frame that drew to the old images has completed. A no-op (two comparisons)
  // on the common path.
  //
  // The handle alone is not a sufficient key: a retired VkSwapchainKHR value
  // can be recycled by the driver, so the array sizes are checked too. That
  // makes a stale-sync mismatch impossible regardless of driver behavior, since
  // the image count is recomputed from live surface caps on every build.
  const VkSwapchainKHR current = swapchain_.handle();
  const uint32_t image_count = swapchain_.image_count();
  if (current == last_swapchain_ && render_finished_.size() == image_count &&
      image_frames_.size() == image_count) {
    return core::Status{};
  }
  render_finished_.clear();
  render_finished_.reserve(image_count);
  for (uint32_t i = 0; i < image_count; ++i) {
    VKC_ASSIGN(core::Semaphore finished,
               core::Semaphore::create(device_.handle()));
    render_finished_.push_back(std::move(finished));
  }
  image_frames_.assign(image_count, 0);
  last_swapchain_ = current;
  return core::Status{};
}

core::Result<std::optional<Frame>> FrameLoop::begin_frame(
    VkExtent2D current_extent) {
  if (current_extent.width == 0 || current_extent.height == 0) {
    // Minimized: nothing to acquire or rebuild; an armed rebuild stays armed
    // for the restore.
    return std::optional<Frame>{};
  }
  if (current_extent.width != swapchain_.requested_extent().width ||
      current_extent.height != swapchain_.requested_extent().height) {
    // The window resized under us; some platforms (MoltenVK in particular)
    // never report OUT_OF_DATE for it. Checked against the swapchain's last
    // *requested* extent (not its surface-clamped one), so a request the
    // surface pins to a different size does not rebuild every tick.
    needs_recreate_ = true;
  }
  // At most one rebuild + one acquire retry per call: rebuild once if armed,
  // then on a stale acquire retry the acquire without a second rebuild. A
  // still-stale result skips the tick rather than spinning -- or churning
  // rebuilds and depth reallocations -- while the surface settles.
  bool rebuilt_this_call = false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (needs_recreate_ && !rebuilt_this_call) {
      const core::Status rebuilt = swapchain_.recreate(current_extent);
      if (!rebuilt.ok()) {
        if (rebuilt.domain() == core::Status::Code::InvalidArgument &&
            swapchain_.valid()) {
          // The surface reported a zero extent mid-rebuild (still minimized):
          // the old chain is intact, so skip this tick and retry later.
          return std::optional<Frame>{};
        }
        // Any other failure destroyed the chain (recreate passes oldSwapchain,
        // which retires it even when creation fails); drop the sync cache key
        // with it.
        last_swapchain_ = VK_NULL_HANDLE;
        return rebuilt;
      }
      // The previous chain was retired and its handle value freed back to the
      // driver, which is free to hand the same value out again. Drop the cache
      // key so ensure_image_sync cannot mistake a recycled handle for the chain
      // it last sized against -- including when the hook below fails and a
      // later retry builds a *third* chain before ensure_image_sync ever
      // observes this one.
      last_swapchain_ = VK_NULL_HANDLE;
      rebuilt_this_call = true;
      // Run the consumer hook *before* clearing needs_recreate_, so a failed
      // resource rebuild leaves the loop armed rather than falsely "in sync".
      // (The rebuild produced a fresh swapchain handle, so the next raw
      // begin_frame's ensure_image_sync refreshes the per-image semaphores.)
      if (recreate_callback_) {
        VKC_TRY(recreate_callback_(swapchain_.extent()));
      }
      needs_recreate_ = false;
    }
    core::Result<Frame> frame = begin_frame();
    if (frame.ok()) {
      return std::optional<Frame>(std::move(frame).value());
    }
    if (swapchain_stale(frame.status())) {
      needs_recreate_ = true;
      continue;
    }
    return frame.status();
  }
  return std::optional<Frame>{};
}

core::Result<Frame> FrameLoop::begin_frame() {
  if (!swapchain_.valid()) {
    return core::Status::invalid_argument(
        "FrameLoop::begin_frame: swapchain is empty (moved from, or a failed "
        "rebuild)");
  }
  // A Swapchain::recreate may have changed the image count since the last
  // frame; resize the per-image sync arrays before indexing them below.
  VKC_TRY(ensure_image_sync());

  const uint64_t number = submitted_ + 1;
  const uint32_t slot = slot_of(number);

  // Keep the CPU at most N frames ahead: wait until this slot's previous frame
  // has retired before reusing its command buffer / semaphore / queries. On
  // its fence, not its number: through MoltenVK the number is reached before
  // the submission's completion handler has run, which is what makes its
  // timestamp queries readable (DECISIONS.md, "Frames are numbered on a
  // timeline").
  VKC_TRY(in_flight_[slot].wait());

  core::Result<uint32_t> acquired =
      swapchain_.acquire_next_image(image_available_[slot].handle());
  if (!acquired.ok()) {
    // OUT_OF_DATE (or another error) flows to the caller, which recreates the
    // swapchain and retries; nothing was consumed, so the retry is this same
    // frame. The slot fence is reset only right before a submit that
    // re-signals it, so it is still signalled here and the retry does not
    // deadlock.
    return acquired.status();
  }
  const uint32_t image_index = acquired.value();

  // The acquired image may still be in use by an earlier frame on another
  // slot when there are more images than in-flight frames; wait for it too.
  if (image_frames_[image_index] != 0) {
    const core::Status waited = timeline_.wait(image_frames_[image_index]);
    if (!waited.ok()) {
      // The acquire above left this slot's semaphore with a pending signal;
      // stand in for the frame before surfacing the error (best-effort: the
      // original error outranks a recovery failure).
      (void)recover_slot(slot, {});
      return waited;
    }
  }

  const VkCommandBuffer cmd = command_buffers_[slot].handle();
  const core::Status begun =
      command_buffers_[slot].begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  if (!begun.ok()) {
    (void)recover_slot(slot, {});
    return begun;
  }

  // Dynamic rendering does not transition images; move it into the attachment
  // layout. UNDEFINED discards the previous (presented) contents, which is fine
  // since the render clears. srcStage is COLOR_ATTACHMENT_OUTPUT (not
  // TOP_OF_PIPE) so the transition is ordered *after* the image-available
  // semaphore — end_frame's submit waits it at that same stage — instead of
  // racing the presentation engine's last read of the image.
  ImageBarrierDesc to_color;
  to_color.image = swapchain_.image(image_index);
  to_color.src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  to_color.dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  to_color.dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_color.new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  cmd_image_barrier(cmd, to_color);

  // Drive the profiler's frame lifecycle (no-op when none is attached). The
  // slot's previous frame was waited for above, so its timestamps are now
  // readable; the command buffer is recording and outside any render pass,
  // where the slot's query-range reset is legal.
  if (profiler_ != nullptr) {
    profiler_->begin_frame(slot, cmd);
  }

  Frame frame;
  frame.cmd = cmd;
  frame.target = &swapchain_.render_target(image_index);
  frame.image_index = image_index;
  frame.slot = slot;
  frame.number = number;
  acquired_image_ = image_index;
  return frame;
}

core::Status FrameLoop::check_points(const Frame& frame) const {
  // The loop's own rules; the core checks the rest as it does for its own
  // submits, and refuses a wait whose value nothing has been submitted to set.
  std::vector<core::TimelinePoint> waits;
  waits.reserve(frame.waits.size());
  for (const FrameWait& wait : frame.waits) {
    if (wait.stages == 0) {
      return core::Status::invalid_argument(
          "FrameLoop::end_frame: a wait has no stages");
    }
    waits.push_back(wait.point);
  }
  for (const core::TimelinePoint& signal : frame.signals) {
    if (signal.semaphore == &timeline_) {
      return core::Status::invalid_argument(
          "FrameLoop::end_frame: only the loop sets its timeline");
    }
  }
  return core::check_timeline_points(device_, waits, frame.signals,
                                     "FrameLoop::end_frame",
                                     core::TimelineWaits::Submitted);
}

void FrameLoop::record_submit(uint64_t number,
                              const std::vector<core::TimelinePoint>& signals) {
  submitted_ = number;
  // Into the core's record of submitted values too, so its checks -- the
  // next frame's and the core's own submits' -- know them.
  core::note_timeline_signals({{&timeline_, number}});
  core::note_timeline_signals(signals);
}

core::Status FrameLoop::end_frame(const Frame& frame) {
  // Frame is a public aggregate, so a hand-built, repeated or other loop's
  // Frame can reach here: take only the one begin_frame last handed out. Its
  // command buffer is this loop's own, which tells it from another loop's.
  const uint32_t slot = slot_of(submitted_ + 1);
  if (!acquired_image_ || frame.number != submitted_ + 1 ||
      frame.image_index != *acquired_image_ || frame.slot != slot ||
      frame.cmd != command_buffers_[slot].handle()) {
    return core::Status::invalid_argument(
        "FrameLoop::end_frame: frame is not the one this loop's begin_frame "
        "last handed out");
  }
  // Spent whatever happens below: a failure before the submit restores the
  // slot, and after it the slot advances.
  acquired_image_.reset();

  // Check both before closing the profiler's frame: discarded commands never
  // write their queries, so their timestamps must not be read on slot reuse.
  const core::Status points = check_points(frame);
  const bool changed_swapchain = swapchain_.handle() != last_swapchain_;
  if (changed_swapchain || !points.ok()) {
    if (changed_swapchain) {
      // A destroyed chain invalidated the command buffer; only a reset or
      // begin leaves that state. An emptied (moved-from) chain needs the same
      // recovery, even while its old images still live in the new owner.
      (void)vkResetCommandBuffer(command_buffers_[slot].handle(), 0);
    } else {
      // A recording command buffer cannot be begun again: end it.
      (void)command_buffers_[slot].end();
    }
    // Consume the acquire and complete this frame's number. Only accepted
    // signals may be set; the refusal outranks a recovery failure.
    if (points.ok()) {
      (void)recover_slot(slot, frame.signals);
    } else {
      (void)recover_slot(slot, {});
      return points;
    }
    return core::Status::invalid_argument(
        "FrameLoop::end_frame: the swapchain was rebuilt or emptied since "
        "begin_frame");
  }

  // Close the profiler's frame (no-op when none is attached): the caller's
  // scopes have finalized into this command buffer, so the per-frame CPU/memory
  // figures can be stamped before the submit below.
  if (profiler_ != nullptr) {
    profiler_->end_frame();
  }

  // Transition the rendered image to PRESENT_SRC.
  ImageBarrierDesc to_present;
  to_present.image = swapchain_.image(frame.image_index);
  to_present.src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  to_present.dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  to_present.src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_present.old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_present.new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  cmd_image_barrier(frame.cmd, to_present);

  const core::Status ended = command_buffers_[slot].end();
  if (!ended.ok()) {
    (void)recover_slot(slot, frame.signals);
    return ended;
  }

  // The acquire and the present keep their binary semaphores; the frame's
  // number and the caller's values ride the same submit.
  const VkSemaphore rendered = render_finished_[frame.image_index].handle();
  SubmitSync sync;
  sync.wait(image_available_[slot].handle(),
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  for (const FrameWait& wait : frame.waits) {
    sync.wait(wait.point.semaphore->handle(), wait.stages, wait.point.value);
  }
  sync.signal(rendered);
  sync.signal(timeline_.handle(), frame.number);
  for (const core::TimelinePoint& signal : frame.signals) {
    sync.signal(signal.semaphore->handle(), signal.value);
  }
  // Reset the in-flight fence immediately before the submit that re-signals it,
  // so every earlier failure leaves it signalled and the slot's next
  // begin_frame never blocks on a fence nothing will signal.
  const core::Status fence_ready = in_flight_[slot].reset();
  if (!fence_ready.ok()) {
    (void)recover_slot(slot, frame.signals);
    return fence_ready;
  }
  const VkResult submitted =
      sync.submit(device_, &frame.cmd, in_flight_[slot].handle());
  if (submitted != VK_SUCCESS) {
    // The failed submit consumed nothing: the acquire signal is still pending
    // and the fence was just reset, so recover_slot restores both.
    (void)recover_slot(slot, frame.signals);
    return core::vk_error(submitted, "vkQueueSubmit");
  }
  record_submit(frame.number, frame.signals);
  image_frames_[frame.image_index] = frame.number;

  core::Status present = swapchain_.present(frame.image_index, rendered);
  if (swapchain_stale(present)) {
    // Arm the managed protocol's rebuild, as recover_slot does; raw callers
    // see the status as ever (they never read needs_recreate_). The protocol
    // is asymmetric (a managed begin_frame(extent), but end_frame stays
    // single), so the raw path's failures have nowhere else to land.
    // TODO: fold the managed begin/end protocol into an app-tier frame driver
    // so end_frame carries no managed state and the raw path is policy-free.
    needs_recreate_ = true;
  }
  // The slot advanced with the number regardless: the work was submitted and
  // will set it, so the slot is reusable next round even when present reports
  // out-of-date.
  return present;
}

core::Result<uint64_t> FrameLoop::completed() const {
  return timeline_.value();
}

core::Status FrameLoop::recover_slot(
    uint32_t slot, const std::vector<core::TimelinePoint>& signals) {
  // The frame's acquired image will not be presented, and only a present or
  // a swapchain rebuild releases an acquired image: arm the extent-taking
  // begin_frame's rebuild, or failed frames exhaust the chain.
  needs_recreate_ = true;
  // A successful acquire left image_available_[slot] with a pending signal that
  // only a queue submit may consume. An empty submit consumes it, sets the
  // frame's number, as the frame would have, and signals the slot fence, so
  // the slot's next begin_frame finds the semaphore free and the numbers stay
  // contiguous. Reached only on a failed frame; `signals` are ones
  // check_points accepted.
  // TODO: a refused frame or changed swapchain runs this, but the other
  // failures and the failed submit below lack test coverage -- no queue/command
  // call fails on a healthy device, so a fault-injection seam is needed to
  // exercise them; today they ship verified only by inspection.
  const uint64_t number = submitted_ + 1;
  SubmitSync sync;
  sync.wait(image_available_[slot].handle(),
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  sync.signal(timeline_.handle(), number);
  for (const core::TimelinePoint& signal : signals) {
    sync.signal(signal.semaphore->handle(), signal.value);
  }
  // The fence must be unsignaled for the submit to signal it.
  core::Status status = in_flight_[slot].reset();
  if (status.ok()) {
    const VkResult submitted =
        sync.submit(device_, nullptr, in_flight_[slot].handle());
    if (submitted == VK_SUCCESS) {
      record_submit(number, signals);
      return core::Status{};
    }
    status = core::vk_error(submitted, "vkQueueSubmit");
  }

  // The queue itself is failing, and the fence may now be unsignaled with
  // nothing that will ever signal it. Replace it with a fresh signaled fence
  // (safe -- no submit references this slot's fence here) so the next
  // begin_frame fails cleanly instead of blocking forever. The acquire
  // semaphore may stay signaled, but only when the queue is already broken,
  // where the next frame errors out anyway.
  VKC_ASSIGN(core::Fence resignaled,
             core::Fence::create(device_.handle(), /*signaled=*/true));
  in_flight_[slot] = std::move(resignaled);
  return status;
}

void FrameLoop::set_profiler(Profiler* profiler) noexcept {
  profiler_ = profiler;
}

void FrameLoop::set_recreate_callback(
    std::function<core::Status(VkExtent2D)> callback) {
  recreate_callback_ = std::move(callback);
}

FrameLoop::~FrameLoop() {
  // Drain the renderer's own queues (`core::Device::wait_idle`: graphics, and
  // present when distinct) so teardown cannot free command buffers / semaphores
  // the GPU still references. Waiting the queues (not this loop's timeline)
  // also covers a present that reported out-of-date and left a render-finished
  // semaphore that no frame number covers. Never device-wide: on a shared
  // adopted device that would idle a sibling library's queues too. Best-effort:
  // errors are unreportable from the destructor and moot on a lost device.
  (void)device_.wait_idle();
}

}  // namespace volumetric_kit::gfx::windowing

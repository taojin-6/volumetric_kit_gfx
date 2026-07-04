// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/windowing/frame_loop.hpp"

#include <utility>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/impl/command.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace volumetric_kit::gfx::windowing {

Result<FrameLoop> FrameLoop::create(const Device& device, Swapchain& swapchain,
                                    uint32_t frames_in_flight) {
  if (frames_in_flight == 0) {
    return Status::invalid_argument(
        "FrameLoop::create: frames_in_flight must be >= 1");
  }
  if (!swapchain.valid()) {
    return Status::invalid_argument("FrameLoop::create: swapchain is empty");
  }

  FrameLoop loop;
  loop.device_ = &device;
  loop.swapchain_ = &swapchain;
  // Seed the managed protocol's resize detection with the built extent, so the
  // first extent-taking begin_frame does not rebuild a fresh swapchain.
  loop.last_requested_extent_ = swapchain.extent();

  VG_ASSIGN(CommandPool pool,
            CommandPool::create(device.handle(), device.graphics_family()));
  loop.pool_ = std::move(pool);

  // Per frame-in-flight slot: a command buffer, an image-available semaphore,
  // and an in-flight fence (created signaled so the first wait does not block).
  for (uint32_t i = 0; i < frames_in_flight; ++i) {
    VG_ASSIGN(CommandBuffer cmd, loop.pool_->allocate_primary());
    loop.command_buffers_.push_back(std::move(cmd));
    VG_ASSIGN(Semaphore available, Semaphore::create(device.handle()));
    loop.image_available_.push_back(std::move(available));
    VG_ASSIGN(Fence fence, Fence::create(device.handle(), /*signaled=*/true));
    loop.in_flight_.push_back(std::move(fence));
  }

  // Per swapchain image: a render-finished semaphore (keyed by image, not slot,
  // to avoid racing the presentation engine) + an in-flight-fence tracking
  // slot. Built here and rebuilt by ensure_image_sync() if a later
  // Swapchain::recreate changes the image count.
  VG_TRY(loop.ensure_image_sync());

  return loop;
}

Status FrameLoop::ensure_image_sync() {
  // Rebuild the per-image sync objects when the swapchain handle changes — i.e.
  // after a Swapchain::recreate produced a fresh chain. Keying on the handle
  // (not just the image count) also covers a same-count rebuild: that still
  // retires the old images, and a failed present (OUT_OF_DATE) can leave a
  // render-finished semaphore signaled, so reusing it would double-signal on
  // the next submit. recreate() idles the device first, so the old per-image
  // objects are drained and safe to replace. A no-op (one handle comparison)
  // on the common path.
  const VkSwapchainKHR current = swapchain_->handle();
  if (current == last_swapchain_) {
    return Status{};
  }
  const uint32_t image_count = swapchain_->image_count();
  render_finished_.clear();
  render_finished_.reserve(image_count);
  for (uint32_t i = 0; i < image_count; ++i) {
    VG_ASSIGN(Semaphore finished, Semaphore::create(device_->handle()));
    render_finished_.push_back(std::move(finished));
  }
  images_in_flight_.assign(image_count, VK_NULL_HANDLE);
  last_swapchain_ = current;
  return Status{};
}

Result<std::optional<Frame>> FrameLoop::begin_frame(VkExtent2D current_extent) {
  if (swapchain_ == nullptr) {
    return Status::invalid_argument("FrameLoop::begin_frame on an empty loop");
  }
  if (current_extent.width == 0 || current_extent.height == 0) {
    // Minimized: nothing to acquire or rebuild; an armed rebuild stays armed
    // for the restore.
    return std::optional<Frame>{};
  }
  if (current_extent.width != last_requested_extent_.width ||
      current_extent.height != last_requested_extent_.height) {
    // The window resized under us; some platforms (MoltenVK in particular)
    // never report OUT_OF_DATE for it.
    needs_recreate_ = true;
  }
  // One rebuild + one acquire retry per call: a second stale result yields a
  // skipped tick rather than looping here while the surface settles.
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (needs_recreate_) {
      const Status rebuilt = swapchain_->recreate(current_extent);
      if (!rebuilt.ok()) {
        if (rebuilt.domain() == Status::Code::InvalidArgument &&
            swapchain_->valid()) {
          // The surface reported a zero extent mid-rebuild (still minimized):
          // the old chain is intact, so skip this tick and retry later.
          return std::optional<Frame>{};
        }
        return rebuilt;
      }
      needs_recreate_ = false;
      last_requested_extent_ = current_extent;
      if (recreate_callback_) {
        VG_TRY(recreate_callback_(swapchain_->extent()));
      }
    }
    Result<Frame> frame = begin_frame();
    if (frame.ok()) {
      return std::optional<Frame>(frame.value());
    }
    if (swapchain_stale(frame.status())) {
      needs_recreate_ = true;
      continue;
    }
    return frame.status();
  }
  return std::optional<Frame>{};
}

Result<Frame> FrameLoop::begin_frame() {
  if (swapchain_ == nullptr || !swapchain_->valid()) {
    return Status::invalid_argument(
        "FrameLoop::begin_frame: swapchain is empty (moved-from, or a failed "
        "rebuild)");
  }
  // A Swapchain::recreate may have changed the image count since the last
  // frame; resize the per-image sync arrays before indexing them below.
  VG_TRY(ensure_image_sync());

  const uint32_t slot = current_slot_;

  // Keep the CPU at most N frames ahead: wait until this slot's previous frame
  // has finished on the GPU before reusing its command buffer / semaphore.
  VG_TRY(in_flight_[slot].wait());

  Result<uint32_t> acquired =
      swapchain_->acquire_next_image(image_available_[slot].handle());
  if (!acquired.ok()) {
    // OUT_OF_DATE (or another error) flows to the caller, which recreates the
    // swapchain and retries. The slot fence is reset only in end_frame (right
    // before the submit that re-signals it), so it is still signalled here and
    // the retry does not deadlock.
    return acquired.status();
  }
  const uint32_t image_index = acquired.value();

  // The acquired image may still be in use by an earlier (different) slot when
  // there are more images than in-flight frames; wait that fence too.
  if (images_in_flight_[image_index] != VK_NULL_HANDLE) {
    const VkResult waited =
        vkWaitForFences(device_->handle(), 1, &images_in_flight_[image_index],
                        VK_TRUE, UINT64_MAX);
    if (waited != VK_SUCCESS) {
      // The acquire above left this slot's semaphore with a pending signal;
      // restore the slot before surfacing the error (best-effort: the original
      // error outranks a recovery failure).
      (void)recover_slot(slot);
      return vk_error(waited, "vkWaitForFences");
    }
  }
  images_in_flight_[image_index] = in_flight_[slot].handle();

  const VkCommandBuffer cmd = command_buffers_[slot].handle();
  const Status begun =
      command_buffers_[slot].begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  if (!begun.ok()) {
    (void)recover_slot(slot);
    return begun;
  }

  // Dynamic rendering does not transition images; move it into the attachment
  // layout. UNDEFINED discards the previous (presented) contents, which is fine
  // since the render clears. srcStage is COLOR_ATTACHMENT_OUTPUT (not
  // TOP_OF_PIPE) so the transition is ordered *after* the image-available
  // semaphore — end_frame's submit waits it at that same stage — instead of
  // racing the presentation engine's last read of the image.
  cmd_image_barrier(cmd, swapchain_->image(image_index),
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  // Drive the profiler's frame lifecycle (no-op when none is attached). The
  // slot's fence was waited above, so its prior submission's timestamps are now
  // readable; the command buffer is recording and outside any render pass,
  // where the slot's query-range reset is legal.
  if (profiler_ != nullptr) {
    profiler_->begin_frame(slot, cmd);
  }

  Frame frame;
  frame.cmd = cmd;
  frame.target = &swapchain_->render_target(image_index);
  frame.image_index = image_index;
  frame.slot = slot;
  return frame;
}

Status FrameLoop::end_frame(const Frame& frame) {
  const uint32_t slot = frame.slot;

  // Close the profiler's frame (no-op when none is attached): the caller's
  // scopes have finalized into this command buffer, so the per-frame CPU/memory
  // figures can be stamped before the submit below.
  if (profiler_ != nullptr) {
    profiler_->end_frame();
  }

  // Transition the rendered image to PRESENT_SRC.
  cmd_image_barrier(frame.cmd, swapchain_->image(frame.image_index),
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  const Status ended = command_buffers_[slot].end();
  if (!ended.ok()) {
    // This frame's acquire signal was never consumed by a submit; restore the
    // slot's sync state so the loop stays usable (best-effort: the original
    // error outranks a recovery failure).
    (void)recover_slot(slot);
    return ended;
  }

  const VkSemaphore wait_sem = image_available_[slot].handle();
  const VkSemaphore signal_sem = render_finished_[frame.image_index].handle();
  const VkPipelineStageFlags wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &wait_sem;
  submit.pWaitDstStageMask = &wait_stage;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &frame.cmd;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &signal_sem;

  // Reset the in-flight fence immediately before the submit that re-signals it.
  // Resetting here (rather than in begin_frame) keeps the fence signalled on
  // every earlier failure or abandoned Frame, so the next begin_frame on this
  // slot never blocks forever on a fence that will never be submitted.
  const Status fence_ready = in_flight_[slot].reset();
  if (!fence_ready.ok()) {
    (void)recover_slot(slot);
    return fence_ready;
  }
  const VkResult submitted = vkQueueSubmit(device_->graphics_queue(), 1,
                                           &submit, in_flight_[slot].handle());
  if (submitted != VK_SUCCESS) {
    // The failed submit consumed nothing: the acquire signal is still pending
    // and the fence was just reset, so recover_slot restores both.
    (void)recover_slot(slot);
    return vk_error(submitted, "vkQueueSubmit");
  }

  Status present = swapchain_->present(frame.image_index, signal_sem);
  if (swapchain_stale(present)) {
    // Arm the managed protocol's rebuild; raw callers see the status as ever.
    needs_recreate_ = true;
  }
  // Advance regardless: the work was submitted and the fence will signal, so
  // the slot is reusable next round even when present reports out-of-date.
  current_slot_ =
      static_cast<uint32_t>((current_slot_ + 1) % in_flight_.size());
  return present;
}

Status FrameLoop::recover_slot(uint32_t slot) {
  // A successful acquire left image_available_[slot] with a pending signal that
  // only a queue submit may consume: an empty submit drains it and re-signals
  // the slot fence, restoring both invariants (semaphore unsignaled, fence
  // signaled) so the next begin_frame on this slot neither trips the validation
  // layers nor deadlocks. Blocks until the drain completes; reached only on a
  // failed frame.
  // TODO: these recovery branches (here and the callers') lack test coverage —
  // no queue/fence call fails on a healthy device, so a fault-injection seam is
  // needed to exercise them; today they ship verified only by inspection.
  const VkSemaphore wait_sem = image_available_[slot].handle();
  const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &wait_sem;
  submit.pWaitDstStageMask = &wait_stage;

  // The fence must be unsignaled for the submit to signal it: reset, submit,
  // then wait. If any step fails the queue itself is failing — fall through.
  Status status = in_flight_[slot].reset();
  if (status.ok()) {
    const VkResult submitted = vkQueueSubmit(
        device_->graphics_queue(), 1, &submit, in_flight_[slot].handle());
    if (submitted == VK_SUCCESS) {
      return in_flight_[slot].wait();
    }
    status = vk_error(submitted, "vkQueueSubmit");
  }

  // The drain never reached the queue, so the fence is now unsignaled with
  // nothing that will ever signal it. Replace it with a fresh signaled fence
  // (safe — no submit references this slot's fence here) so the next
  // begin_frame fails cleanly instead of blocking forever in
  // in_flight_[slot].wait(). The acquire semaphore may stay signaled, but only
  // when the queue is already broken, where the next frame errors out anyway.
  VG_ASSIGN(Fence resignaled,
            Fence::create(device_->handle(), /*signaled=*/true));
  in_flight_[slot] = std::move(resignaled);
  return status;
}

void FrameLoop::set_profiler(Profiler* profiler) noexcept {
  profiler_ = profiler;
}

void FrameLoop::set_recreate_callback(
    std::function<Status(VkExtent2D)> callback) {
  recreate_callback_ = std::move(callback);
}

FrameLoop::~FrameLoop() { drain(); }

void FrameLoop::drain() noexcept {
  if (device_ != nullptr && !in_flight_.empty()) {
    (void)vkDeviceWaitIdle(device_->handle());
  }
}

// Hand-written (not defaulted) because the command buffers free back to the
// pool: destruction order matters, and the move pair must null the borrowed
// pointers on the source so a moved-from loop is fully empty.
FrameLoop::FrameLoop(FrameLoop&& other) noexcept
    : device_(other.device_),
      swapchain_(other.swapchain_),
      pool_(std::move(other.pool_)),
      command_buffers_(std::move(other.command_buffers_)),
      image_available_(std::move(other.image_available_)),
      in_flight_(std::move(other.in_flight_)),
      render_finished_(std::move(other.render_finished_)),
      images_in_flight_(std::move(other.images_in_flight_)),
      last_swapchain_(other.last_swapchain_),
      current_slot_(other.current_slot_),
      profiler_(other.profiler_),
      recreate_callback_(std::move(other.recreate_callback_)),
      needs_recreate_(other.needs_recreate_),
      last_requested_extent_(other.last_requested_extent_) {
  other.device_ = nullptr;
  other.swapchain_ = nullptr;
  other.last_swapchain_ = VK_NULL_HANDLE;
  other.current_slot_ = 0;
  other.profiler_ = nullptr;
  other.recreate_callback_ = nullptr;
  other.needs_recreate_ = false;
  other.last_requested_extent_ = VkExtent2D{};
}

FrameLoop& FrameLoop::operator=(FrameLoop&& other) noexcept {
  if (this != &other) {
    // Wait out our own in-flight frames, then release our resources in
    // dependency order before adopting other's: the command buffers free back
    // to the pool, so they must be destroyed before the pool. (A defaulted
    // move-assign assigns members in declaration order, freeing the pool first
    // while our command buffers still reference it.)
    drain();
    command_buffers_.clear();
    pool_.reset();

    device_ = other.device_;
    swapchain_ = other.swapchain_;
    pool_ = std::move(other.pool_);
    command_buffers_ = std::move(other.command_buffers_);
    image_available_ = std::move(other.image_available_);
    in_flight_ = std::move(other.in_flight_);
    render_finished_ = std::move(other.render_finished_);
    images_in_flight_ = std::move(other.images_in_flight_);
    last_swapchain_ = other.last_swapchain_;
    current_slot_ = other.current_slot_;
    profiler_ = other.profiler_;
    recreate_callback_ = std::move(other.recreate_callback_);
    needs_recreate_ = other.needs_recreate_;
    last_requested_extent_ = other.last_requested_extent_;

    other.device_ = nullptr;
    other.swapchain_ = nullptr;
    other.last_swapchain_ = VK_NULL_HANDLE;
    other.current_slot_ = 0;
    other.profiler_ = nullptr;
    other.recreate_callback_ = nullptr;
    other.needs_recreate_ = false;
    other.last_requested_extent_ = VkExtent2D{};
  }
  return *this;
}

}  // namespace volumetric_kit::gfx::windowing

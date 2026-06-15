// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/windowing/frame_loop.hpp"

#include <utility>

#include "volumetric_kit/gfx/core/device.hpp"
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
  // to avoid racing the presentation engine). TODO: re-create these if a
  // Swapchain::recreate ever changes the image count (stable in practice).
  const uint32_t image_count = swapchain.image_count();
  for (uint32_t i = 0; i < image_count; ++i) {
    VG_ASSIGN(Semaphore finished, Semaphore::create(device.handle()));
    loop.render_finished_.push_back(std::move(finished));
  }
  loop.images_in_flight_.assign(image_count, VK_NULL_HANDLE);

  return loop;
}

Result<Frame> FrameLoop::begin_frame() {
  const uint32_t slot = current_slot_;

  // Keep the CPU at most N frames ahead: wait until this slot's previous frame
  // has finished on the GPU before reusing its command buffer / semaphore.
  VG_TRY(in_flight_[slot].wait());

  Result<uint32_t> acquired =
      swapchain_->acquire_next_image(image_available_[slot].handle());
  if (!acquired.ok()) {
    // OUT_OF_DATE (or another error) flows to the caller, which recreates the
    // swapchain and retries. The slot fence stays signaled, so the retry is
    // safe.
    return acquired.status();
  }
  const uint32_t image_index = acquired.value();

  // The acquired image may still be in use by an earlier (different) slot when
  // there are more images than in-flight frames; wait that fence too.
  if (images_in_flight_[image_index] != VK_NULL_HANDLE) {
    vkWaitForFences(device_->handle(), 1, &images_in_flight_[image_index],
                    VK_TRUE, UINT64_MAX);
  }
  images_in_flight_[image_index] = in_flight_[slot].handle();

  // Reset the fence only now — after a possible early return above — so a frame
  // that never submits cannot leave the fence unsignaled and deadlock.
  VG_TRY(in_flight_[slot].reset());

  const VkCommandBuffer cmd = command_buffers_[slot].handle();
  VG_TRY(command_buffers_[slot].begin(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));

  // Dynamic rendering does not transition images; move it into the attachment
  // layout. UNDEFINED discards the previous (presented) contents, which is fine
  // since the render clears.
  VkImageMemoryBarrier to_color{};
  to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_color.srcAccessMask = 0;
  to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.image = swapchain_->image(image_index);
  to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &to_color);

  Frame frame;
  frame.cmd = cmd;
  frame.target = &swapchain_->render_target(image_index);
  frame.image_index = image_index;
  frame.slot = slot;
  return frame;
}

Status FrameLoop::end_frame(const Frame& frame) {
  const uint32_t slot = frame.slot;

  // Transition the rendered image to PRESENT_SRC.
  VkImageMemoryBarrier to_present{};
  to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_present.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_present.dstAccessMask = 0;
  to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_present.image = swapchain_->image(frame.image_index);
  to_present.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(frame.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_present);

  VG_TRY(command_buffers_[slot].end());

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
  VG_VK_TRY(vkQueueSubmit(device_->graphics_queue(), 1, &submit,
                          in_flight_[slot].handle()));

  Status present = swapchain_->present(frame.image_index, signal_sem);
  // Advance regardless: the work was submitted and the fence will signal, so
  // the slot is reusable next round even when present reports out-of-date.
  current_slot_ =
      static_cast<uint32_t>((current_slot_ + 1) % in_flight_.size());
  return present;
}

}  // namespace volumetric_kit::gfx::windowing

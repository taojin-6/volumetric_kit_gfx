// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_loop.hpp
/// @brief The frames-in-flight render loop: per-frame command buffers + sync
///        that drive a @ref Swapchain's acquire → render → present cycle.

#include <cstdint>
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
/// before reuse). @ref begin_frame waits the slot, acquires an image, and
/// transitions it for rendering; @ref end_frame transitions it for
/// presentation, submits, and presents. Both surface `VK_ERROR_OUT_OF_DATE_KHR`
/// so the caller can
/// @ref Swapchain::recreate.
///
/// @warning The @p device and @p swapchain passed to @ref create must outlive
///          the loop (it borrows both). Idle the device (or drain the loop)
///          before destroying it while frames are in flight.
///
/// @code
/// auto loop = windowing::FrameLoop::create(device, swapchain);
/// while (running) {
///   auto frame = loop.value().begin_frame();
///   if (!frame) { swapchain.recreate(window_extent()); continue; }
///   frame.value().target->begin(frame.value().cmd, clear);
///   // ... bind pipeline, set viewport/scissor, draw ...
///   frame.value().target->end(frame.value().cmd);
///   if (!loop.value().end_frame(frame.value())) swapchain.recreate(...);
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

  ~FrameLoop() = default;
  FrameLoop(FrameLoop&&) noexcept = default;
  FrameLoop& operator=(FrameLoop&&) noexcept = default;
  FrameLoop(const FrameLoop&) = delete;
  FrameLoop& operator=(const FrameLoop&) = delete;

  /// @brief Begin the next frame: wait this slot's fence, acquire an image,
  ///        begin its command buffer, and transition the image into
  ///        `COLOR_ATTACHMENT_OPTIMAL`.
  /// @return The @ref Frame to record into; a non-OK @ref Status carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` (recreate the swapchain and retry) or
  ///         another failed `VkResult`.
  Result<Frame> begin_frame();

  /// @brief End the frame from @ref begin_frame: transition the image to
  ///        `PRESENT_SRC`, end + submit its command buffer, present it, and
  ///        advance to the next slot.
  /// @param frame  The frame returned by @ref begin_frame this iteration.
  /// @return OK on success; a non-OK @ref Status carrying
  ///         `VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR` (recreate the
  ///         swapchain) or another failed `VkResult`.
  Status end_frame(const Frame& frame);

  /// @return The CPU-ahead depth (number of in-flight slots).
  uint32_t frames_in_flight() const noexcept {
    return static_cast<uint32_t>(in_flight_.size());
  }

  /// @return `true` if this owns frame resources.
  bool valid() const noexcept { return !in_flight_.empty(); }

 private:
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
  // Derived state (frames_in_flight() / valid()) reads in_flight_, which
  // empties on move, so the defaulted move pair needs no scalar reset.
  // current_slot_ is a ring position, meaningless (but harmless) on a
  // moved-from loop.
  uint32_t current_slot_ = 0;
};

}  // namespace windowing
}  // namespace volumetric_kit::gfx

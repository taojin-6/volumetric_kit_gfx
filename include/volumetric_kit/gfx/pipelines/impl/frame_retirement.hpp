// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pipelines/impl/frame_retirement.hpp
/// The frame numbers an owner of per-frame resources has been given, and the
/// release of those resources on the frame loop's timeline. Shared by
/// @ref StreamedAtlas and @ref DepthCloudPipeline; included from their
/// headers, not a standalone type.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"

namespace volumetric_kit::gfx::pipelines::detail {

/// @brief Frame numbers on a timeline, given in order, and a @ref RetireQueue
///        on that timeline.
///
/// An owner checks each frame number it is given (@ref check), keeps the
/// newest (@ref advance) and pushes what a frame used to @ref retire. At
/// teardown, @ref reset waits for the newest frame -- or, when that frame
/// never reached a queue, for the device to go idle, since nothing will set
/// its number -- and then runs every deleter, or keeps them all when it
/// cannot tell that the frames completed. Whether a frame reached a queue is
/// read from the core's record of submitted timeline values.
class FrameRetirement {
 public:
  /// @brief Construct an empty one (`valid()` is false).
  FrameRetirement() = default;

  /// @brief Watch @p timeline for frames on @p device.
  /// @param device    The device; borrowed, it must outlive this.
  /// @param timeline  The frame loop's timeline; borrowed by address.
  /// @param owner     Names the owner in teardown's warning; a literal.
  /// @throws std::bad_alloc when the queue cannot be allocated.
  FrameRetirement(const core::Device& device,
                  const core::TimelineSemaphore& timeline, const char* owner)
      : device_(&device),
        timeline_(&timeline),
        owner_(owner),
        retire_(std::make_unique<RetireQueue>(timeline)) {}

  ~FrameRetirement() { reset(); }

  FrameRetirement(FrameRetirement&& other) noexcept { adopt(other); }
  FrameRetirement& operator=(FrameRetirement&& other) noexcept {
    if (this != &other) {
      reset();
      adopt(other);
    }
    return *this;
  }
  FrameRetirement(const FrameRetirement&) = delete;
  FrameRetirement& operator=(const FrameRetirement&) = delete;

  /// @return Whether this watches a timeline.
  bool valid() const noexcept { return timeline_ != nullptr; }
  /// @return The timeline. @pre `valid()`.
  const core::TimelineSemaphore& timeline() const noexcept {
    return *timeline_;
  }
  /// @return The queue whose deleters run once their frames complete.
  ///         @pre `valid()`.
  RetireQueue& retire() noexcept { return *retire_; }
  /// @return The newest frame number given (0 for none).
  std::uint64_t newest() const noexcept { return newest_; }

  /// @brief Check a frame number before using it.
  /// @param frame  The number.
  /// @param call   Names the caller in the message.
  /// @return OK; or `core::Status::Code::InvalidArgument` for 0, or for a
  ///         number below the newest given.
  core::Status check(std::uint64_t frame, const std::string& call) const {
    if (frame == 0) {
      return core::Status::invalid_argument(call +
                                            ": frame numbers start at 1");
    }
    if (frame < newest_) {
      return core::Status::invalid_argument(
          call + ": frame " + std::to_string(frame) + " is below frame " +
          std::to_string(newest_) + ", already given");
    }
    return core::Status{};
  }

  /// @brief Make @p frame the newest given, if it is newer.
  /// @param frame  A number @ref check accepted.
  /// @return Whether it is newer than the newest given before.
  bool advance(std::uint64_t frame) noexcept {
    if (frame <= newest_) {
      return false;
    }
    newest_ = frame;
    return true;
  }

  /// @brief Whether a wait for @p frame returns: the timeline has reached it,
  ///        or a submission that sets it has reached a queue.
  /// @param frame  The number.
  /// @param call   Names the caller in the message.
  /// @return OK; `core::Status::Code::InvalidArgument` when nothing that
  ///         sets it has been submitted; or a backend failure reading the
  ///         timeline. @pre `valid()`.
  core::Status check_submitted(std::uint64_t frame, const char* call) const {
    const std::vector<core::TimelinePoint> point{{timeline_, frame}};
    // Validate the device and semaphore first. Reached is a presentation
    // rule: it cannot distinguish an unfinished submitted frame from a
    // dropped one.
    VKC_TRY(core::check_timeline_points(*device_, point, {}, call));
    // For one valid signal with no waits, InvalidArgument means only that its
    // value is at or below the counter or a recorded signal. This queries the
    // core's submitted-value bound without submitting or recording anything.
    const core::Status signal =
        core::check_timeline_points(*device_, {}, point, call);
    if (signal.domain() == core::Status::Code::InvalidArgument) return {};
    VKC_TRY(signal);
    return core::Status::invalid_argument(std::string(call) + ": frame " +
                                          std::to_string(frame) +
                                          " has not been submitted");
  }

  /// @brief Wait for the newest frame given, run every deleter, and empty
  ///        this.
  ///
  /// A frame that never reached a queue -- it failed before its submit --
  /// will never set its number, so then this waits for the device's queues
  /// instead: every frame that did reach them has completed, and the one that
  /// did not never runs. The deleters run once the wait succeeds, or when it
  /// reports device loss, as nothing will run again. Any other failure is
  /// logged and leaves completion unknown, so the queue is kept whole -- its
  /// deleters neither run nor are destroyed -- rather than freed while the
  /// GPU may still use what they hold (an intentional leak, as
  /// @ref RetireQueue's destruction does). It is not drained either: a frame
  /// that never reached a queue would block that wait forever.
  void reset() noexcept {
    const bool complete =
        timeline_ == nullptr || newest_ == 0 || wait_for_newest();
    if (complete) {
      // The frames that used the queued resources have completed, never ran,
      // or never will.
      if (retire_) retire_->reclaim();
      retire_.reset();
    } else {
      // TODO: the owners' own members -- StreamedAtlas's images,
      // DepthCloudPipeline's blocks and pipeline -- are still destroyed after
      // this, whatever the wait returned.
      (void)retire_.release();
    }
    device_ = nullptr;
    timeline_ = nullptr;
    owner_ = "";
    newest_ = 0;
  }

 private:
  // Waits for the newest frame, or for the device's queues when it never
  // reached one. Returns whether its resources may be freed: the wait
  // succeeded or lost the device. A throw, while reading the record or
  // writing the warning, leaves completion unknown.
  bool wait_for_newest() const noexcept try {
    const core::Status waited = check_submitted(newest_, owner_).ok()
                                    ? timeline_->wait(newest_)
                                    : device_->wait_idle();
    if (waited.ok()) return true;
    const bool lost = core::vk_result(waited) == VK_ERROR_DEVICE_LOST;
    log_message(core::LogLevel::Warning,
                std::string(owner_) + ": waiting for frame " +
                    std::to_string(newest_) + " failed (" + waited.message() +
                    (lost ? "); the device is lost, freeing its resources"
                          : "); its completion is unknown, keeping its "
                            "resources"));
    return lost;
  } catch (...) {
    return false;
  }

  // Takes `other`'s state, leaving it empty; this is empty before.
  void adopt(FrameRetirement& other) noexcept {
    device_ = std::exchange(other.device_, nullptr);
    timeline_ = std::exchange(other.timeline_, nullptr);
    owner_ = std::exchange(other.owner_, "");
    retire_ = std::move(other.retire_);
    newest_ = std::exchange(other.newest_, 0);
  }

  const core::Device* device_ = nullptr;               // borrowed
  const core::TimelineSemaphore* timeline_ = nullptr;  // borrowed
  const char* owner_ = "";
  // On the heap, so a teardown that cannot tell its frames completed keeps
  // the whole queue with release(), allocating nothing.
  std::unique_ptr<RetireQueue> retire_;
  std::uint64_t newest_ = 0;
};

}  // namespace volumetric_kit::gfx::pipelines::detail

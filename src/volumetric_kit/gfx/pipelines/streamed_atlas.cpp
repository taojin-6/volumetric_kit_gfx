// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/streamed_atlas.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/format.hpp"
#include "volumetric_kit/gfx/core/image_update.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"

namespace volumetric_kit::gfx::pipelines {

namespace {

// An update copies into an image whose last frame has completed -- the host
// has seen it on the timeline -- so the copy waits for no earlier read; the
// pipeline samples the picture in the fragment stage.
constexpr ImageUpdateScope kUpdateScope{0,
                                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};

}  // namespace

core::Result<StreamedAtlas> StreamedAtlas::create(
    const HybridMeshPipeline& pipeline, core::Allocator& allocator,
    const core::TimelineSemaphore& timeline, const StreamedAtlasDesc& desc) {
  if (!pipeline.valid()) {
    return core::Status::invalid_argument(
        "StreamedAtlas::create: empty pipeline");
  }
  if (!timeline.valid()) {
    return core::Status::invalid_argument(
        "StreamedAtlas::create: empty timeline");
  }
  if (desc.extent.width == 0 || desc.extent.height == 0 || desc.slots == 0) {
    return core::Status::invalid_argument(
        "StreamedAtlas::create: the extent and slot count must be non-zero");
  }
  const core::Device& device = *pipeline.device_;
  const std::uint32_t max_dim = device.caps().limits().maxImageDimension2D;
  if (desc.extent.width > max_dim || desc.extent.height > max_dim) {
    return core::Status::invalid_argument(
        "StreamedAtlas::create: extent exceeds the device's "
        "maxImageDimension2D");
  }
  if (core::texel_bytes(desc.format) == 0 ||
      core::format_needs_ycbcr_conversion(desc.format)) {
    return core::Status::unsupported(
        "StreamedAtlas::create: the format is not an uncompressed "
        "single-plane color format");
  }
  constexpr VkFormatFeatureFlags kFeatures =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
  if (!device.caps().format_supports(desc.format, VK_IMAGE_TILING_OPTIMAL,
                                     kFeatures)) {
    return core::Status::unsupported(
        "StreamedAtlas::create: the device cannot sample the format with "
        "linear filtering, or copy into it");
  }

  const VkDevice vk = device.handle();
  const VkDescriptorPoolSize pool_size{
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, desc.slots};
  VKC_ASSIGN(core::DescriptorPool pool,
             core::DescriptorPool::create(vk, &pool_size, 1, desc.slots));

  // TRANSFER_SRC keeps a picture copyable (a screenshot, a readback).
  core::ImageDesc image_desc;
  image_desc.extent = desc.extent;
  image_desc.format = desc.format;
  image_desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  std::vector<Slot> slots(desc.slots);
  for (Slot& slot : slots) {
    VKC_ASSIGN(slot.image, allocator.create_image(image_desc));
    VKC_ASSIGN(slot.set,
               pool.allocate(pipeline.pipeline_.descriptor_set_layout(0)));
    // Written once: an update rewrites the image, never the set.
    slot.set.write_combined_image_sampler(
        0, slot.image.view(), pipeline.sampler_->handle(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  StreamedAtlas atlas;
  atlas.device_ = &device;
  atlas.allocator_ = &allocator;
  atlas.desc_ = desc;
  atlas.pool_ = std::move(pool);
  atlas.slots_ = std::move(slots);
  atlas.retire_.emplace(timeline);
  atlas.timeline_ = &timeline;
  return atlas;
}

StreamedAtlas::~StreamedAtlas() { destroy(); }

StreamedAtlas::StreamedAtlas(StreamedAtlas&& other) noexcept
    : device_(other.device_),
      timeline_(other.timeline_),
      allocator_(other.allocator_),
      desc_(other.desc_),
      pool_(std::move(other.pool_)),
      slots_(std::move(other.slots_)),
      retire_(std::move(other.retire_)),
      current_(other.current_),
      newest_(other.newest_) {
  other.device_ = nullptr;
  other.timeline_ = nullptr;
  other.allocator_ = nullptr;
  other.desc_ = kNoDesc;
  other.slots_.clear();
  other.retire_.reset();
  other.current_ = kNoPicture;
  other.newest_ = 0;
}

StreamedAtlas& StreamedAtlas::operator=(StreamedAtlas&& other) noexcept {
  if (this != &other) {
    destroy();
    device_ = other.device_;
    timeline_ = other.timeline_;
    allocator_ = other.allocator_;
    desc_ = other.desc_;
    pool_ = std::move(other.pool_);
    slots_ = std::move(other.slots_);
    retire_ = std::move(other.retire_);
    current_ = other.current_;
    newest_ = other.newest_;
    other.device_ = nullptr;
    other.timeline_ = nullptr;
    other.allocator_ = nullptr;
    other.desc_ = kNoDesc;
    other.slots_.clear();
    other.retire_.reset();
    other.current_ = kNoPicture;
    other.newest_ = 0;
  }
  return *this;
}

core::Status StreamedAtlas::record_update(VkCommandBuffer cmd,
                                          std::uint64_t frame,
                                          const core::Buffer& source,
                                          const VkBufferImageCopy* regions,
                                          std::uint32_t region_count) {
  constexpr const char* kCall = "StreamedAtlas::record_update";
  VKC_ASSIGN(const std::uint32_t slot, take_slot(kCall, frame));
  VKC_TRY(record_image_update(cmd, source, slots_[slot].image, regions,
                              region_count, kUpdateScope)
              .with_context(kCall));
  publish(slot, frame);
  return core::Status{};
}

core::Status StreamedAtlas::record_upload(VkCommandBuffer cmd,
                                          std::uint64_t frame,
                                          const void* pixels,
                                          VkDeviceSize size) {
  constexpr const char* kCall = "StreamedAtlas::record_upload";
  VKC_ASSIGN(const std::uint32_t slot, take_slot(kCall, frame));
  retire_->poll();
  VKC_TRY(record_image_upload(cmd, *allocator_, *retire_, frame,
                              slots_[slot].image, pixels, size, kUpdateScope)
              .with_context(kCall));
  publish(slot, frame);
  return core::Status{};
}

VkDescriptorSet StreamedAtlas::use(std::uint64_t frame) {
  VKC_CHECK(frame != 0, "StreamedAtlas::use: frame numbers start at 1");
  if (current_ == kNoPicture) {
    return VK_NULL_HANDLE;
  }
  // Called once a frame, so the last frames' staging buffers go promptly.
  retire_->poll();
  Slot& slot = slots_[current_];
  slot.last_use = std::max(slot.last_use, frame);
  newest_ = std::max(newest_, frame);
  return slot.set.handle();
}

const core::Image* StreamedAtlas::picture() const noexcept {
  return current_ == kNoPicture ? nullptr : &slots_[current_].image;
}

core::Result<std::uint32_t> StreamedAtlas::take_slot(const char* call,
                                                     std::uint64_t frame) {
  const std::string name = call;
  if (!valid()) {
    return core::Status::invalid_argument(name + ": empty atlas");
  }
  if (frame == 0) {
    return core::Status::invalid_argument(name + ": frame numbers start at 1");
  }
  if (frame < newest_) {
    return core::Status::invalid_argument(
        name + ": frame " + std::to_string(frame) + " is below frame " +
        std::to_string(newest_) + ", already given");
  }

  // The least recently used image, the current picture last among equals.
  const auto later = [this](std::uint32_t a, std::uint32_t b) {
    const Slot& sa = slots_[a];
    const Slot& sb = slots_[b];
    if (sa.last_use != sb.last_use) {
      return sa.last_use > sb.last_use;
    }
    return a == current_ && b != current_;
  };
  std::uint32_t pick = 0;
  for (std::uint32_t i = 1; i < slot_count(); ++i) {
    if (later(pick, i)) {
      pick = i;
    }
  }

  const std::uint64_t last = slots_[pick].last_use;
  if (last == 0) {
    return pick;  // never written or drawn
  }
  VKC_ASSIGN(const std::uint64_t completed, timeline_->value());
  if (last <= completed) {
    return pick;
  }
  if (last >= frame) {
    return core::Status::invalid_argument(
        name + ": every image is in use by frame " + std::to_string(frame) +
        "; deepen the ring (StreamedAtlasDesc::slots)");
  }
  // An earlier frame still uses it: wait for it, as the frame loop waits for
  // a slot's last frame -- once it is known to be submitted, so that the wait
  // returns.
  VKC_TRY(check_submitted(last, call));
  VKC_TRY(timeline_->wait(last));
  return pick;
}

core::Status StreamedAtlas::check_submitted(std::uint64_t frame,
                                            const char* call) const {
  return core::check_timeline_points(*device_, {{timeline_, frame}}, {}, call,
                                     core::TimelineWaits::Submitted);
}

void StreamedAtlas::publish(std::uint32_t slot, std::uint64_t frame) {
  slots_[slot].last_use = frame;
  current_ = slot;
  newest_ = std::max(newest_, frame);
}

void StreamedAtlas::destroy() noexcept {
  if (timeline_ != nullptr && newest_ != 0) {
    // Wait for the newest frame that used the atlas. One that never reached
    // the queue -- it failed before its submit -- will never set its number,
    // so wait for the renderer's queues instead: every frame that did reach
    // them then has completed, and the one that did not never runs.
    const core::Status waited = check_submitted(newest_, "StreamedAtlas").ok()
                                    ? timeline_->wait(newest_)
                                    : device_->wait_idle();
    if (!waited.ok()) {
      // Device loss: nothing will run again, so free regardless, but say so.
      log_message(core::LogLevel::Warning,
                  "StreamedAtlas: waiting for frame " +
                      std::to_string(newest_) + " failed (" + waited.message() +
                      "); freeing the images regardless");
    }
  }
  // The frames that read the staging buffers have completed, or never run.
  if (retire_) {
    retire_->reclaim();
  }
  retire_.reset();
  slots_.clear();
  pool_ = {};
  device_ = nullptr;
  timeline_ = nullptr;
  allocator_ = nullptr;
  desc_ = kNoDesc;
  current_ = kNoPicture;
  newest_ = 0;
}

}  // namespace volumetric_kit::gfx::pipelines

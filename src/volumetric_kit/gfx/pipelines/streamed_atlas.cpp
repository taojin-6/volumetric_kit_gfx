// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/streamed_atlas.hpp"

#include <string>
#include <utility>
#include <vector>

#include "hybrid_mesh_set0.hpp"
#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/format.hpp"
#include "volumetric_kit/gfx/core/image_update.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"

namespace volumetric_kit::gfx::pipelines {

namespace {

// The pipeline samples a picture in the fragment stage. An update copies into
// an image whose last frame the host has seen complete, yet its transition
// still waits for that stage, so the queue itself orders the earlier frames'
// reads before the copy.
constexpr ImageUpdateScope kUpdateScope{VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
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
  if (desc.extent.width == 0 || desc.extent.height == 0 || desc.slots < 2) {
    return core::Status::invalid_argument(
        "StreamedAtlas::create: the extent must be non-zero and at least "
        "two slots are needed to preserve the picture for discard");
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
  const auto pool_sizes = detail::set0_pool_sizes(desc.slots);
  VKC_ASSIGN(core::DescriptorPool pool,
             core::DescriptorPool::create(
                 vk, pool_sizes.data(),
                 static_cast<std::uint32_t>(pool_sizes.size()), desc.slots));

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
    pipeline.write_rgba_set(slot.set, slot.image.view());
  }

  StreamedAtlas atlas;
  atlas.allocator_ = &allocator;
  atlas.desc_ = desc;
  atlas.pool_ = std::move(pool);
  atlas.slots_ = std::move(slots);
  atlas.frames_ = detail::FrameRetirement(device, timeline, "StreamedAtlas");
  return atlas;
}

StreamedAtlas::~StreamedAtlas() { destroy(); }

StreamedAtlas::StreamedAtlas(StreamedAtlas&& other) noexcept
    : allocator_(other.allocator_),
      desc_(other.desc_),
      pool_(std::move(other.pool_)),
      slots_(std::move(other.slots_)),
      frames_(std::move(other.frames_)),
      current_(other.current_),
      undo_(std::move(other.undo_)) {
  other.allocator_ = nullptr;
  other.desc_ = kNoDesc;
  other.slots_.clear();
  other.current_ = kNoPicture;
  other.undo_.clear();
}

StreamedAtlas& StreamedAtlas::operator=(StreamedAtlas&& other) noexcept {
  if (this != &other) {
    destroy();
    allocator_ = other.allocator_;
    desc_ = other.desc_;
    pool_ = std::move(other.pool_);
    slots_ = std::move(other.slots_);
    frames_ = std::move(other.frames_);
    current_ = other.current_;
    undo_ = std::move(other.undo_);
    other.allocator_ = nullptr;
    other.desc_ = kNoDesc;
    other.slots_.clear();
    other.current_ = kNoPicture;
    other.undo_.clear();
  }
  return *this;
}

core::Status StreamedAtlas::record_update(VkCommandBuffer cmd,
                                          std::uint64_t frame,
                                          const core::Buffer& source,
                                          const VkBufferImageCopy* regions,
                                          std::uint32_t region_count) {
  std::vector<ImageCopy> copies;
  if (regions != nullptr) {
    copies.reserve(region_count);
    for (std::uint32_t i = 0; i < region_count; ++i) {
      copies.push_back({&source, regions[i]});
    }
  }
  return record_update(cmd, frame, copies.data(),
                       static_cast<std::uint32_t>(copies.size()));
}

core::Status StreamedAtlas::record_update(VkCommandBuffer cmd,
                                          std::uint64_t frame,
                                          const ImageCopy* copies,
                                          std::uint32_t copy_count) {
  constexpr const char* kCall = "StreamedAtlas::record_update";
  VKC_ASSIGN(const std::uint32_t slot, take_slot(kCall, frame));
  const Undo undo = before_update(slot);
  VKC_TRY(record_image_update(cmd, copies, copy_count, slots_[slot].image,
                              kUpdateScope)
              .with_context(kCall));
  publish(slot, frame, undo);
  return core::Status{};
}

core::Status StreamedAtlas::record_upload(VkCommandBuffer cmd,
                                          std::uint64_t frame,
                                          const void* pixels,
                                          VkDeviceSize size) {
  constexpr const char* kCall = "StreamedAtlas::record_upload";
  VKC_ASSIGN(const std::uint32_t slot, take_slot(kCall, frame));
  const Undo undo = before_update(slot);
  frames_.retire().poll();
  VKC_TRY(record_image_upload(cmd, *allocator_, frames_.retire(), frame,
                              slots_[slot].image, pixels, size, kUpdateScope)
              .with_context(kCall));
  publish(slot, frame, undo);
  return core::Status{};
}

VkDescriptorSet StreamedAtlas::use(std::uint64_t frame) {
  VKC_CHECK(frame != 0, "StreamedAtlas::use: frame numbers start at 1");
  if (current_ == kNoPicture) {
    return VK_NULL_HANDLE;
  }
  // Called once a frame, so the last frames' staging buffers go promptly.
  frames_.retire().poll();
  Slot& slot = slots_[current_];
  if (slot.last_use < frame) {
    if (frame >= frames_.newest()) {
      enter(frame);
      Undo undo;
      undo.slot = current_;
      undo.last_use = slot.last_use;
      undo_.push_back(undo);
    }
    slot.last_use = frame;
  }
  return slot.set.handle();
}

void StreamedAtlas::discard(std::uint64_t frame) {
  VKC_CHECK(frame != 0, "StreamedAtlas::discard: frame numbers start at 1");
  VKC_CHECK(frame >= frames_.newest(),
            "StreamedAtlas::discard: a later frame has been given to the "
            "atlas; discard a frame before giving it the next");
  if (frame != frames_.newest() || undo_.empty()) {
    return;  // the frame gave the atlas nothing
  }
  // A frame that reached a queue -- or a submission in its place -- sets its
  // number, and its commands may have run, so the images it used stay its
  // until the timeline reaches it. One that did not never runs.
  const bool reached =
      frames_.check_submitted(frame, "StreamedAtlas::discard").ok();
  for (auto undo = undo_.rbegin(); undo != undo_.rend(); ++undo) {
    Slot& slot = slots_[undo->slot];
    if (!reached) {
      slot.last_use = undo->last_use;
    }
    if (undo->update) {
      current_ = undo->current;
      slot.image.set_layout(undo->layout);
    }
  }
  undo_.clear();
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
  VKC_TRY(frames_.check(frame, name));

  // Keep the picture from before this frame intact, even if the frame's
  // copies run before its present fails. Restoring an index in discard()
  // cannot restore overwritten texels. Uses before the first update also
  // have undo entries, so find the first update rather than the first entry.
  std::uint32_t preserved = current_;
  if (frame == frames_.newest()) {
    for (const Undo& undo : undo_) {
      if (undo.update) {
        preserved = undo.current;
        break;
      }
    }
  }

  // The least recently used writable image, the current picture last among
  // equals. Creation guarantees at least two slots, so excluding the
  // preserved picture always leaves a candidate.
  const auto later = [this](std::uint32_t a, std::uint32_t b) {
    const Slot& sa = slots_[a];
    const Slot& sb = slots_[b];
    if (sa.last_use != sb.last_use) {
      return sa.last_use > sb.last_use;
    }
    return a == current_ && b != current_;
  };
  std::uint32_t pick = kNoPicture;
  for (std::uint32_t i = 0; i < slot_count(); ++i) {
    if (i != preserved && (pick == kNoPicture || later(pick, i))) {
      pick = i;
    }
  }

  const std::uint64_t last = slots_[pick].last_use;
  if (last == 0) {
    return pick;  // never written or drawn
  }
  VKC_ASSIGN(const std::uint64_t completed, frames_.timeline().value());
  if (last <= completed) {
    return pick;
  }
  if (last >= frame) {
    return core::Status::invalid_argument(
        name + ": every image available for updates is in use by frame " +
        std::to_string(frame) + "; deepen the ring (StreamedAtlasDesc::slots)");
  }
  // An earlier frame still uses it: wait for it, as the frame loop waits for
  // a slot's last frame -- once it is known to be submitted, so that the wait
  // returns.
  VKC_TRY(frames_.check_submitted(last, call));
  VKC_TRY(frames_.timeline().wait(last));
  return pick;
}

StreamedAtlas::Undo StreamedAtlas::before_update(std::uint32_t slot) const {
  Undo undo;
  undo.slot = slot;
  undo.last_use = slots_[slot].last_use;
  undo.update = true;
  undo.current = current_;
  undo.layout = slots_[slot].image.layout();
  return undo;
}

void StreamedAtlas::publish(std::uint32_t slot, std::uint64_t frame,
                            Undo undo) {
  enter(frame);
  undo_.push_back(undo);
  slots_[slot].last_use = frame;
  current_ = slot;
}

void StreamedAtlas::enter(std::uint64_t frame) {
  if (frames_.advance(frame)) {
    undo_.clear();
  }
}

void StreamedAtlas::destroy() noexcept {
  // Waits for the newest frame that used the atlas, or for the queues when
  // it never reached one, and frees the staging buffers; then the images.
  frames_.reset();
  slots_.clear();
  pool_ = {};
  allocator_ = nullptr;
  desc_ = kNoDesc;
  current_ = kNoPicture;
  undo_.clear();
}

}  // namespace volumetric_kit::gfx::pipelines

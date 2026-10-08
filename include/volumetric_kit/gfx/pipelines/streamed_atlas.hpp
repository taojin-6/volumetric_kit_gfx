// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file streamed_atlas.hpp
/// @brief A @ref HybridMeshPipeline atlas that changes while frames are drawn:
///        a ring of images, each reused once the frames that drew it have
///        completed, updated by copies recorded into the frame.

#include <cstdint>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::core {
class Allocator;
class Buffer;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

class HybridMeshPipeline;

/// @brief What a @ref StreamedAtlas holds: its pictures' size and format, and
///        how many it keeps.
struct StreamedAtlasDesc {
  /// Every picture's size in texels; non-zero.
  VkExtent2D extent{};
  /// The pictures' format: uncompressed, single-plane color, which the device
  /// can sample with linear filtering and copy into. `_SRGB` for 8-bit camera
  /// color, so the sampler filters in linear light.
  VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
  /// The ring's depth; non-zero. With one update a frame, the frame loop's
  /// frames in flight plus one never waits (see @ref StreamedAtlas).
  std::uint32_t slots = 3;
};

/// @brief The atlas a live @ref HybridMeshPipeline mesh samples: a ring of
///        images, each with its descriptor set, of which the newest update is
///        the current picture.
///
/// Frames are numbered on a timeline -- `windowing::FrameLoop::timeline`, and
/// each `windowing::Frame::number` -- and the atlas keys everything on those
/// numbers. Each frame that draws the mesh binds @ref use, which marks the
/// current picture's image as used by that frame. An update
/// (@ref record_update from device rows, @ref record_upload from host pixels)
/// is recorded into a frame's command buffer, outside its rendering scope, and
/// copies into the image no unfinished frame uses -- the least recently used,
/// once the timeline has reached its last frame -- which becomes the current
/// picture. A picture a frame in flight draws is therefore never overwritten,
/// so the copy waits on the GPU for nothing earlier. If every image is still in
/// use by an earlier frame, the update waits on the host for the oldest of
/// them; with one update a frame and at least the frame loop's frames in
/// flight plus one images, it never does. Whether a frame's number has been
/// submitted is read from the core's record of submitted timeline values,
/// which `windowing::FrameLoop` adds each frame to (a caller submitting frames
/// itself adds them with `core::note_timeline_signals`).
///
/// Before its first update the atlas has no picture: @ref use returns
/// `VK_NULL_HANDLE`, and the pipeline draws in vertex color. The descriptor
/// sets are made once, so an atlas updated every frame allocates nothing per
/// frame but a host upload's staging buffer, which an internal
/// @ref RetireQueue frees once the frame completes.
///
/// A frame that never reaches the queue is given back with @ref discard;
/// otherwise the next frames bind a picture whose update never ran.
///
/// A mesh's `uv0` index into one picture, so commit a mesh and its picture in
/// the same frame: record the update in the frame that first draws the mesh.
/// An atlas is used from one thread at a time.
///
/// @warning The @p pipeline, @p allocator and @p timeline passed to
///          @ref create must outlive the atlas, as must the device. Destroying
///          (or move-assigning over) an atlas waits until the timeline reaches
///          the newest frame number given to it, or -- when that frame never
///          reached the queue -- until the renderer's queues drain
///          (`core::Device::wait_idle`).
///
/// @code
/// pipelines::StreamedAtlasDesc desc;
/// desc.extent = {1920, 1080};
/// VKC_ASSIGN(pipelines::StreamedAtlas atlas,
///            pipelines::StreamedAtlas::create(pipeline, allocator,
///                                             loop.timeline(), desc));
/// // per frame, before the rendering scope, when a new mesh arrives:
/// VKC_TRY(atlas.record_upload(f.cmd, f.number, bytes.data(), bytes.size()));
/// // inside it:
/// frame.atlas = atlas.use(f.number);
/// pipeline.submit(f.cmd, frame);
/// // after it:
/// core::Status end = loop.end_frame(f);
/// if (!end.ok() && !windowing::swapchain_stale(end)) {
///   atlas.discard(f.number);  // the frame may not have reached the queue
/// }
/// @endcode
class VG_PIPELINES_API StreamedAtlas {
 public:
  /// @brief Construct an empty atlas (owns nothing; `valid()` is false).
  StreamedAtlas() = default;

  /// @brief Create an atlas of @p desc.slots images for @p pipeline.
  /// @param pipeline   The pipeline that draws with it; its sampler is
  ///                   written into the atlas's sets.
  /// @param allocator  Allocates the images now and each host upload's staging
  ///                   buffer later.
  /// @param timeline   The timeline frame numbers are set on, borrowed by
  ///                   address: `windowing::FrameLoop::timeline`.
  /// @param desc       The pictures' size and format, and the ring's depth.
  /// @return The atlas, holding no picture; or
  ///         `core::Status::Code::InvalidArgument` for an empty @p pipeline or
  ///         @p timeline, a zero extent or slot count, or an extent beyond the
  ///         device's `maxImageDimension2D`;
  ///         `core::Status::Code::Unsupported` for a format that is not an
  ///         uncompressed single-plane color format, or that the device
  ///         cannot sample with linear filtering and copy into; or a backend
  ///         failure.
  static core::Result<StreamedAtlas> create(
      const HybridMeshPipeline& pipeline, core::Allocator& allocator,
      const core::TimelineSemaphore& timeline, const StreamedAtlasDesc& desc);

  ~StreamedAtlas();
  StreamedAtlas(StreamedAtlas&& other) noexcept;
  StreamedAtlas& operator=(StreamedAtlas&& other) noexcept;
  StreamedAtlas(const StreamedAtlas&) = delete;
  StreamedAtlas& operator=(const StreamedAtlas&) = delete;

  /// @brief Record a copy of rows of @p source -- one region per tile, say --
  ///        into a free image, which becomes the current picture.
  ///
  /// The copy is recorded as `record_image_update` records it. Texels the
  /// regions do not cover are undefined, so a mesh's `uv0` must address only
  /// texels it wrote.
  /// @param cmd           Frame @p frame 's command buffer, recording,
  ///                      outside a rendering scope.
  /// @param frame         The frame's number; at least every number given
  ///                      before.
  /// @param source        As `record_image_update`; it must outlive frame
  ///                      @p frame 's execution.
  /// @param regions       As `record_image_update`.
  /// @param region_count  The number of @p regions.
  /// @return OK once recorded; `core::Status::Code::InvalidArgument` for an
  ///         empty atlas, a zero @p frame or one below a number already
  ///         given, every image in use by frame @p frame already (too many
  ///         updates in one frame for the ring) or by an earlier frame not
  ///         yet submitted, or what `record_image_update` refuses; or a
  ///         backend failure reading or waiting for the timeline. On an error
  ///         the current picture is unchanged.
  core::Status record_update(VkCommandBuffer cmd, std::uint64_t frame,
                             const core::Buffer& source,
                             const VkBufferImageCopy* regions,
                             std::uint32_t region_count);

  /// @brief Record an upload of host @p pixels -- a whole picture, tightly
  ///        packed -- into a free image, which becomes the current picture.
  ///
  /// The pixels are copied into a staging buffer now, so @p pixels may be
  /// reused when this returns; the staging buffer is freed once frame
  /// @p frame completes.
  /// @param cmd     As @ref record_update.
  /// @param frame   As @ref record_update.
  /// @param pixels  `extent.width * extent.height` texels of the format.
  /// @param size    The byte length of @p pixels.
  /// @return As @ref record_update, plus `core::Status::Code::InvalidArgument`
  ///         for null @p pixels or a @p size other than a picture's, and the
  ///         allocator's failure.
  core::Status record_upload(VkCommandBuffer cmd, std::uint64_t frame,
                             const void* pixels, VkDeviceSize size);

  /// @brief The set frame @p frame binds as the atlas: the current
  ///        picture's, which is then kept until the timeline reaches
  ///        @p frame.
  /// @param frame  The number of the frame that draws with it; non-zero
  ///               (checked with `VKC_CHECK`).
  /// @return The current picture's set; `VK_NULL_HANDLE` before the first
  ///         update or for an empty atlas -- @ref HybridMeshPipeline::submit
  ///         then draws in vertex color.
  VkDescriptorSet use(std::uint64_t frame);

  /// @brief Undo what frame @p frame gave the atlas -- its updates and
  ///        @ref use -- for a frame whose commands may not have run: the
  ///        picture its first update replaced is current again.
  ///
  /// Call it when a frame fails to reach the queue: after
  /// `windowing::FrameLoop::end_frame` returns an error that
  /// `windowing::swapchain_stale` does not classify, or after dropping a frame
  /// you submit yourself; and before giving the atlas a later frame. Whether
  /// the frame, or a submission in its place, reached a queue is read from the
  /// core's record of submitted timeline values, as for an update's wait. If
  /// one did, the frame's commands may have run -- its present failed -- so
  /// the images it used stay in use until the timeline reaches @p frame;
  /// otherwise they are free at once.
  /// @param frame  The frame's number: the newest given to the atlas, or a
  ///               later one, which gave it nothing to undo (checked with
  ///               `VKC_CHECK`).
  void discard(std::uint64_t frame);

  /// @return The current picture -- its image, in `SHADER_READ_ONLY_OPTIMAL`
  ///         once its update has run -- or `nullptr` before the first update.
  const core::Image* picture() const noexcept;
  /// @return Whether an update has been recorded, so there is a picture.
  bool has_picture() const noexcept { return current_ != kNoPicture; }
  /// @return The pictures' size in texels (`{0, 0}` when empty).
  VkExtent2D extent() const noexcept { return desc_.extent; }
  /// @return The pictures' format (`VK_FORMAT_UNDEFINED` when empty).
  VkFormat format() const noexcept { return desc_.format; }
  /// @return The ring's depth (0 when empty).
  std::uint32_t slot_count() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }
  /// @return `true` if this owns an atlas.
  bool valid() const noexcept { return timeline_ != nullptr; }

 private:
  static constexpr std::uint32_t kNoPicture = UINT32_MAX;
  static constexpr StreamedAtlasDesc kNoDesc{{0, 0}, VK_FORMAT_UNDEFINED, 0};

  struct Slot {
    core::Image image;
    core::DescriptorSet set;
    // The newest frame that wrote or drew the image (0 for none): it is
    // reused once the timeline reaches this.
    std::uint64_t last_use = 0;
  };

  // One change the newest frame made, which discard() reverts: a slot's
  // last_use and, for an update, the picture it replaced and the image's
  // layout before it.
  struct Undo {
    std::uint32_t slot = 0;
    std::uint64_t last_use = 0;
    bool update = false;
    std::uint32_t current = kNoPicture;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  };

  // Checks `frame` for an update, and returns the slot it copies into: the
  // least recently used, the current picture last, once its last frame has
  // completed -- waiting for that frame when it is an earlier one.
  core::Result<std::uint32_t> take_slot(const char* call, std::uint64_t frame);
  // OK when the timeline has reached `frame` or a submission that sets it
  // has reached a queue, so that a wait for it returns.
  core::Status check_submitted(std::uint64_t frame, const char* call) const;
  // Makes `slot` the current picture, written by frame `frame`; `undo` holds
  // the slot's state from before the update.
  void publish(std::uint32_t slot, std::uint64_t frame, Undo undo);
  // The state an update of `slot` changes, as it is now.
  Undo before_update(std::uint32_t slot) const;
  // Makes `frame` the newest frame given, starting its undo log when it is
  // newer than the last. `frame` is at least newest_.
  void enter(std::uint64_t frame);
  // Waits for the newest frame given, then frees everything.
  void destroy() noexcept;

  const core::Device* device_ = nullptr;               // borrowed
  const core::TimelineSemaphore* timeline_ = nullptr;  // borrowed
  core::Allocator* allocator_ = nullptr;               // borrowed
  StreamedAtlasDesc desc_ = kNoDesc;
  core::DescriptorPool pool_;
  std::vector<Slot> slots_;
  // Host uploads' staging buffers, each freed once its frame completes.
  std::optional<RetireQueue> retire_;
  std::uint32_t current_ = kNoPicture;
  // The newest frame number given to the atlas (0 for none).
  std::uint64_t newest_ = 0;
  // What frame newest_ changed, oldest first.
  std::vector<Undo> undo_;
};

}  // namespace volumetric_kit::gfx::pipelines

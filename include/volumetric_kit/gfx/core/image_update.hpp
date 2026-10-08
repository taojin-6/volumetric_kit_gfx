// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_update.hpp
/// @brief Rewrite a sampled image inside a frame: copies recorded into the
///        frame's own command buffer, between the layout transitions that
///        order them, from rows of device buffers (@ref record_image_update)
///        or from host pixels staged through a buffer a @ref RetireQueue
///        frees (@ref record_image_upload).

#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/export.hpp"

namespace volumetric_kit::core {
class Allocator;
class Buffer;
class Image;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx {

class RetireQueue;

/// @brief The shader stages an image update is ordered against: the image's
///        readers before the update and after it.
///
/// Both masks are non-empty and hold shader stages only -- `VERTEX_SHADER`,
/// `FRAGMENT_SHADER` and `COMPUTE_SHADER` -- ones the update's queue runs.
struct ImageUpdateScope {
  /// The stages that read the image earlier on the update's queue: the
  /// transition that discards its contents waits for them. It waits even
  /// when the host has seen those reads complete -- a ring slot reused by
  /// frame number (@ref pipelines::StreamedAtlas) -- so that the queue
  /// itself orders them before the copy. Reads on another queue are the
  /// caller's to order.
  VkPipelineStageFlags src_stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  /// The stages that read the image after the update; non-zero. The copy is
  /// made visible to them.
  VkPipelineStageFlags dst_stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
};

/// @brief Record copies of rows of @p source into @p image into @p cmd,
///        between the transitions that order them against the image's
///        readers.
///
/// Records a barrier that discards the image's contents (from `UNDEFINED`),
/// the copies, and a barrier into `SHADER_READ_ONLY_OPTIMAL` for
/// `scope.dst_stages`, which `image.layout()` then records. Texels the regions
/// do not cover are undefined afterwards. Submits nothing: draws recorded
/// after it -- in @p cmd or a later submission on the same queue -- see the
/// new contents. On an error nothing is recorded and the image is unchanged.
///
/// @p source 's writes must be visible to the copy: written by the host before
/// the submission (the core's mapped memory is coherent), recorded earlier on
/// the same queue behind the writer's own barrier, or made on another queue
/// and waited for -- under the frame loop, by a `windowing::Frame::waits`
/// entry at the `TRANSFER` stage.
///
/// @param cmd           A command buffer in the recording state, outside a
///                      rendering scope, for a queue that runs
///                      @p scope 's stages.
/// @param source        A buffer with `TRANSFER_SRC` usage holding the rows;
///                      it must outlive the update's execution.
/// @param image         The image: single-sample, with `TRANSFER_DST` and
///                      `SAMPLED` usage and an uncompressed single-plane
///                      color format, in the layout its `layout()` records.
/// @param regions       The copies: color aspect, within the image's levels,
///                      layers and extent, no two writing a texel in common
///                      (they run with no barrier between them), each
///                      `bufferOffset` a multiple of the texel size and each
///                      row length and image height 0 (tightly packed) or at
///                      least the region's.
/// @param region_count  The number of @p regions; non-zero.
/// @param scope         The stages that read the image before and after.
/// @return OK once recorded; `core::Status::Code::InvalidArgument` for a null
///         @p cmd, an empty @p source or @p image, a missing usage, a
///         multisampled image, no regions, a region outside the image or
///         reading past the end of @p source, two regions that write a
///         texel in common, a misaligned offset, a row
///         length or image height shorter than the region's, or a stage
///         mask that is empty or not shader stages;
///         `core::Status::Code::Unsupported` for a compressed, multi-planar
///         or depth/stencil format.
///
/// @code
/// VkBufferImageCopy tile{};  // one camera's tile of an atlas
/// tile.bufferRowLength = tile_width;
/// tile.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
/// tile.imageOffset = {x, y, 0};
/// tile.imageExtent = {tile_width, tile_height, 1};
/// VKC_TRY(record_image_update(frame.cmd, colour, atlas, &tile, 1));
/// @endcode
VG_CORE_API core::Status record_image_update(
    VkCommandBuffer cmd, const core::Buffer& source, core::Image& image,
    const VkBufferImageCopy* regions, std::uint32_t region_count,
    const ImageUpdateScope& scope = {});

/// @brief One copy into an image from rows of a buffer of its own: the update
///        @ref record_image_update records from several buffers.
struct ImageCopy {
  /// The buffer the rows are in, as @ref record_image_update 's `source`;
  /// non-null.
  const core::Buffer* source = nullptr;
  /// Where they go, as one of @ref record_image_update 's `regions`.
  VkBufferImageCopy region{};
};

/// @brief Record an update of @p image from several buffers -- each camera's
///        tile from that camera's buffer, say -- as one update: the copies
///        between one pair of transitions.
///
/// As the one-buffer @ref record_image_update, except that each copy names
/// its own source, so one update fills regions from buffers that are not one
/// allocation. Every source's writes must be visible to the copy, as that one
/// says.
///
/// @param cmd         As @ref record_image_update.
/// @param copies      The copies, each a source and a region as
///                    @ref record_image_update takes them.
/// @param copy_count  The number of @p copies; non-zero.
/// @param image       As @ref record_image_update.
/// @param scope       As @ref record_image_update.
/// @return As @ref record_image_update, checking each copy's source against
///         its own region; `core::Status::Code::InvalidArgument` also for a
///         null source.
///
/// @code
/// std::vector<ImageCopy> tiles;  // one a camera
/// for (const Camera& c : cameras) tiles.push_back({&c.colour, c.tile});
/// VKC_TRY(record_image_update(frame.cmd, tiles.data(),
///                             static_cast<std::uint32_t>(tiles.size()),
///                             atlas));
/// @endcode
VG_CORE_API core::Status record_image_update(
    VkCommandBuffer cmd, const ImageCopy* copies, std::uint32_t copy_count,
    core::Image& image, const ImageUpdateScope& scope = {});

/// @brief Record an upload of host @p pixels into the whole of @p image 's
///        first level and layer, staged through a buffer that @p retire frees
///        once its timeline reaches @p retire_at.
///
/// The non-blocking counterpart of @ref upload_texture, for pictures that
/// change while frames are drawn: copies @p pixels into a new staging buffer
/// now, then records the copy as @ref record_image_update does. Nothing is
/// submitted or waited for. The other levels and layers are undefined
/// afterwards.
///
/// @param cmd        As @ref record_image_update.
/// @param allocator  Allocates the staging buffer.
/// @param retire     Frees the staging buffer; poll it once per frame.
/// @param retire_at  A value on @p retire 's timeline that is reached once
///                   @p cmd 's work completes: the frame's
///                   `windowing::Frame::number` for a queue on
///                   `windowing::FrameLoop::timeline`.
/// @param image      As @ref record_image_update.
/// @param pixels     Tightly packed rows of the first level:
///                   `width * height * depth` texels.
/// @param size       The byte length of @p pixels.
/// @param scope      The stages that read the image before and after.
/// @return OK once recorded and the staging buffer queued on @p retire;
///         `core::Status::Code::InvalidArgument` for null @p pixels, a
///         @p size other than the first level's, or what
///         @ref record_image_update refuses; or the allocator's failure.
///         On an error nothing is recorded or queued.
///
/// @code
/// RetireQueue retire(loop.timeline());
/// // per frame, before the rendering scope:
/// retire.poll();
/// VKC_TRY(record_image_upload(frame.cmd, allocator, retire, frame.number,
///                             keyframe, rgba.data(), rgba.size()));
/// @endcode
VG_CORE_API core::Status record_image_upload(
    VkCommandBuffer cmd, core::Allocator& allocator, RetireQueue& retire,
    std::uint64_t retire_at, core::Image& image, const void* pixels,
    VkDeviceSize size, const ImageUpdateScope& scope = {});

}  // namespace volumetric_kit::gfx

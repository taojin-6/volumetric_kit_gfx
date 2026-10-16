// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file depth_cloud_pipeline.hpp
/// @brief Points straight from depth buffers and pinhole cameras: one point
///        per depth pixel, read where the producer left it, coloured from a
///        colour camera's buffer or a tint.

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/mat4x4.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/camera/pinhole_camera.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/impl/frame_retirement.hpp"
#include "volumetric_kit/gfx/pipelines/impl/point_draw.hpp"
#include "volumetric_kit/gfx/pipelines/primitive_pipeline.hpp"

namespace volumetric_kit::core {
class Allocator;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

/// @brief A depth image in a buffer, and the camera that took it.
///
/// The buffer holds `camera.width * camera.height` 32-bit floats from
/// @ref offset: metres along the camera's +z, row-major, 0 where the sensor
/// had no return -- recon's `DeviceFrame::depth`, as it is. A depth that is
/// not positive and finite, or outside the bounds, draws nothing.
struct DepthSource {
  /// The depth: a buffer with `STORAGE_BUFFER` usage. Borrowed; see
  /// @ref DepthCloudPipeline for how long it must live and who may write it.
  const core::Buffer* buffer = nullptr;
  /// Where the image starts in @ref buffer, in bytes: a multiple of the
  /// device's `minStorageBufferOffsetAlignment`.
  VkDeviceSize offset = 0;
  /// The depth camera: positive, finite focal lengths, a finite principal
  /// point and a non-zero size.
  camera::PinholeCamera camera{};
  /// The depth camera's pose: its coordinates to world coordinates. Finite.
  glm::mat4 to_world{1.0f};
  /// Depths below this draw nothing; 0 for no bound.
  float min_depth = 0.0f;
  /// Depths above this draw nothing; 0 for no bound.
  float max_depth = 0.0f;
};

/// @brief A colour image in a buffer, and the camera that took it, which a
///        @ref DepthSource's points take their colour from.
///
/// The buffer holds `camera.width * camera.height` 32-bit words from
/// @ref offset, row-major: R', G' and B' sRGB-encoded in the low three bytes,
/// R lowest, as @ref pack_rgba8 packs them -- recon's `DeviceFrame::color`,
/// as it is.
struct ColorSource {
  /// The colour: a buffer with `STORAGE_BUFFER` usage, or null for none, when
  /// every point takes the draw's tint. Borrowed, as @ref DepthSource::buffer.
  const core::Buffer* buffer = nullptr;
  /// Where the image starts in @ref buffer, in bytes, aligned as
  /// @ref DepthSource::offset.
  VkDeviceSize offset = 0;
  /// The colour camera, checked as @ref DepthSource::camera when there is a
  /// buffer.
  camera::PinholeCamera camera{};
  /// The colour camera's pose: its coordinates to world coordinates. Finite
  /// and invertible when there is a buffer.
  glm::mat4 to_world{1.0f};
  /// The high byte of a word is the pixel's coverage: 0 where the colour
  /// camera saw nothing, so a point there takes the tint.
  bool alpha_is_coverage = false;
};

/// @brief How a @ref DepthCloudDraw colours its points.
enum class CloudColor : uint32_t {
  /// Every point takes the tint.
  Solid = 0,
  /// The tint scaled by the colour's luma, Y' = 0.2126 R' + 0.7152 G' +
  /// 0.0722 B' on the encoded values: each camera's points keep their tint
  /// and show the scene's shading.
  TintedLuma = 1,
  /// The colour as the colour camera recorded it.
  True = 2,
};

/// @brief One depth image drawn as points, through one view, into one
///        rectangle of the target.
///
/// Depth pixel `(u, v)` with depth `z` is the depth camera's point
/// `((u - cx) z / fx, (v - cy) z / fy, z)`, moved by `depth.to_world` and
/// then by @ref view_proj. With a colour buffer it takes the colour of the
/// colour camera's pixel it projects to: in front of that camera, inside
/// `[0, width) x [0, height)`, and the nearest pixel centre, the last half
/// pixel clamped, as recon's projection rules. A point with no colour -- no
/// buffer, behind the colour camera, outside its image, or on a pixel of
/// coverage 0 -- takes @ref tint, as does every point in
/// @ref CloudColor::Solid.
struct DepthCloudDraw {
  /// The depth image and its camera.
  DepthSource depth{};
  /// The colour image and its camera; no buffer for none.
  ColorSource color{};
  /// How points are coloured.
  CloudColor mode = CloudColor::Solid;
  /// The tint, an sRGB-encoded @ref pack_rgba8 word; alpha is ignored.
  uint32_t tint = pack_rgba8(255, 255, 255);
  /// Draw every stride-th pixel along each axis, from (0, 0); at least 1.
  uint32_t stride = 1;
  /// The rectangle drawn into, in target pixels, as
  /// @ref PrimitiveDraw::viewport: it may extend past the target, which clips
  /// it, and an empty one draws nothing.
  VkRect2D viewport{};
  /// World to Vulkan clip space: projection * view, as
  /// `camera::Camera::view_proj` gives it. Finite.
  glm::mat4 view_proj{1.0f};
  /// A point's width in pixels: positive and finite, and drawn clamped to
  /// `[1, DepthCloudPipeline::max_point_size()]`.
  float point_size = 1.0f;
};

/// @brief Everything one @ref DepthCloudPipeline::submit records.
///
/// The draw list is borrowed for the duration of the call.
struct DepthCloudFrame {
  VkExtent2D extent{};  ///< The target's size, in pixels.
  /// The frame's number on the timeline @ref DepthCloudPipeline::create was
  /// given, as `windowing::Frame::number`: at least every number given
  /// before, and set on the timeline once the frame's work completes.
  uint64_t number = 0;
  const DepthCloudDraw* draws = nullptr;  ///< The draw list.
  uint32_t draw_count = 0;                ///< The number of @ref draws.
};

/// @brief What @ref DepthCloudPipeline::create builds.
struct DepthCloudPipelineDesc {
  /// The draws one block of per-draw state holds; at least 1. It sets only
  /// the block's size: a submit takes any number of draws, and a frame any
  /// number of submits.
  uint32_t max_draws = 16;
};

/// @brief Draws depth images as points in world space, depth-tested, each
///        straight from the producer's buffer, by the library's own embedded
///        GLSL.
///
/// A draw reads one depth image and, optionally, one colour image, in place:
/// a vertex per depth pixel, with no vertex buffer and no copy. The host
/// folds both cameras' intrinsics and poses and the view into two matrices,
/// so any number of depth cameras, each with its own colour camera, draw in
/// one frame. Points are depth-tested and write depth, so they occlude one
/// another and a @ref PrimitivePipeline's points and lines in the same
/// rendering scope; colours are stored on each target as
/// @ref PrimitivePipeline stores them.
///
/// Each draw's parameters go into host-mapped memory and its buffers into a
/// descriptor set. Sets and parameters come in blocks of `max_draws`: a
/// frame's draws fill as many blocks as they need, and a block returns to
/// the pipeline once the timeline reaches the frame that used it.
///
/// **What the caller guarantees.** Every buffer a draw reads is either
/// `CONCURRENT` across the producer's queue family and the renderer's, or
/// written on the renderer's family. The producer's writes are complete
/// before the frame is submitted: a fence the host waited for, or a
/// timeline value the frame waits for (`windowing::Frame::waits`) that is
/// already reached. The buffers' owners keep them alive, and do not write
/// them, until the timeline reaches the frame's number (a `RetireQueue` on
/// the frame loop's timeline).
///
/// @warning The @p device, @p allocator and @p timeline passed to @ref create
///          must outlive the pipeline. Destroying it waits until the timeline
///          reaches the newest frame number given to @ref submit, or -- when
///          that frame never reached the queue -- until the renderer's queues
///          drain (`core::Device::wait_idle`).
///
/// @code
/// VKC_ASSIGN(std::unique_ptr<pipelines::DepthCloudPipeline> clouds,
///            pipelines::DepthCloudPipeline::create(device, allocator,
///                                                  target.layout(),
///                                                  loop.timeline()));
/// std::vector<pipelines::DepthCloudDraw> draws(cameras.size());
/// for (size_t i = 0; i < cameras.size(); ++i) {
///   draws[i].depth = {frames[i].depth.get(), 0, depth_camera[i],
///                     depth_to_world[i]};
///   draws[i].color = {frames[i].color.get(), 0, color_camera[i],
///                     color_to_world[i], true};
///   draws[i].mode = pipelines::CloudColor::TintedLuma;
///   draws[i].tint = tints[i];
///   draws[i].viewport = {{0, 0}, extent};
///   draws[i].view_proj = orbit.to_camera(fovy, aspect, 0.05f, 50.0f)
///                            .view_proj();
/// }
/// // inside the rendering scope:
/// VKC_TRY(clouds->submit(f.cmd, {extent, f.number, draws.data(),
///                                static_cast<uint32_t>(draws.size())}));
/// @endcode
class VG_PIPELINES_API DepthCloudPipeline {
 public:
  /// @brief Build the pipeline for targets of @p layout, with one block of
  ///        per-draw state.
  /// @param device     The device; it must have enabled the renderer's
  ///                   requirements (@ref device_requirements), may enable
  ///                   `largePoints` for points wider than a pixel, and must
  ///                   have host-mapped device memory
  ///                   (`PhysicalDeviceInfo::device_mapped_memory`).
  /// @param allocator  Allocates each block's parameters, now and as frames
  ///                   need more blocks.
  /// @param layout     The target's signature: exactly one color attachment,
  ///                   of a defined format, and a depth format.
  /// @param timeline   The timeline frame numbers are set on, borrowed by
  ///                   address: `windowing::FrameLoop::timeline`.
  /// @param desc       How many draws a submit takes.
  /// @return The pipeline; `core::Status::Code::InvalidArgument` for a layout
  ///         without exactly one defined color attachment or without a depth
  ///         format, an empty @p timeline, or a zero `max_draws`;
  ///         `core::Status::Code::Unsupported` for a device without the
  ///         renderer's requirements or without host-mapped device memory;
  ///         or a backend failure.
  static core::Result<std::unique_ptr<DepthCloudPipeline>> create(
      const core::Device& device, core::Allocator& allocator,
      const RenderTargetLayout& layout, const core::TimelineSemaphore& timeline,
      const DepthCloudPipelineDesc& desc = {});

  ~DepthCloudPipeline();
  DepthCloudPipeline(const DepthCloudPipeline&) = delete;
  DepthCloudPipeline& operator=(const DepthCloudPipeline&) = delete;
  DepthCloudPipeline(DepthCloudPipeline&&) = delete;
  DepthCloudPipeline& operator=(DepthCloudPipeline&&) = delete;

  /// @return The widest point the device draws, in pixels: 1 without
  ///         `largePoints`, else the top of its `pointSizeRange`.
  float max_point_size() const noexcept { return limits_.max_point_size(); }

  /// @return The draws one block of per-draw state holds.
  uint32_t max_draws() const noexcept { return max_draws_; }

  /// @return The blocks of per-draw state the pipeline holds, in use or
  ///         free, as a measure of its memory: each is `max_draws`
  ///         descriptor sets and as many parameter ranges, at most 256 bytes
  ///         each, of host-mapped device memory. A frame holds
  ///         `ceil(draws drawn / max_draws)` blocks until the timeline
  ///         reaches it.
  uint32_t block_count() const noexcept {
    return static_cast<uint32_t>(blocks_.size());
  }

  /// @brief Record @p frame's draws into @p cmd.
  ///
  /// Checks every draw before recording any, so on an error nothing is
  /// recorded. A draw whose viewport is empty or lies wholly outside the
  /// target is checked, then skipped. The draws take per-draw state from the
  /// block the frame's earlier submits used, while it has room, and then
  /// from free blocks, or new ones when none is free; frame @p frame.number
  /// holds each until the timeline reaches it.
  /// @param cmd    A command buffer in the recording state, inside a
  ///               rendering scope whose target matches the layout
  ///               @ref create was given.
  /// @param frame  The target's size, the frame's number and the draws.
  /// @return OK once recorded; or `core::Status::Code::InvalidArgument` for
  ///         a frame number of 0 or below one already given, a null draw list
  ///         with a non-zero count, or a draw with: a null or empty depth
  ///         buffer, or a non-null colour buffer that is empty; a buffer
  ///         without `STORAGE_BUFFER` usage; a camera of zero size, or with a
  ///         focal length that is not positive and finite or a principal
  ///         point that is not finite; an image beyond its buffer, an offset
  ///         that is not a multiple of `minStorageBufferOffsetAlignment`, or
  ///         an image larger than `maxStorageBufferRange`; a depth bound that
  ///         is negative or not finite, or a `min_depth` above a non-zero
  ///         `max_depth`; an unknown mode; a zero stride; a point size that
  ///         is not positive and finite; a view or pose that is not finite,
  ///         or a colour pose that cannot be inverted; or a viewport beyond
  ///         the device's `maxViewportDimensions` or `viewportBoundsRange`.
  ///         Or the allocator's failure making a block, before anything is
  ///         recorded.
  core::Status submit(VkCommandBuffer cmd, const DepthCloudFrame& frame);

 private:
  // max_draws descriptor sets, and the parameters they bind, one range each.
  struct Block {
    core::DescriptorPool pool;
    std::vector<core::DescriptorSet> sets;
    core::Buffer params;
  };

  DepthCloudPipeline(const core::Device& device, core::Allocator& allocator,
                     const core::TimelineSemaphore& timeline,
                     GraphicsPipeline pipeline, detail::PointDrawLimits limits,
                     uint32_t max_draws, VkDeviceSize params_stride,
                     std::unique_ptr<Block> first);

  // A block of `max_draws` sets of `layout`, each binding its `stride`-byte
  // range of the block's parameters.
  static core::Result<std::unique_ptr<Block>> make_block(
      const core::Device& device, core::Allocator& allocator,
      VkDescriptorSetLayout layout, uint32_t max_draws, VkDeviceSize stride);
  // A block with a free set for frame `frame`: the current one while that
  // frame holds it and it has room, else a free or new one, retiring the
  // current one at its frame.
  core::Result<Block*> block_for(uint64_t frame);

  const core::Device& device_;
  core::Allocator& allocator_;
  GraphicsPipeline pipeline_;
  detail::PointDrawLimits limits_;
  uint32_t max_draws_ = 0;
  VkDeviceSize params_stride_ = 0;  // a draw's share of Block::params
  std::vector<std::unique_ptr<Block>> blocks_;  // every block, owned here
  std::vector<Block*> free_;                    // blocks no frame holds
  Block* current_ = nullptr;    // the block frame current_frame_ draws from
  uint64_t current_frame_ = 0;  // the frame that holds current_
  uint32_t current_used_ = 0;   // the sets of current_ it has taken
  // Declared last, so destroyed first: its teardown waits for the frames
  // that use the blocks, and its deleters return them to free_.
  detail::FrameRetirement frames_;
};

}  // namespace volumetric_kit::gfx::pipelines

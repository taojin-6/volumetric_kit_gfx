// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file primitive_pipeline.hpp
/// @brief Colored points and lines in world space, depth-tested, each draw in
///        a viewport rectangle of its own: point clouds, camera frusta, axes
///        and outlines.

#include <cstdint>
#include <memory>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::core {
class Buffer;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

/// @brief One vertex of a point or a line: a world-space position and an
///        sRGB-encoded color, 16 bytes.
///
/// @code
/// std::vector<pipelines::ColorVertex> cloud(n);
/// cloud[i] = {position, pipelines::pack_rgba8(r, g, b)};
/// @endcode
struct ColorVertex {
  /// World space.
  glm::vec3 position{0.0f, 0.0f, 0.0f};
  /// The color's bytes in memory order R, G, B, A, sRGB-encoded as camera and
  /// 8-bit image bytes are; @ref pack_rgba8 builds it. A is ignored:
  /// primitives draw opaque.
  uint32_t rgba = 0;
};
static_assert(sizeof(ColorVertex) == 16, "ColorVertex is 16 bytes");

/// @brief Pack a color's bytes into @ref ColorVertex::rgba.
///
/// R lands in the lowest byte, which is the first in memory on the
/// little-endian hosts gfx runs on.
/// @param r  Red, sRGB-encoded.
/// @param g  Green, sRGB-encoded.
/// @param b  Blue, sRGB-encoded.
/// @param a  Alpha; ignored by the pipeline.
/// @return The packed color.
constexpr uint32_t pack_rgba8(uint8_t r, uint8_t g, uint8_t b,
                              uint8_t a = 255) noexcept {
  return uint32_t{r} | (uint32_t{g} << 8) | (uint32_t{b} << 16) |
         (uint32_t{a} << 24);
}

/// @brief What a @ref PrimitiveDraw's vertices make.
enum class PrimitiveTopology : uint32_t {
  /// Each vertex is a square point of @ref PrimitiveDraw::point_size pixels.
  Points = 0,
  /// Each pair of vertices is a line segment one pixel wide.
  Lines = 1,
};

/// @brief One run of vertices drawn as points or lines, through one camera,
///        into one rectangle of the target.
///
/// The rectangle is the viewport: normalized device coordinates `[-1, 1]`
/// span it, so @ref view_proj is made for its aspect ratio, and nothing is
/// drawn outside it or outside the target.
struct PrimitiveDraw {
  /// What the vertices make.
  PrimitiveTopology topology = PrimitiveTopology::Points;
  /// The @ref ColorVertex array: a buffer with `VERTEX_BUFFER` usage, such as
  /// `upload_buffer` makes. Borrowed; it must outlive the frame's execution.
  const core::Buffer* vertices = nullptr;
  /// The index of the first vertex drawn.
  uint32_t first_vertex = 0;
  /// The number of vertices drawn: even for @ref PrimitiveTopology::Lines.
  /// 0 draws nothing.
  uint32_t vertex_count = 0;
  /// The rectangle drawn into, in target pixels; it may extend past the
  /// target, which clips it. An empty one draws nothing.
  VkRect2D viewport{};
  /// World to Vulkan clip space: projection * view, as
  /// `camera::Camera::view_proj` gives it.
  glm::mat4 view_proj{1.0f};
  /// A point's width in pixels: positive and finite, and drawn clamped to
  /// `[1, PrimitivePipeline::max_point_size()]`, so always 1 on a device
  /// without `largePoints`. Lines ignore it.
  float point_size = 1.0f;
};

/// @brief Everything @ref PrimitivePipeline::submit records for one frame.
///
/// The draw list is borrowed for the duration of the call.
struct PrimitiveFrame {
  VkExtent2D extent{};                   ///< The target's size, in pixels.
  const PrimitiveDraw* draws = nullptr;  ///< The draw list.
  uint32_t draw_count = 0;               ///< The number of @ref draws.
};

/// @brief Draws @ref ColorVertex runs as points and lines in world space,
///        depth-tested, by the library's own embedded GLSL.
///
/// Holds a point-list and a line-list pipeline, so one frame mixes both:
/// a point cloud and the frusta of the cameras that saw it. Each draw sets
/// its own viewport and scissor and pushes its view-projection and point
/// size, so a frame may show several views side by side. Primitives are
/// depth-tested and write depth, so they occlude one another, and a mesh in
/// the same pass, by distance whatever the draw order.
///
/// Colors are sRGB-encoded bytes. A `_UNORM` target stores them as given,
/// an `_SRGB` one stores the same bytes (the shader decodes and the target
/// encodes), and a float target holds their linear light, as
/// @ref ImagePipeline writes images. Line colors interpolate in the encoded
/// values, alike on every target.
///
/// @warning The @p device passed to @ref create must outlive the pipeline,
///          and the GPU must be done with every frame that drew with it
///          before it is destroyed: retire it through a @ref RetireQueue or
///          after `core::Device::wait_idle`.
///
/// @code
/// VKC_ASSIGN(std::unique_ptr<pipelines::PrimitivePipeline> primitives,
///            pipelines::PrimitivePipeline::create(device, target.layout()));
/// BufferUploadDesc upload;
/// upload.data = cloud.data();
/// upload.size = cloud.size() * sizeof(pipelines::ColorVertex);
/// upload.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
/// VKC_ASSIGN(core::Buffer points, upload_buffer(device, allocator, upload));
/// // each frame, inside the rendering scope:
/// pipelines::PrimitiveDraw draw;
/// draw.vertices = &points;
/// draw.vertex_count = static_cast<uint32_t>(cloud.size());
/// draw.viewport = {{0, 0}, extent};
/// draw.view_proj = camera.view_proj();  // a camera::Camera
/// VKC_TRY(primitives->submit(cmd, {extent, &draw, 1}));
/// @endcode
class VG_PIPELINES_API PrimitivePipeline {
 public:
  /// @brief Build the point and line pipelines for targets of @p layout.
  /// @param device  The device; it must have enabled the renderer's
  ///                requirements (@ref device_requirements), and may enable
  ///                `largePoints` for points wider than a pixel.
  /// @param layout  The target's signature: exactly one color attachment and
  ///                a depth format.
  /// @return The pipeline; or `core::Status::Code::InvalidArgument` for a
  ///         layout without exactly one color attachment or without a depth
  ///         format; `core::Status::Code::Unsupported` for a device without
  ///         the renderer's requirements; or a backend failure.
  static core::Result<std::unique_ptr<PrimitivePipeline>> create(
      const core::Device& device, const RenderTargetLayout& layout);

  ~PrimitivePipeline();
  PrimitivePipeline(const PrimitivePipeline&) = delete;
  PrimitivePipeline& operator=(const PrimitivePipeline&) = delete;
  PrimitivePipeline(PrimitivePipeline&&) = delete;
  PrimitivePipeline& operator=(PrimitivePipeline&&) = delete;

  /// @return The widest point the device draws, in pixels: 1 without
  ///         `largePoints`, else the top of its `pointSizeRange`.
  float max_point_size() const noexcept { return max_point_size_; }

  /// @brief Record @p frame's draws into @p cmd.
  ///
  /// Checks every draw before recording any, so on an error nothing is
  /// recorded. A draw with no vertices is skipped unchecked; one whose
  /// viewport is empty or lies wholly outside the target is checked, then
  /// skipped.
  /// @param cmd    A command buffer in the recording state, inside a
  ///               rendering scope whose target matches the layout
  ///               @ref create was given.
  /// @param frame  The target's size and the draws.
  /// @return OK once recorded; or `core::Status::Code::InvalidArgument` for
  ///         a null draw list with a non-zero count, or a draw with a null or
  ///         empty vertex buffer, one without `VERTEX_BUFFER` usage, a vertex
  ///         range beyond the buffer, an odd vertex count for lines, a point
  ///         size that is not positive and finite for points, or a viewport
  ///         beyond the device's `maxViewportDimensions` or
  ///         `viewportBoundsRange`.
  core::Status submit(VkCommandBuffer cmd, const PrimitiveFrame& frame) const;

 private:
  PrimitivePipeline(GraphicsPipeline points, GraphicsPipeline lines,
                    const VkPhysicalDeviceLimits& limits,
                    bool large_points) noexcept;

  GraphicsPipeline points_;
  GraphicsPipeline lines_;
  float max_point_size_ = 1.0f;
  float max_viewport_[2] = {0.0f, 0.0f};     // maxViewportDimensions
  float viewport_bounds_[2] = {0.0f, 0.0f};  // viewportBoundsRange
};

}  // namespace volumetric_kit::gfx::pipelines

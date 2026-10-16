// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pipelines/impl/point_draw.hpp
/// What the pipelines that draw points share: the widest point the device
/// draws, and the viewport limits a draw is checked against. Included from
/// @ref PrimitivePipeline's and @ref DepthCloudPipeline's headers; not a
/// standalone type.

#include <algorithm>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"

namespace volumetric_kit::gfx::pipelines::detail {

/// @brief Check the target a point pipeline draws into.
/// @param layout  The target's signature.
/// @param call    Names the caller in the message.
/// @return OK for exactly one color attachment, of a defined format, and a
///         depth format, as depth-tested points need; else
///         `core::Status::Code::InvalidArgument`.
inline core::Status check_point_target(const RenderTargetLayout& layout,
                                       const char* call) {
  if (layout.color_count != 1 ||
      layout.color_formats[0] == VK_FORMAT_UNDEFINED) {
    return core::Status::invalid_argument(
        std::string(call) +
        ": the layout must have exactly one color attachment, of a defined "
        "format");
  }
  if (layout.depth_format == VK_FORMAT_UNDEFINED) {
    return core::Status::invalid_argument(
        std::string(call) +
        ": the layout must carry a depth format (points are depth-tested)");
  }
  return core::Status{};
}

/// @brief A device's limits on point draws: the widest point and the
///        viewports it accepts.
class PointDrawLimits {
 public:
  /// @brief Read @p device's limits. Points wider than a pixel need
  ///        `largePoints` enabled on the device, not merely supported.
  /// @param device  The device the pipeline draws on.
  /// @return Its limits.
  static PointDrawLimits of(const core::Device& device) {
    core::DeviceRequirements wide = device_requirements();
    wide.features.largePoints = VK_TRUE;
    const VkPhysicalDeviceLimits& limits = device.caps().limits();
    PointDrawLimits out;
    out.max_point_size_ = device.check_enabled(wide).ok()
                              ? std::max(limits.pointSizeRange[1], 1.0f)
                              : 1.0f;
    out.max_viewport_[0] = static_cast<float>(limits.maxViewportDimensions[0]);
    out.max_viewport_[1] = static_cast<float>(limits.maxViewportDimensions[1]);
    out.viewport_bounds_[0] = limits.viewportBoundsRange[0];
    out.viewport_bounds_[1] = limits.viewportBoundsRange[1];
    return out;
  }

  /// @return The widest point the device draws, in pixels: 1 without
  ///         `largePoints`, else the top of its `pointSizeRange`.
  float max_point_size() const noexcept { return max_point_size_; }

  /// @brief The width a point of @p size is drawn at.
  /// @param size  A positive, finite width in pixels.
  /// @return @p size clamped to `[1, max_point_size()]`.
  float clamp(float size) const noexcept {
    return std::clamp(size, 1.0f, max_point_size_);
  }

  /// @brief Whether the device accepts @p viewport.
  /// @param viewport  A non-empty viewport, in target pixels.
  /// @return `false` past `maxViewportDimensions` or `viewportBoundsRange`.
  bool viewport_fits(const VkRect2D& viewport) const noexcept {
    // Doubles, so an offset plus an extent cannot wrap.
    const double x = viewport.offset.x;
    const double y = viewport.offset.y;
    const double width = viewport.extent.width;
    const double height = viewport.extent.height;
    return width <= max_viewport_[0] && height <= max_viewport_[1] &&
           x >= viewport_bounds_[0] && y >= viewport_bounds_[0] &&
           x + width <= viewport_bounds_[1] &&
           y + height <= viewport_bounds_[1];
  }

 private:
  float max_point_size_ = 1.0f;
  float max_viewport_[2] = {0.0f, 0.0f};     // maxViewportDimensions
  float viewport_bounds_[2] = {0.0f, 0.0f};  // viewportBoundsRange
};

}  // namespace volumetric_kit::gfx::pipelines::detail

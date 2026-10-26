// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file ycbcr.hpp
/// @brief An 8-bit Y'CbCr 4:2:0 picture: how its values code color
///        (@ref YcbcrCoding), and its two planes as images a shader samples
///        where they are (@ref YcbcrPicture).

#include <cstdint>

#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::core {
class Image;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

/// @brief Where a 4:2:0 picture's chroma samples sit relative to its luma.
enum class ChromaSiting : uint32_t {
  /// Between the luma samples on both axes.
  Center = 0,
  /// Level with the left luma sample of each pair, and between the rows.
  Left = 1,
};

/// @brief How a Y'CbCr picture's values code R'G'B': the matrix, the range
///        and the chroma siting.
///
/// The matrix gives R'G'B' in the picture's transfer function, which the
/// family takes as sRGB's.
///
/// @code
/// pipelines::YcbcrCoding bt709;  // BT.709, limited range
/// bt709.kr = 0.2126f;
/// bt709.kb = 0.0722f;
/// bt709.full_range = false;
/// @endcode
struct YcbcrCoding {
  /// The matrix's red weight: BT.601 0.299, BT.709 0.2126.
  float kr = 0.299f;
  /// The matrix's blue weight: BT.601 0.114, BT.709 0.0722. Together with
  /// @ref kr: `kr > 0`, `kb > 0` and `kr + kb < 1`.
  float kb = 0.114f;
  /// Y' in 0..255 rather than 16..235 (and chroma in 0..255 rather than
  /// 16..240): JPEG is full range, broadcast video limited.
  bool full_range = true;
  /// Where the chroma samples sit.
  ChromaSiting chroma_siting = ChromaSiting::Center;
};

/// @brief An 8-bit Y'CbCr 4:2:0 picture's two planes, as images the caller
///        owns and a shader samples where they are, with no copy.
///
/// A camera leaves its picture this way -- ARKit's `capturedImage` is
/// '420f' -- and a plane-by-plane import makes each plane an image. The
/// picture sits at the planes' top-left corner; the planes may be larger than
/// it, and each is addressed by its own size, so odd sizes work.
///
/// Each plane is a single-sample, one-layer 2D image with `SAMPLED` usage and
/// a default view, in the layout its `layout()` records -- `GENERAL` or
/// `SHADER_READ_ONLY_OPTIMAL` -- whenever a frame that samples it runs.
///
/// @code
/// pipelines::YcbcrPicture picture;  // a camera's, BT.601 full range
/// picture.luma = &luma;             // the imported planes
/// picture.chroma = &chroma;
/// picture.extent = {1920, 1440};
/// @endcode
struct YcbcrPicture {
  /// Y': `R8_UNORM`, at least @ref extent.
  const core::Image* luma = nullptr;
  /// Cb then Cr, interleaved: `R8G8_UNORM`, at least half @ref extent on each
  /// axis, rounded up.
  const core::Image* chroma = nullptr;
  /// The picture's size in luma texels; non-zero.
  VkExtent2D extent{};
  /// How its values code R'G'B'.
  YcbcrCoding coding;
};

}  // namespace volumetric_kit::gfx::pipelines

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pinhole_camera.hpp
/// @brief A pinhole camera's intrinsics and image size, as a depth sensor or
///        a colour camera reports them once undistorted.

#include <cstdint>

namespace volumetric_kit::gfx::camera {

/// @brief A pinhole camera: focal lengths and principal point in pixels, and
///        the image's size.
///
/// Pixel centres are at integer coordinates, as OpenCV and recon's cameras
/// have them: a camera-space point `(x, y, z)` with `z > 0` lands at
/// `(fx * x / z + cx, fy * y / z + cy)`, and pixel `(u, v)` is the one whose
/// centre is nearest, inside `[0, width) x [0, height)`. The camera looks
/// down +z, with +x right and +y down the image.
///
/// @code
/// camera::PinholeCamera depth{504.2f, 504.3f, 320.1f, 288.7f, 640, 576};
/// @endcode
struct PinholeCamera {
  float fx = 0.0f;           ///< Focal length along x, in pixels.
  float fy = 0.0f;           ///< Focal length along y, in pixels.
  float cx = 0.0f;           ///< Principal point's x, in pixels.
  float cy = 0.0f;           ///< Principal point's y, in pixels.
  std::uint32_t width = 0;   ///< Image width, in pixels.
  std::uint32_t height = 0;  ///< Image height, in pixels.
};

}  // namespace volumetric_kit::gfx::camera

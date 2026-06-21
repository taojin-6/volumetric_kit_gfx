// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image.hpp
/// @brief Decoded CPU-side image pixels referenced by materials.

#include <cstdint>
#include <string>
#include <vector>

namespace volumetric_kit::gfx::assets {

/// @brief A decoded image held entirely in host memory.
///
/// Pixels are tightly packed, row-major, `width * height * channels` bytes with
/// one byte per channel (8-bit). A loader fills this from an embedded or
/// external image (e.g. via stb_image); a later GPU tier copies @ref pixels
/// into a staging buffer and uploads it to a `VkImage`. Materials never hold an
/// `Image` directly -- they store an index into @ref Model::images.
///
/// @code
/// const assets::Image& img = model.images[material.base_color_texture];
/// upload_rgba8(img.width, img.height, img.pixels.data());  // GPU tier
/// @endcode
struct Image {
  std::string name;            ///< Source URI / embedded name (may be empty).
  std::uint32_t width = 0;     ///< Width in pixels.
  std::uint32_t height = 0;    ///< Height in pixels.
  std::uint32_t channels = 0;  ///< Components per pixel (1=R, 3=RGB, 4=RGBA).
  std::vector<std::uint8_t> pixels;  ///< Row-major 8-bit pixels, no padding.

  /// @return `true` if this holds a non-empty, dimensionally consistent image.
  bool valid() const noexcept {
    return width > 0 && height > 0 && channels > 0 &&
           pixels.size() == static_cast<std::size_t>(width) * height * channels;
  }
};

}  // namespace volumetric_kit::gfx::assets

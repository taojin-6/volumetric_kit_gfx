// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file color.hpp
/// @brief Colours as the pipelines take them, sRGB-encoded RGBA8 words:
///        packing one from its bytes, and a colour for each item of a set.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace volumetric_kit::gfx::pipelines {

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

/// @brief How many of @ref categorical_rgba8's colours are Okabe and Ito's.
inline constexpr std::size_t kCategoricalPaletteSize = 7;

/// @brief A colour for each item of a set -- each camera of a rig, say --
///        that tells the items apart, however many there are.
///
/// Items 0 to 6 take Okabe and Ito's Color Universal Design palette
/// (https://jfly.uni-koeln.de/color/, 2008) without its black, in its order
/// but with yellow and blue swapped, so that the first four differ in
/// chromaticity: orange `#E69F00`, sky blue `#56B4E9`, bluish green
/// `#009E73`, blue `#0072B2`, yellow `#F0E442`, vermillion `#D55E00` and
/// reddish purple `#CC79A7`. Each later item is OKLCH lightness 0.75 and
/// chroma 0.125, which sRGB holds at every hue, at hue 12.5 degrees for item 7
/// and a golden angle (360 / phi^2 = 137.5 degrees) further for each item
/// after it.
///
/// How far apart the colours are in OKLab, where 0.02 is about one
/// just-noticeable difference (CSS Color 4):
/// - Drawn as they are, as frusta and `Solid` points: the first 10 at least
///   0.08, the first 16 at least 0.04, the first 24 at least 0.025, and any 8
///   consecutive ones at least 0.06; by 29 colours, two are within 0.02.
///   Under simulated protan, deutan and tritan vision, only the first 7 are
///   held apart, at least 0.07.
/// - As `TintedLuma` points, which take their lightness from the scene, so
///   at equal luma: the first 12 at least 0.03, but by 13 two are within
///   0.02; for protan and deutan vision only the first 4, at least 0.05.
///
/// @code
/// for (size_t i = 0; i < cameras.size(); ++i) {
///   cameras[i].tint = pipelines::categorical_rgba8(i);
/// }
/// @endcode
/// @param index  The item's index in its set; any value.
/// @return The item's colour as @ref pack_rgba8 packs it, opaque.
///         `IM_COL32` packs the same layout, so ImGui draws it as it is.
inline uint32_t categorical_rgba8(std::size_t index) noexcept {
  constexpr uint32_t kOkabeIto[kCategoricalPaletteSize] = {
      pack_rgba8(230, 159, 0),  pack_rgba8(86, 180, 233),
      pack_rgba8(0, 158, 115),  pack_rgba8(0, 114, 178),
      pack_rgba8(240, 228, 66), pack_rgba8(213, 94, 0),
      pack_rgba8(204, 121, 167)};
  if (index < kCategoricalPaletteSize) {
    return kOkabeIto[index];
  }
  constexpr double kLightness = 0.75;
  constexpr double kChroma = 0.125;
  // The hue in turns, 5/144 + n / phi^2 modulo 1, in 64-bit fixed point
  // (2^64 * 5/144 and 2^64 / phi^2), exact for every n.
  const auto n = static_cast<uint64_t>(index - kCategoricalPaletteSize);
  const double turns = static_cast<double>(uint64_t{0x08E38E38E38E38E4} +
                                           n * uint64_t{0x61C8864680B583EA}) *
                       0x1p-64;
  const double hue = 6.283185307179586 * turns;
  const double a = kChroma * std::cos(hue);
  const double b = kChroma * std::sin(hue);
  // OKLab to linear sRGB (Ottosson, https://bottosson.github.io/posts/oklab/).
  const auto cube = [](double x) { return x * x * x; };
  const double l = cube(kLightness + 0.3963377774 * a + 0.2158037573 * b);
  const double m = cube(kLightness - 0.1055613458 * a - 0.0638541728 * b);
  const double s = cube(kLightness - 0.0894841775 * a - 1.2914855480 * b);
  const auto encode = [](double linear) {
    // The clamp guards rounding alone: no channel leaves [0, 1].
    const double c = std::clamp(linear, 0.0, 1.0);
    const double e =
        c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    return static_cast<uint8_t>(e * 255.0 + 0.5);
  };
  return pack_rgba8(
      encode(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
      encode(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
      encode(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s));
}

}  // namespace volumetric_kit::gfx::pipelines

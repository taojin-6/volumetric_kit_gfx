// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file texture_view_test_support.hpp
/// Test pictures for the texture view's tests: a test card with fine detail,
/// and its Y'CbCr 4:2:0 planes.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/ycbcr.hpp"

namespace vg_test {

// An RGB picture, row-major, 8-bit sRGB-encoded.
struct Rgb {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<std::array<uint8_t, 3>> pixels;
  const std::array<uint8_t, 3>& at(uint32_t x, uint32_t y) const {
    return pixels[static_cast<size_t>(y) * width + x];
  }
};

// A test card with `seed` in it: a colour zone plate on the left half; on
// the right, a one-pixel checker of two colours over colour bars and ramps.
inline Rgb make_card(uint32_t width, uint32_t height, uint32_t seed = 0) {
  Rgb card{width, height, {}};
  card.pixels.resize(static_cast<size_t>(width) * height);
  const double k = 3.14159265 / (2.0 * width);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      std::array<uint8_t, 3>& p =
          card.pixels[static_cast<size_t>(y) * width + x];
      if (x < width / 2) {
        const double r2 =
            static_cast<double>(x) * x + static_cast<double>(y) * y;
        const double wave = 0.5 + 0.5 * std::cos(k * r2 + seed);
        p = {static_cast<uint8_t>(40 + 200 * wave),
             static_cast<uint8_t>(230 - 180 * wave),
             static_cast<uint8_t>(60 + (y * 3 + seed * 17) % 140)};
      } else if (y < height / 2) {
        p = (x + y) % 2 == 0 ? std::array<uint8_t, 3>{250, 230, 40}
                             : std::array<uint8_t, 3>{20, 40, 120};
      } else {
        const uint32_t bar = (x - width / 2) * 8 / (width - width / 2);
        p = {static_cast<uint8_t>((bar & 1) * 200 + (y * 7) % 55),
             static_cast<uint8_t>((bar & 2) * 100 + (x * 5) % 55),
             static_cast<uint8_t>((bar & 4) * 50 + seed * 31 % 55)};
      }
    }
  }
  return card;
}

// `rgb`'s BT.709 full-range Y'CbCr 4:2:0 planes in planes of `luma_size`
// (at least the picture), each chroma sample the mean of its 2 x 2; pixels
// past the picture alternate between the extremes, so a read of them shows.
struct PlaneBytes {
  VkExtent2D luma_size{};
  VkExtent2D chroma_size{};
  std::vector<uint8_t> luma;
  std::vector<uint8_t> chroma;
};
inline PlaneBytes planes_of(const Rgb& rgb, VkExtent2D luma_size) {
  constexpr double kr = 0.2126;
  constexpr double kb = 0.0722;
  PlaneBytes out;
  out.luma_size = luma_size;
  out.chroma_size = {(luma_size.width + 1) / 2, (luma_size.height + 1) / 2};
  out.luma.assign(static_cast<size_t>(luma_size.width) * luma_size.height, 0);
  out.chroma.assign(
      static_cast<size_t>(out.chroma_size.width) * out.chroma_size.height * 2,
      0);
  std::vector<std::array<double, 2>> cbcr(rgb.pixels.size());
  for (uint32_t y = 0; y < luma_size.height; ++y) {
    for (uint32_t x = 0; x < luma_size.width; ++x) {
      const size_t at = static_cast<size_t>(y) * luma_size.width + x;
      if (x >= rgb.width || y >= rgb.height) {
        out.luma[at] = static_cast<uint8_t>(((x + y) % 2) * 255);
        continue;
      }
      const std::array<uint8_t, 3>& p = rgb.at(x, y);
      const double r = p[0] / 255.0;
      const double g = p[1] / 255.0;
      const double b = p[2] / 255.0;
      const double luma = kr * r + (1.0 - kr - kb) * g + kb * b;
      out.luma[at] = static_cast<uint8_t>(std::lround(luma * 255.0));
      cbcr[static_cast<size_t>(y) * rgb.width + x] = {
          (b - luma) / (2.0 * (1.0 - kb)), (r - luma) / (2.0 * (1.0 - kr))};
    }
  }
  for (uint32_t j = 0; j < out.chroma_size.height; ++j) {
    for (uint32_t i = 0; i < out.chroma_size.width; ++i) {
      const size_t at =
          (static_cast<size_t>(j) * out.chroma_size.width + i) * 2;
      std::array<double, 2> sum{0.0, 0.0};
      int n = 0;
      for (uint32_t y = 2 * j; y < std::min(2 * j + 2, rgb.height); ++y) {
        for (uint32_t x = 2 * i; x < std::min(2 * i + 2, rgb.width); ++x) {
          sum[0] += cbcr[static_cast<size_t>(y) * rgb.width + x][0];
          sum[1] += cbcr[static_cast<size_t>(y) * rgb.width + x][1];
          ++n;
        }
      }
      for (int c = 0; c < 2; ++c) {
        out.chroma[at + c] =
            n == 0 ? static_cast<uint8_t>(((i + j) % 2) * 255)
                   : static_cast<uint8_t>(std::clamp(
                         std::lround((sum[c] / n + 0.5) * 255.0), 0L, 255L));
      }
    }
  }
  return out;
}

inline volumetric_kit::gfx::pipelines::YcbcrCoding bt709_full() {
  volumetric_kit::gfx::pipelines::YcbcrCoding c;
  c.kr = 0.2126f;
  c.kb = 0.0722f;
  c.full_range = true;
  c.chroma_siting = volumetric_kit::gfx::pipelines::ChromaSiting::Center;
  return c;
}

}  // namespace vg_test

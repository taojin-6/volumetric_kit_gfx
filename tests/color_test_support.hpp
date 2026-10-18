// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file color_test_support.hpp
/// The tests' reference colour maths, in double precision and independent of
/// the code under test: the sRGB curves and OKLab. Pure CPU.

#include <array>
#include <cmath>

namespace vg_test {

// IEC 61966-2-1's decode, from an encoded value in [0, 1] to linear light.
inline double srgb_to_linear(double encoded) {
  return encoded <= 0.04045 ? encoded / 12.92
                            : std::pow((encoded + 0.055) / 1.055, 2.4);
}

// IEC 61966-2-1's encode, from linear light in [0, 1] to an encoded value.
inline double linear_to_srgb(double linear) {
  return linear <= 0.0031308 ? linear * 12.92
                             : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
}

struct Oklab {
  double L = 0.0;
  double a = 0.0;
  double b = 0.0;
};

// Linear sRGB to OKLab (Ottosson, https://bottosson.github.io/posts/oklab/).
inline Oklab oklab_of(const std::array<double, 3>& linear) {
  const double l =
      std::cbrt(0.4122214708 * linear[0] + 0.5363325363 * linear[1] +
                0.0514459929 * linear[2]);
  const double m =
      std::cbrt(0.2119034982 * linear[0] + 0.6806995451 * linear[1] +
                0.1073969566 * linear[2]);
  const double s =
      std::cbrt(0.0883024619 * linear[0] + 0.2817188376 * linear[1] +
                0.6299787005 * linear[2]);
  return {0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
          1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
          0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s};
}

// The Euclidean distance between two OKLab colours, where 0.02 is about one
// just-noticeable difference (the JND of CSS Color 4's gamut mapping).
inline double distance(const Oklab& p, const Oklab& q) {
  return std::hypot(p.L - q.L, p.a - q.a, p.b - q.b);
}

}  // namespace vg_test

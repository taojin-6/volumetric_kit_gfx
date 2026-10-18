// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// categorical_rgba8: Okabe and Ito's palette, then the OKLCH hue ring, and
// how far apart they are in OKLab, where 0.02 is about one just-noticeable
// difference (the JND of CSS Color 4's gamut mapping). Pure CPU.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>

#include "volumetric_kit/gfx/pipelines/color.hpp"

namespace {

using volumetric_kit::gfx::pipelines::categorical_rgba8;
using volumetric_kit::gfx::pipelines::kCategoricalPaletteSize;
using volumetric_kit::gfx::pipelines::pack_rgba8;

constexpr std::size_t kLast = std::numeric_limits<std::size_t>::max();
constexpr double kPi = 3.14159265358979323846;

using Rgb = std::array<double, 3>;  // linear sRGB

struct Lab {
  double L = 0.0;
  double a = 0.0;
  double b = 0.0;
};

Rgb linear_of(uint32_t rgba) {
  Rgb rgb{};
  for (int i = 0; i < 3; ++i) {
    const double c = static_cast<double>((rgba >> (8 * i)) & 0xFFu) / 255.0;
    rgb[i] = c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
  }
  return rgb;
}

// Linear sRGB to OKLab (Ottosson, https://bottosson.github.io/posts/oklab/).
Lab oklab_of(const Rgb& c) {
  const double l = std::cbrt(0.4122214708 * c[0] + 0.5363325363 * c[1] +
                             0.0514459929 * c[2]);
  const double m = std::cbrt(0.2119034982 * c[0] + 0.6806995451 * c[1] +
                             0.1073969566 * c[2]);
  const double s = std::cbrt(0.0883024619 * c[0] + 0.2817188376 * c[1] +
                             0.6299787005 * c[2]);
  return {0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
          1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
          0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s};
}

Lab oklab(std::size_t index) {
  return oklab_of(linear_of(categorical_rgba8(index)));
}

double distance(const Lab& p, const Lab& q) {
  return std::hypot(p.L - q.L, p.a - q.a, p.b - q.b);
}

// The smallest distance between two of items [first, first + count) fewer
// than `window` apart.
double min_distance(std::size_t first, std::size_t count, std::size_t window) {
  double smallest = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = i + 1; j < count && j - i < window; ++j) {
      smallest =
          std::min(smallest, distance(oklab(first + i), oklab(first + j)));
    }
  }
  return smallest;
}

// Machado, Oliveira and Fernandes, "A Physiologically-based Model for
// Simulation of Color Vision Deficiency" (IEEE TVCG 15(6), 2009): severity
// 1.0, on linear sRGB.
using Matrix = std::array<Rgb, 3>;
constexpr Matrix kProtan = {{{0.152286, 1.052583, -0.204868},
                             {0.114503, 0.786281, 0.099216},
                             {-0.003882, -0.048116, 1.051998}}};
constexpr Matrix kDeutan = {{{0.367322, 0.860646, -0.227968},
                             {0.280085, 0.672501, 0.047413},
                             {-0.011820, 0.042940, 0.968881}}};
constexpr Matrix kTritan = {{{1.255528, -0.076749, -0.178779},
                             {-0.078411, 0.930809, 0.147602},
                             {0.004733, 0.691367, 0.303900}}};

Rgb seen_with(const Matrix& m, const Rgb& c) {
  Rgb seen{};
  for (int i = 0; i < 3; ++i) {
    const double v = m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2];
    seen[i] = std::clamp(v, 0.0, 1.0);
  }
  return seen;
}

TEST(CategoricalColorTest, FirstColorsAreOkabeAndItosWithoutBlack) {
  constexpr uint32_t kOkabeIto[] = {
      pack_rgba8(0xE6, 0x9F, 0x00),  // orange
      pack_rgba8(0x56, 0xB4, 0xE9),  // sky blue
      pack_rgba8(0x00, 0x9E, 0x73),  // bluish green
      pack_rgba8(0xF0, 0xE4, 0x42),  // yellow
      pack_rgba8(0x00, 0x72, 0xB2),  // blue
      pack_rgba8(0xD5, 0x5E, 0x00),  // vermillion
      pack_rgba8(0xCC, 0x79, 0xA7),  // reddish purple
  };
  ASSERT_EQ(kCategoricalPaletteSize, std::size(kOkabeIto));
  for (std::size_t i = 0; i < kCategoricalPaletteSize; ++i) {
    EXPECT_EQ(categorical_rgba8(i), kOkabeIto[i]) << "item " << i;
  }
}

TEST(CategoricalColorTest, IsTheSameForTheSameIndex) {
  // The ring's words, pinned: a changed rule or its arithmetic changes them.
  const std::pair<std::size_t, uint32_t> pinned[] = {
      {7, 0xFFAB8BEEu},  {8, 0xFF71C087u},    {9, 0xFFFDA797u},
      {10, 0xFF5F96ECu}, {1007, 0xFFBC8CE8u}, {1000007, 0xFFA58BEFu},
  };
  for (const auto& [index, word] : pinned) {
    EXPECT_EQ(categorical_rgba8(index), word) << "item " << index;
  }
  if (sizeof(std::size_t) == 8) {
    EXPECT_EQ(categorical_rgba8(kLast), 0xFFC78EE2u);
  }
  for (std::size_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(categorical_rgba8(i), categorical_rgba8(i)) << "item " << i;
  }
}

TEST(CategoricalColorTest, IsOpaque) {
  for (std::size_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(categorical_rgba8(i) >> 24, 0xFFu) << "item " << i;
  }
  EXPECT_EQ(categorical_rgba8(kLast) >> 24, 0xFFu);
}

TEST(CategoricalColorTest, LaterColorsKeepTheirLightnessAndChromaOnTheRing) {
  // An item clipped out of sRGB would leave its lightness or chroma; the
  // tolerances are 8-bit rounding's, with margin.
  const double golden = 180.0 * (3.0 - std::sqrt(5.0));  // 360 / phi^2
  for (std::size_t n = 0; n < 2000; ++n) {
    const Lab lab = oklab(kCategoricalPaletteSize + n);
    EXPECT_NEAR(lab.L, 0.75, 0.005) << "item " << kCategoricalPaletteSize + n;
    EXPECT_NEAR(std::hypot(lab.a, lab.b), 0.125, 0.005)
        << "item " << kCategoricalPaletteSize + n;
    const double hue = std::atan2(lab.b, lab.a) * 180.0 / kPi;
    const double off =
        std::remainder(hue - golden * static_cast<double>(n), 360.0);
    EXPECT_LT(std::abs(off), 2.0) << "item " << kCategoricalPaletteSize + n;
  }
  for (const std::size_t index : {std::size_t{1000007}, kLast}) {
    const Lab lab = oklab(index);
    EXPECT_NEAR(lab.L, 0.75, 0.005) << "item " << index;
    EXPECT_NEAR(std::hypot(lab.a, lab.b), 0.125, 0.005) << "item " << index;
  }
}

TEST(CategoricalColorTest, NoColorIsDark) {
  // The views' backgrounds are dark, and TintedLuma scales by the tint.
  for (std::size_t i = 0; i < 64; ++i) {
    EXPECT_GE(oklab(i).L, 0.5) << "item " << i;
  }
}

TEST(CategoricalColorTest, PaletteColorsAreFarApart) {
  EXPECT_GE(min_distance(0, kCategoricalPaletteSize, kCategoricalPaletteSize),
            0.15);
}

TEST(CategoricalColorTest, PaletteColorsStayApartInColorBlindVision) {
  const std::pair<const char*, const Matrix*> visions[] = {
      {"protan", &kProtan}, {"deutan", &kDeutan}, {"tritan", &kTritan}};
  for (const auto& [name, matrix] : visions) {
    double smallest = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < kCategoricalPaletteSize; ++i) {
      for (std::size_t j = i + 1; j < kCategoricalPaletteSize; ++j) {
        const Lab p =
            oklab_of(seen_with(*matrix, linear_of(categorical_rgba8(i))));
        const Lab q =
            oklab_of(seen_with(*matrix, linear_of(categorical_rgba8(j))));
        smallest = std::min(smallest, distance(p, q));
      }
    }
    EXPECT_GE(smallest, 0.07) << name;
  }
}

TEST(CategoricalColorTest, FirstFourteenAreApart) {
  const std::size_t count = 2 * kCategoricalPaletteSize;
  EXPECT_GE(min_distance(0, count, count), 0.06);
}

TEST(CategoricalColorTest, AnyEightConsecutiveAreApart) {
  for (const std::size_t first :
       {std::size_t{0}, std::size_t{1000000}, kLast - 63}) {
    EXPECT_GE(min_distance(first, 64, 8), 0.06) << "from item " << first;
  }
}

}  // namespace

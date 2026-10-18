// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// categorical_rgba8: Okabe and Ito's palette, then the OKLCH hue ring, and
// how far apart they are in OKLab. Pure CPU.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>

#include "color_test_support.hpp"
#include "volumetric_kit/gfx/pipelines/color.hpp"

namespace {

using volumetric_kit::gfx::pipelines::categorical_rgba8;
using volumetric_kit::gfx::pipelines::kCategoricalPaletteSize;
using volumetric_kit::gfx::pipelines::pack_rgba8;

constexpr std::size_t kLast = std::numeric_limits<std::size_t>::max();
constexpr double kPi = 3.14159265358979323846;

using vg_test::distance;
using vg_test::Oklab;
using vg_test::oklab_of;
using Rgb = std::array<double, 3>;  // linear sRGB

Rgb linear_of(uint32_t rgba) {
  Rgb rgb{};
  for (int i = 0; i < 3; ++i) {
    rgb[i] = vg_test::srgb_to_linear(
        static_cast<double>((rgba >> (8 * i)) & 0xFFu) / 255.0);
  }
  return rgb;
}

Oklab oklab(std::size_t index) {
  return oklab_of(linear_of(categorical_rgba8(index)));
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

// A vision to see colours with: a simulation's matrix, or none for normal
// vision.
struct Vision {
  const char* name;
  const Matrix* matrix;
};
constexpr Vision kNormal = {"normal", nullptr};
constexpr Vision kProtanVision = {"protan", &kProtan};
constexpr Vision kDeutanVision = {"deutan", &kDeutan};
constexpr Vision kTritanVision = {"tritan", &kTritan};

Rgb seen_with(const Vision& vision, const Rgb& c) {
  if (vision.matrix == nullptr) {
    return c;
  }
  const Matrix& m = *vision.matrix;
  Rgb seen{};
  for (int i = 0; i < 3; ++i) {
    const double v = m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2];
    seen[i] = std::clamp(v, 0.0, 1.0);
  }
  return seen;
}

// Item `index` as a TintedLuma point shows it, where the scene sets the
// lightness: its tint scaled to encoded luma 0.3, so that only the tint's
// chromaticity tells it from another. The luma is depth_cloud.vert's, Rec.
// 709's Y' on the encoded values.
Rgb linear_at_equal_luma(std::size_t index) {
  const uint32_t word = categorical_rgba8(index);
  Rgb encoded{};
  for (int i = 0; i < 3; ++i) {
    encoded[i] = static_cast<double>((word >> (8 * i)) & 0xFFu) / 255.0;
  }
  const double luma =
      0.2126 * encoded[0] + 0.7152 * encoded[1] + 0.0722 * encoded[2];
  Rgb rgb{};
  for (int i = 0; i < 3; ++i) {
    rgb[i] = vg_test::srgb_to_linear(encoded[i] * 0.3 / luma);
  }
  return rgb;
}

// The smallest distance between two of the first `count` items, each as
// `lab_of(index)` gives it.
template <typename LabOf>
double min_distance_of(std::size_t count, LabOf lab_of) {
  double smallest = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = i + 1; j < count; ++j) {
      smallest = std::min(smallest, distance(lab_of(i), lab_of(j)));
    }
  }
  return smallest;
}

TEST(CategoricalColorTest, FirstColorsAreOkabeAndItosWithoutBlack) {
  // In Okabe and Ito's order, but with yellow and blue swapped.
  constexpr uint32_t kOkabeIto[] = {
      pack_rgba8(0xE6, 0x9F, 0x00),  // orange
      pack_rgba8(0x56, 0xB4, 0xE9),  // sky blue
      pack_rgba8(0x00, 0x9E, 0x73),  // bluish green
      pack_rgba8(0x00, 0x72, 0xB2),  // blue
      pack_rgba8(0xF0, 0xE4, 0x42),  // yellow
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
      {7, 0xFF988BF2u},    {8, 0xFF83C46Fu},       {9, 0xFFF9A1AAu},
      {10, 0xFF519CE4u},   {11, 0xFFD0C516u},      {12, 0xFFC78EE2u},
      {1007, 0xFFAA8BEEu}, {1000007, 0xFF928CF2u},
  };
  for (const auto& [index, word] : pinned) {
    EXPECT_EQ(categorical_rgba8(index), word) << "item " << index;
  }
  if (sizeof(std::size_t) == 8) {
    EXPECT_EQ(categorical_rgba8(kLast), 0xFFB68CEAu);
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
    const Oklab lab = oklab(kCategoricalPaletteSize + n);
    EXPECT_NEAR(lab.L, 0.75, 0.005) << "item " << kCategoricalPaletteSize + n;
    EXPECT_NEAR(std::hypot(lab.a, lab.b), 0.125, 0.005)
        << "item " << kCategoricalPaletteSize + n;
    const double hue = std::atan2(lab.b, lab.a) * 180.0 / kPi;
    const double off =
        std::remainder(hue - 12.5 - golden * static_cast<double>(n), 360.0);
    EXPECT_LT(std::abs(off), 2.0) << "item " << kCategoricalPaletteSize + n;
  }
  for (const std::size_t index : {std::size_t{1000007}, kLast}) {
    const Oklab lab = oklab(index);
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
  for (const Vision& vision : {kProtanVision, kDeutanVision, kTritanVision}) {
    const double smallest =
        min_distance_of(kCategoricalPaletteSize, [&](std::size_t i) {
          return oklab_of(seen_with(vision, linear_of(categorical_rgba8(i))));
        });
    EXPECT_GE(smallest, 0.07) << vision.name;
  }
}

TEST(CategoricalColorTest, ColorsStayApartAtEqualLuma) {
  const auto smallest = [](std::size_t count, const Vision& vision) {
    return min_distance_of(count, [&](std::size_t i) {
      return oklab_of(seen_with(vision, linear_at_equal_luma(i)));
    });
  };
  EXPECT_GE(smallest(12, kNormal), 0.03);
  // The limit the header states: by 13, two are within 0.02.
  EXPECT_LT(smallest(13, kNormal), 0.02);
  // A rig of 4 keeps its cameras apart in TintedLuma, colour-blind or not.
  for (const Vision& vision : {kNormal, kProtanVision, kDeutanVision}) {
    EXPECT_GE(smallest(4, vision), 0.05) << vision.name;
  }
}

TEST(CategoricalColorTest, FirstColorsAreApart) {
  EXPECT_GE(min_distance(0, 10, 10), 0.08);
  EXPECT_GE(min_distance(0, 16, 16), 0.04);
  EXPECT_GE(min_distance(0, 24, 24), 0.025);
  // The limit the header states: by 29 colours, two are within 0.02.
  EXPECT_LT(min_distance(0, 29, 29), 0.02);
}

TEST(CategoricalColorTest, AnyEightConsecutiveAreApart) {
  for (const std::size_t first :
       {std::size_t{0}, std::size_t{1000000}, kLast - 63}) {
    EXPECT_GE(min_distance(first, 64, 8), 0.06) << "from item " << first;
  }
}

}  // namespace

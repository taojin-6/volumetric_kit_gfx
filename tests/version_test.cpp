// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <string>

#include "volumetric_kit/gfx/version.hpp"

namespace vg = volumetric_kit::gfx;

// The version string must be the dotted composition of the numeric components,
// so a mismatch (e.g. a stale generated header) is caught immediately.
TEST(Version, ComponentsComposeString) {
  const std::string expected = std::to_string(vg::version_major()) + "." +
                               std::to_string(vg::version_minor()) + "." +
                               std::to_string(vg::version_patch());
  EXPECT_EQ(expected, std::string(vg::version_string()));
}

TEST(Version, ComponentsAreNonNegative) {
  EXPECT_GE(vg::version_major(), 0);
  EXPECT_GE(vg::version_minor(), 0);
  EXPECT_GE(vg::version_patch(), 0);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file glm_test_util.hpp
/// gtest checks on glm vectors.

#include <gtest/gtest.h>

#include <glm/vec2.hpp>

namespace vg_test {

// Each component of `actual` within `eps` of `expected`'s.
inline void expect_near(glm::vec2 actual, glm::vec2 expected,
                        float eps = 1e-4f) {
  EXPECT_NEAR(actual.x, expected.x, eps);
  EXPECT_NEAR(actual.y, expected.y, eps);
}

}  // namespace vg_test

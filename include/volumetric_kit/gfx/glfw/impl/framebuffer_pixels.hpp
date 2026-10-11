// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file glfw/impl/framebuffer_pixels.hpp
/// The arithmetic of `glfw::cursor_in_framebuffer`, apart from GLFW so that
/// a high-DPI scale can be tested. Not a public header.

#include <glm/vec2.hpp>

namespace volumetric_kit::gfx::glfw::detail {

// A position in window coordinates, scaled per axis by the framebuffer's size
// over the window's. An axis the window has no size on is left unscaled.
inline glm::vec2 to_framebuffer_pixels(glm::dvec2 position,
                                       glm::ivec2 window_dims,
                                       glm::ivec2 framebuffer_dims) {
  glm::vec2 scale{1.0f, 1.0f};
  for (int axis = 0; axis < 2; ++axis) {
    if (window_dims[axis] > 0) {
      scale[axis] = static_cast<float>(framebuffer_dims[axis]) /
                    static_cast<float>(window_dims[axis]);
    }
  }
  return glm::vec2(position) * scale;
}

}  // namespace volumetric_kit::gfx::glfw::detail

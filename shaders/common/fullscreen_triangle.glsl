// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// One triangle that covers the whole viewport, drawn as three vertices with no
// vertex buffer.

#ifndef VG_COMMON_FULLSCREEN_TRIANGLE_GLSL
#define VG_COMMON_FULLSCREEN_TRIANGLE_GLSL

// The NDC position of vertex `index` (0, 1 or 2; gl_VertexIndex): (-1, -1),
// (3, -1) and (-1, 3), whose triangle contains the [-1, 1] square.
vec2 fullscreen_triangle_ndc(int index) {
  return vec2((index << 1) & 2, index & 2) * 2.0 - 1.0;
}

#endif  // VG_COMMON_FULLSCREEN_TRIANGLE_GLSL

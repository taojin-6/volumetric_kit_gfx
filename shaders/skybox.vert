// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Full-screen triangle (no vertex buffer), placed at the far plane (z == w).
// The skybox pipeline runs with depth test and write disabled, so this depth is
// never tested or stored; the model occludes the sky purely by draw order
// (skybox first, model second). The NDC position is handed to the fragment
// stage, which turns it into a world-space view ray to sample the environment
// cube.

#extension GL_GOOGLE_include_directive : require
#include "common/fullscreen_triangle.glsl"

layout(location = 0) out vec2 ndc;

void main() {
  ndc = fullscreen_triangle_ndc(gl_VertexIndex);
  gl_Position = vec4(ndc, 1.0, 1.0);
}

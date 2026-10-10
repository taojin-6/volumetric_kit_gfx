// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// ImagePipeline's convert pass, vertex stage: one triangle that covers the
// whole target, from gl_VertexIndex alone (no vertex buffer). The fragment
// stage reads its texel position from gl_FragCoord, so nothing is passed on.

#extension GL_GOOGLE_include_directive : require
#include "common/fullscreen_triangle.glsl"

void main() {
  gl_Position = vec4(fullscreen_triangle_ndc(gl_VertexIndex), 0.0, 1.0);
}

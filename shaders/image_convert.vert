// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// ImagePipeline's convert pass, vertex stage: one triangle that covers the
// whole target, from gl_VertexIndex alone (no vertex buffer). The fragment
// stage reads its texel position from gl_FragCoord, so nothing is passed on.

void main() {
  vec2 corner = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}

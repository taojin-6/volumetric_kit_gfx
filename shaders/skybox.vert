// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Full-screen triangle (no vertex buffer): three oversized clip-space corners
// emitted from gl_VertexIndex, placed at the far plane (z == w). The skybox
// pipeline runs with depth test and write disabled, so this depth is never
// tested or stored; the model occludes the sky purely by draw order (skybox
// first, model second). The NDC position is handed to the fragment stage, which
// turns it into a world-space view ray to sample the environment cube.

layout(location = 0) out vec2 ndc;

void main() {
  // (0,0), (2,0), (0,2) -> NDC (-1,-1), (3,-1), (-1,3): covers the viewport.
  const vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  ndc = p * 2.0 - 1.0;
  gl_Position = vec4(ndc, 1.0, 1.0);
}

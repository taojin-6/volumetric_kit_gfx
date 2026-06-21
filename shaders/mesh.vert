// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Vertex-buffer hello-mesh: position and color come from a bound vertex buffer
// (locations 0 and 1), so the draw supplies geometry rather than computing it
// from gl_VertexIndex. Positions are already in clip space (w = 1) -- the depth
// half of the pipeline exercises gl_Position.z without needing a transform.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;

void main() {
  gl_Position = vec4(in_position, 1.0);
  frag_color = in_color;
}

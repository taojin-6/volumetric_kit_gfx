// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Vertex-buffer mesh with an MVP uniform: like mesh.vert, but transforms the
// position by a model-view-projection matrix from descriptor set 0, binding 0.
// The pipeline reflects this uniform buffer into its descriptor-set layout
// automatically (no hand-written layout).

layout(set = 0, binding = 0) uniform Mvp {
  mat4 mvp;
}
u;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;

void main() {
  gl_Position = u.mvp * vec4(in_position, 1.0);
  frag_color = in_color;
}

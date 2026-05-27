// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// The hello-triangle: three vertices and their colors are indexed by
// gl_VertexIndex, so the draw is a bare vkCmdDraw(cmd, 3, 1, 0, 0) with an empty
// pipeline layout -- no vertex buffers and no descriptor sets.

layout(location = 0) out vec3 frag_color;

vec2 positions[3] =
    vec2[](vec2(0.0, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));

vec3 colors[3] =
    vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));

void main() {
  gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
  frag_color = colors[gl_VertexIndex];
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Object-space mesh with a per-draw model + MVP matrix as a push constant (the
// pipeline reflects the range automatically). Positions go to clip space via the
// MVP; the normal and tangent are carried to world space for shading (the normal
// through the inverse-transpose so non-uniform scale still lights right), along
// with the world-space position (for the view vector) and the primary UV.

layout(push_constant) uniform Push {
  mat4 mvp;    // projection * view * model (clip-space transform)
  mat4 model;  // model -> world
}
pc;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_tangent;  // xyz tangent, w handedness (glTF)

layout(location = 0) out vec3 frag_normal;     // world-space, unnormalized
layout(location = 1) out vec2 frag_uv;         // primary texture coordinates
layout(location = 2) out vec3 frag_world_pos;  // world-space position
layout(location = 3) out vec4 frag_tangent;    // xyz world tangent, w handedness

void main() {
  const vec4 world = pc.model * vec4(in_position, 1.0);
  gl_Position = pc.mvp * vec4(in_position, 1.0);
  frag_normal = transpose(inverse(mat3(pc.model))) * in_normal;
  frag_tangent = vec4(mat3(pc.model) * in_tangent.xyz, in_tangent.w);
  frag_uv = in_uv;
  frag_world_pos = world.xyz;
}

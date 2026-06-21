// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Object-space mesh with a per-draw model + MVP matrix supplied as a push
// constant (no descriptor set): the pipeline reflects the push-constant range
// from this block automatically. Positions are transformed to clip space by the
// MVP; the object-space normal is carried to world space for shading, using the
// inverse-transpose of the model's 3x3 so non-uniform scale still lights right.

layout(push_constant) uniform Push {
  mat4 mvp;    // projection * view * model (clip-space transform)
  mat4 model;  // model -> world (for the world-space normal)
}
pc;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 frag_normal;  // world-space, unnormalized

void main() {
  gl_Position = pc.mvp * vec4(in_position, 1.0);
  frag_normal = transpose(inverse(mat3(pc.model))) * in_normal;
}

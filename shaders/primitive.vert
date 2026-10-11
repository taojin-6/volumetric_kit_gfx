// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// PrimitivePipeline, vertex stage: a world-space point or line vertex through
// the draw's view-projection. Both the point and the line pipeline use it; a
// line ignores the point size.

layout(location = 0) in vec3 in_position;  // world space
layout(location = 1) in vec4 in_color;     // sRGB-encoded RGBA8, as UNORM

// Mirrors PushConstants in primitive_pipeline.cpp (std430, 68 bytes).
layout(push_constant) uniform Push {
  mat4 view_proj;    // projection * view
  float point_size;  // pixels, already clamped to the device's range
}
pc;

layout(location = 0) out vec3 frag_color;

void main() {
  gl_Position = pc.view_proj * vec4(in_position, 1.0);
  gl_PointSize = pc.point_size;
  frag_color = in_color.rgb;
}

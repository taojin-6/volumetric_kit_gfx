// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// ImagePipeline's draw, vertex stage: one rectangle of the target, as two
// triangles built from gl_VertexIndex, with the texture coordinates of the
// part of the image it shows. A list rather than a strip: MoltenVK cannot turn
// primitive restart off for a strip, and says so on every pipeline. The host clips the rectangle
// to the draw's viewport, so it never spans more than the target.

// Mirrors DrawPush in image_pipeline.cpp (std430, 40 bytes).
layout(push_constant) uniform Push {
  vec2 target_size;  // the render target, in pixels
  vec2 rect_min;     // the rectangle's top-left corner, in target pixels
  vec2 rect_max;     // its bottom-right corner
  vec2 uv_min;       // the image coordinate at rect_min, normalized
  vec2 uv_max;       // the image coordinate at rect_max, normalized
}
pc;

layout(location = 0) out vec2 frag_uv;

const vec2 kCorners[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
                                 vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0));

void main() {
  vec2 corner = kCorners[gl_VertexIndex];
  vec2 position = mix(pc.rect_min, pc.rect_max, corner);
  gl_Position = vec4(position / pc.target_size * 2.0 - 1.0, 0.0, 1.0);
  frag_uv = mix(pc.uv_min, pc.uv_max, corner);
}

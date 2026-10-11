// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// PrimitivePipeline, fragment stage: the vertex color, opaque. The color is
// sRGB-encoded and interpolated as encoded, so a line shades alike on every
// target.

#extension GL_GOOGLE_include_directive : require
#include "common/srgb.glsl"

// Supplied by PrimitivePipeline::create: whether the target holds linear
// light. An _SRGB target, which encodes on write, and a float target are
// given the decoded color; a UNORM target stores the encoded bytes as given.
layout(constant_id = 0) const bool kLinearTarget = false;

layout(location = 0) in vec3 frag_color;
layout(location = 0) out vec4 out_color;

void main() {
  out_color =
      vec4(kLinearTarget ? srgb_to_linear(frag_color) : frag_color, 1.0);
}

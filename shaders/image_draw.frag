// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// ImagePipeline's draw, fragment stage: samples the display image, whose
// _SRGB view decodes to linear light. The sampler minifies through the mip
// chain (trilinear), with the level chosen per pixel from the screen-space
// derivatives of frag_uv -- so the right level follows the drawn size through
// window resizes, zoom and display scaling with nothing for the host to pick.
// It magnifies with the filter the draw asked for.

#extension GL_GOOGLE_include_directive : require
#include "common/srgb.glsl"

layout(set = 0, binding = 0) uniform sampler2D display;

// Supplied by ImagePipeline::create: whether this stage encodes sRGB itself.
// A UNORM target is given the encoding, so it stores the image's own sRGB
// bytes as an _SRGB target (which encodes on write) does; a float target
// holds linear light, and so is given the samples as they are.
layout(constant_id = 0) const bool kEncodeSrgb = true;

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

void main() {
  vec3 color = texture(display, frag_uv).rgb;
  out_color = vec4(kEncodeSrgb ? linear_to_srgb(color) : color, 1.0);
}

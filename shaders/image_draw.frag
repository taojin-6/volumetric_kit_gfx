// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// ImagePipeline's draw, fragment stage: samples the display image, whose
// _SRGB view decodes to linear light. The sampler minifies through the mip
// chain (trilinear), with the level chosen per pixel from the screen-space
// derivatives of frag_uv -- so the right level follows the drawn size through
// window resizes, zoom and display scaling with nothing for the host to pick.
// It magnifies with the filter the draw asked for.

layout(set = 0, binding = 0) uniform sampler2D display;

// Supplied by ImagePipeline::create: whether color attachment 0 encodes sRGB
// on write. A UNORM target is given the encoding itself, so either kind of
// target stores the image's own sRGB bytes.
layout(constant_id = 0) const bool kSrgbTarget = false;

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

// The sRGB encode (IEC 61966-2-1).
vec3 linear_to_srgb(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
             greaterThan(c, vec3(0.0031308)));
}

void main() {
  vec3 color = texture(display, frag_uv).rgb;
  out_color = vec4(kSrgbTarget ? color : linear_to_srgb(color), 1.0);
}

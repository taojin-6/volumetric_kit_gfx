// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The sRGB transfer curve (IEC 61966-2-1), exact and piecewise -- the curve an
// _SRGB format decodes when sampled and encodes when written, so a shader that
// converts by hand agrees with one that leaves it to the format.
// tests/shader_common_test.cpp checks it against the reference curve.
//
// TODO: include volumetric_kit_core's sRGB curve, shared with recon's
// color_common.glsl and color_space.hpp, once the core has one.

#ifndef VG_COMMON_SRGB_GLSL
#define VG_COMMON_SRGB_GLSL

// The decode: sRGB-encoded values to linear light.
vec3 srgb_to_linear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)),
             greaterThan(c, vec3(0.04045)));
}

// The encode: linear light to sRGB-encoded values, the inverse of
// srgb_to_linear.
vec3 linear_to_srgb(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
             greaterThan(c, vec3(0.0031308)));
}

#endif  // VG_COMMON_SRGB_GLSL

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Tone mapping from linear HDR to the displayable [0, 1].

#ifndef VG_COMMON_TONEMAP_GLSL
#define VG_COMMON_TONEMAP_GLSL

// Narkowicz's fit of the ACES filmic curve, on linear values.
vec3 tonemap_aces(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

#endif  // VG_COMMON_TONEMAP_GLSL

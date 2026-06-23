// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Sample the environment cubemap along the per-pixel world-space view ray. The
// ray is reconstructed by unprojecting the far-plane NDC position through the
// inverse view-projection (the same matrix the model is drawn with, so the sky
// stays registered with the scene) and pointing it away from the camera. The
// cube stores linear HDR color; this tone-maps it before output (the sRGB target
// encodes on write).

layout(location = 0) in vec2 ndc;

layout(push_constant) uniform Push {
  mat4 inv_view_proj;
  vec4 camera_pos;  // .xyz world-space eye
}
pc;

layout(set = 0, binding = 0) uniform samplerCube sky;

layout(location = 0) out vec4 out_color;

// Narkowicz ACES filmic tone-mapping approximation (operates on linear HDR).
// Duplicated in model.frag (no shader #include path) -- keep in sync.
vec3 tonemap_aces(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  // Unproject the far-plane NDC point to world space. A perspective
  // view-projection yields world.w > 0 here, so the perspective divide is safe;
  // this skybox assumes such a projection (not an orthographic/oblique one).
  const vec4 world = pc.inv_view_proj * vec4(ndc, 1.0, 1.0);
  const vec3 dir = normalize(world.xyz / world.w - pc.camera_pos.xyz);
  out_color = vec4(tonemap_aces(texture(sky, dir).rgb), 1.0);
}

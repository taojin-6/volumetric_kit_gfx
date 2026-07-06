// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Reconstruction hybrid mesh, fragment stage. Albedo comes from one of two
// sources per triangle: the projective-texturing atlas (sampled at uv0) where a
// triangle won a camera, or the interpolated per-vertex color (the TSDF
// fallback) where uv0 is the (-1, -1) sentinel. Optionally lit by a single
// world-space directional light plus a constant ambient term (kFlagLit); unlit
// passes the albedo straight through. Two-sided: the normal is flipped for back
// faces since the pipeline does not cull.

layout(location = 0) in vec3 frag_normal;  // world space
layout(location = 1) in vec2 frag_uv;      // atlas uv, or (< 0) = vertex color
layout(location = 2) in vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D atlas_tex;

layout(push_constant) uniform Push {
  mat4 view_proj;
  vec4 light;  // xyz world-space light direction; w > 0.5 = apply shading
}
pc;

layout(location = 0) out vec4 out_color;

void main() {
  // uv0 < 0 is the "no atlas coordinate" sentinel -> use the vertex color.
  vec3 albedo =
      frag_uv.x < 0.0 ? frag_color.rgb : texture(atlas_tex, frag_uv).rgb;

  // light.w > 0.5 requests lit shading; otherwise pass the albedo through flat.
  if (pc.light.w > 0.5) {
    const vec3 n = normalize(gl_FrontFacing ? frag_normal : -frag_normal);
    const vec3 l = normalize(pc.light.xyz);
    const float ambient = 0.25;
    albedo *= ambient + (1.0 - ambient) * max(dot(n, l), 0.0);
  }

  out_color = vec4(albedo, 1.0);
}

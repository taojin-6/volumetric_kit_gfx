// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Reconstruction hybrid mesh, fragment stage. Albedo comes from one of two
// sources, chosen by the `flat` selector the vertex stage resolved from uv0: the
// projective-texturing atlas (sampled at uv0) where a triangle won a camera, or
// the interpolated per-vertex color (the TSDF fallback) otherwise. The atlas is
// sampled unconditionally -- in uniform control flow, so its implicit-LOD screen
// derivatives are always well defined -- and the unused result is simply not
// selected. Optionally lit by a single world-space directional light plus a
// constant ambient term (light.w > 0.5, i.e. the kHybridMeshLit flag); unlit
// passes the albedo straight through. Two-sided: the normal is flipped for back
// faces since the pipeline does not cull.

layout(location = 0) in vec3 frag_normal;  // world space
layout(location = 1) in vec2 frag_uv;      // atlas uv (0,0 on the vertex-color path)
layout(location = 2) in vec4 frag_color;
layout(location = 3) flat in uint frag_use_vertex_color;

layout(set = 0, binding = 0) uniform sampler2D atlas_tex;

layout(push_constant) uniform Push {
  mat4 view_proj;
  vec4 light;  // xyz world-space direction TO the light (unit); w > 0.5 = shade
}
pc;

layout(location = 0) out vec4 out_color;

void main() {
  // Sample the atlas unconditionally (uniform control flow keeps the LOD
  // derivatives defined), then select the source the vertex stage resolved.
  vec3 atlas_albedo = texture(atlas_tex, frag_uv).rgb;
  vec3 albedo = frag_use_vertex_color != 0u ? frag_color.rgb : atlas_albedo;

  // light.w > 0.5 requests lit shading; otherwise pass the albedo through flat.
  if (pc.light.w > 0.5) {
    vec3 raw = gl_FrontFacing ? frag_normal : -frag_normal;
    // Guard the normalize: a zero/degenerate interpolated normal would yield a
    // NaN that max() does not reliably clamp. Fall back to pure ambient (n = 0
    // makes the diffuse term vanish).
    vec3 n = dot(raw, raw) > 0.0 ? normalize(raw) : vec3(0.0);
    // pc.light.xyz is the pre-normalized world-space direction to the light.
    const float ambient = 0.25;
    albedo *= ambient + (1.0 - ambient) * max(dot(n, pc.light.xyz), 0.0);
  }

  out_color = vec4(albedo, 1.0);
}

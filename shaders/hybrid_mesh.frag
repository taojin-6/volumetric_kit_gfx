// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Reconstruction hybrid mesh, fragment stage. Albedo comes from one of two
// sources, chosen by the `flat` selector the vertex stage resolved from uv0: the
// projective-texturing atlas (sampled at uv0) where a triangle won a camera, or
// the interpolated per-vertex color (the TSDF fallback) otherwise. Whenever
// albedo is shown the atlas is sampled unconditionally -- in uniform control
// flow, so its implicit-LOD screen derivatives are always well defined -- and
// the unused result is simply not selected. The kHybridMeshVertexColor flag
// selects the vertex color everywhere (HybridMeshPipeline::submit sets it when
// it binds its fallback for a frame with no atlas). Optionally lit by a single
// world-space directional light plus a constant ambient term (the
// kHybridMeshLit flag); unlit passes the albedo straight through. Two-sided:
// the normal is flipped for back faces since the pipeline does not cull. The
// kHybridMeshNormals debug view replaces all of that with the world-space
// normal encoded as a color.

layout(location = 0) in vec3 frag_normal;  // world space
layout(location = 1) in vec2 frag_uv;      // atlas uv (0,0 on the vertex-color path)
layout(location = 2) in vec4 frag_color;
layout(location = 3) flat in uint frag_use_vertex_color;

layout(set = 0, binding = 0) uniform sampler2D atlas_tex;

layout(push_constant) uniform Push {
  mat4 view_proj;
  vec3 light_dir;  // world-space direction TO the light (unit)
  uint flags;      // HybridMeshFlags
}
pc;

// Supplied by HybridMeshPipeline::create: the HybridMeshFlags bits, taken from
// the host enum so it is their one definition (a 0 here switches the mode off),
// and whether color attachment 0 encodes sRGB on write.
layout(constant_id = 0) const uint kFlagLit = 0u;      // kHybridMeshLit
layout(constant_id = 1) const uint kFlagNormals = 0u;  // kHybridMeshNormals
layout(constant_id = 2) const bool kSrgbTarget = false;
layout(constant_id = 3) const uint kFlagVertexColor = 0u;  // kHybridMeshVertexColor

layout(location = 0) out vec4 out_color;

// normalize() that maps a zero/degenerate vector to 0 instead of the NaN it
// would otherwise yield (which max() does not reliably clamp).
vec3 safe_normalize(vec3 v) {
  return dot(v, v) > 0.0 ? normalize(v) : vec3(0.0);
}

// The sRGB decode (IEC 61966-2-1): an sRGB target encodes what is written, so
// writing srgb_to_linear(c) stores c itself.
vec3 srgb_to_linear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)),
             greaterThan(c, vec3(0.04045)));
}

void main() {
  // Normal debug view: the mesh's own normal, deliberately NOT flipped for back
  // faces, so a region whose normals point the wrong way reads as a color jump
  // instead of being masked the way lit shading masks it. (Winding alone does
  // not change this output; lit shading is what shows that.) A zero normal
  // encodes to mid-grey. `flags` is a push constant, so this branch is
  // dynamically uniform and the atlas sample below stays in uniform control
  // flow.
  if ((pc.flags & kFlagNormals) != 0u) {
    vec3 encoded = safe_normalize(frag_normal) * 0.5 + 0.5;
    out_color = vec4(kSrgbTarget ? srgb_to_linear(encoded) : encoded, 1.0);
    return;
  }

  // Sample the atlas unconditionally (uniform control flow keeps the LOD
  // derivatives defined), then select the source the vertex stage resolved,
  // or the vertex color everywhere when the frame asks for it.
  vec3 atlas_albedo = texture(atlas_tex, frag_uv).rgb;
  bool use_vertex_color =
      frag_use_vertex_color != 0u || (pc.flags & kFlagVertexColor) != 0u;
  vec3 albedo = use_vertex_color ? frag_color.rgb : atlas_albedo;

  // The lit flag requests lit shading; otherwise pass the albedo through flat.
  if ((pc.flags & kFlagLit) != 0u) {
    // A degenerate normal falls back to pure ambient (n = 0 makes the diffuse
    // term vanish).
    vec3 n = safe_normalize(gl_FrontFacing ? frag_normal : -frag_normal);
    // pc.light_dir is the pre-normalized world-space direction to the light.
    const float ambient = 0.25;
    albedo *= ambient + (1.0 - ambient) * max(dot(n, pc.light_dir), 0.0);
  }

  out_color = vec4(albedo, 1.0);
}

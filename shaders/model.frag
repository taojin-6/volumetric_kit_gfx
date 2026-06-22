// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Base-color (albedo) shading: the mesh's base-color texture lit by a key
// directional light plus ambient and a crude hemispherical sky/ground fill. The
// base color is sampled from a combined image/sampler (set 0, binding 0) that
// the pipeline reflects automatically; the example binds the material's
// base-color map there, or a 1x1 white fallback for an untextured material.
// Two-sided -- the normal is flipped for back faces -- because the pipeline does
// not cull, so interiors of an open mesh still light sensibly.
// TODO: full metallic-roughness PBR -- base_color_factor, the metallic/
// roughness/normal/occlusion/emissive maps, and IBL -- arrives with the PBR
// spine; this samples base color only.

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;

layout(set = 0, binding = 0) uniform sampler2D base_color;

layout(location = 0) out vec4 out_color;

void main() {
  vec3 geo_n = normalize(frag_normal);  // geometric world-space normal
  // Two-sided key light: flip the shading normal toward the viewer for back
  // faces (the pipeline does not cull).
  vec3 n = gl_FrontFacing ? geo_n : -geo_n;

  const vec3 light_dir = normalize(vec3(0.5, 0.8, 0.6));  // world-space key
  // Base color is stored sRGB-encoded; the sampler returns it linearized.
  vec3 albedo = texture(base_color, frag_uv).rgb;

  float key = max(dot(n, light_dir), 0.0);  // directional diffuse
  // Sky/ground ambient from the geometric orientation, so a surface's up/down
  // reads the same viewed from either side.
  float hemi = 0.5 + 0.5 * geo_n.y;

  // Output is linear; an sRGB swapchain encodes it on write.
  vec3 color = albedo * (0.12 + 0.70 * key + 0.18 * hemi);
  out_color = vec4(color, 1.0);
}

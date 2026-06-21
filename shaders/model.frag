// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Untextured normal shading so a loaded mesh's form reads: a key directional
// light plus ambient and a crude hemispherical sky/ground fill, over a neutral
// base color. Two-sided -- the normal is flipped for back faces -- because the
// pipeline does not cull, so interiors of an open mesh still light sensibly.
// TODO: PBR + material-texture shading (the IBL spine); this is the first-look,
// untextured normal shading.

layout(location = 0) in vec3 frag_normal;

layout(location = 0) out vec4 out_color;

void main() {
  vec3 geo_n = normalize(frag_normal);  // geometric world-space normal
  // Two-sided key light: flip the shading normal toward the viewer for back
  // faces (the pipeline does not cull).
  vec3 n = gl_FrontFacing ? geo_n : -geo_n;

  const vec3 light_dir = normalize(vec3(0.5, 0.8, 0.6));  // world-space key
  const vec3 base = vec3(0.80, 0.78, 0.76);

  float key = max(dot(n, light_dir), 0.0);  // directional diffuse
  // Sky/ground ambient from the geometric orientation, so a surface's up/down
  // reads the same viewed from either side.
  float hemi = 0.5 + 0.5 * geo_n.y;

  // Output is linear; an sRGB swapchain encodes it on write.
  vec3 color = base * (0.12 + 0.70 * key + 0.18 * hemi);
  out_color = vec4(color, 1.0);
}

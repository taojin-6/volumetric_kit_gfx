// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Untextured normal shading so a loaded mesh's form reads: a key directional
// light plus ambient and a crude hemispherical sky/ground fill, over a neutral
// base color. Two-sided -- the normal is flipped for back faces -- because the
// pipeline does not cull, so interiors of an open mesh still light sensibly.
// PBR + material textures arrive with the IBL spine; this is the first look.

layout(location = 0) in vec3 frag_normal;

layout(location = 0) out vec4 out_color;

void main() {
  vec3 n = normalize(frag_normal);
  if (!gl_FrontFacing) {
    n = -n;
  }

  const vec3 light_dir = normalize(vec3(0.5, 0.8, 0.6));  // world-space key
  const vec3 base = vec3(0.80, 0.78, 0.76);

  float key = max(dot(n, light_dir), 0.0);  // directional diffuse
  float hemi = 0.5 + 0.5 * n.y;             // sky (up) brighter than ground

  // Output is linear; an sRGB swapchain encodes it on write.
  vec3 color = base * (0.12 + 0.70 * key + 0.18 * hemi);
  out_color = vec4(color, 1.0);
}

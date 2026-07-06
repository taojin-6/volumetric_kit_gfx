// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Reconstruction hybrid mesh, vertex stage. The interleaved vertices arrive in
// WORLD space (as the reconstruction mesh tier emits them), so the pipeline
// applies only the camera's view-projection -- there is no per-mesh model
// matrix. Each vertex's uv0 is either a valid atlas coordinate (projective
// texturing won that triangle a camera) or the (-1, -1) sentinel meaning "use
// the per-vertex color" (the TSDF vertex-color fallback); the fragment stage
// branches on it. The mesh emits independent triangles, so all three vertices of
// a triangle share the same sentinel -- the interpolated uv0 never mixes an
// atlas coordinate with the sentinel.

layout(location = 0) in vec3 in_position;  // world space
layout(location = 1) in vec3 in_normal;    // world space
layout(location = 2) in vec2 in_uv0;       // atlas uv, or (< 0) = vertex color
layout(location = 3) in vec4 in_color;     // per-vertex color (TSDF fallback)

layout(push_constant) uniform Push {
  mat4 view_proj;  // projection * view (vertices are already world-space)
  vec4 light;      // xyz world-space light direction; w > 0.5 = apply shading
}
pc;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out vec4 frag_color;

void main() {
  gl_Position = pc.view_proj * vec4(in_position, 1.0);
  frag_normal = in_normal;
  frag_uv = in_uv0;
  frag_color = in_color;
}

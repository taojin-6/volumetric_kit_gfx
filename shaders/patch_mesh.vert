// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Reconstruction patch-atlas mesh, vertex stage. Interleaved vertices arrive in
// WORLD space (as the reconstruction mesh tier emits them), so this applies only
// the camera's view-projection -- no per-mesh model matrix.
//
// The one thing it does that the hybrid pipeline's vertex stage does not is
// forward the world position. The fragment stage needs it to recover the
// fragment's barycentric coordinate inside its triangle, which is how a
// per-TRIANGLE patch is addressed without a per-vertex uv -- and a per-vertex uv
// is exactly what a patch atlas cannot have, since a vertex shared between six
// triangles would need six of them.
//
// uv0 is not read at all here. A mesh drawn through this pipeline carries its
// colour in the atlas, not in a texture coordinate.

layout(location = 0) in vec3 in_position;  // world space
layout(location = 1) in vec3 in_normal;    // world space
layout(location = 2) in vec4 in_color;     // per-vertex colour (TSDF fallback)

layout(push_constant) uniform Push {
  mat4 view_proj;  // projection * view (vertices are already world-space)
  vec4 light;      // xyz world-space direction TO the light; w > 0.5 = shade
  uvec4 patch_shape;     // x = texels per patch leg, y = texels per patch, zw unused
}
pc;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec3 frag_world;
layout(location = 2) out vec4 frag_color;

void main() {
  gl_Position = pc.view_proj * vec4(in_position, 1.0);
  frag_normal = in_normal;
  frag_world = in_position;
  frag_color = in_color;
}

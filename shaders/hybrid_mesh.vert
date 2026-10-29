// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Reconstruction hybrid mesh, vertex stage. The interleaved vertices arrive in
// WORLD space (as the reconstruction mesh tier emits them), so the pipeline
// applies only the camera's view-projection -- there is no per-mesh model
// matrix. (World-space input is also what keeps the push block within the
// 128-byte guaranteed maxPushConstantsSize: a per-draw model + normal matrix
// alongside the light would not fit.)
//
// Each vertex's uv0 is either a valid atlas coordinate -- in [0, 1], or over
// two pictures uv0.y in [0, 2) (HybridMeshTextureSet) -- where projective
// texturing won that triangle a camera, or carries a negative x -- recon emits
// the (-1, -1) sentinel -- meaning "use the per-vertex color" (the TSDF
// vertex-color fallback). The choice is resolved HERE and forwarded as a `flat`
// selector, so the fragment stage never branches on an interpolated value: a
// triangle takes its provoking vertex's class as a whole. A triangle that mixes
// the two classes (which the mesh is not meant to emit) is then shaded
// uniformly rather than split mid-triangle by an interpolated uv crossing zero.
// Sentinel vertices forward an in-range uv (0, 0) so a fragment quad straddling
// an atlas/color seam never derives the atlas LOD from the out-of-range
// sentinel. The picture is resolved here too, from the decoded uv0.y: from 1
// up, picture 1, so a triangle takes its provoking vertex's picture as well.
//
// The tangent slot carries a second atlas view, (uv_B.x, uv_B.y, w_B, unused),
// which the fragment stage blends with uv0's by the weight w_B. It is
// interpolated like uv0, as one vec3: a mesh of small triangles draws
// measurably faster on Apple GPUs with one interpolant than with a vec2 and a
// float. The slot's placeholder, (1, 0, 0, 1), has w_B = 0: no second view.
// Only the pipeline for meshes that carry one reads frag_second
// (hybrid_mesh.frag's kSecondView).

layout(location = 0) in vec3 in_position;  // world space
layout(location = 1) in vec3 in_normal;    // world space
layout(location = 2) in vec2 in_uv0;       // atlas uv, or x<0 = vertex color
layout(location = 3) in vec4 in_color;     // per-vertex color (TSDF fallback)
layout(location = 4) in vec4 in_tangent;   // (uv_B, w_B, unused): second view

layout(push_constant) uniform Push {
  mat4 view_proj;  // projection * view (vertices are already world-space)
  vec3 light_dir;  // world-space direction TO the light (fragment stage)
  uint flags;      // HybridMeshFlags (fragment stage)
}
pc;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out vec4 frag_color;
// Bit 0: the vertex color; bit 1: picture 1.
layout(location = 3) flat out uint frag_select;
layout(location = 4) out vec3 frag_second;  // (uv_B, w_B): the second view

void main() {
  gl_Position = pc.view_proj * vec4(in_position, 1.0);
  frag_normal = in_normal;

  // Negative x means "no atlas for this vertex" -> vertex color.
  bool use_vertex_color = in_uv0.x < 0.0;
  // A negative uv0 CARRIES its atlas coordinate as -uv - 1, so decode rather
  // than substitute. Both forms are handled by the one expression: the classic
  // (-1, -1) sentinel decodes to (0, 0), which is exactly the stand-in this
  // used to substitute unconditionally, so a producer that emits it sees no
  // change at all.
  //
  // It matters on a triangle whose vertices disagree about class, which is what
  // a producer emits along the edge of a textured region once visibility is
  // decided per vertex (recon's projective texturer, on a shared-vertex mesh).
  // Such a triangle takes its provoking vertex's class for its whole face, so
  // when that class is "atlas" the uv still interpolates toward the other two
  // vertices -- and substituting (0, 0) for them dragged the interpolation to
  // the atlas origin, smearing the corner of the image across the face. With
  // the real coordinate carried through, the same triangle samples the
  // neighbourhood it actually covers and the error is bounded to texture
  // bleeding about one triangle past the silhouette.
  //
  // Still in range for the LOD derivative reason the substitution existed for:
  // a decoded uv is a real atlas coordinate, not the out-of-range sentinel.
  frag_uv = use_vertex_color ? (-in_uv0 - 1.0) : in_uv0;
  frag_select = (use_vertex_color ? 1u : 0u) | (frag_uv.y >= 1.0 ? 2u : 0u);
  frag_color = in_color;
  frag_second = in_tangent.xyz;
}

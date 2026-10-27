// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Reconstruction hybrid mesh, fragment stage. Albedo comes from one of two
// sources, chosen by the `flat` selector the vertex stage resolved from uv0: the
// projective-texturing atlas (sampled at uv0) where a triangle won a camera, or
// the interpolated per-vertex color (the TSDF fallback) otherwise. Whenever
// albedo is shown the atlas is sampled unconditionally -- in uniform control
// flow, so its implicit-LOD screen derivatives are always well defined -- and
// the unused result is simply not selected. Over an RGBA atlas, a second view,
// sampled at uv_B the same way, is blended in by its weight w_B (both from the
// tangent slot; see blend_views). The kHybridMeshVertexColor flag selects the
// vertex color everywhere (HybridMeshPipeline::submit sets it when it binds
// its fallback for a frame with no atlas). Optionally lit by a single
// world-space directional light plus a constant ambient term (the
// kHybridMeshLit flag); unlit passes the albedo straight through. Two-sided:
// the normal is flipped for back faces since the pipeline does not cull. The
// kHybridMeshNormals debug view replaces all of that with the world-space
// normal encoded as a color.

#extension GL_GOOGLE_include_directive : require
#include "common/srgb.glsl"
#include "common/ycbcr.glsl"

layout(location = 0) in vec3 frag_normal;  // world space
layout(location = 1) in vec2 frag_uv;      // atlas uv (0,0 on the vertex-color path)
layout(location = 2) in vec4 frag_color;
layout(location = 3) flat in uint frag_use_vertex_color;
layout(location = 4) in vec2 frag_uv_b;  // the second view's atlas uv
layout(location = 5) in float frag_w_b;  // its weight (0: none)

// Set 0, as HybridMeshTextureSet, StreamedAtlas and the pipeline's fallback
// write it (hybrid_mesh_set0.hpp): an RGBA image, or a camera picture's two
// Y'CbCr 4:2:0 planes, as the Picture uniform says.
layout(set = 0, binding = 0) uniform sampler2D plane0;  // RGBA, or the luma
layout(set = 0, binding = 1) uniform sampler2D plane1;  // the chroma, or unused
// Mirrors PictureUniform in hybrid_mesh_set0.hpp (std140, 32 bytes).
layout(set = 0, binding = 2) uniform Picture {
  uint two_plane;      // 0: plane0 is RGBA; 1: Y'CbCr 4:2:0 planes
  uint full_range;     // two-plane: Y' in 0..255 rather than 16..235
  float kr;            // two-plane: the matrix's red weight
  float kb;            // two-plane: the matrix's blue weight
  vec2 extent;         // two-plane: the picture's size in luma texels
  uint chroma_siting;  // two-plane: ChromaSiting
  uint padding;        // to std140's 16-byte multiple
}
picture;

layout(push_constant) uniform Push {
  mat4 view_proj;
  vec3 light_dir;  // world-space direction TO the light (unit)
  uint flags;      // HybridMeshFlags
}
pc;

// Supplied by HybridMeshPipeline::create: the HybridMeshFlags bits, taken from
// the host enum so it is their one definition (a 0 here switches the mode off),
// and whether color attachment 0 holds linear light. An _SRGB target, which
// encodes on write, and a float target are given linear light; a UNORM target
// stores what is written, so it is given the encoding.
layout(constant_id = 0) const uint kFlagLit = 0u;      // kHybridMeshLit
layout(constant_id = 1) const uint kFlagNormals = 0u;  // kHybridMeshNormals
layout(constant_id = 2) const bool kLinearTarget = false;
layout(constant_id = 3) const uint kFlagVertexColor = 0u;  // kHybridMeshVertexColor

layout(location = 0) out vec4 out_color;

// normalize() that maps a zero/degenerate vector to 0 instead of the NaN it
// would otherwise yield (which max() does not reliably clamp).
vec3 safe_normalize(vec3 v) {
  return dot(v, v) > 0.0 ? normalize(v) : vec3(0.0);
}

// Two atlas views of a fragment, `a` at uv0 and `b` at uv_B, mixed in linear
// light by the weight `w` (clamped to [0, 1]), each also weighted by its alpha,
// the atlas's coverage (linear in an _SRGB format, which decodes rgb alone):
//   ((1 - w) a.a a.rgb + w b.a b.rgb) / ((1 - w) a.a + w b.a).
// w = 0, which the tangent slot's (1, 0, 0, 1) placeholder carries, and a zero
// denominator give a.rgb exactly, so a mesh with no second view draws as if
// it had none.
vec3 blend_views(vec4 a, vec4 b, float w) {
  w = clamp(w, 0.0, 1.0);
  float wa = (1.0 - w) * a.a;
  float wb = w * b.a;
  float sum = wa + wb;
  return (w > 0.0 && sum > 0.0) ? (wa * a.rgb + wb * b.rgb) / sum : a.rgb;
}

// Albedo from set 0 at `uv`, in linear light. An RGBA image is sampled as its
// sampler says (an _SRGB one decodes); a camera picture's two planes are
// filtered at level 0 where they are, converted, then decoded from sRGB.
// `two_plane` is the same for the whole draw, so the branch does not diverge.
// Over an RGBA image, uv0's view is blended with the second view's.
vec3 atlas_albedo(vec2 uv) {
  if (picture.two_plane == 0u) {
    return blend_views(texture(plane0, uv), texture(plane0, frag_uv_b),
                       frag_w_b);
  }
  vec2 p = uv * picture.extent;
  float y = textureLod(plane0,
                       ycbcr_luma_uv(p, picture.extent,
                                     vec2(textureSize(plane0, 0))),
                       0.0)
                .r;
  vec2 cbcr = textureLod(plane1,
                         ycbcr_chroma_uv(p, picture.chroma_siting,
                                         picture.extent,
                                         vec2(textureSize(plane1, 0))),
                         0.0)
                  .rg;
  return srgb_to_linear(ycbcr_to_rgb(y, cbcr, picture.kr, picture.kb,
                                     picture.full_range != 0u));
}

void main() {
  // Normal debug view: the mesh's own normal, deliberately NOT flipped for back
  // faces, so a region whose normals point the wrong way reads as a color jump
  // instead of being masked the way lit shading masks it. (Winding alone does
  // not change this output; lit shading is what shows that.) A zero normal
  // encodes to mid-grey. The encoding is the color to show, so a target that
  // takes linear light is given srgb_to_linear(c), which an sRGB target stores
  // as c itself. `flags` is a push constant, so this branch is dynamically
  // uniform and the atlas sample below stays in uniform control flow.
  if ((pc.flags & kFlagNormals) != 0u) {
    vec3 encoded = safe_normalize(frag_normal) * 0.5 + 0.5;
    out_color = vec4(kLinearTarget ? srgb_to_linear(encoded) : encoded, 1.0);
    return;
  }

  // Sample the atlas unconditionally (uniform control flow keeps the LOD
  // derivatives defined), then select the source the vertex stage resolved,
  // or the vertex color everywhere when the frame asks for it.
  vec3 atlas = atlas_albedo(frag_uv);
  bool use_vertex_color =
      frag_use_vertex_color != 0u || (pc.flags & kFlagVertexColor) != 0u;
  vec3 albedo = use_vertex_color ? frag_color.rgb : atlas;

  // The lit flag requests lit shading; otherwise pass the albedo through flat.
  if ((pc.flags & kFlagLit) != 0u) {
    // A degenerate normal falls back to pure ambient (n = 0 makes the diffuse
    // term vanish).
    vec3 n = safe_normalize(gl_FrontFacing ? frag_normal : -frag_normal);
    // pc.light_dir is the pre-normalized world-space direction to the light.
    const float ambient = 0.25;
    albedo *= ambient + (1.0 - ambient) * max(dot(n, pc.light_dir), 0.0);
  }

  // Albedo and shading are linear light; a UNORM target is given them encoded.
  out_color = vec4(kLinearTarget ? albedo : linear_to_srgb(albedo), 1.0);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HybridMeshPipeline's set 0, as HybridMeshTextureSet, StreamedAtlas and the
// pipeline's fallback write it (hybrid_mesh_set0.hpp): an RGBA image, or one
// or two Y'CbCr 4:2:0 pictures' planes, as the Picture uniform says. With the
// sampling of a picture, which hybrid_mesh.frag and texture_view.frag share,
// so the texture view shows what the mesh shows.

#ifndef VG_COMMON_HYBRID_MESH_SET0_GLSL
#define VG_COMMON_HYBRID_MESH_SET0_GLSL

#include "common/ycbcr.glsl"

layout(set = 0, binding = 0) uniform sampler2D plane0;  // RGBA, or 0's luma
layout(set = 0, binding = 1) uniform sampler2D plane1;  // 0's chroma, or unused
layout(set = 0, binding = 3) uniform sampler2D luma1;    // 1's luma, or unused
layout(set = 0, binding = 4) uniform sampler2D chroma1;  // 1's chroma, or unused
// One picture's coding; mirrors PictureCoding in hybrid_mesh_set0.hpp.
struct Coding {
  uint full_range;     // Y' in 0..255 rather than 16..235
  float kr;            // the matrix's red weight
  float kb;            // the matrix's blue weight
  uint chroma_siting;  // ChromaSiting
  vec2 extent;         // the picture's size in luma pixels
};
// Mirrors PictureUniform in hybrid_mesh_set0.hpp (std140, 80 bytes).
layout(set = 0, binding = 2) uniform Picture {
  uint pictures;  // 0: plane0 is RGBA; n: n Y'CbCr 4:2:0 pictures
  Coding coding[2];
}
picture;

// R'G'B' of the picture `luma` and `chroma` hold, coded as `c`, at `uv`: its
// two planes filtered at level 0 where they are, then converted.
vec3 picture_rgb(sampler2D luma, sampler2D chroma, Coding c, vec2 uv) {
  vec2 p = uv * c.extent;
  float y =
      textureLod(luma, ycbcr_luma_uv(p, c.extent, vec2(textureSize(luma, 0))),
                 0.0)
          .r;
  vec2 cbcr = textureLod(chroma,
                         ycbcr_chroma_uv(p, c.chroma_siting, c.extent,
                                         vec2(textureSize(chroma, 0))),
                         0.0)
                  .rg;
  return ycbcr_to_rgb(y, cbcr, c.kr, c.kb, c.full_range != 0u);
}

#endif  // VG_COMMON_HYBRID_MESH_SET0_GLSL

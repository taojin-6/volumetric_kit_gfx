// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// HybridMeshTextureView's draw, fragment stage: one picture of a
// HybridMeshPipeline set-0 texture, drawn flat (image_draw.vert gives the
// rectangle and its uv, stacked as the mesh's uv0: picture p is v in
// [p, p + 1)) and filtered as hybrid_mesh_texture_view.hpp says, at the
// level the host picks per draw. Level 0 is read through the mesh shader's
// own sampling (common/hybrid_mesh_set0.glsl), so the view shows what the
// mesh shows.

#extension GL_GOOGLE_include_directive : require
#include "common/hybrid_mesh_set0.glsl"
#include "common/srgb.glsl"

// Levels 1 and up of pictures 0 and 1 (PicturePyramid), _SRGB, so a sample
// is linear light; or the pipeline's placeholder, never read then.
layout(set = 1, binding = 0) uniform sampler2D pyramid0;
layout(set = 1, binding = 1) uniform sampler2D pyramid1;

// Mirrors ViewPush in hybrid_mesh_texture_view.cpp (std430, 52 bytes): the
// first 40 bytes are image_draw.vert's.
layout(push_constant) uniform Push {
  layout(offset = 40) float footprint;  // F: picture pixels per target pixel
  int level;     // -1: the nearest pixel; 0: the picture; L: pyramid L - 1
  uint picture;  // p, from the draw's uv
}
pc;

// Supplied by HybridMeshTextureView::create: whether color attachment 0
// takes linear light (_SRGB or float); a UNORM target is given the encoding.
layout(constant_id = 0) const bool kLinearTarget = false;

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

// Linear light of the set's picture `p`, of `extent` pixels, at `pix`, in its
// pixels, held to its pixel centres: an RGBA image as sampled (an _SRGB one
// decodes), a picture through the mesh's conversion, decoded from sRGB.
vec3 level0(uint p, vec2 pix, vec2 extent) {
  vec2 uv = clamp(pix, vec2(0.5), extent - 0.5) / extent;
  if (picture.pictures == 0u) {
    return textureLod(plane0, uv, 0.0).rgb;
  }
  vec3 rgb = p == 1u ? picture_rgb(luma1, chroma1, picture.coding[1], uv)
                     : picture_rgb(plane0, plane1, picture.coding[0], uv);
  return srgb_to_linear(rgb);
}

// Linear light of pixel `at` of picture `p`'s `level` (0, the picture, of
// `extent`; L, pyramid level L - 1, of `size`), held to the level.
vec3 pixel(uint p, int level, ivec2 at, vec2 extent, ivec2 size) {
  if (level == 0) {
    return level0(p, vec2(at) + 0.5, extent);
  }
  ivec2 held = clamp(at, ivec2(0), size - 1);
  return p == 1u ? texelFetch(pyramid1, held, level - 1).rgb
                 : texelFetch(pyramid0, held, level - 1).rgb;
}

void main() {
  // The picture the draw names, the nearest one the set holds; an RGBA image
  // is one picture.
  uint p = picture.pictures == 0u ? 0u : min(pc.picture, picture.pictures - 1u);
  vec2 extent = picture.pictures == 0u ? vec2(textureSize(plane0, 0))
                                       : picture.coding[p].extent;
  vec2 pix = (frag_uv - vec2(0.0, float(p))) * extent;

  vec3 rgb;
  if (pc.level < 0) {
    rgb = level0(p, floor(pix) + 0.5, extent);
  } else {
    // The footprint in the level's pixels, which span the picture as the
    // level 0 pixels do, and the pixels it touches; at most 6 an axis, even
    // where an odd size floors a level's.
    ivec2 size = pc.level == 0
                     ? ivec2(extent)
                     : (p == 1u ? textureSize(pyramid1, pc.level - 1)
                                : textureSize(pyramid0, pc.level - 1));
    vec2 to_level = vec2(size) / extent;
    vec2 lo = (pix - 0.5 * pc.footprint) * to_level;
    vec2 hi = (pix + 0.5 * pc.footprint) * to_level;
    ivec2 first = ivec2(floor(lo));
    ivec2 last = min(ivec2(ceil(hi)) - 1, first + 5);
    rgb = vec3(0.0);
    float area = 0.0;
    for (int y = first.y; y <= last.y; ++y) {
      float wy = min(hi.y, float(y + 1)) - max(lo.y, float(y));
      for (int x = first.x; x <= last.x; ++x) {
        float w = wy * (min(hi.x, float(x + 1)) - max(lo.x, float(x)));
        rgb += w * pixel(p, pc.level, ivec2(x, y), extent, size);
        area += w;
      }
    }
    rgb /= area;
  }
  out_color = vec4(kLinearTarget ? rgb : linear_to_srgb(rgb), 1.0);
}

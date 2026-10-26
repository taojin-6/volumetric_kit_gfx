// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// An 8-bit Y'CbCr 4:2:0 picture sampled as two planes -- the luma (R8) and
// the interleaved chroma (R8G8, Cb then Cr) at half its size, rounded up --
// and converted to R'G'B': gfx's one home for that maths, which
// HybridMeshPipeline's fragment stage and ImagePipeline's convert pass share.
// The picture sits at each plane's top-left corner, and a plane may be
// larger than the picture's part of it. Positions are in luma texels, a
// texel's center at +0.5. hybrid_mesh_texture_set_test checks the result
// against a double-precision reference.

#ifndef VG_COMMON_YCBCR_GLSL
#define VG_COMMON_YCBCR_GLSL

// ChromaSiting's Left, as the host numbers it; anything else is Center.
const uint kYcbcrSitingLeft = 1u;

// The luma plane's coordinate for picture position `p`: held to the
// picture's texel centers, so a filter never reads the plane beyond the
// picture, and normalized by the plane's own size.
vec2 ycbcr_luma_uv(vec2 p, vec2 extent, vec2 plane_size) {
  return clamp(p, vec2(0.5), extent - 0.5) / plane_size;
}

// The chroma plane's coordinate for picture position `p`. Chroma sited
// between the luma samples (Center) puts luma position p at chroma position
// p / 2; left-sited chroma sits level with the left luma sample of each pair,
// a quarter of a chroma texel further on. Held to the picture's chroma texel
// centers -- half its extent, rounded up -- and normalized by the plane's own
// size, so an odd plane, or one larger than the picture, works.
vec2 ycbcr_chroma_uv(vec2 p, uint siting, vec2 extent, vec2 plane_size) {
  vec2 at = p * 0.5 + vec2(siting == kYcbcrSitingLeft ? 0.25 : 0.0, 0.0);
  return clamp(at, vec2(0.5), ceil(extent * 0.5) - 0.5) / plane_size;
}

// Y' and (Cb, Cr), as UNORM planes read them, to R'G'B' in [0, 1]: the range
// expanded -- full, or limited range's 16..235 and 16..240 -- then the Kr/Kb
// matrix, then a clamp, as a saturated chroma or a code outside limited range
// leaves the unit cube.
vec3 ycbcr_to_rgb(float y, vec2 cbcr, float kr, float kb, bool full_range) {
  float luma;
  vec2 chroma;
  if (full_range) {
    luma = y;
    chroma = cbcr - 128.0 / 255.0;
  } else {
    luma = (y * 255.0 - 16.0) / 219.0;
    chroma = (cbcr * 255.0 - 128.0) / 224.0;
  }
  float kg = 1.0 - kr - kb;
  float r = luma + 2.0 * (1.0 - kr) * chroma.y;
  float b = luma + 2.0 * (1.0 - kb) * chroma.x;
  float g = (luma - kr * r - kb * b) / kg;
  return clamp(vec3(r, g, b), 0.0, 1.0);
}

#endif  // VG_COMMON_YCBCR_GLSL

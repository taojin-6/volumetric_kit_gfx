// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// Reconstruction patch-atlas mesh, fragment stage. Albedo comes from a small
// per-TRIANGLE texture patch the producer accumulated across frames, or from the
// interpolated per-vertex colour where no frame ever observed that patch.
//
// THE ADDRESSING IS PER PRIMITIVE, WHICH IS THE WHOLE POINT. A patch belongs to
// a triangle, so its three corners need three distinct texture coordinates --
// and a vertex shared between up to six triangles cannot carry them. The
// hybrid pipeline's per-vertex uv0 therefore cannot express this at all. Here
// the atlas offset comes from `gl_PrimitiveID` (under an indexed indirect draw
// with firstIndex 0 that is exactly the producer's arena triangle slot) and the
// position within the patch comes from the fragment's barycentric coordinate,
// recovered below. Nothing is stored per vertex.
//
// BARYCENTRICS ARE RECOVERED, NOT READ. VK_KHR_fragment_shader_barycentric
// would hand them over directly and is supported on the Apple targets, but it
// is a device feature -- so it would have to be threaded through every
// embedder's device creation, including the neutral bootstrap that merges two
// libraries' requirements. Recovering them from the interpolated world position
// and the triangle's three corners costs three buffer fetches and a 2x2 solve,
// needs no feature, and is perspective-correct because the interpolated
// position is.
//
// THE BUFFERS ARE FLAT SCALAR ARRAYS, deliberately. A `Vertex` struct mirror
// would need GL_EXT_scalar_block_layout to match the producer's packing --
// std430 pads a vec3 in a struct array to 16 bytes, which the 64-byte host
// layout does not -- and that is another feature dependency for no gain. A
// `float[]` indexed by hand has no padding rules to get wrong.

layout(location = 0) in vec3 frag_normal;  // world space
layout(location = 1) in vec3 frag_world;   // world space
layout(location = 2) in vec4 frag_color;

// The atlas: one uint per texel, canonical-encoded R'G'B' in the low three
// bytes and the accumulated observation weight in the high one. Patch `t`
// occupies [t * texels_per_patch, (t+1) * texels_per_patch).
layout(set = 0, binding = 0, std430) readonly buffer Patches { uint patches[]; };
// Three per triangle, into the vertex array below.
layout(set = 0, binding = 1, std430) readonly buffer Indices { uint indices[]; };
// The producer's interleaved vertices as raw floats: 16 per vertex (64 bytes),
// position at 0..2. See the note above on why this is not a struct.
layout(set = 0, binding = 2, std430) readonly buffer Vertices {
  float vertex_data[];
};

layout(push_constant) uniform Push {
  mat4 view_proj;
  vec4 light;   // xyz world-space direction TO the light (unit); w > 0.5 = lit
  uvec4 patch_shape;  // x = texels per patch leg, y = texels per patch, zw unused
}
pc;

layout(location = 0) out vec4 out_color;

const uint kFloatsPerVertex = 16u;

vec3 vertex_position(uint v) {
  uint base = v * kFloatsPerVertex;
  return vec3(vertex_data[base + 0u], vertex_data[base + 1u],
              vertex_data[base + 2u]);
}

// Exact piecewise sRGB decode, one channel. The producer stores canonical
// encoded 8-bit and this pipeline shades in linear, so the decode has to happen
// here -- there is no _SRGB image format doing it in hardware, which is the one
// thing a buffer atlas gives up. Exact rather than pow(x, 2.2), matching the
// producer's own curve; the two must agree or a round trip drifts.
float srgb_to_linear(float e) {
  return e <= 0.04045 ? e / 12.92 : pow((e + 0.055) / 1.055, 2.4);
}

void main() {
  // The triangle this fragment belongs to, and its three world-space corners.
  uint tri = uint(gl_PrimitiveID);
  vec3 p0 = vertex_position(indices[tri * 3u + 0u]);
  vec3 p1 = vertex_position(indices[tri * 3u + 1u]);
  vec3 p2 = vertex_position(indices[tri * 3u + 2u]);

  // Barycentrics of the interpolated position inside that triangle, by the
  // standard projection onto the triangle's own basis.
  vec3 e1 = p1 - p0;
  vec3 e2 = p2 - p0;
  vec3 pv = frag_world - p0;
  float d11 = dot(e1, e1);
  float d12 = dot(e1, e2);
  float d22 = dot(e2, e2);
  float dp1 = dot(pv, e1);
  float dp2 = dot(pv, e2);
  float denom = d11 * d22 - d12 * d12;

  // `denom` is four times the squared area, so a degenerate triangle makes it
  // zero. The producer retires geometry by collapsing a triangle to a single
  // vertex, and such a triangle is culled before rasterisation -- so this
  // should be unreachable. Guarded anyway: the alternative is a NaN
  // coordinate, which indexes the atlas anywhere at all.
  bool use_vertex_color = !(denom > 0.0);
  vec3 albedo = frag_color.rgb;

  if (!use_vertex_color) {
    float b1 = (d22 * dp1 - d12 * dp2) / denom;
    float b2 = (d11 * dp2 - d12 * dp1) / denom;
    uint leg = pc.patch_shape.x;
    float span = float(leg - 1u);
    // Nearest texel. Clamped into the triangular domain rather than trusted:
    // interpolation at a shared edge can put a fragment a hair outside its own
    // triangle, and i + j must stay under `leg` or the row-major triangular
    // index runs into the next patch.
    //
    // TODO: bilinear within the patch. Nearest is visibly blocky when a
    // triangle covers many pixels, which is the near-field case; the fix is an
    // interpolation over the triangular lattice rather than a square one.
    uint i = uint(clamp(b1 * span + 0.5, 0.0, span));
    uint j = uint(clamp(b2 * span + 0.5, 0.0, span));
    if (i + j > leg - 1u) {
      // Push the fragment back onto the hypotenuse, keeping whichever
      // coordinate is larger -- the corner it is actually nearest.
      uint over = i + j - (leg - 1u);
      if (i >= j) {
        i -= min(over, i);
      } else {
        j -= min(over, j);
      }
    }
    uint texel = patches[tri * pc.patch_shape.y + (j * leg - j * (j - 1u) / 2u + i)];

    // A weight of zero means no frame ever observed this texel -- the producer
    // leaves such a texel entirely zero, so its colour bits are not merely
    // unhelpful but meaningless. Fall back to the colour the volume fused,
    // which is what the surface looked like before any camera resolved it.
    if ((texel >> 24u) == 0u) {
      use_vertex_color = true;
    } else {
      vec3 encoded = vec3(float(texel & 0xFFu), float((texel >> 8u) & 0xFFu),
                          float((texel >> 16u) & 0xFFu)) /
                     255.0;
      albedo = vec3(srgb_to_linear(encoded.r), srgb_to_linear(encoded.g),
                    srgb_to_linear(encoded.b));
    }
  }

  // light.w > 0.5 requests lit shading; otherwise pass the albedo through flat.
  // Two-sided, since the pipeline does not cull.
  if (pc.light.w > 0.5) {
    vec3 raw = gl_FrontFacing ? frag_normal : -frag_normal;
    // Guard the normalize: a zero/degenerate interpolated normal would yield a
    // NaN that max() does not reliably clamp. Fall back to pure ambient.
    vec3 n = dot(raw, raw) > 0.0 ? normalize(raw) : vec3(0.0);
    const float ambient = 0.25;
    albedo *= ambient + (1.0 - ambient) * max(dot(n, pc.light.xyz), 0.0);
  }

  out_color = vec4(albedo, 1.0);
}

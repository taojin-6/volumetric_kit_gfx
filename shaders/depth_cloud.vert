// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// DepthCloudPipeline, vertex stage: one point per depth pixel -- every
// stride-th along each axis -- read where the producer left it. There is no
// vertex input: vertex i is pixel (i % columns, i / columns) * stride. The
// host folds the cameras' intrinsics and poses into the two matrices: one
// moves (u z, v z, z, 1) into clip space, the other into a colour camera's
// pixels. This stage keeps only recon's pixel rule, copied from its
// nearest_pixel (tsdf_common.glsl), which gfx cannot include: a change to
// recon's rule must change this copy. primitive.frag shades the point.

// Read-only: gfx does not enable vertexPipelineStoresAndAtomics, without which
// a vertex stage's storage buffers must not be writable
// (VUID-RuntimeSpirv-NonWritable-06341).
layout(set = 0, binding = 0, std430) readonly buffer Depth {
  float depth[];  // metres, row-major; 0 where there was no return
};
layout(set = 0, binding = 1, std430) readonly buffer Color {
  uint color[];  // R'G'B'A8 words, R in the lowest byte
};

// Mirrors Params in depth_cloud_pipeline.cpp (std140, 176 bytes).
layout(set = 0, binding = 2, std140) uniform Params {
  // (u z, v z, z, 1) of depth pixel (u, v) at depth z to clip space.
  mat4 depth_to_clip;
  // The same to (x, y, w, 1): colour pixel (x / w, y / w) at depth w.
  mat4 depth_to_color;
  uvec4 grid;   // depth width, columns, stride, colour width
  uvec4 style;  // mode, tint (an RGBA8 word), flags, colour height
  vec4 range;   // nearest depth, farthest finite depth, point size, unused
}
p;

// CloudColor.
const uint kSolid = 0u;
const uint kTrue = 2u;
// Params::style.z.
const uint kHasColor = 1u;
const uint kAlphaIsCoverage = 2u;

layout(location = 0) out vec3 frag_color;  // sRGB-encoded

void main() {
  uint i = uint(gl_VertexIndex);
  uvec2 px = uvec2(i % p.grid.y, i / p.grid.y) * p.grid.z;
  float z = depth[px.y * p.grid.x + px.x];
  gl_PointSize = p.range.z;
  frag_color = vec3(0.0);
  // Negated, so a NaN fails it: no return, NaN or a depth out of range puts
  // the point outside the clip volume, which drops it.
  if (!(z > 0.0 && z >= p.range.x && z <= p.range.y)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec4 h = vec4(vec2(px) * z, z, 1.0);
  gl_Position = p.depth_to_clip * h;

  vec3 tint = unpackUnorm4x8(p.style.y).rgb;
  frag_color = tint;
  if (p.style.x == kSolid || (p.style.z & kHasColor) == 0u) {
    return;
  }
  // recon's rule: in front of the camera, inside [0, w) x [0, h), and the
  // nearest pixel with the last half pixel clamped. The direct comparisons
  // fail on NaN, so a NaN takes the tint.
  vec4 c = p.depth_to_color * h;
  uvec2 size = uvec2(p.grid.w, p.style.w);
  if (!(c.z > 0.0)) {
    return;
  }
  vec2 q = c.xy / c.z;
  if (!(q.x >= 0.0 && q.x < float(size.x) && q.y >= 0.0 &&
        q.y < float(size.y))) {
    return;
  }
  uvec2 n = min(uvec2(floor(q + 0.5)), size - 1u);
  uint word = color[n.y * size.x + n.x];
  if ((p.style.z & kAlphaIsCoverage) != 0u && (word >> 24) == 0u) {
    return;  // the colour camera saw nothing there
  }
  vec3 rgb = unpackUnorm4x8(word).rgb;
  // TintedLuma: the tint scaled by Rec. 709's luma of the encoded colour.
  frag_color = p.style.x == kTrue
                   ? rgb
                   : tint * dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// ImagePipeline's convert pass, fragment stage: turns one texel of a source
// image into display color, written as linear light into level 0 of an _SRGB
// display image (which encodes it), before that image's mip chain is built.
// Converting first is what makes the chain right: each mapping below is
// nonlinear -- the sRGB decode, the Y'CbCr matrix and its clamp, the color
// ramp -- so averaging the source and then converting would not give the
// average of what the viewer sees.
//
// The target is the same size as the source, so each fragment is one texel,
// read with texelFetch at gl_FragCoord. Only NV12's chroma plane is sampled
// with filtering, to bring it up to luma resolution.

#extension GL_GOOGLE_include_directive : require
#include "common/srgb.glsl"
#include "common/ycbcr.glsl"

layout(set = 0, binding = 0) uniform sampler2D plane0;  // color, grey, ramp or luma
layout(set = 0, binding = 1) uniform sampler2D plane1;  // NV12 CbCr; else plane0, nearest

// Mirrors ConvertPush in image_pipeline.cpp (std430, 40 bytes).
layout(push_constant) uniform Push {
  uint mapping;        // ImageMapping: 0 color, 1 grey, 2 ramp, 3 NV12
  uint srgb_encoded;   // color / grey / NV12: the values are sRGB-encoded
  float value_scale;   // ramp: the stored value is texel * value_scale
  float ramp_min;      // ramp: the stored value at the ramp's first color
  float ramp_max;      // ramp: the stored value at the ramp's last color
  uint zero_is_empty;  // ramp: a stored 0 is "no data", shown black
  float kr;            // NV12: the matrix's red weight
  float kb;            // NV12: the matrix's blue weight
  uint full_range;     // NV12: Y in 0..255 rather than 16..235
  uint chroma_siting;  // NV12: ChromaSiting
}
pc;

layout(location = 0) out vec4 out_color;

const uint kColor = 0u;
const uint kGrey = 1u;
const uint kRamp = 2u;
const uint kNv12 = 3u;

// The ramp: five sRGB-encoded stops, dark violet through blue, green and
// yellow to dark red, interpolated linearly between them.
vec3 ramp(float t) {
  const vec3 stops[5] =
      vec3[5](vec3(0.19, 0.07, 0.23), vec3(0.27, 0.51, 0.95),
              vec3(0.10, 0.88, 0.45), vec3(0.96, 0.80, 0.15),
              vec3(0.64, 0.04, 0.01));
  float x = clamp(t, 0.0, 1.0) * 4.0;
  // Clamped on both sides: the caller screens out NaN, whose clamp and int()
  // are undefined, but the index must stay in bounds whatever it is given.
  int i = clamp(int(x), 0, 3);
  return mix(stops[i], stops[i + 1], x - float(i));
}

void main() {
  ivec2 texel = ivec2(gl_FragCoord.xy);
  vec4 sample0 = texelFetch(plane0, texel, 0);

  if (pc.mapping == kRamp) {
    float value = sample0.r * pc.value_scale;
    // A float map's NaN is "no data" whatever zero_is_empty says.
    if (isnan(value) || (pc.zero_is_empty != 0u && value == 0.0)) {
      out_color = vec4(0.0, 0.0, 0.0, 1.0);
      return;
    }
    float t = (value - pc.ramp_min) / (pc.ramp_max - pc.ramp_min);
    out_color = vec4(srgb_to_linear(ramp(t)), 1.0);
    return;
  }

  vec3 rgb;
  if (pc.mapping == kNv12) {
    // The planes are the picture's own size, the luma's center at the
    // fragment's.
    vec2 at = ycbcr_chroma_uv(gl_FragCoord.xy, pc.chroma_siting,
                              vec2(textureSize(plane0, 0)),
                              vec2(textureSize(plane1, 0)));
    rgb = ycbcr_to_rgb(sample0.r, textureLod(plane1, at, 0.0).rg, pc.kr,
                       pc.kb, pc.full_range != 0u);
  } else if (pc.mapping == kGrey) {
    // NaN, which only R32_SFLOAT holds, shows black: its clamp is undefined.
    rgb = vec3(isnan(sample0.r) ? 0.0 : clamp(sample0.r, 0.0, 1.0));
  } else {
    rgb = clamp(sample0.rgb, 0.0, 1.0);
  }
  out_color = vec4(pc.srgb_encoded != 0u ? srgb_to_linear(rgb) : rgb, 1.0);
}

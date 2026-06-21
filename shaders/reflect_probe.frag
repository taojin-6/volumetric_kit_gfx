// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// A probe for the SPIR-V reflection tests: declares one uniform buffer, one
// combined image sampler, and a push-constant block -- each actually used, so
// the compiler keeps them and reflection has something to recover.

layout(set = 0, binding = 0) uniform Globals {
  vec4 color;
}
globals;

layout(set = 0, binding = 1) uniform sampler2D tex;

layout(push_constant) uniform Push {
  vec2 offset;
}
push;

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

void main() {
  out_color = globals.color * texture(tex, in_uv + push.offset);
}

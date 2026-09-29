// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// A probe for GraphicsPipelineDesc::fragment_specialization: the output red is
// a specialization constant, so the rendered pixel shows whether the pipeline
// applied the caller's value or kept this default. Pairs with mesh.vert, whose
// color output it accepts but ignores.

layout(constant_id = 0) const float kRed = 0.0;

layout(location = 0) in vec3 frag_color;

layout(location = 0) out vec4 out_color;

void main() { out_color = vec4(kRed, 0.0, 0.0, 1.0); }

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#version 450

// glTF 2.0 metallic-roughness PBR, lit by one directional key light plus
// image-based ambient. Material factors + the five maps come from a per-material
// set (set 1: a UBO at binding 0, then base-color / metallic-roughness / normal
// / occlusion / emissive samplers at 1..5); the scene set (set 0) carries the
// camera and the IBL textures (irradiance + prefiltered-specular cubes + a BRDF
// LUT). All are reflected into the pipeline layout automatically. The example
// binds a 1x1 white (or flat-normal) fallback for any map a material omits, so
// the factor alone applies. Two-sided: the normal is flipped for back faces
// because the pipeline does not cull.

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;
layout(location = 2) in vec3 frag_world_pos;
layout(location = 3) in vec4 frag_tangent;

layout(set = 0, binding = 0) uniform Scene {
  vec4 camera_pos;  // .xyz world-space eye, .w = prefiltered-specular max LOD
}
scene;
layout(set = 0, binding = 1) uniform samplerCube irradiance_cube;  // diffuse IBL
layout(set = 0, binding = 2) uniform samplerCube prefilter_cube;  // specular IBL
layout(set = 0, binding = 3) uniform sampler2D brdf_lut;  // BRDF integration

layout(set = 1, binding = 0) uniform Material {
  vec4 base_color_factor;
  vec4 emissive_factor;  // .rgb
  float metallic_factor;
  float roughness_factor;
  float normal_scale;
  float occlusion_strength;
}
mat;
layout(set = 1, binding = 1) uniform sampler2D base_color_tex;
layout(set = 1, binding = 2) uniform sampler2D metallic_roughness_tex;
layout(set = 1, binding = 3) uniform sampler2D normal_tex;
layout(set = 1, binding = 4) uniform sampler2D occlusion_tex;
layout(set = 1, binding = 5) uniform sampler2D emissive_tex;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;

// Trowbridge-Reitz GGX normal distribution.
float distribution_ggx(float n_dot_h, float roughness) {
  const float a = roughness * roughness;
  const float a2 = a * a;
  const float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-7);
}

// Smith geometry term with the Schlick-GGX direct-lighting remap of roughness.
float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
  const float r = roughness + 1.0;
  const float k = (r * r) / 8.0;
  const float gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
  const float gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
  return gv * gl;
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
  return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Fresnel with a roughness-aware ceiling for the IBL ambient term (Karis): a
// rough surface's grazing reflectance is pulled back toward F0.
vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
  const vec3 ceiling = max(vec3(1.0 - roughness), f0);
  return f0 + (ceiling - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Perturb the world normal by the tangent-space normal map, building the TBN
// from the interpolated normal + tangent (Gram-Schmidt re-orthonormalized).
vec3 shading_normal(vec3 n_geo) {
  vec3 sampled = texture(normal_tex, frag_uv).xyz * 2.0 - 1.0;
  sampled.xy *= mat.normal_scale;
  const vec3 n = normalize(n_geo);
  // Gram-Schmidt the tangent against n. When the mesh has no tangents (the
  // (1,0,0,1) default) or the tangent is parallel to n, the residual is ~0;
  // fall back to an arbitrary perpendicular so normalize() never divides by
  // zero -- a NaN there poisons the whole frame (0*NaN is still NaN, so even a
  // flat normal map could not recover it).
  vec3 t = frag_tangent.xyz - dot(frag_tangent.xyz, n) * n;
  if (dot(t, t) < 1e-8) {
    t = abs(n.x) < 0.9 ? cross(vec3(1.0, 0.0, 0.0), n)
                       : cross(vec3(0.0, 1.0, 0.0), n);
  }
  t = normalize(t);
  const vec3 b = cross(n, t) * frag_tangent.w;
  return normalize(mat3(t, b, n) * sampled);
}

void main() {
  const vec3 albedo =
      (mat.base_color_factor * texture(base_color_tex, frag_uv)).rgb;

  const vec3 mr = texture(metallic_roughness_tex, frag_uv).rgb;
  const float metallic = mat.metallic_factor * mr.b;  // glTF: B = metallic
  const float roughness =
      clamp(mat.roughness_factor * mr.g, 0.04, 1.0);  // G = roughness
  const float ao =
      mix(1.0, texture(occlusion_tex, frag_uv).r, mat.occlusion_strength);
  const vec3 emissive =
      mat.emissive_factor.rgb * texture(emissive_tex, frag_uv).rgb;

  // Flip the geometric normal toward the viewer for back faces (no culling),
  // then apply the normal map.
  const vec3 geo_n = gl_FrontFacing ? frag_normal : -frag_normal;
  const vec3 n = shading_normal(geo_n);
  const vec3 v = normalize(scene.camera_pos.xyz - frag_world_pos);

  // One world-space directional key light. Intensity is tuned for the /PI
  // diffuse; IBL will supply the real environment response.
  const vec3 light_dir = normalize(vec3(0.5, 0.8, 0.6));
  const vec3 light_color = vec3(3.0);
  const vec3 h = normalize(v + light_dir);

  const float n_dot_v = max(dot(n, v), 1e-4);
  const float n_dot_l = max(dot(n, light_dir), 0.0);
  const float n_dot_h = max(dot(n, h), 0.0);
  const float v_dot_h = max(dot(v, h), 0.0);

  const vec3 f0 = mix(vec3(0.04), albedo, metallic);
  const float d = distribution_ggx(n_dot_h, roughness);
  const float g = geometry_smith(n_dot_v, n_dot_l, roughness);
  const vec3 f = fresnel_schlick(v_dot_h, f0);
  const vec3 specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
  const vec3 kd = (vec3(1.0) - f) * (1.0 - metallic);
  const vec3 direct = (kd * albedo / PI + specular) * light_color * n_dot_l;

  // Image-based ambient (split-sum): diffuse from the irradiance cube, specular
  // from the roughness-prefiltered cube weighted by the BRDF LUT.
  const vec3 ibl_f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
  const vec3 kd_ibl = (vec3(1.0) - ibl_f) * (1.0 - metallic);
  const vec3 diffuse_ibl = texture(irradiance_cube, n).rgb * albedo;
  const vec3 reflected = reflect(-v, n);
  const vec3 prefiltered =
      textureLod(prefilter_cube, reflected, roughness * scene.camera_pos.w).rgb;
  const vec2 brdf = texture(brdf_lut, vec2(n_dot_v, roughness)).rg;
  const vec3 specular_ibl = prefiltered * (ibl_f * brdf.x + brdf.y);
  const vec3 ambient = (kd_ibl * diffuse_ibl + specular_ibl) * ao;

  // Output is linear; the sRGB target encodes it on write.
  // TODO: honor alpha mode -- mask `discard` (needs a device feature/SPIR-V
  // capability) and blending; today every fragment is written fully opaque.
  const vec3 color = ambient + direct + emissive;
  out_color = vec4(color, 1.0);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file material.hpp
/// @brief glTF metallic-roughness material, CPU-side and GPU-API-free.

#include <cstdint>
#include <string>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace volumetric_kit::gfx::assets {

/// @brief Sentinel for "no texture": a texture index field equal to this names
///        no image. Compare against it before indexing @ref Model::images.
inline constexpr std::uint32_t kNoTexture = 0xFFFFFFFFu;

/// @brief How a material's alpha channel is interpreted (glTF `alphaMode`).
enum class AlphaMode {
  Opaque,  ///< Alpha is ignored; the surface is fully opaque.
  Mask,    ///< Fragments with `alpha < cutoff` are discarded.
  Blend,   ///< Alpha composites the surface over the background.
};

/// @brief A glTF 2.0 metallic-roughness material.
///
/// Factors are plain values; every texture is an index into @ref Model::images
/// (or @ref kNoTexture when absent) -- no GPU handle is created at load time.
/// Channel conventions follow the glTF spec: base color is sRGB; metallic
/// (B) / roughness (G) and occlusion (R) are linear; the normal map is a
/// tangent-space XYZ encoded in [0,1].
///
/// @code
/// const assets::Material& m = model.materials[mesh.material];
/// glm::vec4 albedo = m.base_color_factor;
/// if (m.base_color_texture != assets::kNoTexture)
///   albedo *= sample(model.images[m.base_color_texture], uv);  // shading
/// @endcode
struct Material {
  std::string name;  ///< Material name (may be empty).

  // --- Base color (albedo) ---
  glm::vec4 base_color_factor{1.0f, 1.0f, 1.0f,
                              1.0f};              ///< Linear RGBA multiplier.
  std::uint32_t base_color_texture = kNoTexture;  ///< sRGB albedo texture.

  // --- Metallic-roughness ---
  float metallic_factor = 1.0f;                           ///< [0,1] metalness.
  float roughness_factor = 1.0f;                          ///< [0,1] roughness.
  std::uint32_t metallic_roughness_texture = kNoTexture;  ///< B=metal, G=rough.

  // --- Normal map ---
  std::uint32_t normal_texture = kNoTexture;  ///< Tangent-space normal map.
  float normal_scale = 1.0f;                  ///< Scales the sampled XY normal.

  // --- Ambient occlusion ---
  std::uint32_t occlusion_texture = kNoTexture;  ///< R channel = occlusion.
  float occlusion_strength = 1.0f;               ///< [0,1] occlusion blend.

  // --- Emission ---
  glm::vec3 emissive_factor{0.0f, 0.0f, 0.0f};  ///< Linear RGB emission.
  std::uint32_t emissive_texture = kNoTexture;  ///< sRGB emissive texture.

  // --- Alpha + culling ---
  AlphaMode alpha_mode = AlphaMode::Opaque;  ///< Alpha interpretation.
  float alpha_cutoff = 0.5f;                 ///< Threshold when @ref alpha_mode
                                             ///< is @ref AlphaMode::Mask.
  bool double_sided = false;                 ///< Disable back-face culling.
};

}  // namespace volumetric_kit::gfx::assets

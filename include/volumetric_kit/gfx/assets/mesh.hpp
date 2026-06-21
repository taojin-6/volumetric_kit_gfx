// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh.hpp
/// @brief Triangle mesh: interleaved vertices + a 32-bit index buffer.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace volumetric_kit::gfx::assets {

/// @brief One vertex, interleaved (array-of-structs).
///
/// Interleaved -- not parallel arrays -- because this is the layout a single
/// `VkVertexInputBindingDescription` consumes directly: the later GPU tier
/// uploads `std::vector<Vertex>` to one vertex buffer with no repacking, and a
/// vertex's attributes share a cache line during draw. @ref color is always
/// present; when the source has no vertex colors a loader fills opaque white,
/// so a shader can read the slot unconditionally.
struct Vertex {
  glm::vec3 position{0.0f};                   ///< Object-space position.
  glm::vec3 normal{0.0f, 0.0f, 1.0f};         ///< Object-space unit normal.
  glm::vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f};  ///< xyz tangent, w = handedness.
  glm::vec2 uv0{0.0f};                        ///< Primary texture coordinates.
  glm::vec4 color{1.0f};  ///< Vertex color (white if unset).
};

/// @brief A triangle mesh: one vertex stream, one index stream, one material.
///
/// Indices are 32-bit and address @ref vertices; their count is a multiple of
/// three (triangle list). @ref material indexes @ref Model::materials, or is
/// @ref kNoMaterial when the source assigned none. @ref transform is the node-
/// local model matrix glTF attaches at the node that referenced this mesh;
/// composing it with parent node transforms is the consumer's job (the @ref
/// Scene node tree is data only).
///
/// @code
/// for (const assets::Mesh& mesh : model.meshes) {
///   upload_vertices(mesh.vertices);            // GPU tier
///   upload_indices(mesh.indices);
///   bind_material(model.materials[mesh.material]);
/// }
/// @endcode
struct Mesh {
  /// @brief Sentinel @ref material value meaning "no material assigned".
  static constexpr std::uint32_t kNoMaterial = 0xFFFFFFFFu;

  std::string name;              ///< Mesh / primitive name (may be empty).
  std::vector<Vertex> vertices;  ///< Interleaved vertex stream.
  std::vector<std::uint32_t> indices;    ///< Triangle-list indices into @ref
                                         ///< vertices (size % 3 == 0).
  std::uint32_t material = kNoMaterial;  ///< Index into @ref Model::materials.
  glm::mat4 transform{1.0f};  ///< Node-local model matrix (identity if
                              ///< none).

  /// @return Number of triangles (`indices.size() / 3`).
  std::size_t triangle_count() const noexcept { return indices.size() / 3; }
};

}  // namespace volumetric_kit::gfx::assets

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file scene.hpp
/// @brief Flat node tree (transforms + child/mesh indices). Data only.

#include <cstdint>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

namespace volumetric_kit::gfx::assets {

/// @brief One node in the scene graph: a local transform and index references.
///
/// References are stored as flat indices -- @ref children into @ref
/// Scene::nodes, @ref mesh into @ref Model::meshes -- so the structure is a
/// trivially copyable value with no pointers and no ownership. This is
/// deliberately data only: the library defines no runtime scene-graph behavior
/// (no traversal, no transform caching). A consumer that wants world transforms
/// walks @ref children itself, composing @ref transform down the tree.
struct Node {
  /// @brief Sentinel @ref mesh value meaning "this node draws nothing".
  static constexpr std::uint32_t kNoMesh = 0xFFFFFFFFu;

  std::string name;                     ///< Node name (may be empty).
  glm::mat4 transform{1.0f};            ///< Node-local transform (identity if
                                        ///< none).
  std::uint32_t mesh = kNoMesh;         ///< Index into @ref Model::meshes.
  std::vector<std::uint32_t> children;  ///< Child indices into @ref
                                        ///< Scene::nodes.
};

/// @brief A flat node array plus the indices of its roots.
///
/// The node tree is stored flat in @ref nodes; @ref roots lists the top-level
/// node indices (a glTF scene's `nodes`). There is no behavior here -- a
/// renderer reads this to know which meshes to draw and under what transform.
///
/// @code
/// for (std::uint32_t root : model.scene.roots)
///   draw_node(model, root, glm::mat4(1.0f));  // consumer-defined traversal
/// @endcode
struct Scene {
  std::string name;                  ///< Scene name (may be empty).
  std::vector<Node> nodes;           ///< All nodes, flat.
  std::vector<std::uint32_t> roots;  ///< Top-level node indices into @ref
                                     ///< nodes.
};

}  // namespace volumetric_kit::gfx::assets

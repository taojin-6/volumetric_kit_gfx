// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file model.hpp
/// @brief The aggregate CPU-side asset: meshes, materials, images, scene.

#include <vector>

#include "volumetric_kit/gfx/assets/image.hpp"
#include "volumetric_kit/gfx/assets/material.hpp"
#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/assets/point_cloud.hpp"
#include "volumetric_kit/gfx/assets/scene.hpp"

namespace volumetric_kit::gfx::assets {

/// @brief Everything loaded from one asset file, CPU-side and GPU-API-free.
///
/// A loader (e.g. @ref load_gltf) fills this from a single file: @ref meshes
/// and @ref point_clouds reference @ref materials by index; materials reference
/// @ref images by index; @ref scene's nodes reference @ref meshes by index.
/// Because every cross-reference is an index into a vector here, the whole
/// aggregate is a self-contained value -- copyable, with no internal pointers
/// and no GPU handles. A later tier walks it to build vertex/index buffers and
/// textures.
///
/// @code
/// std::string err;
/// if (auto model = io::load_gltf("scene.glb", &err)) {
///   for (const assets::Mesh& m : model->meshes) upload(m);  // GPU tier
/// } else {
///   log_message(LogLevel::Error, err);
/// }
/// @endcode
struct Model {
  std::vector<Mesh> meshes;              ///< Triangle meshes.
  std::vector<PointCloud> point_clouds;  ///< Point clouds (e.g. future PLY).
  std::vector<Material> materials;  ///< Shared materials (indexed by mesh).
  std::vector<Image> images;        ///< Decoded textures (indexed by mat).
  Scene scene;                      ///< Node tree referencing @ref meshes.
};

}  // namespace volumetric_kit::gfx::assets

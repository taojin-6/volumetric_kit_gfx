// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file gltf_loader.hpp
/// @brief Load a glTF 2.0 file (`.gltf` or `.glb`) into an assets @ref Model.

#include <optional>
#include <string>
#include <string_view>

#include "volumetric_kit/gfx/assets/export.hpp"
#include "volumetric_kit/gfx/assets/model.hpp"

namespace volumetric_kit::gfx::assets {

/// @brief Load a glTF 2.0 asset from disk into a CPU-side @ref Model.
///
/// Parses the file with tinygltf, converts every mesh primitive to a @ref Mesh
/// (interleaved @ref Vertex stream + 32-bit indices), every glTF material to a
/// @ref Material, and decodes every referenced image to an @ref Image. Both the
/// JSON form (`.gltf` + external buffers/images) and the binary container
/// (`.glb`) are accepted; the format is chosen from the file extension.
///
/// Error handling is by value -- the assets tier does not depend on `core`, so
/// it does not use `vg::Status`/`vg::Result`. A failed load returns
/// `std::nullopt`; when @p error is non-null it receives a human-readable
/// reason (parser error, unsupported feature, missing file). Warnings from the
/// parser are not fatal and are not surfaced here.
///
/// @param path   Filesystem path to a `.gltf` or `.glb` file.
/// @param error  Optional out-param; on failure, set to the failure reason
///               (left unchanged on success). Pass `nullptr` to ignore it.
/// @return The loaded @ref Model on success, or `std::nullopt` on failure.
///
/// @code
/// std::string err;
/// std::optional<assets::Model> model = assets::load_gltf("box.glb", &err);
/// if (!model) { handle_error(err); return; }
/// use(*model);
/// @endcode
VG_ASSETS_API std::optional<Model> load_gltf(std::string_view path,
                                             std::string* error = nullptr);

// TODO: OBJ loader (load_obj) -- thin per-format loader into this same Model.
// TODO: PLY loader (load_ply) -- produces PointCloud for scanned/point data.
// TODO: assimp catch-all loader -- optional, covers the long tail of formats.

}  // namespace volumetric_kit::gfx::assets

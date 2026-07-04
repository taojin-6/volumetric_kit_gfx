// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file gltf_loader.hpp
/// @brief Load a glTF 2.0 file (`.gltf` or `.glb`) into an @ref
///        volumetric_kit::gfx::assets::Model.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/gfx/assets/model.hpp"
#include "volumetric_kit/gfx/io/export.hpp"

namespace volumetric_kit::gfx::io {

/// @brief Load a glTF 2.0 asset from disk into a CPU-side @ref assets::Model.
///
/// Parses the file with tinygltf, converts every mesh primitive to a @ref
/// assets::Mesh (interleaved @ref assets::Vertex stream + 32-bit indices),
/// every glTF material to a @ref assets::Material, and decodes every referenced
/// image to an @ref assets::Image. Both the JSON form (`.gltf` + external
/// buffers/images) and the binary container (`.glb`) are accepted; the format
/// is chosen from the file extension.
///
/// Error handling is by value -- the io tier does not depend on `core`, so it
/// does not use `vg::Status`/`vg::Result`. A failed load returns
/// `std::nullopt`; when @p error is non-null it receives a human-readable
/// reason (parser error, unsupported feature, missing file). Warnings from the
/// parser are not fatal and are not surfaced here.
///
/// A load can also succeed *partially*: a mesh primitive the converter cannot
/// represent is dropped from the returned model rather than failing the load.
/// Pass @p warnings to observe such drops -- a silently incomplete model is
/// otherwise indistinguishable from a complete one.
///
/// @note Sparse accessors and accessors without a bufferView (both spec-valid)
///       are not supported yet; a primitive whose POSITION or index accessor
///       uses either is dropped and reported through @p warnings.
///
/// @param path      Filesystem path to a `.gltf` or `.glb` file.
/// @param error     Optional out-param; on failure, set to the failure reason
///                  (left unchanged on success). Pass `nullptr` to ignore it.
/// @param warnings  Optional out-param; one message is appended per dropped
///                  primitive, naming the mesh, the primitive index, and the
///                  reason. Never cleared, and untouched when nothing is
///                  dropped. Pass `nullptr` to ignore drops.
/// @return The loaded @ref assets::Model on success, or `std::nullopt` on
///         failure.
///
/// @code
/// std::string err;
/// std::vector<std::string> warnings;
/// std::optional<assets::Model> model =
///     io::load_gltf("box.glb", &err, &warnings);
/// if (!model) { handle_error(err); return; }
/// for (const std::string& w : warnings) log_warning(w);
/// use(*model);
/// @endcode
VG_IO_API std::optional<assets::Model> load_gltf(
    std::string_view path, std::string* error = nullptr,
    std::vector<std::string>* warnings = nullptr);

// TODO: OBJ loader (load_obj) -- thin per-format loader into this same Model.
// TODO: PLY loader (load_ply) -- produces PointCloud for scanned/point data.
// TODO: assimp catch-all loader -- optional, covers the long tail of formats.

}  // namespace volumetric_kit::gfx::io

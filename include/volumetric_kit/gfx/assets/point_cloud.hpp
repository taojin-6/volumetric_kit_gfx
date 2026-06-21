// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file point_cloud.hpp
/// @brief Point cloud: positions plus arbitrary named per-point channels.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

namespace volumetric_kit::gfx::assets {

/// @brief A named, flat per-point attribute channel.
///
/// @ref values is `point_count * component_count` floats, point-major: point
/// `i` occupies `[i * component_count, (i + 1) * component_count)`. Naming the
/// channel (rather than fixing a struct) lets a loader preserve format-specific
/// attributes -- e.g. a PLY file's `intensity`, `confidence`, or a custom
/// scalar -- without the model needing to know them in advance.
struct PointAttribute {
  std::string name;              ///< Channel name (e.g. "color", "intensity").
  std::uint32_t components = 0;  ///< Floats per point (1=scalar, 3=vec3, ...).
  std::vector<float> values;     ///< Point-major, `count * components` floats.
};

/// @brief A point cloud: a position list plus zero or more named channels.
///
/// The model carries this type so a point-cloud source (e.g. a future PLY
/// loader) drops in without reshaping the model -- the glTF loader produces
/// @ref Mesh, not point clouds. Standard channels ("color", "normal") have
/// convenience accessors; everything else is reachable by name via @ref
/// attribute, so non-standard properties survive a load round-trip.
///
/// @code
/// assets::PointCloud cloud;
/// cloud.positions = scan_points;
/// cloud.add_attribute("intensity", 1, scan_intensity);
/// if (const auto* a = cloud.attribute("intensity"))
///   shade_by_scalar(a->values);
/// @endcode
struct PointCloud {
  std::string name;                        ///< Cloud name (may be empty).
  std::vector<glm::vec3> positions;        ///< One position per point.
  std::vector<PointAttribute> attributes;  ///< Named per-point channels.

  /// @return Number of points (`positions.size()`).
  std::size_t size() const noexcept { return positions.size(); }

  /// @brief Append a named channel.
  /// @param name        Channel name (need not be unique, but @ref attribute
  ///                    returns the first match).
  /// @param components  Floats per point.
  /// @param values      Point-major data; its size should be
  ///                    `size() * components`.
  void add_attribute(std::string name, std::uint32_t components,
                     std::vector<float> values) {
    attributes.push_back({std::move(name), components, std::move(values)});
  }

  /// @brief Find a channel by name.
  /// @param name  Channel name to look up.
  /// @return Pointer to the first matching channel, or `nullptr` if absent.
  const PointAttribute* attribute(std::string_view name) const noexcept {
    for (const PointAttribute& a : attributes) {
      if (a.name == name) return &a;
    }
    return nullptr;
  }

  /// @return The "color" channel (3 or 4 components), or `nullptr` if absent.
  const PointAttribute* color() const noexcept { return attribute("color"); }
  /// @return The "normal" channel (3 components), or `nullptr` if absent.
  const PointAttribute* normal() const noexcept { return attribute("normal"); }
};

}  // namespace volumetric_kit::gfx::assets

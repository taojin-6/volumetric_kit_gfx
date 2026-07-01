// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file orientation.hpp
/// @brief Shared level-horizon primitives for the camera tier -- the world-up
///        axis, the pole-clamp and min-gap constants, and the roll-free basis.
///
/// Both @ref volumetric_kit::gfx::camera::CameraRig and @ref
/// volumetric_kit::gfx::camera::OrbitCamera keep a horizon-level pose whose
/// vertical angle is held shy of the poles. These are the pieces they share, so
/// the two controllers agree on which axis is "up", how close to the pole they
/// clamp, and how the right/up view-plane axes fall out of a forward direction.
///
/// Internal header (`impl/`); consumers include `camera_rig.hpp` /
/// `orbit_camera.hpp`, which pull this in.

#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
//
#include <glm/geometric.hpp>  // cross, length, normalize
#include <glm/vec3.hpp>

namespace volumetric_kit::gfx::camera {

/// World up for the whole camera tier: the axis the level-horizon basis keeps
/// level and the elevation poles live on.
inline constexpr glm::vec3 kWorldUp(0.0f, 1.0f, 0.0f);

/// Pole-clamp magnitude: the vertical angle (pitch / elevation) is held in
/// `[-kPoleClampRadians, +kPoleClampRadians]`, just shy of a quarter turn, so
/// `forward` never aligns with @ref kWorldUp and the level basis never
/// degenerates.
inline constexpr float kPoleClampRadians = 1.5620697f;  // (pi/2) - ~0.0087 rad

/// Smallest pivot gap (a rig's focus distance, an orbit camera's distance) so
/// the eye and the pivot never coincide and collapse the view ray.
inline constexpr float kMinPivotDistance = 0.001f;

/// @brief The roll-free right/up view-plane axes for a forward direction.
struct LevelBasis {
  glm::vec3 right;  ///< Horizontal right axis (perpendicular to world up).
  glm::vec3 up;     ///< Up axis completing the right-handed basis.
};

/// @brief Derive the roll-free @ref LevelBasis for a forward direction, world
///        +Y up.
/// @param forward  View direction; need not be unit length.
/// @return `right` horizontal (perpendicular to @ref kWorldUp), `up` completing
///         the basis. Pole-safe: if @p forward is (anti)parallel to
///         @ref kWorldUp the horizon heading is undefined, so world +Z seeds
///         `right` instead -- callers that clamp shy of the pole never hit that
///         path.
inline LevelBasis level_basis(const glm::vec3& forward) {
  const glm::vec3 f = glm::normalize(forward);
  glm::vec3 right = glm::cross(f, kWorldUp);
  if (glm::length(right) < 1e-6f) {
    // forward is (anti)parallel to world up; seed the heading from world +Z.
    right = glm::cross(f, glm::vec3(0.0f, 0.0f, 1.0f));
  }
  right = glm::normalize(right);
  return {right, glm::cross(right, f)};
}

}  // namespace volumetric_kit::gfx::camera

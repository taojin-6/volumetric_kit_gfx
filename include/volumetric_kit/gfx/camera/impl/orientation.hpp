// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file orientation.hpp
/// @brief Controller math for the camera tier -- the min-gap constant, the
///        spherical parametrization, the pivot-distance clamp and the Camera
///        baking they share, and @ref volumetric_kit::gfx::camera::CameraRig's
///        world up, pole clamp and roll-free basis.
///
/// @ref volumetric_kit::gfx::camera::CameraRig and @ref
/// volumetric_kit::gfx::camera::OrbitCamera drive the same motion verbs over
/// different state (a quaternion pose vs. spherical angles). The shared pieces
/// are defined once so the controllers agree on what the spherical angles mean
/// and how a distance write survives a NaN -- structurally, rather than by
/// keeping two copies of the math in step. They do not share an up or a clamp:
/// the orbit camera's up is settable and its angles have no stop.
///
/// Internal header (`impl/`); consumers include `camera_rig.hpp` /
/// `orbit_camera.hpp`, which pull this in.

#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
//
#include <algorithm>  // std::max
#include <cmath>      // std::cos, std::sin

#include <glm/geometric.hpp>  // cross, length, normalize
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace volumetric_kit::gfx::camera {

/// @ref CameraRig's world up: the axis its level-horizon basis keeps level and
/// its pitch poles live on. Also the up an @ref OrbitCamera starts with.
inline constexpr glm::vec3 kWorldUp(0.0f, 1.0f, 0.0f);

/// @ref CameraRig's pole clamp: its pitch in level-horizon mode is held in
/// `[-kPoleClampRadians, +kPoleClampRadians]`, just shy of a quarter turn, so
/// `forward` never aligns with @ref kWorldUp and the level basis never
/// degenerates.
inline constexpr float kPoleClampRadians = 1.5620697f;  // (pi/2) - ~0.0087 rad

/// Smallest pivot gap (a rig's focus distance, an orbit camera's distance) so
/// the eye and the pivot never coincide and collapse the view ray.
inline constexpr float kMinPivotDistance = 0.001f;

/// @brief Floor a pivot distance (a rig's focus distance, an orbit camera's
///        distance) to @ref kMinPivotDistance.
/// @param distance  Proposed new distance, straight from input arithmetic; may
///                  be zero, negative, or NaN.
/// @return @p distance, or @ref kMinPivotDistance if it does not compare
///         greater. The floor comes *first* on purpose: `std::max` returns its
///         first argument when neither compares greater, so a NaN clamps to
///         the floor here, whereas `std::max(distance, kMinPivotDistance)`
///         would return the NaN and permanently poison the pose it feeds.
inline float clamp_pivot_distance(float distance) noexcept {
  return std::max(kMinPivotDistance, distance);
}

/// @brief The unit direction at `azimuth` about +Y and `elevation` above the
///        horizontal, anchored on +Z.
/// @param azimuth    Yaw, in radians: 0 points along +Z; a positive turn swings
///                   toward +X.
/// @param elevation  Pitch, in radians: 0 is horizontal; positive lifts toward
///                   +Y.
/// @return The unit direction for those angles.
///
/// One parametrization serves both controllers, negated between them:
/// @ref OrbitCamera offsets its eye *along* it, in the level frame of its up
/// (`eye = target + distance * level_frame * spherical_direction(azimuth,
/// elevation)`; with a +Y up that frame is the identity), while @ref
/// CameraRig's level look ray runs *back down* it (`forward =
/// -spherical_direction(yaw, -elevation)`, so forward is world -Z at rest).
/// Negating both the vector and the elevation keeps the vertical sense shared:
/// the pitch that tilts a rig's view down is the elevation that raises a
/// turntable eye.
inline glm::vec3 spherical_direction(float azimuth, float elevation) {
  const float cos_elevation = std::cos(elevation);
  return glm::vec3(cos_elevation * std::sin(azimuth), std::sin(elevation),
                   cos_elevation * std::cos(azimuth));
}

/// @brief The roll-free right/up view-plane axes for a forward direction.
struct LevelBasis {
  glm::vec3 right;  ///< Horizontal right axis (perpendicular to world up).
  glm::vec3 up;     ///< Up axis completing the right-handed basis.
};

/// @brief Derive @ref CameraRig's roll-free @ref LevelBasis for a forward
///        direction, world +Y up.
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

/// @brief Pair a controller's view matrix with a perspective projection.
/// @param view          The world->view matrix (see @ref Camera::set_view).
/// @param fovy_radians  Vertical field of view, in radians (`(0, pi)`).
/// @param aspect        Viewport width / height (> 0).
/// @param z_near        Near plane distance (`> 0`).
/// @param z_far         Far plane distance (`> z_near`).
/// @return A @ref Camera carrying the given view and frustum.
inline Camera bake_camera(const glm::mat4& view, float fovy_radians,
                          float aspect, float z_near, float z_far) {
  Camera camera;
  camera.set_view(view);
  camera.set_perspective(fovy_radians, aspect, z_near, z_far);
  return camera;
}

}  // namespace volumetric_kit::gfx::camera

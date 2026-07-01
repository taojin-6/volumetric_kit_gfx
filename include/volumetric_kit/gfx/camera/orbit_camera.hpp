// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file orbit_camera.hpp
/// @brief A turntable controller that orbits, pans, and dollies around a point.
///
/// @ref OrbitCamera holds the *state* a turntable / "arcball-lite" interaction
/// drives -- a `target` point and a spherical offset `{distance, azimuth,
/// elevation}` -- and derives the eye position and view matrix from it. It owns
/// no GPU resources and depends only on glm. Feed pointer deltas into
/// @ref orbit / @ref pan / @ref dolly, then read @ref view or bake a full
/// @ref to_camera (adding a projection) for submission.

#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/camera/export.hpp"
#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
#include "volumetric_kit/gfx/camera/impl/orientation.hpp"
//
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace volumetric_kit::gfx::camera {

/// @brief Turntable camera state: a target plus a spherical offset to the eye.
///
/// The eye sits on a sphere of radius @ref distance around @ref target, at
/// @ref azimuth (yaw, radians, around world +Y) and @ref elevation (pitch,
/// radians, measured from the horizontal plane). At `azimuth == 0,
/// elevation == 0` the eye is directly in front of the target on +Z, looking
/// toward -Z, with world +Y up.
///
/// @ref elevation is clamped just shy of the poles (@ref kMaxElevation) so the
/// view direction never becomes parallel to world up -- which would collapse
/// the look-at basis (gimbal flip). The world up is +Y.
///
/// @code
/// using namespace volumetric_kit::gfx::camera;
/// OrbitCamera orbit;
/// orbit.set_target({0, 0, 0});
/// orbit.set_distance(5.0f);
/// orbit.orbit(glm::radians(30.0f), glm::radians(20.0f));  // drag to rotate
/// orbit.dolly(-1.0f);                                     // wheel to zoom in
/// Camera cam = orbit.to_camera(glm::radians(60.0f), 16.0f / 9, 0.1f, 100.0f);
/// @endcode
class VG_CAMERA_API OrbitCamera {
 public:
  /// @brief Elevation clamp magnitude: just under a quarter turn (`~89.5°`).
  ///
  /// @ref elevation is held in `[-kMaxElevation, +kMaxElevation]` so the eye
  /// never reaches a pole, where `forward` would align with world up and the
  /// look-at basis would degenerate. Shares the camera tier's
  /// @ref kPoleClampRadians with @ref CameraRig::kMaxPitch.
  static constexpr float kMaxElevation = kPoleClampRadians;

  /// @brief Smallest allowed @ref distance, so the eye cannot reach the target.
  static constexpr float kMinDistance = kMinPivotDistance;

  /// Construct a default controller: target at the origin, unit distance,
  /// azimuth and elevation zero (eye in front on +Z).
  OrbitCamera() = default;

  /// @brief Rotate the eye around the target.
  /// @param delta_azimuth    Yaw change, in radians (added to @ref azimuth).
  /// @param delta_elevation  Pitch change, in radians; the result is clamped to
  ///                         `[-kMaxElevation, +kMaxElevation]`.
  void orbit(float delta_azimuth, float delta_elevation);

  /// @brief Slide the target (and the eye with it) in the view plane.
  /// @param delta_right  Motion along the camera's right axis, in world units;
  ///                     `+` moves the target right, so the scene appears to
  ///                     move left.
  /// @param delta_up     Motion along the camera's up axis, in world units; `+`
  ///                     moves the target up.
  ///
  /// The deltas are world-unit distances in the current view plane (right / up
  /// derived from the present orientation); scale them by @ref distance at the
  /// call site for a zoom-consistent screen feel.
  void pan(float delta_right, float delta_up);

  /// @brief Move the eye toward or away from the target along the view ray.
  /// @param delta  Change in @ref distance, in world units; `-` moves closer
  ///               (zoom in), `+` moves away. The result is clamped to at least
  ///               @ref kMinDistance.
  void dolly(float delta);

  /// @brief Scale @ref distance multiplicatively (a wheel-style zoom).
  /// @param factor  Positive multiplier applied to @ref distance (`< 1` zooms
  ///                in, `> 1` zooms out). The result is clamped to at least
  ///                @ref kMinDistance.
  /// @pre @p factor is > 0.
  void zoom(float factor);

  /// @brief Set the point the camera orbits.
  /// @param target  New world-space target.
  void set_target(const glm::vec3& target) noexcept { target_ = target; }
  /// @brief Set the eye's distance from the target.
  /// @param distance  New radius; clamped to at least @ref kMinDistance.
  void set_distance(float distance) noexcept;
  /// @brief Set the yaw angle.
  /// @param azimuth  New azimuth, in radians.
  void set_azimuth(float azimuth) noexcept { azimuth_ = azimuth; }
  /// @brief Set the pitch angle.
  /// @param elevation  New elevation, in radians; clamped to
  ///                   `[-kMaxElevation, +kMaxElevation]`.
  void set_elevation(float elevation) noexcept;

  /// @return The world-space point the camera orbits.
  const glm::vec3& target() const noexcept { return target_; }
  /// @return The eye's distance from the target.
  float distance() const noexcept { return distance_; }
  /// @return The yaw angle, in radians.
  float azimuth() const noexcept { return azimuth_; }
  /// @return The pitch angle, in radians.
  float elevation() const noexcept { return elevation_; }

  /// @return The world-space eye position derived from the current state.
  glm::vec3 eye() const;

  /// @return The world->view matrix looking from @ref eye at @ref target with
  ///         world +Y up.
  glm::mat4 view() const;

  /// @brief Bake a @ref Camera by pairing this view with a perspective frustum.
  /// @param fovy_radians  Vertical field of view, in radians (`(0, pi)`).
  /// @param aspect        Viewport width / height (> 0).
  /// @param z_near        Near plane distance (`> 0`).
  /// @param z_far         Far plane distance (`> z_near`).
  /// @return A camera with this controller's view and the given projection.
  Camera to_camera(float fovy_radians, float aspect, float z_near,
                   float z_far) const;

 private:
  glm::vec3 target_ = glm::vec3(0.0f);
  float distance_ = 1.0f;
  float azimuth_ = 0.0f;
  float elevation_ = 0.0f;
};

}  // namespace volumetric_kit::gfx::camera

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file orbit_camera.hpp
/// @brief A free turntable that orbits, pans and dollies around a point.
///
/// @ref OrbitCamera holds a `target`, a `distance`, the world's up and two
/// angles about it, and derives the eye and the view matrix from them. Drags
/// turn the eye about the world's up (yaw) and about the camera's own right
/// (pitch) with no stop: the eye passes over the top and under the bottom,
/// where the picture is upside down, and the view never rolls. It owns no GPU
/// resources and depends only on glm. Feed drags into @ref drag_orbit /
/// @ref drag_pan and wheel notches or a pinch into @ref zoom (or angles and
/// world units into @ref orbit / @ref pan / @ref dolly), then read @ref view or
/// bake a full @ref to_camera (adding a projection) for submission.

#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/camera/export.hpp"
#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
#include "volumetric_kit/gfx/camera/impl/orientation.hpp"
//
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace volumetric_kit::gfx::camera {

/// @brief Free-turntable state: a target, a distance, the world's up and two
///        angles about it.
///
/// At `azimuth == 0, elevation == 0` the camera is level and world +X (its
/// part across the up; `up x +Z` for an up along x) points right on screen.
/// With the default +Y up the eye is on +Z looking toward -Z; with +Z up, on
/// the -Y side looking toward +Y; with -Y up, on the -Z side looking toward +Z.
///
/// Neither angle is clamped; both are kept in `[-pi, pi]`. Past
/// `|elevation| == pi / 2` the eye has gone over a pole and the picture is
/// upside down. The camera's right stays across the world's up, so the view
/// never rolls; the view is built from the angles, not a look-at, so it is
/// defined at the poles too.
///
/// The camera is a value: keep a copy of the start pose and assign it back to
/// reset the view.
///
/// @code
/// using namespace volumetric_kit::gfx::camera;
/// OrbitCamera orbit;
/// orbit.set_world_up({0, 0, 1});     // the stream's world: +z up
/// orbit.set_target({0, 0, 0.85f});
/// orbit.set_distance(3.0f);
/// const OrbitCamera home = orbit;    // the start view
/// const glm::vec2 drag = (cursor - last_cursor) / viewport_height;
/// orbit.drag_orbit(drag);                                     // left drag
/// orbit.drag_pan(drag, fovy);                                 // right drag
/// orbit.zoom(std::pow(OrbitCamera::kZoomPerNotch, notches));  // wheel
/// if (double_click) orbit = home;                             // reset
/// Camera cam = orbit.to_camera(fovy, 16.0f / 9, 0.1f, 100.0f);
/// @endcode
class VG_CAMERA_API OrbitCamera {
 public:
  /// @brief Smallest allowed @ref distance, so the eye cannot reach the target.
  ///        Shares the camera tier's @ref kMinPivotDistance with
  ///        @ref CameraRig::kMinFocusDistance.
  static constexpr float kMinDistance = kMinPivotDistance;

  /// @brief Radians the eye turns per viewport height dragged: half a turn.
  static constexpr float kDragRadians = 3.14159265f;

  /// @brief One wheel notch's zoom: `zoom(std::pow(kZoomPerNotch, notches))`.
  static constexpr float kZoomPerNotch = 0.9f;

  /// Target at the origin, unit distance, world +Y up, angles zero (eye on +Z).
  OrbitCamera() = default;

  /// @brief Turn the eye around the target.
  /// @param delta_yaw    Radians about @ref world_up; `+` swings the eye toward
  ///                     the camera's right. Past a pole it is subtracted from
  ///                     @ref azimuth, so that stays true; the sign is chosen
  ///                     per call from the current elevation.
  /// @param delta_pitch  Radians about the camera's right, added to
  ///                     @ref elevation; `+` swings the eye toward the camera's
  ///                     up. Not clamped.
  ///
  /// A non-finite delta is ignored. A zero delta changes nothing.
  void orbit(float delta_yaw, float delta_pitch);

  /// @brief Slide the target (and the eye with it) in the view plane.
  /// @param delta_right  Motion along the camera's right axis, in world units;
  ///                     `+` moves the target right, so the scene appears to
  ///                     move left.
  /// @param delta_up     Motion along the camera's up axis, in world units; `+`
  ///                     moves the target up on screen, also when the picture
  ///                     is upside down.
  ///
  /// The deltas are world-unit distances in the current view plane (right / up
  /// derived from the present orientation); @ref drag_pan takes a drag on
  /// screen instead. A move that would leave the target non-finite is ignored.
  void pan(float delta_right, float delta_up);

  /// @brief Move the eye toward or away from the target along the view ray.
  /// @param delta  Change in @ref distance, in world units; `-` moves closer
  ///               (zoom in), `+` moves away. The result is clamped to at least
  ///               @ref kMinDistance.
  void dolly(float delta);

  /// @brief Scale @ref distance multiplicatively (a wheel or pinch zoom).
  /// @param factor  Multiplier applied to @ref distance (`< 1` zooms in, `> 1`
  ///                zooms out); a pinch is `1 / scale`. The result is clamped
  ///                to at least @ref kMinDistance. A non-finite or
  ///                non-positive factor (a pinch can report 0) is ignored.
  void zoom(float factor);

  /// @brief Orbit by a drag; the scene follows the pointer, upright or upside
  ///        down.
  /// @param drag  The move in viewport heights, +x right, +y down: a cursor or
  ///              finger delta over the viewport's height in the same unit, so
  ///              window pixels, framebuffer pixels and UIKit points agree.
  ///
  /// @ref kDragRadians per height on both axes: a drag down raises the eye;
  /// half a height from level reaches the top pole, one height the far side. A
  /// non-finite drag (a delta over a zero height) is ignored.
  void drag_orbit(glm::vec2 drag);

  /// @brief Pan by a drag; the point at the target's depth stays under it.
  /// @param drag          As @ref drag_orbit.
  /// @param fovy_radians  The vertical field of view the view is drawn with
  ///                      (`(0, pi)`).
  ///
  /// A non-finite drag or field of view is ignored.
  void drag_pan(glm::vec2 drag, float fovy_radians);

  /// @brief Set the point the camera orbits.
  /// @param target  New world-space target.
  void set_target(const glm::vec3& target) noexcept { target_ = target; }
  /// @brief Set the eye's distance from the target.
  /// @param distance  New radius; clamped to at least @ref kMinDistance. An
  ///                  infinite one, here or from @ref dolly or @ref zoom, is
  ///                  ignored.
  void set_distance(float distance) noexcept;

  /// @brief Set the world's up: the axis the camera turns about and keeps
  ///        upright on screen.
  /// @param up  Any length: +Y for normalized mesh sequences, +Z for a rig's
  ///            tag world. Ignored when it has no finite, non-zero length.
  ///
  /// The target, distance and angles are kept, now measured about the new up,
  /// so the view re-aims at the same azimuth and elevation and the call
  /// commutes with @ref set_azimuth / @ref set_elevation. A target that depends
  /// on the up (a point above the floor) is the caller's to move.
  void set_world_up(const glm::vec3& up) noexcept;

  /// @brief Set the yaw about @ref world_up.
  /// @param azimuth  Radians, kept in `[-pi, pi]`; non-finite is ignored.
  void set_azimuth(float azimuth) noexcept;
  /// @brief Set the pitch above the plane across @ref world_up.
  /// @param elevation  Radians, kept in `[-pi, pi]`; beyond `+-pi / 2` the eye
  ///                   is past a pole, upside down; non-finite is ignored.
  void set_elevation(float elevation) noexcept;

  /// @return The world-space point the camera orbits.
  const glm::vec3& target() const noexcept { return target_; }
  /// @return The eye's distance from the target.
  float distance() const noexcept { return distance_; }
  /// @return The unit world up.
  const glm::vec3& world_up() const noexcept { return world_up_; }
  /// @return The yaw angle, in radians.
  float azimuth() const noexcept { return azimuth_; }
  /// @return The pitch angle, in radians.
  float elevation() const noexcept { return elevation_; }

  /// @return The world-space eye position derived from the current state.
  glm::vec3 eye() const;

  /// @return The world->view matrix: the camera's right, up and back axes as
  ///         its rows. Defined at every angle, the poles included.
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
  glm::vec3 world_up_ = kWorldUp;
  float azimuth_ = 0.0f;
  float elevation_ = 0.0f;
};

}  // namespace volumetric_kit::gfx::camera

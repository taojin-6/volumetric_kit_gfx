// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera_rig.hpp
/// @brief A 6-DoF camera rig: one pose driven by rigid-motion verbs, from which
///        orbit and fly controls both fall out.
///
/// @ref CameraRig holds a single rigid pose -- a world-space `position` plus an
/// `orientation` quaternion -- and a `focus_distance` naming a pivot point
/// ahead of the eye. Input is applied through motion *verbs* (@ref orbit, @ref
/// look,
/// @ref move_local, @ref pan, @ref zoom, @ref snap_turn), not by picking a
/// camera "type": mouse-drag maps to @ref orbit (swing around the pivot),
/// WASDQE maps to @ref move_local (translate along the local axes), and
/// mouse-look maps to @ref look (rotate in place). It owns no GPU resources and
/// depends only on glm.
///
/// This is the app-controlled *stage* transform. For a mono window read
/// @ref to_camera (view + projection ready to submit); for stereo/XR read
/// @ref stage_transform and compose the runtime's per-eye poses under it.
///
/// @par Horizon
/// With @ref level_horizon set (the default), turning stays yaw-about-world-up
/// with pitch clamped shy of the poles (@ref kMaxPitch), so the horizon never
/// rolls -- the comfortable feel for a viewer or a VR locomotion rig. Clear it
/// for free 6-DoF (roll accumulates; no pole clamp) -- e.g. inspecting a volume
/// from an arbitrary angle.

#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/camera/export.hpp"
#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
#include "volumetric_kit/gfx/camera/impl/orientation.hpp"
//
#include <glm/gtc/quaternion.hpp>  // glm::quat
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace volumetric_kit::gfx::camera {

/// @brief A single rigid camera pose plus a focus pivot, driven by motion
/// verbs.
///
/// The eye sits at @ref position looking down its local `-Z` with local `+Y`
/// up, rotated by @ref orientation. The pivot -- what @ref orbit swings around
/// and
/// @ref zoom moves toward -- lies @ref focus_distance ahead along the view ray
/// (@ref focus_point). A freshly constructed rig sits at the origin looking
/// down world `-Z`, world `+Y` up, with unit focus distance.
///
/// @code
/// using namespace volumetric_kit::gfx::camera;
/// CameraRig rig;
/// rig.set_position({0, 1, 4});
/// rig.set_focus({0, 0, 0});           // aim at the model
/// rig.orbit(dx * 0.01f, dy * 0.01f);  // mouse drag -> orbit
/// rig.move_local({0, 0, dz});         // 'W' -> fly forward
/// rig.zoom(0.9f);                     // wheel -> dolly in
/// Camera cam = rig.to_camera(glm::radians(60.0f), 16.0f / 9, 0.1f, 100.0f);
/// @endcode
class VG_CAMERA_API CameraRig {
 public:
  /// @brief Pitch clamp magnitude in @ref level_horizon mode: just under a
  ///        quarter turn (`~89.5°`).
  ///
  /// The look direction's elevation is held in `[-kMaxPitch, +kMaxPitch]` so it
  /// never reaches a pole, where `forward` would align with world up and the
  /// level-horizon basis would degenerate. Ignored when @ref level_horizon is
  /// cleared. Shares the camera tier's @ref kPoleClampRadians with
  /// @ref OrbitCamera::kMaxElevation.
  static constexpr float kMaxPitch = kPoleClampRadians;

  /// @brief Smallest allowed @ref focus_distance, so the pivot never reaches
  /// the
  ///        eye (which would collapse the view ray).
  static constexpr float kMinFocusDistance = kMinPivotDistance;

  /// Construct the default rig: at the origin, looking down world `-Z` with
  /// world `+Y` up, unit focus distance, level horizon.
  CameraRig() = default;

  // --- motion verbs --------------------------------------------------------

  /// @brief Swing the eye around the focus pivot (the drag-to-rotate verb).
  /// @param delta_yaw    Yaw change, in radians; `+` turns left (counter-
  ///                     clockwise about world up), matching @ref OrbitCamera.
  /// @param delta_pitch  Pitch change, in radians; `+` raises the eye over the
  ///                     pivot (tilting the view down), matching
  ///                     @ref OrbitCamera's elevation sign. In
  ///                     @ref level_horizon mode it is clamped to
  ///                     `[-kMaxPitch, +kMaxPitch]`.
  ///
  /// @ref focus_point and @ref focus_distance are preserved: the eye moves
  /// along a sphere around the pivot while the look direction stays aimed at
  /// it.
  void orbit(float delta_yaw, float delta_pitch);

  /// @brief Rotate the look direction in place, leaving @ref position fixed
  /// (the
  ///        mouse-look / turn verb). The pivot rides with the new look ray.
  /// @param delta_yaw    Yaw change, in radians (see @ref orbit for the sign).
  /// @param delta_pitch  Pitch change, in radians; shares @ref orbit's sign, so
  ///                     `+` tilts the view down (clamped in
  ///                     @ref level_horizon mode). A mouse-look mapping that
  ///                     wants drag-up to look up negates the pointer delta.
  void look(float delta_yaw, float delta_pitch);

  /// @brief Translate the eye along its own axes (the WASDQE verb).
  /// @param delta  Motion in local units: `x` along @ref right, `y` along
  ///               @ref up, `z` along @ref forward (`+z` flies forward). The
  ///               pivot rides along, so the view direction is unchanged.
  void move_local(const glm::vec3& delta);

  /// @brief Slide the eye within its view plane (the screen-space pan verb).
  /// @param delta_right  Motion along @ref right, in world units.
  /// @param delta_up     Motion along @ref up, in world units.
  ///
  /// Equivalent to @ref move_local with no forward component. Scale the deltas
  /// by @ref focus_distance at the call site for a zoom-consistent screen feel.
  void pan(float delta_right, float delta_up);

  /// @brief Dolly the eye toward or away from the pivot (the wheel-zoom verb).
  /// @param factor  Positive multiplier applied to @ref focus_distance (`< 1`
  ///                moves closer, `> 1` moves away). @ref focus_point is held
  ///                fixed; the result is clamped to at least
  ///                @ref kMinFocusDistance.
  /// @pre @p factor is > 0.
  void zoom(float factor);

  /// @brief Yaw the rig in place about world up (the VR comfort snap-turn
  /// verb).
  /// @param delta_yaw  Yaw change, in radians. Always level (about world up),
  ///                   independent of @ref level_horizon, and leaves
  ///                   @ref position fixed.
  void snap_turn(float delta_yaw);

  // --- state ---------------------------------------------------------------

  /// @brief Move the eye without changing where it looks.
  /// @param position  New world-space eye position.
  void set_position(const glm::vec3& position) noexcept {
    position_ = position;
  }

  /// @brief Aim the eye exactly at a world point, level, and set
  ///        @ref focus_distance to the gap so @ref focus_point lands on it.
  /// @param target  World-space point to look at. Ignored if it coincides with
  ///                @ref position (which has no well-defined direction).
  ///
  /// The resulting orientation is roll-free (horizon level) whatever
  /// @ref level_horizon is; a target directly overhead/underfoot is aimed at
  /// exactly, but the next @ref level_horizon turn then re-clamps shy of the
  /// pole.
  void set_focus(const glm::vec3& target);

  /// @brief Set the pivot distance ahead of the eye along the view ray.
  /// @param distance  New focus distance; clamped to at least
  ///                  @ref kMinFocusDistance.
  void set_focus_distance(float distance) noexcept;

  /// @brief Choose the horizon behavior of the rotation verbs.
  /// @param level  `true` keeps the horizon level (yaw/pitch, pole-clamped);
  ///               `false` allows free 6-DoF rotation with roll.
  /// @note Re-enabling level mode from a near-vertical free-mode orientation
  ///       re-levels on the next @ref orbit / @ref look: any accumulated roll
  ///       is dropped and, within a pole margin, the heading snaps (the
  ///       horizontal yaw is undefined when @ref forward is vertical).
  void set_level_horizon(bool level) noexcept { level_horizon_ = level; }

  // --- reads ---------------------------------------------------------------

  /// @return The world-space eye position.
  const glm::vec3& position() const noexcept { return position_; }
  /// @return The eye orientation (a rotation taking local axes to world axes).
  const glm::quat& orientation() const noexcept { return orientation_; }
  /// @return The pivot distance ahead of the eye along the view ray.
  float focus_distance() const noexcept { return focus_distance_; }
  /// @return Whether the rotation verbs keep the horizon level (see
  ///         @ref set_level_horizon).
  bool level_horizon() const noexcept { return level_horizon_; }

  /// @return The unit forward axis (the view direction, local `-Z` in world).
  glm::vec3 forward() const;
  /// @return The unit right axis (local `+X` in world).
  glm::vec3 right() const;
  /// @return The unit up axis (local `+Y` in world).
  glm::vec3 up() const;

  /// @return The world-space pivot: @ref position plus @ref forward scaled by
  ///         @ref focus_distance.
  glm::vec3 focus_point() const;

  /// @return The rig pose as a local->world (eye-to-world) transform. Compose a
  ///         runtime per-eye pose under this and invert for a view matrix; for
  ///         a mono view prefer @ref to_camera.
  glm::mat4 stage_transform() const;

  /// @brief Bake a @ref Camera for a single (mono) view.
  /// @param fovy_radians  Vertical field of view, in radians (`(0, pi)`).
  /// @param aspect        Viewport width / height (> 0).
  /// @param z_near        Near plane distance (`> 0`).
  /// @param z_far         Far plane distance (`> z_near`).
  /// @return A camera pairing this rig's view with the given perspective
  ///         frustum.
  Camera to_camera(float fovy_radians, float aspect, float z_near,
                   float z_far) const;

 private:
  // Apply a yaw/pitch rotation to orientation_ per the level_horizon_ policy;
  // the verbs layer position handling (orbit/look) on top.
  void rotate(float delta_yaw, float delta_pitch);

  glm::vec3 position_ = glm::vec3(0.0f);
  glm::quat orientation_ =
      glm::quat(1.0f, 0.0f, 0.0f, 0.0f);  // identity (w,x,y,z)
  float focus_distance_ = 1.0f;
  bool level_horizon_ = true;
};

}  // namespace volumetric_kit::gfx::camera

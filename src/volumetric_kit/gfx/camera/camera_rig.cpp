// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/camera/camera_rig.hpp"

#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
#include "volumetric_kit/gfx/camera/impl/orientation.hpp"  // shared controller math
//
#include <algorithm>  // std::clamp
#include <cmath>      // std::asin, std::atan2

#include <glm/ext/matrix_transform.hpp>  // translate
#include <glm/geometric.hpp>             // length, normalize
#include <glm/gtc/quaternion.hpp>        // angleAxis, quat_cast, mat4_cast
#include <glm/matrix.hpp>                // inverse

namespace volumetric_kit::gfx::camera {

namespace {

// A roll-free orientation looking along `forward` with world +Y up. Pole-safe
// via level_basis, so an exactly-vertical `forward` (e.g. from set_focus) is
// handled; the level-horizon verbs clamp shy of the pole and never reach it.
glm::quat level_orientation(const glm::vec3& forward) {
  const glm::vec3 f = glm::normalize(forward);
  const LevelBasis basis = level_basis(f);
  // Camera basis columns: X = right, Y = up, Z = backward (-forward).
  return glm::normalize(glm::quat_cast(glm::mat3(basis.right, basis.up, -f)));
}

// The roll-free orientation whose forward points at `yaw` (about world up) and
// `elevation` (angle above the horizontal): the level look ray runs back down
// the shared spherical direction (see spherical_direction for the sign
// relationship), so forward is world -Z at (0, 0), a positive yaw turns left,
// and a positive elevation lifts forward toward world up. rotate() maps the
// public pitch sign onto `elevation`.
glm::quat level_look(float yaw, float elevation) {
  return level_orientation(-spherical_direction(yaw, -elevation));
}

}  // namespace

void CameraRig::rotate(float delta_yaw, float delta_pitch) {
  // Positive delta_pitch tilts the view down -- raising the eye when orbiting.
  // `elevation` below measures forward's lift toward world +Y, the opposite
  // sense, so negate the incoming pitch.
  const float delta_elevation = -delta_pitch;
  if (level_horizon_) {
    // Decompose the current look into yaw + elevation, advance them, clamp the
    // elevation shy of the poles, and rebuild a roll-free orientation.
    const glm::vec3 f = forward();
    const float elevation = std::asin(std::clamp(f.y, -1.0f, 1.0f));
    // TODO: yaw is undefined when forward is vertical (reachable only by
    // entering level mode from a near-vertical free / set_focus pose), so the
    // heading snaps to 0; a stored heading would preserve it.
    const float yaw = std::atan2(-f.x, -f.z);
    orientation_ = level_look(
        yaw + delta_yaw,
        std::clamp(elevation + delta_elevation, -kMaxPitch, kMaxPitch));
  } else {
    // Free 6-DoF: compose in the local frame, so repeated yaw + pitch
    // accumulate roll and the pitch is unclamped (the look ray may pass the
    // poles).
    const glm::quat rot = glm::angleAxis(delta_yaw, up()) *
                          glm::angleAxis(delta_elevation, right());
    orientation_ = glm::normalize(rot * orientation_);
  }
}

void CameraRig::orbit(float delta_yaw, float delta_pitch) {
  const glm::vec3 pivot = focus_point();
  rotate(delta_yaw, delta_pitch);
  // Swing the eye back onto the sphere so the pivot and focus distance hold.
  position_ = pivot - forward() * focus_distance_;
}

void CameraRig::look(float delta_yaw, float delta_pitch) {
  rotate(delta_yaw, delta_pitch);
}

void CameraRig::move_local(const glm::vec3& delta) {
  // right()*x + up()*y + forward()*z, collapsed into one rotation: forward() is
  // local -Z, so negate the local delta's z before rotating it into the world.
  position_ += orientation_ * glm::vec3(delta.x, delta.y, -delta.z);
}

void CameraRig::pan(float delta_right, float delta_up) {
  move_local(glm::vec3(delta_right, delta_up, 0.0f));
}

void CameraRig::zoom(float factor) {
  const glm::vec3 pivot = focus_point();
  focus_distance_ = clamp_pivot_distance(focus_distance_ * factor);
  position_ = pivot - forward() * focus_distance_;
}

void CameraRig::snap_turn(float delta_yaw) {
  // Always level (about world up), independent of level_horizon_.
  orientation_ =
      glm::normalize(glm::angleAxis(delta_yaw, kWorldUp) * orientation_);
}

void CameraRig::set_focus(const glm::vec3& target) {
  const glm::vec3 to_target = target - position_;
  const float distance = glm::length(to_target);
  if (distance <= kMinFocusDistance) {
    return;  // no well-defined look direction toward a coincident point
  }
  focus_distance_ = distance;
  // Aim exactly along the target ray (roll-free, pole-safe) so focus_point()
  // lands on the target.
  orientation_ = level_orientation(to_target / distance);
}

void CameraRig::set_focus_distance(float distance) noexcept {
  focus_distance_ = clamp_pivot_distance(distance);
}

glm::vec3 CameraRig::forward() const {
  return orientation_ * glm::vec3(0.0f, 0.0f, -1.0f);
}

glm::vec3 CameraRig::right() const {
  return orientation_ * glm::vec3(1.0f, 0.0f, 0.0f);
}

glm::vec3 CameraRig::up() const {
  return orientation_ * glm::vec3(0.0f, 1.0f, 0.0f);
}

glm::vec3 CameraRig::focus_point() const {
  return position_ + forward() * focus_distance_;
}

glm::mat4 CameraRig::stage_transform() const {
  return glm::translate(glm::mat4(1.0f), position_) *
         glm::mat4_cast(orientation_);
}

Camera CameraRig::to_camera(float fovy_radians, float aspect, float z_near,
                            float z_far) const {
  // The view matrix is the inverse of the eye-to-world stage pose.
  return bake_camera(glm::inverse(stage_transform()), fovy_radians, aspect,
                     z_near, z_far);
}

}  // namespace volumetric_kit::gfx::camera

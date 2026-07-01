// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/camera/camera_rig.hpp"

#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
//
#include <algorithm>  // std::clamp, std::max
#include <cmath>      // std::asin, std::atan2, std::cos, std::sin

#include <glm/ext/matrix_transform.hpp>  // translate
#include <glm/geometric.hpp>             // cross, length, normalize
#include <glm/gtc/quaternion.hpp>        // angleAxis, quat_cast, mat4_cast
#include <glm/matrix.hpp>                // inverse

namespace volumetric_kit::gfx::camera {

namespace {

// World up. Shared by the level-horizon basis and the elevation poles, so the
// clamp and the horizon agree on which axis "up" is.
constexpr glm::vec3 kWorldUp(0.0f, 1.0f, 0.0f);

// A roll-free orientation looking along `forward` with world +Y up. The caller
// guarantees `forward` is not parallel to world up (level-horizon pitch is
// clamped shy of the poles), so the cross product never degenerates.
glm::quat level_orientation(const glm::vec3& forward) {
  const glm::vec3 f = glm::normalize(forward);
  const glm::vec3 right = glm::normalize(glm::cross(f, kWorldUp));
  const glm::vec3 up = glm::cross(right, f);
  // Camera basis columns: X = right, Y = up, Z = backward (-forward).
  return glm::normalize(glm::quat_cast(glm::mat3(right, up, -f)));
}

// The roll-free orientation for a look direction given as yaw (about world up)
// and pitch (elevation). At yaw 0 / pitch 0 the look direction is world -Z, so
// a positive yaw turns left and a positive pitch tilts up -- matching
// OrbitCamera.
glm::quat level_look(float yaw, float pitch) {
  const float cos_pitch = std::cos(pitch);
  const glm::vec3 forward(-cos_pitch * std::sin(yaw), std::sin(pitch),
                          -cos_pitch * std::cos(yaw));
  return level_orientation(forward);
}

}  // namespace

void CameraRig::rotate(float delta_yaw, float delta_pitch) {
  if (level_horizon_) {
    // Decompose the current look into yaw + elevation, advance them, clamp the
    // pitch shy of the poles, and rebuild a roll-free orientation.
    const glm::vec3 f = forward();
    const float pitch = std::asin(std::clamp(f.y, -1.0f, 1.0f));
    const float yaw = std::atan2(-f.x, -f.z);
    orientation_ =
        level_look(yaw + delta_yaw,
                   std::clamp(pitch + delta_pitch, -kMaxPitch, kMaxPitch));
  } else {
    // Free 6-DoF: compose in the local frame, so repeated yaw + pitch
    // accumulate roll and the pitch is unclamped (the look ray may pass the
    // poles).
    const glm::quat rot =
        glm::angleAxis(delta_yaw, up()) * glm::angleAxis(delta_pitch, right());
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
  position_ += right() * delta.x + up() * delta.y + forward() * delta.z;
}

void CameraRig::pan(float delta_right, float delta_up) {
  position_ += right() * delta_right + up() * delta_up;
}

void CameraRig::zoom(float factor) {
  const glm::vec3 pivot = focus_point();
  focus_distance_ = std::max(focus_distance_ * factor, kMinFocusDistance);
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
  const glm::vec3 dir = to_target / distance;
  // Aim level, clamping elevation shy of the pole so the basis stays defined.
  const float pitch = std::clamp(std::asin(std::clamp(dir.y, -1.0f, 1.0f)),
                                 -kMaxPitch, kMaxPitch);
  orientation_ = level_look(std::atan2(-dir.x, -dir.z), pitch);
}

void CameraRig::set_focus_distance(float distance) noexcept {
  focus_distance_ = std::max(distance, kMinFocusDistance);
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
  Camera camera;
  camera.set_view(glm::inverse(stage_transform()));
  camera.set_perspective(fovy_radians, aspect, z_near, z_far);
  return camera;
}

}  // namespace volumetric_kit::gfx::camera

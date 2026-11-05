// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/camera/orbit_camera.hpp"

#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
#include "volumetric_kit/gfx/camera/impl/orientation.hpp"  // shared controller math
//
#include <cmath>  // std::cos, std::isfinite, std::remainder, std::sin, std::tan

#include <glm/geometric.hpp>  // cross, dot, length, normalize
#include <glm/gtc/constants.hpp>
#include <glm/mat3x3.hpp>
#include <glm/matrix.hpp>  // transpose

namespace volumetric_kit::gfx::camera {
namespace {

// A NaN or infinite move would poison the state for good, so a drag that
// makes one is ignored.
bool is_finite(glm::vec2 v) { return std::isfinite(v.x) && std::isfinite(v.y); }

float wrap_angle(float angle) {
  return std::remainder(angle, 2.0f * glm::pi<float>());
}

// Columns right, up, back at azimuth 0, elevation 0: world +X stays right, so
// the rule is continuous near +Y, +Z and -Y, and singular only near +-X.
glm::mat3 level_frame(const glm::vec3& up) {
  glm::vec3 right = glm::vec3(1.0f, 0.0f, 0.0f) - up.x * up;
  if (glm::dot(right, right) < 1e-6f) {
    right = glm::cross(up, glm::vec3(0.0f, 0.0f, 1.0f));
  }
  right = glm::normalize(right);
  return glm::mat3(right, up, glm::cross(right, up));
}

// The camera's axes in world coordinates, columns right, up, back. Right never
// leaves the plane across the up, whatever the angles, so the view cannot roll.
glm::mat3 camera_axes(const glm::vec3& up, float azimuth, float elevation) {
  const glm::vec3 right(std::cos(azimuth), 0.0f, -std::sin(azimuth));
  return level_frame(up) *
         glm::mat3(
             right,
             spherical_direction(azimuth, elevation + glm::half_pi<float>()),
             spherical_direction(azimuth, elevation));
}

}  // namespace

void OrbitCamera::orbit(float delta_yaw, float delta_pitch) {
  if (!std::isfinite(delta_yaw) || !std::isfinite(delta_pitch)) return;
  // Past a pole the eye's motion along the camera's right reverses for a rising
  // azimuth, so the yaw flips with it. Each call takes its sign from where the
  // eye is, so a drag across a pole is continuous.
  const float sign = std::cos(elevation_) < 0.0f ? -1.0f : 1.0f;
  azimuth_ = wrap_angle(azimuth_ + sign * delta_yaw);
  elevation_ = wrap_angle(elevation_ + delta_pitch);
}

void OrbitCamera::pan(float delta_right, float delta_up) {
  const glm::mat3 axes = camera_axes(world_up_, azimuth_, elevation_);
  target_ += axes[0] * delta_right + axes[1] * delta_up;
}

void OrbitCamera::dolly(float delta) { set_distance(distance_ + delta); }

void OrbitCamera::zoom(float factor) {
  if (!std::isfinite(factor) || factor <= 0.0f) return;
  set_distance(distance_ * factor);
}

void OrbitCamera::drag_orbit(glm::vec2 drag) {
  // The eye turns against the drag, so the scene turns with it: a drag right
  // swings the eye left, and a drag down raises it.
  const glm::vec2 turn = kDragRadians * drag;
  if (!is_finite(turn)) return;
  orbit(-turn.x, turn.y);
}

void OrbitCamera::drag_pan(glm::vec2 drag, float fovy_radians) {
  // The world height the viewport spans at the target's depth.
  const float height = 2.0f * distance_ * std::tan(0.5f * fovy_radians);
  const glm::vec2 move = height * drag;
  if (!is_finite(move)) return;
  pan(-move.x, move.y);
}

void OrbitCamera::set_distance(float distance) noexcept {
  distance_ = clamp_pivot_distance(distance);
}

void OrbitCamera::set_world_up(const glm::vec3& up) noexcept {
  const float length = glm::length(up);
  if (!std::isfinite(length) || length <= 0.0f) return;
  world_up_ = up / length;
}

void OrbitCamera::set_azimuth(float azimuth) noexcept {
  if (std::isfinite(azimuth)) azimuth_ = wrap_angle(azimuth);
}

void OrbitCamera::set_elevation(float elevation) noexcept {
  if (std::isfinite(elevation)) elevation_ = wrap_angle(elevation);
}

glm::vec3 OrbitCamera::eye() const {
  // The target->eye offset runs *along* the shared spherical direction, in the
  // level frame of the up; at azimuth 0 / elevation 0 that is the frame's back.
  return target_ + distance_ * (level_frame(world_up_) *
                                spherical_direction(azimuth_, elevation_));
}

glm::mat4 OrbitCamera::view() const {
  const glm::mat3 rotation =
      glm::transpose(camera_axes(world_up_, azimuth_, elevation_));
  glm::mat4 world_to_view(rotation);
  world_to_view[3] = glm::vec4(-(rotation * eye()), 1.0f);
  return world_to_view;
}

Camera OrbitCamera::to_camera(float fovy_radians, float aspect, float z_near,
                              float z_far) const {
  return bake_camera(view(), fovy_radians, aspect, z_near, z_far);
}

}  // namespace volumetric_kit::gfx::camera

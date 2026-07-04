// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/camera/orbit_camera.hpp"

#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
#include "volumetric_kit/gfx/camera/impl/orientation.hpp"  // shared controller math
//
#include <algorithm>  // std::clamp

#include <glm/ext/matrix_transform.hpp>  // lookAt

namespace volumetric_kit::gfx::camera {

void OrbitCamera::orbit(float delta_azimuth, float delta_elevation) {
  azimuth_ += delta_azimuth;
  set_elevation(elevation_ + delta_elevation);
}

void OrbitCamera::pan(float delta_right, float delta_up) {
  // Slide the target within the current view plane; forward points from the eye
  // toward the target.
  const LevelBasis basis = level_basis(target_ - eye());
  target_ += basis.right * delta_right + basis.up * delta_up;
}

void OrbitCamera::dolly(float delta) { set_distance(distance_ + delta); }

void OrbitCamera::zoom(float factor) { set_distance(distance_ * factor); }

void OrbitCamera::set_distance(float distance) noexcept {
  distance_ = clamp_pivot_distance(distance);
}

void OrbitCamera::set_elevation(float elevation) noexcept {
  elevation_ = std::clamp(elevation, -kMaxElevation, kMaxElevation);
}

glm::vec3 OrbitCamera::eye() const {
  // The target->eye offset runs *along* the shared spherical direction; at
  // azimuth 0 / elevation 0 that is +Z, so the eye sits in front of the target.
  return target_ + distance_ * spherical_direction(azimuth_, elevation_);
}

glm::mat4 OrbitCamera::view() const {
  return glm::lookAt(eye(), target_, kWorldUp);
}

Camera OrbitCamera::to_camera(float fovy_radians, float aspect, float z_near,
                              float z_far) const {
  return bake_camera(view(), fovy_radians, aspect, z_near, z_far);
}

}  // namespace volumetric_kit::gfx::camera

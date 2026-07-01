// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/camera/orbit_camera.hpp"

#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
#include "volumetric_kit/gfx/camera/impl/orientation.hpp"  // kWorldUp, level_basis
//
#include <algorithm>  // std::clamp, std::max
#include <cmath>      // std::cos, std::sin

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
  distance_ = std::max(distance, kMinDistance);
}

void OrbitCamera::set_elevation(float elevation) noexcept {
  elevation_ = std::clamp(elevation, -kMaxElevation, kMaxElevation);
}

glm::vec3 OrbitCamera::eye() const {
  // Spherical -> cartesian offset from the target. elevation is measured from
  // the horizontal plane; at azimuth 0 / elevation 0 the offset is +Z, so the
  // eye sits in front of the target.
  const float cos_elev = std::cos(elevation_);
  const glm::vec3 offset(distance_ * cos_elev * std::sin(azimuth_),
                         distance_ * std::sin(elevation_),
                         distance_ * cos_elev * std::cos(azimuth_));
  return target_ + offset;
}

glm::mat4 OrbitCamera::view() const {
  return glm::lookAt(eye(), target_, kWorldUp);
}

Camera OrbitCamera::to_camera(float fovy_radians, float aspect, float z_near,
                              float z_far) const {
  Camera camera;
  camera.set_view(view());
  camera.set_perspective(fovy_radians, aspect, z_near, z_far);
  return camera;
}

}  // namespace volumetric_kit::gfx::camera

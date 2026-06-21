// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/camera/camera.hpp"

#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
//
#include <glm/ext/matrix_clip_space.hpp>  // perspective, ortho
#include <glm/ext/matrix_transform.hpp>   // lookAt

namespace volumetric_kit::gfx::camera {

namespace {

// glm's projections target OpenGL's framebuffer (Y up); Vulkan's is Y down.
// Negating proj[1][1] flips clip-space Y once at the source, so the same shader
// renders right-side up under Vulkan without a per-vertex flip or a
// negative-height viewport. (GLM_FORCE_DEPTH_ZERO_TO_ONE -- see glm_config.hpp
// -- has already mapped depth into [0, 1]; this only touches Y.)
glm::mat4 flip_y(glm::mat4 proj) {
  proj[1][1] = -proj[1][1];
  return proj;
}

}  // namespace

Camera Camera::look_at_perspective(const glm::vec3& eye,
                                   const glm::vec3& center, const glm::vec3& up,
                                   float fovy_radians, float aspect,
                                   float z_near, float z_far) {
  Camera camera;
  camera.set_view(eye, center, up);
  camera.set_perspective(fovy_radians, aspect, z_near, z_far);
  return camera;
}

Camera Camera::look_at_ortho(const glm::vec3& eye, const glm::vec3& center,
                             const glm::vec3& up, float left, float right,
                             float bottom, float top, float z_near,
                             float z_far) {
  Camera camera;
  camera.set_view(eye, center, up);
  camera.set_ortho(left, right, bottom, top, z_near, z_far);
  return camera;
}

void Camera::set_view(const glm::vec3& eye, const glm::vec3& center,
                      const glm::vec3& up) {
  view_ = glm::lookAt(eye, center, up);
}

void Camera::set_perspective(float fovy_radians, float aspect, float z_near,
                             float z_far) {
  proj_ = flip_y(glm::perspective(fovy_radians, aspect, z_near, z_far));
}

void Camera::set_ortho(float left, float right, float bottom, float top,
                       float z_near, float z_far) {
  proj_ = flip_y(glm::ortho(left, right, bottom, top, z_near, z_far));
}

}  // namespace volumetric_kit::gfx::camera

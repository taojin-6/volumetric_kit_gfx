// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera.hpp
/// @brief CPU-side view + projection math for the renderer, in Vulkan clip
///        space.
///
/// @ref Camera is a plain value type: a view matrix (where the camera is and
/// what it looks at) paired with a projection matrix (how that view maps into
/// clip space). It owns no GPU resources and never touches Vulkan -- the camera
/// tier depends only on glm. Consumers feed @ref Camera::view_proj into a
/// uniform/push-constant and submit it through a pipeline.
///
/// @par Clip-space convention
/// The projection targets Vulkan's clip space, not OpenGL's:
/// - **Depth range [0, 1]** (not GL's [-1, 1]). The header forces glm's
///   `GLM_FORCE_DEPTH_ZERO_TO_ONE` *before* including glm, so every glm
///   `perspective`/`ortho` here emits a `z'` in `[0, 1]` -- the range
///   `VkViewport::{minDepth, maxDepth}` defaults to and `VK_ATTACHMENT_LOAD_OP`
///   clears against.
/// - **Y points down in framebuffer space.** glm's projections assume GL's
///   Y-up normalized device coordinates; Vulkan's framebuffer origin is the top
///   left with +Y downward. The projection matrices below therefore negate
///   `proj[1][1]`, which flips clip-space Y once at the source -- so the same
///   GLSL renders identically without a `gl_Position.y = -gl_Position.y` patch
///   or a negative-height viewport. Front faces stay counter-clockwise in
///   object space; the Y flip inverts winding in framebuffer space, so a
///   pipeline using these matrices should cull with
///   `VK_FRONT_FACE_COUNTER_CLOCKWISE`.

#include "volumetric_kit/gfx/camera/impl/glm_config.hpp"
//
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace volumetric_kit::gfx::camera {

/// @brief A view matrix paired with a Vulkan-clip-space projection matrix.
///
/// Build one from a factory -- @ref look_at_perspective for a perspective
/// frustum or @ref look_at_ortho for an orthographic box -- then read
/// @ref view, @ref proj, or the combined @ref view_proj. The pieces can also be
/// replaced independently with @ref set_view / @ref set_perspective /
/// @ref set_ortho (e.g. to animate the projection while a controller drives the
/// view). All matrices follow the clip-space convention documented on this
/// header (depth `[0, 1]`, framebuffer Y down).
///
/// @code
/// using namespace volumetric_kit::gfx::camera;
/// Camera cam = Camera::look_at_perspective(
///     /*eye=*/{0.0f, 0.0f, 5.0f}, /*center=*/{0.0f, 0.0f, 0.0f},
///     /*up=*/{0.0f, 1.0f, 0.0f},
///     /*fovy_radians=*/glm::radians(60.0f), /*aspect=*/16.0f / 9.0f,
///     /*z_near=*/0.1f, /*z_far=*/100.0f);
/// glm::mat4 mvp = cam.view_proj() * model;  // feed to a uniform / push const
/// @endcode
class Camera {
 public:
  /// Construct an identity camera (identity view and projection).
  Camera() = default;

  /// @brief Build a camera with a look-at view and a perspective projection.
  /// @param eye           World-space position of the camera.
  /// @param center        World-space point the camera looks at.
  /// @param up            World-space up direction (need not be unit length;
  ///                      must not be parallel to `center - eye`).
  /// @param fovy_radians  Vertical field of view, in radians; must be in
  ///                      `(0, pi)`.
  /// @param aspect        Viewport width / height; must be > 0.
  /// @param z_near        Near plane distance; must satisfy `0 < z_near`.
  /// @param z_far         Far plane distance; must satisfy `z_near < z_far`.
  /// @return A camera whose @ref view_proj maps the frustum into Vulkan clip
  ///         space.
  /// @pre The preconditions on each argument hold; this performs CPU math only
  ///      and does not validate them.
  static Camera look_at_perspective(const glm::vec3& eye,
                                    const glm::vec3& center,
                                    const glm::vec3& up, float fovy_radians,
                                    float aspect, float z_near, float z_far);

  /// @brief Build a camera with a look-at view and an orthographic projection.
  /// @param eye     World-space position of the camera.
  /// @param center  World-space point the camera looks at.
  /// @param up      World-space up direction (see @ref look_at_perspective).
  /// @param left    Left clip plane (world units at the view plane).
  /// @param right   Right clip plane; must satisfy `left != right`.
  /// @param bottom  Bottom clip plane.
  /// @param top     Top clip plane; must satisfy `bottom != top`.
  /// @param z_near  Near plane distance.
  /// @param z_far   Far plane distance; must satisfy `z_near != z_far`.
  /// @return A camera whose @ref view_proj maps the box into Vulkan clip space.
  static Camera look_at_ortho(const glm::vec3& eye, const glm::vec3& center,
                              const glm::vec3& up, float left, float right,
                              float bottom, float top, float z_near,
                              float z_far);

  /// @brief Replace the view with a look-at transform.
  /// @param eye     World-space camera position.
  /// @param center  World-space point looked at.
  /// @param up      World-space up direction (see @ref look_at_perspective).
  void set_view(const glm::vec3& eye, const glm::vec3& center,
                const glm::vec3& up);

  /// @brief Replace the view with an arbitrary world->view matrix.
  /// @param view  A view matrix (e.g. produced by a controller).
  void set_view(const glm::mat4& view) { view_ = view; }

  /// @brief Replace the projection with a perspective frustum.
  /// @param fovy_radians  Vertical field of view, in radians (`(0, pi)`).
  /// @param aspect        Viewport width / height (> 0).
  /// @param z_near        Near plane distance (`> 0`).
  /// @param z_far         Far plane distance (`> z_near`).
  void set_perspective(float fovy_radians, float aspect, float z_near,
                       float z_far);

  /// @brief Replace the projection with an orthographic box.
  /// @param left    Left clip plane.
  /// @param right   Right clip plane (`!= left`).
  /// @param bottom  Bottom clip plane.
  /// @param top     Top clip plane (`!= bottom`).
  /// @param z_near  Near plane distance.
  /// @param z_far   Far plane distance (`!= z_near`).
  void set_ortho(float left, float right, float bottom, float top, float z_near,
                 float z_far);

  /// @return The world->view (camera) matrix.
  const glm::mat4& view() const noexcept { return view_; }
  /// @return The view->clip (projection) matrix, in Vulkan clip space.
  const glm::mat4& proj() const noexcept { return proj_; }
  /// @return The combined world->clip matrix, `proj() * view()`.
  glm::mat4 view_proj() const { return proj_ * view_; }

 private:
  glm::mat4 view_ = glm::mat4(1.0f);
  glm::mat4 proj_ = glm::mat4(1.0f);
};

}  // namespace volumetric_kit::gfx::camera

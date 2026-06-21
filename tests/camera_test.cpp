// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Pure-CPU tests for the camera tier: known-value checks on the view/projection
// math and the orbit controller. No Vulkan device is needed, so these run on
// every platform.

#include <gtest/gtest.h>

#include <cmath>

#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/camera/orbit_camera.hpp"
//
#include <glm/geometric.hpp>  // length
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>  // radians
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace cam = volumetric_kit::gfx::camera;

namespace {

// Project a world point and apply the perspective divide -> normalized device
// coordinates (the clip-space convention the Camera bakes in).
glm::vec3 to_ndc(const cam::Camera& camera, const glm::vec3& world) {
  const glm::vec4 clip = camera.view_proj() * glm::vec4(world, 1.0f);
  return glm::vec3(clip) / clip.w;
}

}  // namespace

// A point on the look-at axis sits at the center of the screen: NDC x == y ==
// 0.
TEST(Camera, AxisPointProjectsToScreenCenter) {
  const cam::Camera camera = cam::Camera::look_at_perspective(
      /*eye=*/{0.0f, 0.0f, 5.0f}, /*center=*/{0.0f, 0.0f, 0.0f},
      /*up=*/{0.0f, 1.0f, 0.0f}, glm::radians(60.0f), /*aspect=*/16.0f / 9.0f,
      /*z_near=*/0.1f, /*z_far=*/100.0f);

  const glm::vec3 ndc = to_ndc(camera, {0.0f, 0.0f, 0.0f});
  EXPECT_NEAR(ndc.x, 0.0f, 1e-5f);
  EXPECT_NEAR(ndc.y, 0.0f, 1e-5f);
}

// Vulkan depth is [0, 1]: a point at the near plane maps to z == 0, one at the
// far plane to z == 1. (Proves GLM_FORCE_DEPTH_ZERO_TO_ONE is in effect.)
TEST(Camera, DepthRangeIsZeroToOneAtPlanes) {
  constexpr float kNear = 0.1f;
  constexpr float kFar = 100.0f;
  const cam::Camera camera = cam::Camera::look_at_perspective(
      /*eye=*/{0.0f, 0.0f, 5.0f}, /*center=*/{0.0f, 0.0f, 0.0f},
      /*up=*/{0.0f, 1.0f, 0.0f}, glm::radians(60.0f), /*aspect=*/1.0f, kNear,
      kFar);

  // Eye is at z = 5 looking toward -z, so the near/far planes are at world
  // z = 5 - kNear and z = 5 - kFar.
  const glm::vec3 at_near = to_ndc(camera, {0.0f, 0.0f, 5.0f - kNear});
  const glm::vec3 at_far = to_ndc(camera, {0.0f, 0.0f, 5.0f - kFar});
  EXPECT_NEAR(at_near.z, 0.0f, 1e-4f);
  EXPECT_NEAR(at_far.z, 1.0f, 1e-4f);
}

// Vulkan's framebuffer Y points down, so the baked-in proj[1][1] flip sends a
// world point *above* the look-at center to *negative* NDC y (and below ->
// positive). This is the GL-vs-Vulkan difference made observable.
TEST(Camera, FramebufferYIsFlipped) {
  const cam::Camera camera = cam::Camera::look_at_perspective(
      /*eye=*/{0.0f, 0.0f, 5.0f}, /*center=*/{0.0f, 0.0f, 0.0f},
      /*up=*/{0.0f, 1.0f, 0.0f}, glm::radians(60.0f), /*aspect=*/1.0f,
      /*z_near=*/0.1f, /*z_far=*/100.0f);

  const glm::vec3 above = to_ndc(camera, {0.0f, 1.0f, 0.0f});  // world +Y
  const glm::vec3 below = to_ndc(camera, {0.0f, -1.0f, 0.0f});
  EXPECT_LT(above.y, 0.0f);
  EXPECT_GT(below.y, 0.0f);
  // X is unaffected by the Y flip: a world +X point stays on the +X NDC side.
  const glm::vec3 right = to_ndc(camera, {1.0f, 0.0f, 0.0f});
  EXPECT_GT(right.x, 0.0f);
}

// A symmetric frustum maps the frustum-edge ray to the NDC edge (|y| == 1): at
// the look-at depth, a point tan(fovy/2) * dist above center is exactly at the
// top edge, which the Y flip sends to y == -1.
TEST(Camera, FrustumEdgeMapsToNdcEdge) {
  constexpr float kFovy = glm::radians(60.0f);
  constexpr float kDist = 5.0f;  // eye z -> origin
  const cam::Camera camera = cam::Camera::look_at_perspective(
      /*eye=*/{0.0f, 0.0f, kDist}, /*center=*/{0.0f, 0.0f, 0.0f},
      /*up=*/{0.0f, 1.0f, 0.0f}, kFovy, /*aspect=*/1.0f, /*z_near=*/0.1f,
      /*z_far=*/100.0f);

  const float top = std::tan(kFovy * 0.5f) * kDist;
  const glm::vec3 ndc = to_ndc(camera, {0.0f, top, 0.0f});
  EXPECT_NEAR(ndc.y, -1.0f, 1e-4f);  // top edge, flipped
}

// Orthographic projection is linear: the box center maps to the NDC origin and
// a corner of the X/Y extents maps to the NDC corner (with the Y flip applied).
TEST(Camera, OrthographicMapsBoxToNdc) {
  const cam::Camera camera = cam::Camera::look_at_ortho(
      /*eye=*/{0.0f, 0.0f, 5.0f}, /*center=*/{0.0f, 0.0f, 0.0f},
      /*up=*/{0.0f, 1.0f, 0.0f}, /*left=*/-2.0f, /*right=*/2.0f,
      /*bottom=*/-2.0f, /*top=*/2.0f, /*z_near=*/0.1f, /*z_far=*/100.0f);

  EXPECT_NEAR(to_ndc(camera, {0.0f, 0.0f, 0.0f}).x, 0.0f, 1e-5f);
  EXPECT_NEAR(to_ndc(camera, {0.0f, 0.0f, 0.0f}).y, 0.0f, 1e-5f);
  // Right edge -> +1 in x; top edge -> -1 in y (flipped).
  const glm::vec3 corner = to_ndc(camera, {2.0f, 2.0f, 0.0f});
  EXPECT_NEAR(corner.x, 1.0f, 1e-5f);
  EXPECT_NEAR(corner.y, -1.0f, 1e-5f);
}

// set_view(eye, center, up) then a separately-set projection composes to the
// same matrix as the look_at_perspective factory.
TEST(Camera, SettersComposeLikeFactory) {
  const glm::vec3 eye(1.0f, 2.0f, 3.0f);
  const glm::vec3 center(0.0f, 0.0f, 0.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);

  const cam::Camera factory = cam::Camera::look_at_perspective(
      eye, center, up, glm::radians(45.0f), 1.5f, 0.1f, 50.0f);

  cam::Camera built;
  built.set_view(eye, center, up);
  built.set_perspective(glm::radians(45.0f), 1.5f, 0.1f, 50.0f);

  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      EXPECT_NEAR(built.view_proj()[c][r], factory.view_proj()[c][r], 1e-6f);
    }
  }
}

// Orbit spherical math: azimuth 0 / elevation 0 puts the eye in front (+Z); a
// quarter turn of azimuth swings it onto +X; elevation up swings it toward +Y.
TEST(OrbitCamera, EyePositionFromSphericalAngles) {
  cam::OrbitCamera orbit;
  orbit.set_target({0.0f, 0.0f, 0.0f});
  orbit.set_distance(4.0f);

  // Default: in front of the target on +Z.
  EXPECT_NEAR(orbit.eye().x, 0.0f, 1e-5f);
  EXPECT_NEAR(orbit.eye().y, 0.0f, 1e-5f);
  EXPECT_NEAR(orbit.eye().z, 4.0f, 1e-5f);

  // Quarter turn of azimuth -> on +X.
  orbit.set_azimuth(glm::half_pi<float>());
  EXPECT_NEAR(orbit.eye().x, 4.0f, 1e-4f);
  EXPECT_NEAR(orbit.eye().y, 0.0f, 1e-4f);
  EXPECT_NEAR(orbit.eye().z, 0.0f, 1e-4f);

  // Elevation toward the pole -> mostly +Y (clamped just short of exactly +Y).
  orbit.set_azimuth(0.0f);
  orbit.set_elevation(glm::half_pi<float>());  // clamps to kMaxElevation
  EXPECT_GT(orbit.eye().y, 3.99f);             // ~4 * sin(89.5deg)
  EXPECT_LT(orbit.eye().y, 4.0f);
}

// The eye is always at `distance` from the target, regardless of the angles.
TEST(OrbitCamera, EyeKeepsDistanceFromTarget) {
  cam::OrbitCamera orbit;
  orbit.set_target({1.0f, -2.0f, 3.0f});
  orbit.set_distance(7.5f);
  orbit.orbit(glm::radians(123.0f), glm::radians(34.0f));

  EXPECT_NEAR(glm::length(orbit.eye() - orbit.target()), 7.5f, 1e-4f);
}

// Elevation is clamped to +/-kMaxElevation so the look-at basis never
// degenerates at the poles.
TEST(OrbitCamera, ElevationClampsAtPoles) {
  cam::OrbitCamera orbit;
  orbit.orbit(0.0f, /*delta_elevation=*/10.0f);  // way past +pi/2
  EXPECT_FLOAT_EQ(orbit.elevation(), cam::OrbitCamera::kMaxElevation);

  orbit.orbit(0.0f, /*delta_elevation=*/-100.0f);  // way past -pi/2
  EXPECT_FLOAT_EQ(orbit.elevation(), -cam::OrbitCamera::kMaxElevation);
}

// Distance is clamped to a positive minimum so the eye can never reach the
// target (which would collapse the view ray).
TEST(OrbitCamera, DistanceClampsToPositiveMinimum) {
  cam::OrbitCamera orbit;
  orbit.set_distance(1.0f);
  orbit.dolly(-100.0f);  // far past zero
  EXPECT_FLOAT_EQ(orbit.distance(), cam::OrbitCamera::kMinDistance);

  orbit.set_distance(10.0f);
  orbit.zoom(0.0f);  // multiplicative collapse
  EXPECT_FLOAT_EQ(orbit.distance(), cam::OrbitCamera::kMinDistance);
}

// dolly moves the eye along the view ray without rotating it: at the default
// orientation (+Z), dollying changes only the eye's z, and target is unmoved.
TEST(OrbitCamera, DollyMovesEyeAlongViewRay) {
  cam::OrbitCamera orbit;
  orbit.set_distance(5.0f);
  orbit.dolly(-2.0f);  // move closer
  EXPECT_FLOAT_EQ(orbit.distance(), 3.0f);
  EXPECT_NEAR(orbit.eye().z, 3.0f, 1e-5f);
  EXPECT_EQ(orbit.target(), glm::vec3(0.0f));
}

// pan slides the target (and eye) within the view plane. At the default
// orientation, forward is -Z and world up is +Y, so right is +X: a +right pan
// moves the target along +X, and a +up pan moves it along +Y; Z is unchanged.
TEST(OrbitCamera, PanSlidesTargetInViewPlane) {
  cam::OrbitCamera orbit;
  orbit.set_target({0.0f, 0.0f, 0.0f});
  orbit.set_distance(5.0f);
  orbit.pan(/*delta_right=*/2.0f, /*delta_up=*/3.0f);

  EXPECT_NEAR(orbit.target().x, 2.0f, 1e-5f);
  EXPECT_NEAR(orbit.target().y, 3.0f, 1e-5f);
  EXPECT_NEAR(orbit.target().z, 0.0f, 1e-5f);
  // The eye rides along with the target, so the offset is preserved.
  EXPECT_NEAR(orbit.eye().x, 2.0f, 1e-5f);
  EXPECT_NEAR(orbit.eye().y, 3.0f, 1e-5f);
  EXPECT_NEAR(orbit.eye().z, 5.0f, 1e-5f);
}

// to_camera pairs the controller's view with a perspective projection: a point
// at the target lands at the screen center.
TEST(OrbitCamera, ToCameraCentersOnTarget) {
  cam::OrbitCamera orbit;
  orbit.set_target({2.0f, 1.0f, -3.0f});
  orbit.set_distance(6.0f);
  orbit.orbit(glm::radians(40.0f), glm::radians(15.0f));

  const cam::Camera camera =
      orbit.to_camera(glm::radians(55.0f), 1.0f, 0.1f, 100.0f);
  const glm::vec3 ndc = to_ndc(camera, orbit.target());
  EXPECT_NEAR(ndc.x, 0.0f, 1e-4f);
  EXPECT_NEAR(ndc.y, 0.0f, 1e-4f);
}

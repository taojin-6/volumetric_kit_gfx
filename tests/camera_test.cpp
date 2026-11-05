// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Pure-CPU tests for the camera tier: known-value checks on the view/projection
// math and the orbit controller. No Vulkan device is needed, so these run on
// every platform.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/camera/orbit_camera.hpp"
//
#include <glm/ext/matrix_transform.hpp>  // lookAt
#include <glm/geometric.hpp>             // cross, dot, length, normalize
#include <glm/gtc/constants.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>  // radians
#include <glm/vec2.hpp>
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

// A world point's place on screen in viewport heights from the centre, +x
// right and +y down: the unit OrbitCamera's drags take. NDC spans 2 across the
// height and 2 across the width, which is `aspect` heights.
glm::vec2 in_heights(const cam::Camera& camera, float aspect,
                     const glm::vec3& world) {
  const glm::vec3 ndc = to_ndc(camera, world);
  return 0.5f * glm::vec2(ndc.x * aspect, ndc.y);
}

// An orbit camera off every axis, so a drag's sign or axis mix-up shows.
cam::OrbitCamera oblique_orbit() {
  cam::OrbitCamera orbit;
  orbit.set_target({1.0f, -2.0f, 3.0f});
  orbit.set_distance(5.0f);
  orbit.orbit(glm::radians(30.0f), glm::radians(20.0f));
  return orbit;
}

// oblique_orbit() upright and upside down, past the top pole and past the
// bottom one, about +Y and about +Z.
std::vector<cam::OrbitCamera> start_poses() {
  std::vector<cam::OrbitCamera> poses;
  for (const glm::vec3 up : {glm::vec3(0.0f, 1.0f, 0.0f),  //
                             glm::vec3(0.0f, 0.0f, 1.0f)}) {
    for (const float elevation :
         {glm::radians(20.0f), glm::radians(160.0f), glm::radians(-20.0f),
          glm::radians(-160.0f)}) {
      cam::OrbitCamera orbit = oblique_orbit();
      orbit.set_world_up(up);
      orbit.set_elevation(elevation);
      poses.push_back(orbit);
    }
  }
  return poses;
}

// The camera's axes in world coordinates: the rows of its view's rotation.
struct Axes {
  glm::vec3 right;
  glm::vec3 up;
  glm::vec3 back;
};

Axes view_axes(const cam::OrbitCamera& orbit) {
  const glm::mat4 view = orbit.view();
  return {{view[0][0], view[1][0], view[2][0]},
          {view[0][1], view[1][1], view[2][1]},
          {view[0][2], view[1][2], view[2][2]}};
}

void expect_vec_near(const glm::vec3& actual, const glm::vec3& expected,
                     float tolerance) {
  EXPECT_NEAR(actual.x, expected.x, tolerance);
  EXPECT_NEAR(actual.y, expected.y, tolerance);
  EXPECT_NEAR(actual.z, expected.z, tolerance);
}

void expect_mat_near(const glm::mat4& actual, const glm::mat4& expected,
                     float tolerance) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      EXPECT_NEAR(actual[c][r], expected[c][r], tolerance);
    }
  }
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

// An off-center orthographic box has a Y translation as well as a scale.
// Both must be flipped to keep its center and edges inside the viewport.
TEST(Camera, AsymmetricOrthographicMapsBoxToNdc) {
  for (const glm::vec4 bounds : {glm::vec4(0.0f, 640.0f, 0.0f, 480.0f),
                                 glm::vec4(-20.0f, 40.0f, -30.0f, -10.0f)}) {
    SCOPED_TRACE(bounds.z);
    const cam::Camera camera = cam::Camera::look_at_ortho(
        {0.0f, 0.0f, 5.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, bounds.x,
        bounds.y, bounds.z, bounds.w, 0.1f, 100.0f);
    const glm::vec3 center = to_ndc(
        camera,
        {(bounds.x + bounds.y) * 0.5f, (bounds.z + bounds.w) * 0.5f, 0.0f});
    EXPECT_NEAR(center.x, 0.0f, 1e-5f);
    EXPECT_NEAR(center.y, 0.0f, 1e-5f);
    const glm::vec3 bottom_left = to_ndc(camera, {bounds.x, bounds.z, 0.0f});
    EXPECT_NEAR(bottom_left.x, -1.0f, 1e-5f);
    EXPECT_NEAR(bottom_left.y, 1.0f, 1e-5f);
    const glm::vec3 top_right = to_ndc(camera, {bounds.y, bounds.w, 0.0f});
    EXPECT_NEAR(top_right.x, 1.0f, 1e-5f);
    EXPECT_NEAR(top_right.y, -1.0f, 1e-5f);
  }
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

  // Elevation at the pole -> straight up, with no clamp.
  orbit.set_azimuth(0.0f);
  orbit.set_elevation(glm::half_pi<float>());
  EXPECT_FLOAT_EQ(orbit.elevation(), glm::half_pi<float>());
  expect_vec_near(orbit.eye(), {0.0f, 4.0f, 0.0f}, 1e-5f);
}

// The eye is always at `distance` from the target, regardless of the angles.
TEST(OrbitCamera, EyeKeepsDistanceFromTarget) {
  cam::OrbitCamera orbit;
  orbit.set_target({1.0f, -2.0f, 3.0f});
  orbit.set_distance(7.5f);
  orbit.orbit(glm::radians(123.0f), glm::radians(34.0f));

  EXPECT_NEAR(glm::length(orbit.eye() - orbit.target()), 7.5f, 1e-4f);
}

// The setters keep an angle in [-pi, pi], at the same place on its circle, and
// ignore a non-finite one; a huge finite turn stays in range.
TEST(OrbitCamera, AnglesWrapAndIgnoreNonFinite) {
  constexpr float kTurn = glm::two_pi<float>();
  cam::OrbitCamera orbit;
  orbit.set_azimuth(0.5f + 3.0f * kTurn);
  EXPECT_NEAR(orbit.azimuth(), 0.5f, 1e-5f);
  orbit.set_elevation(2.0f - kTurn);
  EXPECT_NEAR(orbit.elevation(), 2.0f, 1e-5f);

  const float azimuth = orbit.azimuth();
  const float elevation = orbit.elevation();
  for (const float angle : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity()}) {
    SCOPED_TRACE(angle);
    orbit.set_azimuth(angle);
    orbit.set_elevation(angle);
    EXPECT_EQ(orbit.azimuth(), azimuth);
    EXPECT_EQ(orbit.elevation(), elevation);
  }

  orbit.orbit(1e30f, -3e38f);
  EXPECT_LE(std::fabs(orbit.azimuth()), glm::pi<float>());
  EXPECT_LE(std::fabs(orbit.elevation()), glm::pi<float>());
  EXPECT_TRUE(std::isfinite(orbit.view()[3][2]));
}

// Distance is clamped to a positive minimum so the eye can never reach the
// target (which would collapse the view ray).
TEST(OrbitCamera, DistanceClampsToPositiveMinimum) {
  cam::OrbitCamera orbit;
  orbit.set_distance(1.0f);
  orbit.dolly(-100.0f);  // far past zero
  EXPECT_FLOAT_EQ(orbit.distance(), cam::OrbitCamera::kMinDistance);

  orbit.set_distance(10.0f);
  orbit.zoom(1e-6f);  // multiplicative collapse
  EXPECT_FLOAT_EQ(orbit.distance(), cam::OrbitCamera::kMinDistance);
}

// A NaN distance write -- via set_distance or dolly -- clamps to the floor
// instead of poisoning the state (std::max floors it -- the arg order matters
// for NaN), mirroring CameraRig's NonFiniteFocusDistanceClampsToFloor. zoom
// ignores a NaN factor (ZoomIgnoresNonPositiveAndNonFiniteFactors).
TEST(OrbitCamera, NonFiniteDistanceClampsToFloor) {
  const float nan = std::numeric_limits<float>::quiet_NaN();

  cam::OrbitCamera orbit;
  orbit.set_distance(nan);
  EXPECT_FLOAT_EQ(orbit.distance(), cam::OrbitCamera::kMinDistance);
  EXPECT_TRUE(std::isfinite(orbit.eye().x));
  EXPECT_TRUE(std::isfinite(orbit.eye().y));
  EXPECT_TRUE(std::isfinite(orbit.eye().z));

  orbit.dolly(nan);  // kMinDistance + NaN is NaN again; the clamp re-floors it
  EXPECT_FLOAT_EQ(orbit.distance(), cam::OrbitCamera::kMinDistance);

  // The controller stays usable: a subsequent valid verb yields a finite eye.
  orbit.dolly(2.0f);
  EXPECT_NEAR(orbit.distance(), 2.0f + cam::OrbitCamera::kMinDistance, 1e-6f);
  EXPECT_TRUE(std::isfinite(orbit.eye().x));
  EXPECT_TRUE(std::isfinite(orbit.eye().y));
  EXPECT_TRUE(std::isfinite(orbit.eye().z));
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

// to_camera routes its projection arguments through bake_camera into
// set_perspective; ToCameraCentersOnTarget only checks a target-axis point
// (which lands at NDC (0,0) for any centered frustum) and so is blind to which
// slot each argument reaches. Pin all four: depth at the near/far planes fixes
// z_near and z_far (and their order), the vertical frustum edge fixes fovy, and
// the horizontal edge fixes aspect.
TEST(OrbitCamera, ToCameraForwardsProjectionArguments) {
  constexpr float kFovy = glm::radians(60.0f);
  constexpr float kAspect = 1.6f;  // != 1 and != kFovy, so a fovy/aspect swap
                                   // is observable
  constexpr float kNear = 0.1f;
  constexpr float kFar = 100.0f;
  constexpr float kDist = 5.0f;

  cam::OrbitCamera orbit;  // default angles: eye on +Z at kDist, looking -Z
  orbit.set_target({0.0f, 0.0f, 0.0f});
  orbit.set_distance(kDist);
  const cam::Camera camera = orbit.to_camera(kFovy, kAspect, kNear, kFar);

  // Eye at z = kDist looking -Z, so the near/far planes sit at world
  // z = kDist - kNear and kDist - kFar: depth 0 at near, 1 at far. Swapping
  // z_near/z_far inverts this.
  EXPECT_NEAR(to_ndc(camera, {0.0f, 0.0f, kDist - kNear}).z, 0.0f, 1e-4f);
  EXPECT_NEAR(to_ndc(camera, {0.0f, 0.0f, kDist - kFar}).z, 1.0f, 1e-4f);

  // A point tan(fovy/2)*kDist above center is at the top frustum edge, which
  // the Vulkan Y flip sends to NDC y == -1: sensitive to fovy.
  const float half = std::tan(kFovy * 0.5f);
  EXPECT_NEAR(to_ndc(camera, {0.0f, half * kDist, 0.0f}).y, -1.0f, 1e-4f);

  // The horizontal half-extent scales the vertical by aspect, so a point
  // tan(fovy/2)*aspect*kDist to the right maps to NDC x == +1: sensitive to
  // aspect (and breaks if fovy lands in aspect's slot).
  EXPECT_NEAR(to_ndc(camera, {half * kAspect * kDist, 0.0f, 0.0f}).x, 1.0f,
              1e-4f);
}

// The scene follows the pointer: a drag right, left, down or up carries a point
// in front of the target that way on screen, by either verb, upright and upside
// down. The direction is checked, not just a sign, so a swapped axis fails too.
TEST(OrbitCamera, DragsMoveTheSceneWithThePointer) {
  constexpr float kFovy = glm::radians(50.0f);
  constexpr float kAspect = 1.5f;
  for (const cam::OrbitCamera& start : start_poses()) {
    for (const bool pan : {false, true}) {
      for (const glm::vec2 drag :
           {glm::vec2(0.05f, 0.0f), glm::vec2(-0.05f, 0.0f),
            glm::vec2(0.0f, 0.05f), glm::vec2(0.0f, -0.05f)}) {
        SCOPED_TRACE(testing::Message()
                     << (pan ? "drag_pan " : "drag_orbit ") << drag.x << ", "
                     << drag.y << " at elevation " << start.elevation()
                     << ", up z " << start.world_up().z);
        cam::OrbitCamera orbit = start;
        const glm::vec3 in_front = 0.5f * (orbit.target() + orbit.eye());
        const glm::vec2 before = in_heights(
            orbit.to_camera(kFovy, kAspect, 0.1f, 100.0f), kAspect, in_front);
        if (pan) {
          orbit.drag_pan(drag, kFovy);
        } else {
          orbit.drag_orbit(drag);
        }
        const glm::vec2 moved =
            in_heights(orbit.to_camera(kFovy, kAspect, 0.1f, 100.0f), kAspect,
                       in_front) -
            before;
        EXPECT_GT(glm::dot(glm::normalize(moved), glm::normalize(drag)), 0.95f);
      }
    }
  }
}

// With the default +Y up and the eye short of a pole, the view is the look-at
// turntable's, for every consumer that never sets an up or passes a pole.
TEST(OrbitCamera, DefaultUpMatchesTheLookAtTurntable) {
  const glm::vec3 target(1.0f, -2.0f, 3.0f);
  constexpr float kDistance = 5.0f;
  for (int azimuth_deg = -180; azimuth_deg <= 180; azimuth_deg += 30) {
    for (int elevation_deg = -85; elevation_deg <= 85; elevation_deg += 5) {
      SCOPED_TRACE(testing::Message() << azimuth_deg << ", " << elevation_deg);
      const float azimuth = glm::radians(static_cast<float>(azimuth_deg));
      const float elevation = glm::radians(static_cast<float>(elevation_deg));
      cam::OrbitCamera orbit;
      orbit.set_target(target);
      orbit.set_distance(kDistance);
      orbit.set_azimuth(azimuth);
      orbit.set_elevation(elevation);
      const glm::vec3 eye =
          target + kDistance * cam::spherical_direction(azimuth, elevation);
      expect_vec_near(orbit.eye(), eye, 1e-5f);
      expect_mat_near(orbit.view(),
                      glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f)),
                      2e-5f);
    }
  }
}

// A vertical drag has no stop: half a height from level reaches a pole, where
// the view is still defined and aimed at the target, and one height the far
// side, upside down. Two heights are a full turn, back to the start view.
TEST(OrbitCamera, DragOrbitPassesOverThePoles) {
  constexpr float kFovy = glm::radians(50.0f);
  constexpr float kD = 2.0f;
  const auto expect_pose = [&](const cam::OrbitCamera& orbit,
                               const glm::vec3& eye, const glm::vec3& up) {
    expect_vec_near(orbit.eye(), eye, 1e-5f);
    expect_vec_near(view_axes(orbit).up, up, 1e-5f);
    const cam::Camera camera = orbit.to_camera(kFovy, 1.5f, 0.1f, 100.0f);
    EXPECT_NEAR(to_ndc(camera, orbit.target()).x, 0.0f, 1e-5f);
    EXPECT_NEAR(to_ndc(camera, orbit.target()).y, 0.0f, 1e-5f);
  };

  cam::OrbitCamera orbit;
  orbit.set_distance(kD);
  const cam::OrbitCamera level = orbit;
  orbit.drag_orbit({0.0f, 0.5f});
  expect_pose(orbit, {0.0f, kD, 0.0f}, {0.0f, 0.0f, -1.0f});
  orbit.drag_orbit({0.0f, 0.5f});
  expect_pose(orbit, {0.0f, 0.0f, -kD}, {0.0f, -1.0f, 0.0f});

  cam::OrbitCamera below = level;
  below.drag_orbit({0.0f, -0.5f});
  expect_pose(below, {0.0f, -kD, 0.0f}, {0.0f, 0.0f, 1.0f});

  cam::OrbitCamera around = level;
  for (int step = 0; step < 200; ++step) around.drag_orbit({0.0f, 0.01f});
  expect_mat_near(around.view(), level.view(), 1e-4f);
}

// However a seeded run of drags and pans goes, over the poles and back, the
// camera's right stays across the up, its axes stay a right-handed orthonormal
// set, the distance holds and the angles stay in range.
TEST(OrbitCamera, OrbitNeverRolls) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  const auto random_drag = [&] {
    const float x = unit(rng);
    const float y = unit(rng);
    return glm::vec2(x, y);
  };
  for (const glm::vec3 up :
       {glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, -1.0f, 0.0f),
        glm::normalize(glm::vec3(0.3f, -0.9f, 0.3f))}) {
    SCOPED_TRACE(testing::Message() << up.x << ", " << up.y << ", " << up.z);
    cam::OrbitCamera orbit;
    orbit.set_world_up(2.5f * up);
    orbit.set_target({0.3f, 0.1f, -0.2f});
    orbit.set_distance(4.0f);
    for (int i = 0; i < 2000; ++i) {
      // Mostly small drags, with a long one now and then to jump a pole.
      orbit.drag_orbit((i % 20 == 0 ? 3.0f : 0.05f) * random_drag());
      if (i % 7 == 0) orbit.drag_pan(0.1f * random_drag(), glm::radians(50.0f));

      const Axes axes = view_axes(orbit);
      ASSERT_NEAR(glm::dot(axes.right, up), 0.0f, 1e-5f);
      ASSERT_NEAR(glm::length(axes.right), 1.0f, 1e-5f);
      ASSERT_NEAR(glm::length(axes.up), 1.0f, 1e-5f);
      ASSERT_NEAR(glm::dot(axes.right, axes.up), 0.0f, 1e-5f);
      ASSERT_NEAR(glm::length(glm::cross(axes.right, axes.up) - axes.back),
                  0.0f, 1e-5f);
      ASSERT_NEAR(glm::length(orbit.eye() - orbit.target()), 4.0f, 1e-4f);
      ASSERT_LE(std::fabs(orbit.azimuth()), glm::pi<float>());
      ASSERT_LE(std::fabs(orbit.elevation()), glm::pi<float>());
    }
  }
}

// A closed drag loop returns to the start view, upright or upside down, as long
// as it does not cross a pole. One that does lands turned: its sideways legs
// run opposite ways in azimuth on the two sides of the pole, so they add.
TEST(OrbitCamera, ClosedDragLoopsReturnToTheStart) {
  for (const float elevation : {0.3f, 2.4f}) {
    SCOPED_TRACE(elevation);
    cam::OrbitCamera orbit;
    orbit.set_world_up({0.0f, 0.0f, 1.0f});
    orbit.set_target({1.0f, 2.0f, 3.0f});
    orbit.set_distance(3.0f);
    orbit.set_azimuth(0.7f);
    orbit.set_elevation(elevation);
    const glm::mat4 start = orbit.view();

    // A square of 0.1 heights a side.
    for (const glm::vec2 side :
         {glm::vec2(1.0f, 0.0f), glm::vec2(0.0f, 1.0f), glm::vec2(-1.0f, 0.0f),
          glm::vec2(0.0f, -1.0f)}) {
      for (int step = 0; step < 10; ++step) orbit.drag_orbit(0.01f * side);
    }
    expect_mat_near(orbit.view(), start, 1e-4f);

    // A circle of 0.1 heights' radius.
    constexpr int kSteps = 64;
    for (int step = 0; step < kSteps; ++step) {
      const float from = glm::two_pi<float>() * static_cast<float>(step) /
                         static_cast<float>(kSteps);
      const float to = glm::two_pi<float>() * static_cast<float>(step + 1) /
                       static_cast<float>(kSteps);
      orbit.drag_orbit(0.1f * glm::vec2(std::cos(to) - std::cos(from),
                                        std::sin(to) - std::sin(from)));
    }
    expect_mat_near(orbit.view(), start, 1e-4f);
  }

  constexpr float kSideways = 0.2f;
  for (const float over_a_pole : {0.6f, -0.6f}) {
    SCOPED_TRACE(over_a_pole);
    cam::OrbitCamera crossing;
    crossing.set_world_up({0.0f, 0.0f, 1.0f});
    crossing.drag_orbit({0.0f, over_a_pole});
    crossing.drag_orbit({kSideways, 0.0f});
    crossing.drag_orbit({0.0f, -over_a_pole});
    crossing.drag_orbit({-kSideways, 0.0f});
    EXPECT_NEAR(crossing.azimuth(), 2.0f * kSideways * glm::pi<float>(), 1e-5f);
    EXPECT_NEAR(crossing.elevation(), 0.0f, 1e-5f);
  }
}

// The yaw's sign comes from the elevation a call starts at, not the one it ends
// at, on either way across a pole.
TEST(OrbitCamera, OrbitTakesItsYawSignFromTheStartingElevation) {
  cam::OrbitCamera orbit;
  orbit.set_elevation(1.5f);
  orbit.orbit(0.2f, 0.2f);
  EXPECT_FLOAT_EQ(orbit.azimuth(), 0.2f);
  EXPECT_FLOAT_EQ(orbit.elevation(), 1.7f);

  orbit.orbit(0.2f, -0.2f);
  EXPECT_NEAR(orbit.azimuth(), 0.0f, 1e-6f);
  EXPECT_FLOAT_EQ(orbit.elevation(), 1.5f);
}

// A set up stands the scene upright: unit whatever the length given, the level
// frame's sides as the class says, the same view in either order of setters. A
// bad up is ignored, and an up along +-X, the rule's fallback, stays defined.
TEST(OrbitCamera, WorldUpStandsTheSceneUpright) {
  cam::OrbitCamera orbit;
  EXPECT_EQ(orbit.world_up(), glm::vec3(0.0f, 1.0f, 0.0f));
  orbit.set_world_up({0.0f, 0.0f, 2.0f});
  EXPECT_EQ(orbit.world_up(), glm::vec3(0.0f, 0.0f, 1.0f));
  orbit.set_target({0.0f, 0.0f, 0.85f});
  orbit.set_distance(3.0f);
  expect_vec_near(orbit.eye(), {0.0f, -3.0f, 0.85f}, 1e-5f);
  expect_vec_near(view_axes(orbit).right, {1.0f, 0.0f, 0.0f}, 1e-6f);
  expect_vec_near(view_axes(orbit).up, {0.0f, 0.0f, 1.0f}, 1e-6f);

  cam::OrbitCamera down;
  down.set_world_up({0.0f, -1.0f, 0.0f});
  expect_vec_near(down.eye(), {0.0f, 0.0f, -1.0f}, 1e-6f);
  expect_vec_near(view_axes(down).right, {1.0f, 0.0f, 0.0f}, 1e-6f);
  expect_vec_near(view_axes(down).up, {0.0f, -1.0f, 0.0f}, 1e-6f);

  const glm::vec3 oblique(0.2f, 0.3f, 0.9f);
  cam::OrbitCamera up_first;
  up_first.set_world_up(oblique);
  up_first.set_azimuth(1.1f);
  up_first.set_elevation(2.2f);
  cam::OrbitCamera up_last;
  up_last.set_azimuth(1.1f);
  up_last.set_elevation(2.2f);
  up_last.set_world_up(oblique);
  EXPECT_EQ(up_first.view(), up_last.view());

  // The target stays too: one that depends on the up is the caller's to move.
  cam::OrbitCamera keeps_target;
  keeps_target.set_target({1.0f, 2.0f, 3.0f});
  keeps_target.set_world_up(oblique);
  EXPECT_EQ(keeps_target.target(), glm::vec3(1.0f, 2.0f, 3.0f));

  // The heading off the exact axes: world +X's part across the up, up to where
  // the up is about 1e-3 rad off X, and inside that the fallback's up x +Z.
  const auto right_at = [](const glm::vec3& up) {
    cam::OrbitCamera at;
    at.set_world_up(up);
    return view_axes(at).right;
  };
  const glm::vec3 tilted = glm::normalize(glm::vec3(0.3f, -0.9f, 0.3f));
  expect_vec_near(
      right_at(tilted),
      glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f) - tilted.x * tilted), 1e-6f);
  expect_vec_near(right_at({std::cos(0.01f), 0.0f, std::sin(0.01f)}),
                  {std::sin(0.01f), 0.0f, -std::cos(0.01f)}, 1e-6f);
  expect_vec_near(right_at({1.0f, 0.0f, 1e-4f}), {0.0f, -1.0f, 0.0f}, 1e-6f);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  for (const glm::vec3 bad :
       {glm::vec3(0.0f), glm::vec3(kNan, 0.0f, 0.0f),
        glm::vec3(0.0f, kInf, 0.0f), glm::vec3(0.0f, 0.0f, -kInf)}) {
    SCOPED_TRACE(testing::Message() << bad.x << ", " << bad.y << ", " << bad.z);
    orbit.set_world_up(bad);
    EXPECT_EQ(orbit.world_up(), glm::vec3(0.0f, 0.0f, 1.0f));
  }

  // Any length, down to the smallest float and up to the largest.
  for (const float length :
       {std::numeric_limits<float>::denorm_min(), 3e-23f, 1e-22f, 1e30f,
        std::numeric_limits<float>::max()}) {
    for (const glm::vec3 axis :
         {glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f),
          glm::vec3(0.0f, 0.0f, 1.0f)}) {
      SCOPED_TRACE(testing::Message() << length << " along " << axis.x << ", "
                                      << axis.y << ", " << axis.z);
      cam::OrbitCamera any_length;
      any_length.set_world_up(length * axis);
      EXPECT_EQ(any_length.world_up(), axis);
    }
  }

  for (const float sign : {1.0f, -1.0f}) {
    SCOPED_TRACE(sign);
    const glm::vec3 up(sign, 0.0f, 0.0f);
    cam::OrbitCamera along_x;
    along_x.set_world_up(up);
    along_x.set_elevation(0.4f);
    const Axes axes = view_axes(along_x);
    EXPECT_NEAR(glm::dot(axes.right, up), 0.0f, 1e-6f);
    EXPECT_NEAR(glm::length(axes.right), 1.0f, 1e-6f);
    EXPECT_GT(glm::dot(axes.up, up), 0.0f);  // level, and upright
    expect_vec_near(glm::cross(up, glm::vec3(0.0f, 0.0f, 1.0f)), axes.right,
                    1e-6f);
    expect_vec_near(along_x.eye() - along_x.target(), axes.back, 1e-6f);
  }
}

// An up near +-X keeps the camera's right across it, also just outside the
// rule's fallback, where X - up.x * up cancels in float.
TEST(OrbitCamera, RightStaysAcrossAnUpNearX) {
  constexpr int kAround = 64;
  for (const float off_x : {1e-4f, 1.01e-3f, 2e-3f, 1e-2f, 3e-2f}) {
    for (const float sign : {1.0f, -1.0f}) {
      for (int step = 0; step < kAround; ++step) {
        const float around = glm::two_pi<float>() * static_cast<float>(step) /
                             static_cast<float>(kAround);
        const glm::vec3 up(sign * std::cos(off_x),
                           std::sin(off_x) * std::cos(around),
                           std::sin(off_x) * std::sin(around));
        SCOPED_TRACE(testing::Message()
                     << up.x << ", " << up.y << ", " << up.z);
        cam::OrbitCamera orbit;
        orbit.set_world_up(up);
        const Axes axes = view_axes(orbit);
        EXPECT_NEAR(glm::dot(axes.right, orbit.world_up()), 0.0f, 1e-6f);
        EXPECT_NEAR(glm::length(axes.back), 1.0f, 1e-6f);
      }
    }
  }
}

// One viewport height dragged turns the eye exactly kDragRadians, half a turn;
// a vertical drag turns the elevation at the same rate.
TEST(OrbitCamera, OneViewportHeightTurnsDragRadians) {
  EXPECT_FLOAT_EQ(cam::OrbitCamera::kDragRadians, glm::pi<float>());

  cam::OrbitCamera orbit;
  orbit.set_azimuth(0.25f);
  orbit.set_elevation(-0.5f);
  orbit.drag_orbit({1.0f, 0.0f});
  EXPECT_FLOAT_EQ(orbit.azimuth(), 0.25f - cam::OrbitCamera::kDragRadians);
  EXPECT_FLOAT_EQ(orbit.elevation(), -0.5f);

  orbit.drag_orbit({0.0f, 0.25f});
  EXPECT_FLOAT_EQ(orbit.azimuth(), 0.25f - cam::OrbitCamera::kDragRadians);
  EXPECT_FLOAT_EQ(orbit.elevation(),
                  -0.5f + 0.25f * cam::OrbitCamera::kDragRadians);
}

// drag_pan keeps the points at the target's depth under the pointer: through
// to_camera, each moves on screen by exactly the drag, at any field of view,
// upright and upside down.
TEST(OrbitCamera, DragPanKeepsTargetDepthPointsUnderThePointer) {
  constexpr float kAspect = 1.5f;
  for (const cam::OrbitCamera& start : start_poses()) {
    for (const glm::vec2 drag :
         {glm::vec2(0.125f, 0.0f), glm::vec2(0.0f, -0.3f),
          glm::vec2(0.125f, -0.3f)}) {
      for (const float fovy : {glm::radians(30.0f), glm::radians(75.0f)}) {
        SCOPED_TRACE(testing::Message()
                     << fovy << " at elevation " << start.elevation()
                     << ", up z " << start.world_up().z << ", drag " << drag.x
                     << ", " << drag.y);
        cam::OrbitCamera orbit = start;
        const Axes axes = view_axes(orbit);
        const glm::vec3 points[] = {
            orbit.target(),
            orbit.target() + 0.75f * axes.right - 0.5f * axes.up};
        const cam::Camera before = orbit.to_camera(fovy, kAspect, 0.1f, 100.0f);

        orbit.drag_pan(drag, fovy);
        const cam::Camera after = orbit.to_camera(fovy, kAspect, 0.1f, 100.0f);
        for (const glm::vec3& point : points) {
          const glm::vec2 moved = in_heights(after, kAspect, point) -
                                  in_heights(before, kAspect, point);
          EXPECT_NEAR(moved.x, drag.x, 1e-5f);
          EXPECT_NEAR(moved.y, drag.y, 1e-5f);
        }
        EXPECT_FLOAT_EQ(orbit.distance(), 5.0f);
      }
    }
  }
}

// zoom ignores a factor it cannot scale by -- a pinch can report 0 -- so the
// distance keeps its value.
TEST(OrbitCamera, ZoomIgnoresNonPositiveAndNonFiniteFactors) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  cam::OrbitCamera orbit;
  orbit.set_distance(4.0f);
  for (const float factor : {0.0f, -0.0f, -0.5f, -kInf, kInf,
                             std::numeric_limits<float>::quiet_NaN()}) {
    SCOPED_TRACE(factor);
    orbit.zoom(factor);
    EXPECT_FLOAT_EQ(orbit.distance(), 4.0f);
  }
}

// A non-finite drag -- a delta over a zero height -- or field of view leaves
// the state as it was instead of poisoning it, by any verb. So does a turn too
// small to register, and a zero one.
TEST(OrbitCamera, DragsIgnoreNonFiniteInput) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kFovy = glm::radians(50.0f);
  const cam::OrbitCamera start = oblique_orbit();
  const auto expect_unchanged = [&](const cam::OrbitCamera& orbit) {
    EXPECT_EQ(orbit.target(), start.target());
    EXPECT_EQ(orbit.distance(), start.distance());
    EXPECT_EQ(orbit.world_up(), start.world_up());
    EXPECT_EQ(orbit.azimuth(), start.azimuth());
    EXPECT_EQ(orbit.elevation(), start.elevation());
    EXPECT_TRUE(orbit.view() == start.view());
  };
  for (const glm::vec2 drag : {glm::vec2(kNan, 0.0f), glm::vec2(0.0f, kNan),
                               glm::vec2(kInf, 0.0f), glm::vec2(0.0f, -kInf)}) {
    SCOPED_TRACE(testing::Message() << drag.x << ", " << drag.y);
    cam::OrbitCamera orbit = start;
    orbit.drag_orbit(drag);
    expect_unchanged(orbit);
    orbit.drag_pan(drag, kFovy);
    expect_unchanged(orbit);
    orbit.orbit(drag.x, drag.y);
    expect_unchanged(orbit);
  }
  for (const float fovy : {kNan, kInf}) {
    SCOPED_TRACE(fovy);
    cam::OrbitCamera orbit = start;
    orbit.drag_pan({0.1f, 0.0f}, fovy);
    expect_unchanged(orbit);
  }

  cam::OrbitCamera orbit = start;
  orbit.orbit(0.0f, 0.0f);
  expect_unchanged(orbit);
  orbit.drag_orbit({0.0f, 0.0f});
  expect_unchanged(orbit);
  orbit.orbit(1e-25f, 0.0f);
  expect_unchanged(orbit);
}

// A move that would leave the target or the distance non-finite is ignored, by
// any verb and also when its deltas are finite, so a later good move still
// works.
TEST(OrbitCamera, MovesToANonFiniteStateAreIgnored) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kMax = std::numeric_limits<float>::max();
  const cam::OrbitCamera start = oblique_orbit();
  for (const glm::vec2 delta :
       {glm::vec2(kNan, 0.0f), glm::vec2(0.0f, kNan), glm::vec2(kInf, 0.0f),
        glm::vec2(0.0f, -kInf)}) {
    SCOPED_TRACE(testing::Message() << delta.x << ", " << delta.y);
    cam::OrbitCamera orbit = start;
    orbit.pan(delta.x, delta.y);
    EXPECT_EQ(orbit.target(), start.target());
  }

  cam::OrbitCamera panned;
  panned.pan(kMax, kMax);
  const glm::vec3 reached = panned.target();
  panned.pan(kMax, kMax);
  EXPECT_EQ(panned.target(), reached);
  panned.pan(-kMax, -kMax);
  EXPECT_NE(panned.target(), reached);

  cam::OrbitCamera orbit = start;
  orbit.set_distance(kInf);
  EXPECT_EQ(orbit.distance(), start.distance());
  orbit.dolly(kInf);
  EXPECT_EQ(orbit.distance(), start.distance());
  orbit.zoom(kMax);
  EXPECT_EQ(orbit.distance(), start.distance());
  EXPECT_TRUE(orbit.view() == start.view());
  orbit.zoom(0.5f);
  EXPECT_FLOAT_EQ(orbit.distance(), 0.5f * start.distance());
}

// A wheel notch in, then one out, returns to the start distance.
TEST(OrbitCamera, WheelNotchRoundTrips) {
  cam::OrbitCamera orbit;
  orbit.set_distance(5.0f);
  orbit.zoom(std::pow(cam::OrbitCamera::kZoomPerNotch, 1.0f));  // zooms in
  EXPECT_FLOAT_EQ(orbit.distance(), 4.5f);

  orbit.zoom(std::pow(cam::OrbitCamera::kZoomPerNotch, -1.0f));
  EXPECT_NEAR(orbit.distance(), 5.0f, 1e-5f);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Pure-CPU tests for CameraRig: known-value checks on the motion verbs, the
// level-horizon invariant, and the free-roll fallback. No Vulkan device is
// needed, so these run on every platform.

#include <gtest/gtest.h>

#include <cmath>

#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/camera/camera_rig.hpp"
//
#include <glm/geometric.hpp>  // length, normalize
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>  // radians
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace cam = volumetric_kit::gfx::camera;

namespace {

// Project a world point through the camera and apply the perspective divide.
glm::vec3 to_ndc(const cam::Camera& camera, const glm::vec3& world) {
  const glm::vec4 clip = camera.view_proj() * glm::vec4(world, 1.0f);
  return glm::vec3(clip) / clip.w;
}

void expect_vec_near(const glm::vec3& got, const glm::vec3& want, float eps) {
  EXPECT_NEAR(got.x, want.x, eps);
  EXPECT_NEAR(got.y, want.y, eps);
  EXPECT_NEAR(got.z, want.z, eps);
}

}  // namespace

// A default rig sits at the origin looking down world -Z, +Y up, with the pivot
// one unit ahead.
TEST(CameraRig, DefaultPose) {
  const cam::CameraRig rig;
  expect_vec_near(rig.position(), {0.0f, 0.0f, 0.0f}, 1e-6f);
  expect_vec_near(rig.forward(), {0.0f, 0.0f, -1.0f}, 1e-6f);
  expect_vec_near(rig.right(), {1.0f, 0.0f, 0.0f}, 1e-6f);
  expect_vec_near(rig.up(), {0.0f, 1.0f, 0.0f}, 1e-6f);
  EXPECT_FLOAT_EQ(rig.focus_distance(), 1.0f);
  expect_vec_near(rig.focus_point(), {0.0f, 0.0f, -1.0f}, 1e-6f);
}

// move_local translates along the eye's own axes: x -> right, y -> up,
// z -> forward. At the default pose those are +X, +Y, -Z.
TEST(CameraRig, MoveLocalTranslatesAlongLocalAxes) {
  cam::CameraRig rig;
  rig.move_local({2.0f, 3.0f, 5.0f});
  expect_vec_near(rig.position(), {2.0f, 3.0f, -5.0f}, 1e-5f);
}

// look rotates the view in place: position is fixed, forward turns. A +half-pi
// yaw from the default swings forward from -Z onto -X (positive yaw turns
// left).
TEST(CameraRig, LookRotatesInPlace) {
  cam::CameraRig rig;
  rig.look(glm::half_pi<float>(), 0.0f);
  expect_vec_near(rig.position(), {0.0f, 0.0f, 0.0f}, 1e-6f);
  expect_vec_near(rig.forward(), {-1.0f, 0.0f, 0.0f}, 1e-5f);
}

// move_local is frame-relative: after turning to face -X, flying "forward"
// (+z local) moves the eye along -X.
TEST(CameraRig, MoveLocalIsFrameRelative) {
  cam::CameraRig rig;
  rig.look(glm::half_pi<float>(), 0.0f);  // now facing -X
  rig.move_local({0.0f, 0.0f, 1.0f});     // forward
  expect_vec_near(rig.position(), {-1.0f, 0.0f, 0.0f}, 1e-5f);
}

// orbit swings the eye around the pivot: a +half-pi yaw from the default moves
// the eye onto the sphere while the pivot and look direction stay put.
TEST(CameraRig, OrbitSwingsEyeAroundPivot) {
  cam::CameraRig rig;  // pivot at (0, 0, -1)
  rig.orbit(glm::half_pi<float>(), 0.0f);
  expect_vec_near(rig.forward(), {-1.0f, 0.0f, 0.0f}, 1e-5f);
  expect_vec_near(rig.focus_point(), {0.0f, 0.0f, -1.0f}, 1e-5f);
  expect_vec_near(rig.position(), {1.0f, 0.0f, -1.0f}, 1e-5f);
}

// orbit preserves the pivot point and the focus distance for arbitrary angles.
TEST(CameraRig, OrbitKeepsPivotAndDistance) {
  cam::CameraRig rig;
  rig.set_focus_distance(5.0f);  // pivot at (0, 0, -5)
  rig.orbit(glm::radians(37.0f), glm::radians(21.0f));

  expect_vec_near(rig.focus_point(), {0.0f, 0.0f, -5.0f}, 1e-4f);
  EXPECT_NEAR(glm::length(rig.position() - rig.focus_point()), 5.0f, 1e-4f);
}

// In level-horizon mode the pitch is clamped shy of the poles, so forward never
// reaches straight up/down and the horizon basis never degenerates.
TEST(CameraRig, PitchClampsAtPolesWhenLevel) {
  const float max_y = std::sin(cam::CameraRig::kMaxPitch);

  cam::CameraRig rig;
  rig.look(0.0f, 10.0f);  // way past +pi/2
  EXPECT_NEAR(rig.forward().y, max_y, 1e-5f);
  EXPECT_LT(rig.forward().y, 1.0f);

  rig.look(0.0f, -100.0f);  // way past -pi/2
  EXPECT_NEAR(rig.forward().y, -max_y, 1e-5f);
}

// Level-horizon keeps the roll at zero: however the rig is yawed and pitched,
// the right axis stays horizontal (no y component).
TEST(CameraRig, LevelHorizonKeepsRightHorizontal) {
  cam::CameraRig rig;
  rig.look(glm::radians(50.0f), glm::radians(30.0f));
  rig.look(glm::radians(-20.0f), glm::radians(-70.0f));
  rig.orbit(glm::radians(80.0f), glm::radians(15.0f));
  EXPECT_NEAR(rig.right().y, 0.0f, 1e-6f);
}

// Clearing level_horizon allows free 6-DoF: pitch is unclamped, so the look ray
// can tip past the pole and point back the way it came (forward.z > 0). The
// same input clamps when level.
TEST(CameraRig, FreeModeAllowsPitchPastThePole) {
  cam::CameraRig free_rig;
  free_rig.set_level_horizon(false);
  free_rig.look(0.0f, 3.0f);  // ~172 degrees of pitch
  EXPECT_GT(free_rig.forward().z, 0.5f);

  cam::CameraRig level_rig;
  level_rig.look(0.0f, 3.0f);
  EXPECT_LT(level_rig.forward().z,
            0.0f);  // clamped, still in the -Z hemisphere
}

// zoom dollies the eye toward the pivot without moving the pivot; a factor < 1
// shortens the focus distance and pulls the eye in.
TEST(CameraRig, ZoomHoldsPivotAndMovesEye) {
  cam::CameraRig rig;
  rig.set_focus_distance(4.0f);  // pivot at (0, 0, -4)
  rig.zoom(0.5f);

  EXPECT_FLOAT_EQ(rig.focus_distance(), 2.0f);
  expect_vec_near(rig.focus_point(), {0.0f, 0.0f, -4.0f}, 1e-5f);
  expect_vec_near(rig.position(), {0.0f, 0.0f, -2.0f}, 1e-5f);
}

// Focus distance is clamped to a positive minimum, so the pivot can never reach
// the eye (which would collapse the view ray).
TEST(CameraRig, FocusDistanceClampsToPositiveMinimum) {
  cam::CameraRig rig;
  rig.set_focus_distance(-3.0f);
  EXPECT_FLOAT_EQ(rig.focus_distance(), cam::CameraRig::kMinFocusDistance);

  rig.set_focus_distance(10.0f);
  rig.zoom(0.0f);  // multiplicative collapse
  EXPECT_FLOAT_EQ(rig.focus_distance(), cam::CameraRig::kMinFocusDistance);
}

// set_focus aims the eye at a world point and sets the focus distance to the
// gap, so the pivot lands exactly on that point and to_camera centers on it.
TEST(CameraRig, SetFocusAimsAtTargetAndCenters) {
  cam::CameraRig rig;
  rig.set_position({3.0f, 2.0f, 1.0f});
  rig.set_focus({0.0f, 0.0f, 0.0f});

  EXPECT_NEAR(rig.focus_distance(), std::sqrt(14.0f), 1e-5f);
  expect_vec_near(rig.forward(), glm::normalize(glm::vec3(-3.0f, -2.0f, -1.0f)),
                  1e-5f);
  expect_vec_near(rig.focus_point(), {0.0f, 0.0f, 0.0f}, 1e-4f);

  const cam::Camera camera =
      rig.to_camera(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  const glm::vec3 ndc = to_ndc(camera, {0.0f, 0.0f, 0.0f});
  EXPECT_NEAR(ndc.x, 0.0f, 1e-4f);
  EXPECT_NEAR(ndc.y, 0.0f, 1e-4f);
}

// stage_transform is the eye-to-world pose: it maps the local origin to the
// eye position and the local -Z direction to the world forward axis.
TEST(CameraRig, StageTransformIsEyeToWorldPose) {
  cam::CameraRig rig;
  rig.set_position({-2.0f, 5.0f, 3.0f});
  rig.look(glm::radians(35.0f), glm::radians(10.0f));

  const glm::mat4 pose = rig.stage_transform();
  const glm::vec3 mapped_origin = glm::vec3(pose * glm::vec4(0, 0, 0, 1));
  const glm::vec3 mapped_forward = glm::vec3(pose * glm::vec4(0, 0, -1, 0));
  expect_vec_near(mapped_origin, rig.position(), 1e-5f);
  expect_vec_near(mapped_forward, rig.forward(), 1e-5f);
}

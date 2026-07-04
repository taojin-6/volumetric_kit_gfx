// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Pure-CPU tests for CameraRig: known-value checks on the motion verbs, the
// level-horizon invariant, and the free-roll fallback. No Vulkan device is
// needed, so these run on every platform.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/camera/camera_rig.hpp"
#include "volumetric_kit/gfx/camera/orbit_camera.hpp"  // cross-check orbit sign
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
// reaches straight up/down and the horizon basis never degenerates. Positive
// pitch tilts the view down (OrbitCamera's sign), so it clamps at the bottom.
TEST(CameraRig, PitchClampsAtPolesWhenLevel) {
  const float max_y = std::sin(cam::CameraRig::kMaxPitch);

  cam::CameraRig rig;
  rig.look(0.0f, 10.0f);  // way past the bottom pole (+pitch tilts down)
  EXPECT_NEAR(rig.forward().y, -max_y, 1e-5f);
  EXPECT_GT(rig.forward().y, -1.0f);

  rig.look(0.0f, -100.0f);  // way past the top pole
  EXPECT_NEAR(rig.forward().y, max_y, 1e-5f);
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

// orbit's +pitch matches OrbitCamera's +elevation: it raises the eye over the
// pivot (tilting the view down). Both controllers build on the shared
// spherical_direction parametrization (negated between them), so this
// cross-check guards that shared math and its negated uses against a sign
// regression.
TEST(CameraRig, OrbitPitchRaisesEyeLikeOrbitCamera) {
  cam::CameraRig rig;  // origin, pivot at (0, 0, -1)
  rig.orbit(0.0f, glm::radians(30.0f));
  EXPECT_GT(rig.position().y, 0.0f);  // eye rose above the pivot
  EXPECT_LT(rig.forward().y, 0.0f);   // view tilted down toward the pivot

  cam::OrbitCamera orbit;  // same +elevation raises the turntable eye
  orbit.set_distance(1.0f);
  orbit.orbit(0.0f, glm::radians(30.0f));
  EXPECT_GT(orbit.eye().y, 0.0f);
}

// set_focus aims exactly even at a target directly overhead: the pole-safe
// basis keeps focus_point() on the target instead of clamping the aim shy of
// vertical.
TEST(CameraRig, SetFocusAimsExactlyAtVerticalTarget) {
  cam::CameraRig rig;
  rig.set_position({0.0f, 0.0f, 0.0f});
  rig.set_focus({0.0f, 10.0f, 0.0f});  // straight up

  EXPECT_FLOAT_EQ(rig.focus_distance(), 10.0f);
  expect_vec_near(rig.forward(), {0.0f, 1.0f, 0.0f}, 1e-5f);
  expect_vec_near(rig.focus_point(), {0.0f, 10.0f, 0.0f}, 1e-4f);
}

// pan slides the eye within its view plane along right()/up(); at the default
// pose those are +X/+Y with Z unchanged, and it follows the frame after a turn.
TEST(CameraRig, PanSlidesInViewPlane) {
  cam::CameraRig rig;
  rig.pan(2.0f, 3.0f);
  expect_vec_near(rig.position(), {2.0f, 3.0f, 0.0f}, 1e-5f);

  cam::CameraRig turned;
  turned.look(glm::half_pi<float>(), 0.0f);  // face -X; right() is now -Z
  turned.pan(1.0f, 0.0f);
  expect_vec_near(turned.position(), {0.0f, 0.0f, -1.0f}, 1e-5f);
}

// snap_turn yaws in place about world up: position is fixed and a +half-pi turn
// swings forward from -Z onto -X, staying level and independent of the mode.
TEST(CameraRig, SnapTurnYawsAboutWorldUpInPlace) {
  cam::CameraRig rig;
  rig.set_position({1.0f, 2.0f, 3.0f});
  rig.snap_turn(glm::half_pi<float>());
  expect_vec_near(rig.position(), {1.0f, 2.0f, 3.0f}, 1e-6f);  // fixed
  expect_vec_near(rig.forward(), {-1.0f, 0.0f, 0.0f}, 1e-5f);  // -Z -> -X
  EXPECT_NEAR(rig.right().y, 0.0f, 1e-6f);                     // stays level

  // Independent of level_horizon: still yaws in place with roll present.
  cam::CameraRig free_rig;
  free_rig.set_level_horizon(false);
  free_rig.look(0.3f, 0.4f);  // introduce roll
  const glm::vec3 before = free_rig.position();
  free_rig.snap_turn(glm::half_pi<float>());
  expect_vec_near(free_rig.position(), before, 1e-6f);
}

// A NaN or non-positive zoom / focus-distance clamps to the floor instead of
// poisoning the pose (std::max floors it -- the arg order matters for NaN).
TEST(CameraRig, NonFiniteFocusDistanceClampsToFloor) {
  const float nan = std::numeric_limits<float>::quiet_NaN();

  cam::CameraRig rig;
  rig.set_focus_distance(4.0f);
  rig.zoom(nan);
  EXPECT_FLOAT_EQ(rig.focus_distance(), cam::CameraRig::kMinFocusDistance);
  EXPECT_TRUE(std::isfinite(rig.position().x));
  EXPECT_TRUE(std::isfinite(rig.position().y));
  EXPECT_TRUE(std::isfinite(rig.position().z));

  rig.set_focus_distance(nan);
  EXPECT_FLOAT_EQ(rig.focus_distance(), cam::CameraRig::kMinFocusDistance);
}

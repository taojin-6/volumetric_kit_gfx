// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ImageView2D: the fit, the image <-> target mapping and its inverse, zooming
// about a fixed point, panning, the zoom limits, and what a viewport resize
// keeps. Pure CPU.

#include <gtest/gtest.h>

#include <glm/vec2.hpp>

#include "volumetric_kit/gfx/camera/image_view_2d.hpp"

namespace {

using volumetric_kit::gfx::camera::ImageView2D;

constexpr float kEps = 1e-4f;

void expect_near(glm::vec2 actual, glm::vec2 expected) {
  EXPECT_NEAR(actual.x, expected.x, kEps);
  EXPECT_NEAR(actual.y, expected.y, kEps);
}

TEST(ImageView2DTest, FitsAWideImageCenteredWithBarsAboveAndBelow) {
  // A 400 x 100 image in a 200 x 200 viewport at (10, 20): the width limits
  // the scale to 0.5, so the image is 200 x 50, centered vertically.
  const ImageView2D view({400.0f, 100.0f}, {10.0f, 20.0f}, {200.0f, 200.0f});
  EXPECT_FLOAT_EQ(view.scale(), 0.5f);
  EXPECT_FLOAT_EQ(view.zoom(), 1.0f);
  expect_near(view.origin(), {10.0f, 20.0f + 75.0f});
  expect_near(view.image_to_target({400.0f, 100.0f}), {210.0f, 20.0f + 125.0f});
}

TEST(ImageView2DTest, TargetToImageInvertsImageToTarget) {
  ImageView2D view({640.0f, 480.0f}, {0.0f, 0.0f}, {1000.0f, 700.0f});
  view.zoom_about({300.0f, 200.0f}, 3.0f);
  view.pan({-17.0f, 41.0f});
  const glm::vec2 points[] = {{0.0f, 0.0f}, {12.5f, 400.25f}, {639.0f, 1.0f}};
  for (const glm::vec2& p : points) {
    expect_near(view.target_to_image(view.image_to_target(p)), p);
  }
}

TEST(ImageView2DTest, ZoomAboutKeepsThePointUnderTheCursor) {
  ImageView2D view({1920.0f, 1080.0f}, {0.0f, 0.0f}, {960.0f, 540.0f});
  const glm::vec2 cursor{700.0f, 120.0f};
  const glm::vec2 under = view.target_to_image(cursor);
  view.zoom_about(cursor, 4.0f);
  EXPECT_FLOAT_EQ(view.zoom(), 4.0f);
  EXPECT_FLOAT_EQ(view.scale(), 2.0f);  // the fit was 0.5
  expect_near(view.image_to_target(under), cursor);
}

TEST(ImageView2DTest, PanMovesTheImageByTheDrag) {
  ImageView2D view({100.0f, 100.0f}, {0.0f, 0.0f}, {200.0f, 200.0f});
  const glm::vec2 before = view.origin();
  view.pan({15.0f, -5.0f});
  expect_near(view.origin(), before + glm::vec2{15.0f, -5.0f});
}

TEST(ImageView2DTest, ResizeKeepsTheCenterAndScalesWithTheViewport) {
  ImageView2D view({1000.0f, 1000.0f}, {0.0f, 0.0f}, {500.0f, 500.0f});
  view.zoom_about({100.0f, 100.0f}, 2.0f);
  const glm::vec2 center = view.center();
  const float scale = view.scale();

  // Half the window: the same part of the image, at half the scale.
  view.set_viewport({0.0f, 0.0f}, {250.0f, 250.0f});
  expect_near(view.center(), center);
  EXPECT_FLOAT_EQ(view.zoom(), 2.0f);
  EXPECT_FLOAT_EQ(view.scale(), scale * 0.5f);
  expect_near(view.image_to_target(center), {125.0f, 125.0f});
}

TEST(ImageView2DTest, AnEmptyViewportHasZeroScaleAndMapsToTheCenter) {
  ImageView2D view({64.0f, 64.0f}, {0.0f, 0.0f}, {128.0f, 128.0f});
  view.set_viewport({0.0f, 0.0f}, {0.0f, 0.0f});  // minimized
  EXPECT_FLOAT_EQ(view.scale(), 0.0f);
  expect_near(view.target_to_image({5.0f, 5.0f}), view.center());
  view.zoom_about({5.0f, 5.0f}, 2.0f);  // ignored, not a division by zero
  EXPECT_FLOAT_EQ(view.zoom(), 1.0f);
}

TEST(ImageView2DTest, ZoomIsClampedToTheLimits) {
  ImageView2D view({100.0f, 100.0f}, {0.0f, 0.0f}, {100.0f, 100.0f});
  view.set_zoom_limits(0.5f, 8.0f);
  view.zoom_about({50.0f, 50.0f}, 100.0f);
  EXPECT_FLOAT_EQ(view.scale(), 8.0f);
  view.zoom_about({50.0f, 50.0f}, 1e-4f);
  EXPECT_FLOAT_EQ(view.zoom(), 0.5f);
}

TEST(ImageView2DTest, AFitBeyondMaxScaleIsStillAllowed) {
  // A 4 x 4 image fitted to 400 x 400 is 100 pixels per texel, past the
  // default limit; the fit stands, and zooming in stops there.
  ImageView2D view({4.0f, 4.0f}, {0.0f, 0.0f}, {400.0f, 400.0f});
  EXPECT_FLOAT_EQ(view.scale(), 100.0f);
  view.zoom_about({200.0f, 200.0f}, 2.0f);
  EXPECT_FLOAT_EQ(view.scale(), 100.0f);
}

TEST(ImageView2DTest, FitAndANewImageResetTheView) {
  ImageView2D view({100.0f, 50.0f}, {0.0f, 0.0f}, {100.0f, 100.0f});
  view.zoom_about({10.0f, 10.0f}, 3.0f);
  view.fit();
  EXPECT_FLOAT_EQ(view.zoom(), 1.0f);
  expect_near(view.center(), {50.0f, 25.0f});

  view.zoom_about({10.0f, 10.0f}, 3.0f);
  view.set_image_size({20.0f, 40.0f});
  EXPECT_FLOAT_EQ(view.zoom(), 1.0f);
  EXPECT_FLOAT_EQ(view.scale(), 2.5f);
  expect_near(view.center(), {10.0f, 20.0f});
}

}  // namespace

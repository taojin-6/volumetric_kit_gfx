// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include "volumetric_kit/gfx/assets/point_cloud.hpp"

namespace assets = volumetric_kit::gfx::assets;

// The asset data model is header-only and loader-free: this test links only
// gfx_assets (no gfx_io, no tinygltf), proving the model stands alone.

// PointCloud is CPU-only and independent of any loader: construct one, attach a
// named channel, and read it back -- the path the future PLY loader relies on.
TEST(PointCloud, NamedAttributeChannelRoundTrips) {
  assets::PointCloud cloud;
  cloud.name = "scan";
  cloud.positions = {{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}};
  EXPECT_EQ(cloud.size(), 2u);

  cloud.add_attribute("intensity", 1, {0.25f, 0.75f});
  cloud.add_attribute("color", 3, {1, 0, 0, 0, 1, 0});

  const assets::PointAttribute* intensity = cloud.attribute("intensity");
  ASSERT_NE(intensity, nullptr);
  EXPECT_EQ(intensity->components, 1u);
  ASSERT_EQ(intensity->values.size(), 2u);
  EXPECT_FLOAT_EQ(intensity->values[1], 0.75f);

  // The convenience accessor finds the standard "color" channel...
  const assets::PointAttribute* color = cloud.color();
  ASSERT_NE(color, nullptr);
  EXPECT_EQ(color->components, 3u);
  EXPECT_EQ(color->values.size(), 6u);

  // ...and absent channels return null rather than aborting.
  EXPECT_EQ(cloud.normal(), nullptr);
  EXPECT_EQ(cloud.attribute("confidence"), nullptr);
}

TEST(PointCloud, DefaultIsEmpty) {
  assets::PointCloud cloud;
  EXPECT_EQ(cloud.size(), 0u);
  EXPECT_TRUE(cloud.attributes.empty());
  EXPECT_EQ(cloud.color(), nullptr);
}

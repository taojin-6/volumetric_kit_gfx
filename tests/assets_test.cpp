// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/gfx/assets/gltf_loader.hpp"
#include "volumetric_kit/gfx/assets/model.hpp"
#include "volumetric_kit/gfx/assets/point_cloud.hpp"

namespace assets = volumetric_kit::gfx::assets;

namespace {

// Absolute path to the Box.glb fixture, baked in by CMake (see
// tests/CMakeLists.txt). Empty if the fixture was not available at configure
// time -- the glTF tests then skip rather than fail.
#ifdef VG_ASSETS_DIR
constexpr const char* kAssetsDir = VG_ASSETS_DIR;
#else
constexpr const char* kAssetsDir = "";
#endif

std::string box_path() { return std::string(kAssetsDir) + "/Box.glb"; }

// True when the Box.glb fixture was fetched at configure time. When it is
// missing (e.g. CI had no network), the fixture-dependent tests skip rather
// than fail -- the PointCloud tests still exercise the model with no fixture.
bool have_fixture() { return std::string(kAssetsDir).size() > 0; }

}  // namespace

// The Khronos Box sample is a single mesh of 24 vertices (6 faces x 4, no
// shared corners because each face needs its own normal) and 36 indices (12
// triangles).
TEST(GltfLoader, LoadsBoxGlb) {
  if (!have_fixture()) GTEST_SKIP() << "Box.glb fixture unavailable";
  std::string err;
  std::optional<assets::Model> model = assets::load_gltf(box_path(), &err);
  ASSERT_TRUE(model.has_value()) << "load_gltf failed: " << err;

  ASSERT_EQ(model->meshes.size(), 1u);
  const assets::Mesh& mesh = model->meshes.front();
  EXPECT_EQ(mesh.vertices.size(), 24u);
  EXPECT_EQ(mesh.indices.size(), 36u);
  EXPECT_EQ(mesh.triangle_count(), 12u);

  // Every index must address a real vertex.
  for (std::uint32_t i : mesh.indices) {
    EXPECT_LT(i, mesh.vertices.size());
  }

  // The Box has one material; its base color is the red the sample defines
  // (0.8, 0.0, 0.0, 1.0), and it is opaque/single-sided.
  ASSERT_EQ(model->materials.size(), 1u);
  const assets::Material& mat = model->materials.front();
  EXPECT_EQ(mesh.material, 0u);
  EXPECT_NEAR(mat.base_color_factor.r, 0.8f, 1e-3f);
  EXPECT_NEAR(mat.base_color_factor.g, 0.0f, 1e-3f);
  EXPECT_NEAR(mat.base_color_factor.b, 0.0f, 1e-3f);
  EXPECT_NEAR(mat.base_color_factor.a, 1.0f, 1e-3f);
  EXPECT_EQ(mat.alpha_mode, assets::AlphaMode::Opaque);
  EXPECT_FALSE(mat.double_sided);

  // The Box has no textures; the model carries no decoded images.
  EXPECT_TRUE(model->images.empty());
  EXPECT_EQ(mat.base_color_texture, assets::kNoTexture);

  // The scene references the mesh through its node tree (data only).
  EXPECT_FALSE(model->scene.nodes.empty());
}

TEST(GltfLoader, MissingFileReportsError) {
  std::string err;
  std::optional<assets::Model> model =
      assets::load_gltf("/no/such/dir/does_not_exist.glb", &err);
  EXPECT_FALSE(model.has_value());
  EXPECT_FALSE(err.empty());
}

TEST(GltfLoader, UnsupportedExtensionReportsError) {
  std::string err;
  std::optional<assets::Model> model = assets::load_gltf("model.fbx", &err);
  EXPECT_FALSE(model.has_value());
  EXPECT_NE(err.find(".fbx"), std::string::npos);
}

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

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <glm/geometric.hpp>  // length
#include <glm/mat4x4.hpp>

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

// True when the Box.glb fixture is present and readable *now*. VG_ASSETS_DIR is
// baked in at configure time, but the file can be removed afterwards (e.g. `git
// clean` of the gitignored tests/assets/), so re-check at runtime: a missing
// fixture skips the glTF tests rather than failing them. The PointCloud tests
// still exercise the model with no fixture.
bool have_fixture() {
  if (std::string(kAssetsDir).empty()) return false;
  std::ifstream f(box_path(), std::ios::binary);
  return f.good();
}

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

  // The Box ships per-face normals; every one decodes to a unit vector. This
  // exercises the attribute-decode path -- a stride/count regression would fail
  // here, not merely change a vertex count.
  for (const assets::Vertex& v : mesh.vertices) {
    EXPECT_NEAR(glm::length(v.normal), 1.0f, 1e-3f);
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

  // The scene references the mesh through its node tree. The Box has two nodes:
  // the root carries the model matrix (a -90deg rotation about X) and parents
  // the second, which draws mesh 0. The transform lives on the node hierarchy,
  // never on the (shared) mesh.
  ASSERT_EQ(model->scene.nodes.size(), 2u);
  ASSERT_EQ(model->scene.roots.size(), 1u);
  EXPECT_EQ(model->scene.roots.front(), 0u);

  const assets::Node& root = model->scene.nodes[0];
  EXPECT_EQ(root.mesh, assets::Node::kNoMesh);
  ASSERT_EQ(root.children.size(), 1u);
  EXPECT_EQ(root.children.front(), 1u);
  // glm is column-major (transform[col][row]); the -90deg X rotation puts 0 on
  // the diagonal and +/-1 off it -- proving node_local_transform decoded it.
  EXPECT_NEAR(root.transform[1][1], 0.0f, 1e-3f);
  EXPECT_NEAR(root.transform[1][2], -1.0f, 1e-3f);
  EXPECT_NEAR(root.transform[2][1], 1.0f, 1e-3f);

  const assets::Node& mesh_node = model->scene.nodes[1];
  EXPECT_EQ(mesh_node.mesh, 0u);        // points at the single mesh...
  EXPECT_EQ(mesh_node.mesh_count, 1u);  // ...as a one-element range
  EXPECT_NEAR(mesh_node.transform[0][0], 1.0f, 1e-3f);  // local identity
  EXPECT_NEAR(mesh_node.transform[1][1], 1.0f, 1e-3f);
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

// A structurally valid glTF whose POSITION accessor claims far more elements
// than its buffer holds must be handled gracefully: the loader rejects the
// over-long accessor instead of reading past the buffer (which the sanitizers
// CI job would flag) and drops the unbuildable primitive. The buffer is 36
// bytes -- room for 3 VEC3<float> -- but the accessor claims count 1000.
TEST(GltfLoader, RejectsOutOfBoundsAccessor) {
  const std::string gltf = R"({
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": 36, "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
    "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1000, "type": "VEC3"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
    "nodes": [{"mesh": 0}],
    "scenes": [{"nodes": [0]}],
    "scene": 0
  })";
  const std::string path = std::string(testing::TempDir()) + "vg_oob.gltf";
  {
    std::ofstream(path) << gltf;
  }

  std::string err;
  std::optional<assets::Model> model = assets::load_gltf(path, &err);
  std::remove(path.c_str());

  // The JSON parses, so the load itself succeeds; the over-long primitive is
  // safely skipped, leaving no mesh -- and, crucially, no out-of-bounds read.
  ASSERT_TRUE(model.has_value()) << "load_gltf failed: " << err;
  EXPECT_TRUE(model->meshes.empty());
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

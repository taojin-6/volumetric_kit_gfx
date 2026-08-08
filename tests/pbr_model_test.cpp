// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/pbr_model.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <utility>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/assets/model.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/io/gltf_loader.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace assets = volumetric_kit::gfx::assets;
namespace io = volumetric_kit::gfx::io;
namespace pipelines = volumetric_kit::gfx::pipelines;

// Absolute path to the Box.glb fixture, baked in by CMake (see
// tests/CMakeLists.txt). Empty if the fixture was not available at configure
// time -- the glTF-backed test then skips rather than fails.
#ifdef VG_ASSETS_DIR
constexpr const char* kAssetsDir = VG_ASSETS_DIR;
#else
constexpr const char* kAssetsDir = "";
#endif

std::string box_path() { return std::string(kAssetsDir) + "/Box.glb"; }

// True when the Box.glb fixture is present and readable *now* (the file can be
// removed after configure time; see io_test.cpp).
bool have_fixture() {
  if (std::string(kAssetsDir).empty()) return false;
  std::ifstream f(box_path(), std::ios::binary);
  return f.good();
}

// One triangle under one root node -- the smallest drawable model. The mesh
// names no material, so a created PbrModel resolves it to the fallback.
assets::Model one_mesh_model() {
  assets::Mesh mesh;
  mesh.vertices.resize(3);
  mesh.vertices[0].position = {0.0f, 0.0f, 0.0f};
  mesh.vertices[1].position = {1.0f, 0.0f, 0.0f};
  mesh.vertices[2].position = {0.0f, 1.0f, 0.0f};
  mesh.indices = {0, 1, 2};

  assets::Model model;
  model.meshes.push_back(std::move(mesh));
  assets::Node node;
  node.mesh = 0;
  node.mesh_count = 1;
  model.scene.nodes.push_back(std::move(node));
  model.scene.roots.push_back(0);
  return model;
}

// A minimal triangle mesh naming material @p material (kNoMaterial for none).
assets::Mesh tri_mesh(std::uint32_t material = assets::Mesh::kNoMaterial) {
  assets::Mesh mesh;
  mesh.vertices.resize(3);
  mesh.vertices[0].position = {0.0f, 0.0f, 0.0f};
  mesh.vertices[1].position = {1.0f, 0.0f, 0.0f};
  mesh.vertices[2].position = {0.0f, 1.0f, 0.0f};
  mesh.indices = {0, 1, 2};
  mesh.material = material;
  return mesh;
}

// One triangle whose material references two images: a 3-channel base-color
// (an sRGB slot, exercising RGB->RGBA padding) and a 4-channel normal map (a
// linear slot). Drives the full texture path: to_rgba8, the sRGB-vs-UNORM
// per-slot policy, and tex_for resolving to a real uploaded image.
assets::Model textured_model() {
  assets::Model model;
  model.meshes.push_back(tri_mesh(/*material=*/0));

  assets::Image base;
  base.width = 2;
  base.height = 2;
  base.channels = 3;  // RGB -> to_rgba8 pads opaque alpha
  base.pixels.assign(2u * 2u * 3u, 0x80);
  model.images.push_back(std::move(base));

  assets::Image normal;
  normal.width = 2;
  normal.height = 2;
  normal.channels = 4;
  normal.pixels.assign(2u * 2u * 4u, 0x80);
  model.images.push_back(std::move(normal));

  assets::Material mat;
  mat.base_color_texture = 0;  // -> sRGB
  mat.normal_texture = 1;      // -> linear
  model.materials.push_back(mat);

  assets::Node node;
  node.mesh = 0;
  node.mesh_count = 1;
  model.scene.nodes.push_back(std::move(node));
  model.scene.roots.push_back(0);
  return model;
}

// An allocator + a PbrPipeline (for the reflected material layout), on top of
// the shared device fixture. Skips wholesale when no Vulkan device is present.
class PbrModelTest : public VulkanDeviceTest {
 protected:
  // Upload records copies + layout transitions + descriptor writes, so run
  // under the validation layer with teeth (on CI, where the layer is present).
  bool wants_validation() const override { return true; }

  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;  // no device, or the base SetUp failed fatally
    }
    auto allocator = vg::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());

    vg::RenderTargetLayout layout;
    layout.color_formats[0] = VK_FORMAT_R8G8B8A8_SRGB;
    layout.color_count = 1;
    layout.depth_format = VK_FORMAT_D32_SFLOAT;
    auto pipeline = pipelines::PbrPipeline::create(device(), layout);
    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    pipeline_.emplace(std::move(pipeline).value());
  }

  pipelines::PbrModel make_model(const assets::Model& model) {
    auto made =
        pipelines::PbrModel::create(*device_, *allocator_, *pipeline_, model);
    EXPECT_TRUE(made.ok()) << made.status().message();
    return made.ok() ? std::move(made).value() : pipelines::PbrModel{};
  }

  std::optional<vg::Allocator> allocator_;
  std::optional<pipelines::PbrPipeline> pipeline_;
};

}  // namespace

// pbr_material_desc is the one home of the factor mapping: every factor lands
// field-by-field, and the views/sampler stay null for the caller to resolve.
// Pure CPU, so it runs without a device.
TEST(PbrMaterialDescTest, MapsEveryFactor) {
  assets::Material m;
  m.base_color_factor = glm::vec4(0.1f, 0.2f, 0.3f, 0.4f);
  m.emissive_factor = glm::vec3(0.5f, 0.6f, 0.7f);
  m.metallic_factor = 0.25f;
  m.roughness_factor = 0.75f;
  m.normal_scale = 1.5f;
  m.occlusion_strength = 0.125f;

  const pipelines::PbrMaterialDesc d = pipelines::pbr_material_desc(m);
  EXPECT_EQ(d.base_color_factor, m.base_color_factor);
  EXPECT_EQ(d.emissive_factor, m.emissive_factor);
  EXPECT_EQ(d.metallic_factor, m.metallic_factor);
  EXPECT_EQ(d.roughness_factor, m.roughness_factor);
  EXPECT_EQ(d.normal_scale, m.normal_scale);
  EXPECT_EQ(d.occlusion_strength, m.occlusion_strength);
  EXPECT_EQ(d.base_color, VK_NULL_HANDLE);
  EXPECT_EQ(d.metallic_roughness, VK_NULL_HANDLE);
  EXPECT_EQ(d.normal, VK_NULL_HANDLE);
  EXPECT_EQ(d.occlusion, VK_NULL_HANDLE);
  EXPECT_EQ(d.emissive, VK_NULL_HANDLE);
  EXPECT_EQ(d.sampler, VK_NULL_HANDLE);
}

// The real thing end to end: the Khronos Box.glb (one mesh, one material,
// one node) through load_gltf -> PbrModel::create.
TEST_F(PbrModelTest, CreatesFromBoxGlb) {
  if (!have_fixture()) GTEST_SKIP() << "Box.glb fixture unavailable";
  std::string err;
  std::optional<assets::Model> model = io::load_gltf(box_path(), &err);
  ASSERT_TRUE(model.has_value()) << "load_gltf failed: " << err;

  pipelines::PbrModel gpu = make_model(*model);
  EXPECT_TRUE(gpu.valid());
  ASSERT_EQ(gpu.draws().size(), 1u);
  EXPECT_NE(gpu.draws()[0].mesh, nullptr);
  EXPECT_NE(gpu.draws()[0].material, nullptr);
  EXPECT_EQ(gpu.mesh_count(), static_cast<uint32_t>(model->meshes.size()));
  // One material per source material plus the shared fallback.
  EXPECT_EQ(gpu.material_count(),
            static_cast<uint32_t>(model->materials.size()) + 1u);
}

// A material with real texture maps drives the full upload path (to_rgba8, the
// sRGB-vs-UNORM per-slot policy, tex_for -> a real image) -- which Box.glb (no
// images) never exercises. Under validation-with-teeth, a wrong format/barrier
// for the uploaded maps fails the test.
TEST_F(PbrModelTest, UploadsTexturedMaterial) {
  pipelines::PbrModel gpu = make_model(textured_model());
  EXPECT_TRUE(gpu.valid());
  ASSERT_EQ(gpu.draws().size(), 1u);
  EXPECT_NE(gpu.draws()[0].mesh, nullptr);
  ASSERT_NE(gpu.draws()[0].material, nullptr);
  EXPECT_TRUE(gpu.draws()[0].material->valid());
  EXPECT_EQ(gpu.material_count(), 2u);  // the one source material + fallback
}

// The node tree composes parent * child down to world space: a mesh under a
// child node inherits its parent's transform. Guards the multiply order and
// that the parent transform is not dropped (Box.glb has a -90deg root rotation
// no other test checks).
TEST_F(PbrModelTest, ComposesNodeWorldTransforms) {
  glm::mat4 parent_xf(1.0f);
  parent_xf[3] = glm::vec4(10.0f, 0.0f, 0.0f, 1.0f);  // translate (10,0,0)
  glm::mat4 child_xf(1.0f);
  child_xf[3] = glm::vec4(0.0f, 20.0f, 0.0f, 1.0f);  // translate (0,20,0)

  assets::Model model;
  model.meshes.push_back(tri_mesh());
  assets::Node child;  // node 0: holds the mesh, its own transform
  child.mesh = 0;
  child.mesh_count = 1;
  child.transform = child_xf;
  model.scene.nodes.push_back(std::move(child));
  assets::Node parent;  // node 1: parent of node 0
  parent.transform = parent_xf;
  parent.children = {0};
  model.scene.nodes.push_back(std::move(parent));
  model.scene.roots.push_back(1);

  pipelines::PbrModel gpu = make_model(model);
  ASSERT_EQ(gpu.draws().size(), 1u);
  EXPECT_EQ(gpu.draws()[0].world, parent_xf * child_xf);
}

// Material index resolution: a real index binds that material; kNoMaterial and
// an out-of-range index both bind the shared fallback (and the out-of-range one
// must not index materials_ out of bounds).
TEST_F(PbrModelTest, MaterialIndexResolvesRealAndFallback) {
  assets::Model model;
  model.meshes.push_back(tri_mesh(/*material=*/0));             // real
  model.meshes.push_back(tri_mesh(assets::Mesh::kNoMaterial));  // none
  model.meshes.push_back(tri_mesh(/*material=*/5));  // out of range -> fallback
  model.materials.push_back(assets::Material{});     // one source material
  for (std::uint32_t i = 0; i < 3; ++i) {
    assets::Node node;
    node.mesh = i;
    node.mesh_count = 1;
    model.scene.nodes.push_back(std::move(node));
    model.scene.roots.push_back(i);
  }

  pipelines::PbrModel gpu = make_model(model);
  ASSERT_EQ(gpu.draws().size(), 3u);
  for (const pipelines::PbrDraw& d : gpu.draws()) {
    ASSERT_NE(d.material, nullptr);
    EXPECT_TRUE(d.material->valid());
  }
  // Draw 0 binds the real material; draws 1 (kNoMaterial) and 2 (out of range)
  // both bind the one shared fallback.
  EXPECT_NE(gpu.draws()[0].material, gpu.draws()[1].material);
  EXPECT_EQ(gpu.draws()[1].material, gpu.draws()[2].material);
}

// A malformed node graph -- a cycle plus out-of-range child/mesh indices --
// must terminate (each node visits once) and keep the acyclic draws.
TEST_F(PbrModelTest, NodeCycleTerminates) {
  assets::Model model = one_mesh_model();
  model.scene.nodes[0].children = {1, 99};  // 99: out of range, skipped
  assets::Node loop;
  loop.children = {0};  // cycle back to the root
  model.scene.nodes.push_back(std::move(loop));

  pipelines::PbrModel gpu = make_model(model);
  EXPECT_TRUE(gpu.valid());
  EXPECT_EQ(gpu.draws().size(), 1u);  // node 0's mesh, exactly once
}

// A mesh with no material draws with the generated fallback material.
TEST_F(PbrModelTest, FallbackMaterialForMaterialLessMesh) {
  pipelines::PbrModel gpu = make_model(one_mesh_model());
  EXPECT_TRUE(gpu.valid());
  ASSERT_EQ(gpu.draws().size(), 1u);
  EXPECT_NE(gpu.draws()[0].material, nullptr);
  EXPECT_TRUE(gpu.draws()[0].material->valid());
  EXPECT_GE(gpu.material_count(), 1u);  // at least the fallback
}

// A model with no meshes is a legal no-op: create succeeds, the model is
// valid (it owns the fallback material), and there is nothing to draw.
TEST_F(PbrModelTest, EmptyModelIsValidWithZeroDraws) {
  pipelines::PbrModel gpu = make_model(assets::Model{});
  EXPECT_TRUE(gpu.valid());
  EXPECT_TRUE(gpu.draws().empty());
  EXPECT_EQ(gpu.mesh_count(), 0u);
  EXPECT_EQ(gpu.material_count(), 1u);  // the fallback alone
}

TEST_F(PbrModelTest, RejectsInvalidPipeline) {
  const pipelines::PbrPipeline empty;  // default-constructed: valid() is false
  auto made = pipelines::PbrModel::create(*device_, *allocator_, empty,
                                          one_mesh_model());
  ASSERT_FALSE(made.ok());
  EXPECT_EQ(made.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(PbrModelTest, RejectsNullDevice) {
  // A moved-from Device is the only way to hold one with a null handle.
  vg::Device stolen = std::move(*device_);
  auto made = pipelines::PbrModel::create(*device_, *allocator_, *pipeline_,
                                          one_mesh_model());
  EXPECT_FALSE(made.ok());
  EXPECT_EQ(made.status().domain(), vg::Status::Code::InvalidArgument);
  *device_ = std::move(stolen);  // restore for TearDown
}

TEST_F(PbrModelTest, MoveLeavesSourceEmpty) {
  pipelines::PbrModel source = make_model(one_mesh_model());
  ASSERT_TRUE(source.valid());
  const pipelines::PbrDraw first = source.draws()[0];

  pipelines::PbrModel moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  ASSERT_EQ(moved.draws().size(), 1u);
  // The draws borrow the model's own vectors, whose elements keep their
  // addresses across a move -- the borrowed pointers must survive intact.
  EXPECT_EQ(moved.draws()[0].mesh, first.mesh);
  EXPECT_EQ(moved.draws()[0].material, first.material);
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(source.draws().empty());
  EXPECT_EQ(source.mesh_count(), 0u);
  EXPECT_EQ(source.material_count(), 0u);
}

TEST_F(PbrModelTest, MoveAssignOverLiveLeavesSourceEmpty) {
  pipelines::PbrModel dst = make_model(one_mesh_model());
  pipelines::PbrModel src = make_model(one_mesh_model());
  ASSERT_TRUE(dst.valid());
  ASSERT_TRUE(src.valid());

  dst = std::move(src);  // frees dst's meshes/textures/materials, adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_EQ(dst.draws().size(), 1u);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(src.draws().empty());
  EXPECT_EQ(src.mesh_count(), 0u);
  EXPECT_EQ(src.material_count(), 0u);
}

TEST_F(PbrModelTest, SelfMoveAssignIsSafe) {
  pipelines::PbrModel model = make_model(one_mesh_model());
  ASSERT_TRUE(model.valid());

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must keep the model intact.
  pipelines::PbrModel* alias = &model;
  model = std::move(*alias);
  EXPECT_TRUE(model.valid());
  EXPECT_EQ(model.draws().size(), 1u);
  EXPECT_EQ(model.material_count(), 1u);
}

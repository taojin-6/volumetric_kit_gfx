// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>

#include <glm/vec3.hpp>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/ibl.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_scene.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

// A PbrPipeline (for the reflected set-0 layout), an allocator, a sampler, and
// a 1x1 texture whose view stands in for the three IBL maps. Skips with the
// base fixture when no Vulkan device is present.
class PbrSceneTest : public VulkanDeviceTest {
 protected:
  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;  // no device, or the base SetUp failed fatally
    }
    auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());

    vg::RenderTargetLayout layout;
    layout.color_formats[0] = VK_FORMAT_R8G8B8A8_SRGB;
    layout.color_count = 1;
    layout.depth_format = VK_FORMAT_D32_SFLOAT;
    auto pipeline = pipelines::PbrPipeline::create(device(), layout);
    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    pipeline_.emplace(std::move(pipeline).value());

    auto sampler = vg::Sampler::create(device());
    ASSERT_TRUE(sampler.ok()) << sampler.status().message();
    sampler_.emplace(std::move(sampler).value());

    const uint8_t white[4] = {255, 255, 255, 255};
    vg::ImageUploadDesc d;
    d.extent = {1, 1};
    d.format = VK_FORMAT_R8G8B8A8_UNORM;
    d.pixels = white;
    d.size = sizeof(white);
    auto tex = vg::upload_texture(*device_, *allocator_, d);
    ASSERT_TRUE(tex.ok()) << tex.status().message();
    tex_.emplace(std::move(tex).value());
  }

  VkDescriptorSetLayout scene_layout() const {
    return pipeline_->descriptor_set_layout(0);
  }

  // A fully-populated desc: the same 1x1 view for all three IBL maps (a
  // create/move test never draws, so the view type does not matter).
  pipelines::PbrSceneDesc full_desc() const {
    pipelines::PbrSceneDesc d;
    d.irradiance = tex_->view();
    d.prefilter = tex_->view();
    d.brdf_lut = tex_->view();
    d.sampler = sampler_->handle();
    return d;
  }

  // Writes slot `slot`'s camera in a submit of its own, so a recorded update
  // runs (and validation sees it).
  void set_camera(const pipelines::PbrScene& scene, uint32_t slot,
                  const glm::vec3& eye) {
    const vg::Status status = device_->submit_single_time(
        [&](VkCommandBuffer cmd) { scene.set_camera(cmd, slot, eye, 4.0f); });
    EXPECT_TRUE(status.ok()) << status.message();
  }

  std::optional<vkc::Allocator> allocator_;
  std::optional<pipelines::PbrPipeline> pipeline_;
  std::optional<vg::Sampler> sampler_;
  std::optional<vkc::Image> tex_;
};

// The same fixture under validation-with-teeth: the base TearDown fails the
// test on any captured VUID, which is what gives the submit() guards below
// something to assert against.
class PbrSubmitTest : public PbrSceneTest {
 protected:
  bool wants_validation() const override { return true; }

  // A minimal two-triangle quad (4 default vertices, 6 indices).
  static volumetric_kit::gfx::assets::Mesh quad() {
    volumetric_kit::gfx::assets::Mesh mesh;
    mesh.vertices.resize(4);
    mesh.indices = {0, 1, 2, 0, 2, 3};
    return mesh;
  }

  // The set-1 counterpart of full_desc(): the same 1x1 view in every slot.
  pipelines::PbrMaterialDesc material_desc() const {
    pipelines::PbrMaterialDesc d;
    d.base_color = tex_->view();
    d.metallic_roughness = tex_->view();
    d.normal = tex_->view();
    d.occlusion = tex_->view();
    d.emissive = tex_->view();
    d.sampler = sampler_->handle();
    return d;
  }

  // A material whose factors went up through a finished batch of its own.
  vg::Result<pipelines::PbrMaterial> make_material(
      const pipelines::PbrMaterialDesc& desc) {
    VG_ASSIGN(vg::UploadBatch batch,
              vg::UploadBatch::begin(*device_, *allocator_));
    VG_ASSIGN(pipelines::PbrMaterial material,
              pipelines::PbrMaterial::create(
                  device(), batch, pipeline_->descriptor_set_layout(1), desc));
    VG_TRY(batch.finish());
    return material;
  }

  // Records `frame` through the fixture's pipeline into a throwaway primary
  // command buffer, outside any render pass -- enough for the set-0 bind under
  // test, and legal on its own when submit() correctly records nothing.
  void record_submit(const pipelines::PbrFrame& frame) {
    auto pool = vg::CommandPool::create(device(), device_->queue_family());
    ASSERT_TRUE(pool.ok()) << pool.status().message();
    auto cmd = pool.value().allocate_primary();
    ASSERT_TRUE(cmd.ok()) << cmd.status().message();
    ASSERT_TRUE(
        cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    pipeline_->submit(cmd.value().handle(), frame);
    ASSERT_TRUE(cmd.value().end().ok());
  }
};

}  // namespace

TEST_F(PbrSceneTest, CreatesSet0) {
  auto scene = pipelines::PbrScene::create(device(), *allocator_,
                                           scene_layout(), full_desc());
  ASSERT_TRUE(scene.ok()) << scene.status().message();
  EXPECT_TRUE(scene.value().valid());
  EXPECT_EQ(scene.value().frames_in_flight(), 1u);  // the default ring depth
  EXPECT_NE(scene.value().descriptor_set(), VK_NULL_HANDLE);
  set_camera(scene.value(), 0, glm::vec3(0.0f, 0.0f, 3.0f));
}

// One camera UBO + descriptor set per frame-in-flight slot. Distinct set
// handles are the observable proxy for the ring's anti-race property: each set
// comes from its own OwnedDescriptorSet -- hence its own UBO buffer -- so
// writing one slot cannot touch another's. (PbrScene exposes no UBO read-back
// to assert that directly; AGENTS.md prefers behavior tests over private-state
// backdoors.) Both slots accept a write; an out-of-range slot is a no-op.
TEST_F(PbrSceneTest, RingsUboPerFrameInFlight) {
  auto scene = pipelines::PbrScene::create(device(), *allocator_,
                                           scene_layout(), full_desc(),
                                           /*frames_in_flight=*/2);
  ASSERT_TRUE(scene.ok()) << scene.status().message();
  EXPECT_EQ(scene.value().frames_in_flight(), 2u);
  EXPECT_NE(scene.value().descriptor_set(0), VK_NULL_HANDLE);
  EXPECT_NE(scene.value().descriptor_set(1), VK_NULL_HANDLE);
  EXPECT_NE(scene.value().descriptor_set(0), scene.value().descriptor_set(1));
  EXPECT_EQ(scene.value().descriptor_set(2), VK_NULL_HANDLE);
  set_camera(scene.value(), 0, glm::vec3(1.0f, 0.0f, 0.0f));
  set_camera(scene.value(), 1, glm::vec3(0.0f, 1.0f, 0.0f));
  // Out-of-range slot: guarded no-op, not a write past the ring.
  set_camera(scene.value(), 2, glm::vec3(0.0f));
}

// PbrScene::create defaults frames_in_flight to 1 while FrameLoop::create and
// WindowedAppConfig default to 2, so the documented `pbr_frame.slot = f.slot`
// wiring hands submit() a slot the scene's UBO ring does not have.
// descriptor_set(1) is then VK_NULL_HANDLE, and binding that trips
// VUID-vkCmdBindDescriptorSets-pDescriptorSets-parameter -- so the frame must
// be dropped whole rather than half-bound.
TEST_F(PbrSubmitTest, DropsFrameWhoseSlotOutrunsTheSceneRing) {
  auto scene = pipelines::PbrScene::create(device(), *allocator_,
                                           scene_layout(), full_desc());
  ASSERT_TRUE(scene.ok()) << scene.status().message();
  ASSERT_EQ(scene.value().frames_in_flight(), 1u);
  ASSERT_EQ(scene.value().descriptor_set(1), VK_NULL_HANDLE);

  pipelines::PbrFrame frame;
  frame.scene = &scene.value();
  frame.slot = 1;  // one past the ring: the frame loop's second slot
  frame.extent = {64, 64};
  ASSERT_NO_FATAL_FAILURE(record_submit(frame));
}

// The same guard from the other side: no scene at all. model.frag reads set 0
// unconditionally, so recording draws against an unbound set is invalid usage;
// submit() must record nothing instead of skipping only the bind.
//
// The draw list is what makes this observable: with zero draws both the fixed
// and unfixed paths record only legal commands. One real draw separates them --
// recording it here, outside any render pass, is itself a VUID, so the captured
// error proves the draw was recorded rather than dropped.
TEST_F(PbrSubmitTest, DropsFrameWithNoScene) {
  auto mesh = pipelines::upload_mesh(*device_, *allocator_, quad());
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();
  auto material = make_material(material_desc());
  ASSERT_TRUE(material.ok()) << material.status().message();

  pipelines::PbrDraw draw;
  draw.mesh = &mesh.value();
  draw.material = &material.value();

  pipelines::PbrFrame frame;
  frame.scene = nullptr;
  frame.extent = {64, 64};
  frame.draws = &draw;
  frame.draw_count = 1;
  ASSERT_NO_FATAL_FAILURE(record_submit(frame));
}

TEST_F(PbrSceneTest, RejectsZeroFramesInFlight) {
  auto scene =
      pipelines::PbrScene::create(device(), *allocator_, scene_layout(),
                                  full_desc(), /*frames_in_flight=*/0);
  ASSERT_FALSE(scene.ok());
  EXPECT_EQ(scene.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(PbrSceneTest, RejectsNullMap) {
  pipelines::PbrSceneDesc d = full_desc();
  d.prefilter = VK_NULL_HANDLE;  // the shader samples every IBL map
  auto scene =
      pipelines::PbrScene::create(device(), *allocator_, scene_layout(), d);
  ASSERT_FALSE(scene.ok());
  EXPECT_EQ(scene.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(PbrSceneTest, RejectsNullLayout) {
  auto scene = pipelines::PbrScene::create(device(), *allocator_,
                                           VK_NULL_HANDLE, full_desc());
  ASSERT_FALSE(scene.ok());
  EXPECT_EQ(scene.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(PbrSceneTest, MoveLeavesSourceEmpty) {
  auto made = pipelines::PbrScene::create(device(), *allocator_, scene_layout(),
                                          full_desc());
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::PbrScene source = std::move(made).value();
  ASSERT_TRUE(source.valid());

  pipelines::PbrScene moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_NE(moved.descriptor_set(), VK_NULL_HANDLE);
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.descriptor_set(), VK_NULL_HANDLE);
}

TEST_F(PbrSceneTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto a = pipelines::PbrScene::create(device(), *allocator_, scene_layout(),
                                       full_desc());
  auto b = pipelines::PbrScene::create(device(), *allocator_, scene_layout(),
                                       full_desc());
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  pipelines::PbrScene dst = std::move(a).value();
  pipelines::PbrScene src = std::move(b).value();

  dst = std::move(src);  // frees dst's pool + UBO, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(src.descriptor_set(), VK_NULL_HANDLE);
}

TEST_F(PbrSceneTest, SelfMoveAssignIsSafe) {
  auto made = pipelines::PbrScene::create(device(), *allocator_, scene_layout(),
                                          full_desc());
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::PbrScene scene = std::move(made).value();

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must keep the scene intact.
  pipelines::PbrScene* alias = &scene;
  scene = std::move(*alias);
  EXPECT_TRUE(scene.valid());
  EXPECT_NE(scene.descriptor_set(), VK_NULL_HANDLE);
}

// submit() with no draws records only state (bind pipeline + viewport + the
// scene set), which is valid outside a render scope -- exercises the set-0 bind
// path end to end. The full draw path is covered below.
TEST_F(PbrSceneTest, SubmitBindsSceneWithoutDraws) {
  auto made = pipelines::PbrScene::create(device(), *allocator_, scene_layout(),
                                          full_desc());
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::PbrScene scene = std::move(made).value();
  set_camera(scene, 0, glm::vec3(0.0f, 0.0f, 3.0f));

  pipelines::PbrFrame frame;
  frame.extent = {16, 16};
  frame.scene = &scene;
  frame.slot = 0;
  frame.draws = nullptr;
  frame.draw_count = 0;

  const vg::Status status = device_->submit_single_time(
      [&](VkCommandBuffer cmd) { pipeline_->submit(cmd, frame); });
  EXPECT_TRUE(status.ok()) << status.message();
}

// The camera reaches the shader from either home of its uniform buffer:
// device-mapped memory written in place, or the device-only fallback written
// by a recorded update, which a device without device-mapped memory takes.
// One lit quad is drawn under a fixed view-projection, so only the camera
// uniform moves the shading (the view vector of the specular highlight): the
// same eye renders the same pixel from both placements, and another eye a
// different one -- which proves the write landed, not a stale or zeroed
// buffer. Under validation, so a recorded update inside the render pass, or
// a missing barrier the layer catches, fails the test.
TEST_F(PbrSubmitTest, CameraReachesTheShaderFromEitherPlacement) {
  vg::OffscreenTargetDesc target_desc;
  target_desc.extent = {16, 16};
  target_desc.color_format = VK_FORMAT_R8G8B8A8_SRGB;  // the fixture's layout
  target_desc.depth_format = VK_FORMAT_D32_SFLOAT;
  auto target = vg::OffscreenTarget::create(*allocator_, target_desc);
  ASSERT_TRUE(target.ok()) << target.status().message();

  pipelines::IblBakeDesc bake;
  bake.irradiance_size = 4;
  bake.prefilter_size = 8;
  bake.prefilter_mip_levels = 2;
  bake.prefilter_samples = 8;
  bake.brdf_lut_size = 8;
  bake.brdf_lut_samples = 16;
  auto ibl = pipelines::bake_ibl(
      *device_, *allocator_,
      [](const glm::vec3& d) { return glm::vec3(0.2f) + 0.3f * d; }, bake);
  ASSERT_TRUE(ibl.ok()) << ibl.status().message();

  // A quad facing +Z across the middle of clip space, drawn with an identity
  // view-projection.
  volumetric_kit::gfx::assets::Mesh mesh;
  const glm::vec3 corners[4] = {{-0.5f, -0.5f, 0.5f},
                                {0.5f, -0.5f, 0.5f},
                                {0.5f, 0.5f, 0.5f},
                                {-0.5f, 0.5f, 0.5f}};
  for (const glm::vec3& corner : corners) {
    volumetric_kit::gfx::assets::Vertex v;
    v.position = corner;
    mesh.vertices.push_back(v);
  }
  mesh.indices = {0, 1, 2, 0, 2, 3};
  auto gpu_mesh = pipelines::upload_mesh(*device_, *allocator_, mesh);
  ASSERT_TRUE(gpu_mesh.ok()) << gpu_mesh.status().message();
  pipelines::PbrMaterialDesc material_desc = this->material_desc();
  material_desc.metallic_factor = 0.0f;
  material_desc.roughness_factor = 0.3f;
  auto material = make_material(material_desc);
  ASSERT_TRUE(material.ok()) << material.status().message();
  pipelines::PbrDraw draw;
  draw.mesh = &gpu_mesh.value();
  draw.material = &material.value();

  // Renders one frame with the camera at `eye` and returns the center pixel.
  const auto render = [&](const pipelines::PbrScene& scene,
                          const glm::vec3& eye) -> uint32_t {
    pipelines::PbrFrame frame;
    frame.extent = target_desc.extent;
    frame.scene = &scene;
    frame.draws = &draw;
    frame.draw_count = 1;
    const vg::Status status =
        device_->submit_single_time([&](VkCommandBuffer cmd) {
          scene.set_camera(cmd, 0, eye, ibl.value().prefilter_max_lod);
          target.value().prepare(cmd);
          const vg::RenderTarget rt = target.value().target();
          rt.begin(cmd, vg::RenderTargetBeginInfo{});
          pipeline_->submit(cmd, frame);
          rt.end(cmd);
          target.value().record_readback(cmd);
        });
    EXPECT_TRUE(status.ok()) << status.message();
    const auto* pixels = static_cast<const uint8_t*>(target.value().pixels());
    uint32_t center = 0;
    std::memcpy(&center, pixels + (8 * 16 + 8) * 4, sizeof(center));
    return center;
  };

  pipelines::PbrSceneDesc scene_desc = ibl.value().scene_desc();
  auto mapped = pipelines::PbrScene::create(device(), *allocator_,
                                            scene_layout(), scene_desc);
  ASSERT_TRUE(mapped.ok()) << mapped.status().message();
  scene_desc.camera_memory = pipelines::FrameUniformMemory::DeviceOnly;
  auto device_only = pipelines::PbrScene::create(device(), *allocator_,
                                                 scene_layout(), scene_desc);
  ASSERT_TRUE(device_only.ok()) << device_only.status().message();

  const glm::vec3 front(0.0f, 0.0f, 3.0f);
  const glm::vec3 grazing(3.0f, 0.0f, 0.6f);
  const uint32_t front_mapped = render(mapped.value(), front);
  const uint32_t front_device_only = render(device_only.value(), front);
  const uint32_t grazing_device_only = render(device_only.value(), grazing);
  EXPECT_EQ(front_mapped, front_device_only);
  EXPECT_NE(front_device_only, grazing_device_only);
  // The quad was drawn: not the cleared (zero) color.
  EXPECT_NE(front_mapped, 0u);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_pipeline.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

// A PbrPipeline (for the reflected set-1 layout), a sampler, and a 1x1 texture
// whose view stands in for all five maps.
class PbrMaterialTest : public vg_test::RendererDeviceTest {
 protected:
  void SetUp() override {
    RendererDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;  // no device, or the base SetUp failed fatally
    }

    vg::RenderTargetLayout layout;
    layout.color_formats[0] = VK_FORMAT_R8G8B8A8_SRGB;
    layout.color_count = 1;
    layout.depth_format = VK_FORMAT_D32_SFLOAT;
    auto pipeline = pipelines::PbrPipeline::create(device().handle(), layout);
    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    pipeline_.emplace(std::move(pipeline).value());

    auto sampler = vg::Sampler::create(device().handle());
    ASSERT_TRUE(sampler.ok()) << sampler.status().message();
    sampler_.emplace(std::move(sampler).value());

    const uint8_t white[4] = {255, 255, 255, 255};
    vg::ImageUploadDesc d;
    d.extent = {1, 1};
    d.format = VK_FORMAT_R8G8B8A8_UNORM;
    d.pixels = white;
    d.size = sizeof(white);
    auto tex = vg::upload_texture(device(), allocator(), d);
    ASSERT_TRUE(tex.ok()) << tex.status().message();
    tex_.emplace(std::move(tex).value());
  }

  VkDescriptorSetLayout material_layout() const {
    return pipeline_->descriptor_set_layout(1);
  }

  // A fully-populated desc: the same 1x1 view in every slot (a create/move test
  // never draws, so the slot's color space does not matter).
  pipelines::PbrMaterialDesc full_desc() const {
    pipelines::PbrMaterialDesc d;
    d.base_color = tex_->view();
    d.metallic_roughness = tex_->view();
    d.normal = tex_->view();
    d.occlusion = tex_->view();
    d.emissive = tex_->view();
    d.sampler = sampler_->handle();
    return d;
  }

  // Builds a material on a batch of its own and finishes it, so the factor
  // upload runs (under the base fixture's validation capture).
  vkc::Result<pipelines::PbrMaterial> make(
      VkDescriptorSetLayout layout, const pipelines::PbrMaterialDesc& d) {
    VKC_ASSIGN(vg::UploadBatch batch,
               vg::UploadBatch::begin(device(), allocator()));
    VKC_ASSIGN(
        pipelines::PbrMaterial material,
        pipelines::PbrMaterial::create(device().handle(), batch, layout, d));
    VKC_TRY(batch.finish());
    return material;
  }

  std::optional<pipelines::PbrPipeline> pipeline_;
  std::optional<vg::Sampler> sampler_;
  std::optional<vkc::Image> tex_;
};

}  // namespace

TEST_F(PbrMaterialTest, CreatesSet1) {
  auto mat = make(material_layout(), full_desc());
  ASSERT_TRUE(mat.ok()) << mat.status().message();
  EXPECT_TRUE(mat.value().valid());
  EXPECT_NE(mat.value().descriptor_set(), VK_NULL_HANDLE);
}

TEST_F(PbrMaterialTest, RejectsNullMap) {
  pipelines::PbrMaterialDesc d = full_desc();
  d.normal = VK_NULL_HANDLE;  // the shader samples every slot
  auto mat = make(material_layout(), d);
  ASSERT_FALSE(mat.ok());
  EXPECT_EQ(mat.status().domain(), vkc::Status::Code::InvalidArgument);
}

// Validation precedes the factor upload, so a refused material queues nothing
// and the batch it was given still finishes.
TEST_F(PbrMaterialTest, RefusedMaterialLeavesTheBatchUsable) {
  auto batch = vg::UploadBatch::begin(device(), allocator());
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  pipelines::PbrMaterialDesc d = full_desc();
  d.sampler = VK_NULL_HANDLE;
  auto mat = pipelines::PbrMaterial::create(device().handle(), batch.value(),
                                            material_layout(), d);
  ASSERT_FALSE(mat.ok());
  EXPECT_EQ(mat.status().domain(), vkc::Status::Code::InvalidArgument);
  const vkc::Status finished = batch.value().finish();
  EXPECT_TRUE(finished.ok()) << finished.message();
}

// create_all builds every material on one shared factor upload: each gets a
// set of its own, and the batch submits them all. That each draw reads its own
// factors is PbrSubmitTest.PackedMaterialsEachReadTheirOwnFactors.
TEST_F(PbrMaterialTest, CreatesManyMaterialsOnOneUpload) {
  auto batch = vg::UploadBatch::begin(device(), allocator());
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  std::vector<pipelines::PbrMaterialDesc> descs(3, full_desc());
  descs[1].roughness_factor = 0.25f;
  descs[2].metallic_factor = 0.0f;
  auto materials = pipelines::PbrMaterial::create_all(
      device().handle(), batch.value(), material_layout(), descs);
  ASSERT_TRUE(materials.ok()) << materials.status().message();
  const vkc::Status finished = batch.value().finish();
  ASSERT_TRUE(finished.ok()) << finished.message();

  ASSERT_EQ(materials.value().size(), descs.size());
  for (const pipelines::PbrMaterial& material : materials.value()) {
    EXPECT_TRUE(material.valid());
    EXPECT_NE(material.descriptor_set(), VK_NULL_HANDLE);
  }
  EXPECT_NE(materials.value()[0].descriptor_set(),
            materials.value()[1].descriptor_set());
  EXPECT_NE(materials.value()[1].descriptor_set(),
            materials.value()[2].descriptor_set());

  // The materials share the factor buffer: dropping one leaves the rest
  // whole.
  materials.value().erase(materials.value().begin());
  EXPECT_TRUE(materials.value().front().valid());
}

// A refused create_all -- no materials, or one bad desc among good ones --
// queues nothing, so the batch it was given still finishes.
TEST_F(PbrMaterialTest, RefusedCreateAllLeavesTheBatchUsable) {
  auto batch = vg::UploadBatch::begin(device(), allocator());
  ASSERT_TRUE(batch.ok()) << batch.status().message();

  auto none = pipelines::PbrMaterial::create_all(
      device().handle(), batch.value(), material_layout(), {});
  ASSERT_FALSE(none.ok());
  EXPECT_EQ(none.status().domain(), vkc::Status::Code::InvalidArgument);

  std::vector<pipelines::PbrMaterialDesc> descs(3, full_desc());
  descs[2].emissive = VK_NULL_HANDLE;
  auto mixed = pipelines::PbrMaterial::create_all(
      device().handle(), batch.value(), material_layout(), descs);
  ASSERT_FALSE(mixed.ok());
  EXPECT_EQ(mixed.status().domain(), vkc::Status::Code::InvalidArgument);

  const vkc::Status finished = batch.value().finish();
  EXPECT_TRUE(finished.ok()) << finished.message();
}

TEST_F(PbrMaterialTest, RejectsNullLayout) {
  auto mat = make(VK_NULL_HANDLE, full_desc());
  ASSERT_FALSE(mat.ok());
  EXPECT_EQ(mat.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(PbrMaterialTest, MoveLeavesSourceEmpty) {
  auto made = make(material_layout(), full_desc());
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::PbrMaterial source = std::move(made).value();
  ASSERT_TRUE(source.valid());

  pipelines::PbrMaterial moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_NE(moved.descriptor_set(), VK_NULL_HANDLE);
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.descriptor_set(), VK_NULL_HANDLE);
}

TEST_F(PbrMaterialTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto a = make(material_layout(), full_desc());
  auto b = make(material_layout(), full_desc());
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  pipelines::PbrMaterial dst = std::move(a).value();
  pipelines::PbrMaterial src = std::move(b).value();

  dst = std::move(src);  // frees dst's pool + UBO, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(src.descriptor_set(), VK_NULL_HANDLE);
}

TEST_F(PbrMaterialTest, SelfMoveAssignIsSafe) {
  auto made = make(material_layout(), full_desc());
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::PbrMaterial mat = std::move(made).value();

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must keep the material intact.
  pipelines::PbrMaterial* alias = &mat;
  mat = std::move(*alias);
  EXPECT_TRUE(mat.valid());
  EXPECT_NE(mat.descriptor_set(), VK_NULL_HANDLE);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_pipeline.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

// A PbrPipeline (for the reflected set-1 layout), an allocator, a sampler, and
// a 1x1 texture whose view stands in for all five maps. Skips with the base
// fixture when no Vulkan device is present.
class PbrMaterialTest : public VulkanDeviceTest {
 protected:
  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (IsSkipped()) {
      return;
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

  std::optional<vg::Allocator> allocator_;
  std::optional<pipelines::PbrPipeline> pipeline_;
  std::optional<vg::Sampler> sampler_;
  std::optional<vg::Texture> tex_;
};

}  // namespace

TEST_F(PbrMaterialTest, CreatesSet1) {
  auto mat = pipelines::PbrMaterial::create(device(), *allocator_,
                                            material_layout(), full_desc());
  ASSERT_TRUE(mat.ok()) << mat.status().message();
  EXPECT_TRUE(mat.value().valid());
  EXPECT_NE(mat.value().descriptor_set(), VK_NULL_HANDLE);
}

TEST_F(PbrMaterialTest, RejectsNullMap) {
  pipelines::PbrMaterialDesc d = full_desc();
  d.normal = VK_NULL_HANDLE;  // the shader samples every slot
  auto mat = pipelines::PbrMaterial::create(device(), *allocator_,
                                            material_layout(), d);
  ASSERT_FALSE(mat.ok());
  EXPECT_EQ(mat.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(PbrMaterialTest, RejectsNullLayout) {
  auto mat = pipelines::PbrMaterial::create(device(), *allocator_,
                                            VK_NULL_HANDLE, full_desc());
  ASSERT_FALSE(mat.ok());
  EXPECT_EQ(mat.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(PbrMaterialTest, MoveLeavesSourceEmpty) {
  auto made = pipelines::PbrMaterial::create(device(), *allocator_,
                                             material_layout(), full_desc());
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
  auto a = pipelines::PbrMaterial::create(device(), *allocator_,
                                          material_layout(), full_desc());
  auto b = pipelines::PbrMaterial::create(device(), *allocator_,
                                          material_layout(), full_desc());
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
  auto made = pipelines::PbrMaterial::create(device(), *allocator_,
                                             material_layout(), full_desc());
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::PbrMaterial mat = std::move(made).value();

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must keep the material intact.
  pipelines::PbrMaterial* alias = &mat;
  mat = std::move(*alias);
  EXPECT_TRUE(mat.valid());
  EXPECT_NE(mat.descriptor_set(), VK_NULL_HANDLE);
}

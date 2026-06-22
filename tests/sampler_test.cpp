// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <limits>
#include <utility>

#include "volumetric_kit/gfx/core/sampler.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

using SamplerTest = VulkanDeviceTest;

}  // namespace

TEST_F(SamplerTest, CreatesWithDefaults) {
  auto sampler = vg::Sampler::create(device());
  ASSERT_TRUE(sampler.ok()) << sampler.status().message();
  EXPECT_TRUE(sampler.value().valid());
  EXPECT_NE(sampler.value().handle(), VK_NULL_HANDLE);
}

TEST_F(SamplerTest, CreatesWithCustomDesc) {
  vg::SamplerDesc desc;
  desc.mag_filter = VK_FILTER_NEAREST;
  desc.min_filter = VK_FILTER_NEAREST;
  desc.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  desc.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  desc.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  desc.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  desc.max_lod = 4.0f;

  auto sampler = vg::Sampler::create(device(), desc);
  ASSERT_TRUE(sampler.ok()) << sampler.status().message();
  EXPECT_TRUE(sampler.value().valid());
}

TEST_F(SamplerTest, CreateRejectsNullDevice) {
  EXPECT_EQ(vg::Sampler::create(VK_NULL_HANDLE).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(SamplerTest, CreateRejectsMaxLodBelowMinLod) {
  vg::SamplerDesc desc;
  desc.min_lod = 4.0f;
  desc.max_lod = 1.0f;  // empty LOD range — rejected up front
  EXPECT_EQ(vg::Sampler::create(device(), desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(SamplerTest, CreateRejectsNanLod) {
  vg::SamplerDesc desc;
  // NaN compares false against everything, so it would slip past a bare
  // max_lod < min_lod test; the explicit isnan check must catch it.
  desc.min_lod = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(vg::Sampler::create(device(), desc).status().domain(),
            vg::Status::Code::InvalidArgument);
}

TEST_F(SamplerTest, MoveLeavesSourceEmpty) {
  auto created = vg::Sampler::create(device());
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::Sampler source = std::move(created).value();
  ASSERT_NE(source.handle(), VK_NULL_HANDLE);

  vg::Sampler moved(std::move(source));
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.handle(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(source.valid());
}

TEST_F(SamplerTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto a = vg::Sampler::create(device());
  auto b = vg::Sampler::create(device());
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Sampler dst = std::move(a).value();
  vg::Sampler src = std::move(b).value();

  dst = std::move(src);  // frees dst's original sampler, then adopts src's
  EXPECT_NE(dst.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(src.handle(), VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(SamplerTest, SelfMoveAssignIsSafe) {
  auto created = vg::Sampler::create(device());
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::Sampler sampler = std::move(created).value();

  // Launder through a pointer to dodge -Wself-move under -Werror; the
  // this != &other guard must keep the sampler intact.
  vg::Sampler* alias = &sampler;
  sampler = std::move(*alias);
  EXPECT_NE(sampler.handle(), VK_NULL_HANDLE);
}

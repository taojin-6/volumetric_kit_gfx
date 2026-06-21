// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

#include "spirv_test_util.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "vulkan_test_fixture.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

// Creating a real module needs a device, so these tests use the shared
// VulkanDeviceTest fixture (skips when the runner has none). Modules are local
// to each test, so they tear down before the fixture's device. The triangle
// vertex module is loaded via the shared vg_test::load_module helper.
using ShaderTest = VulkanDeviceTest;

}  // namespace

// --- Validation rejects: no device needed (checked before the Vulkan call) ---

TEST(ShaderModuleTest, NullCodeRejected) {
  auto module = vg::ShaderModule::create(VK_NULL_HANDLE, nullptr, 4);
  ASSERT_FALSE(module.ok());
  EXPECT_EQ(module.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST(ShaderModuleTest, ZeroSizeRejected) {
  const uint32_t word = 0;
  auto module = vg::ShaderModule::create(VK_NULL_HANDLE, &word, 0);
  ASSERT_FALSE(module.ok());
  EXPECT_EQ(module.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST(ShaderModuleTest, MisalignedSizeRejected) {
  const uint32_t word = 0;
  // 6 bytes is not a whole number of SPIR-V words -- reject before Vulkan.
  auto module = vg::ShaderModule::create(VK_NULL_HANDLE, &word, 6);
  ASSERT_FALSE(module.ok());
  EXPECT_EQ(module.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST(ShaderModuleTest, DefaultConstructedIsEmpty) {
  vg::ShaderModule module;
  EXPECT_FALSE(module.valid());
  EXPECT_EQ(module.handle(), VK_NULL_HANDLE);
}

// --- Real module creation + move semantics: needs a device ------------------

TEST_F(ShaderTest, LoadsTriangleVertexShader) {
  vg::ShaderModule module = vg_test::load_module(device(), "triangle.vert.spv");
  EXPECT_TRUE(module.valid());
  EXPECT_NE(module.handle(), VK_NULL_HANDLE);
}

// --- SPIR-V reflection: the descriptor interface recovered at create() -------

TEST_F(ShaderTest, ReflectsDescriptorBindingsAndPushConstants) {
  vg::ShaderModule module =
      vg_test::load_module(device(), "reflect_probe.frag.spv");
  ASSERT_TRUE(module.valid());

  // The probe is a fragment shader (stage recovered from the execution model).
  EXPECT_EQ(module.stage(), VK_SHADER_STAGE_FRAGMENT_BIT);

  // It declares exactly two descriptor-bound resources: a uniform buffer at
  // set 0 / binding 0 and a combined image sampler at set 0 / binding 1.
  const std::vector<vg::ReflectedResource>& res = module.resources();
  ASSERT_EQ(res.size(), 2u);

  const vg::ReflectedResource* ubo = nullptr;
  const vg::ReflectedResource* sampler = nullptr;
  for (const vg::ReflectedResource& r : res) {
    if (r.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
      ubo = &r;
    } else if (r.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      sampler = &r;
    }
  }

  ASSERT_NE(ubo, nullptr);
  EXPECT_EQ(ubo->set, 0u);
  EXPECT_EQ(ubo->binding, 0u);
  EXPECT_EQ(ubo->count, 1u);
  EXPECT_NE(ubo->stages & VK_SHADER_STAGE_FRAGMENT_BIT, 0u);

  ASSERT_NE(sampler, nullptr);
  EXPECT_EQ(sampler->set, 0u);
  EXPECT_EQ(sampler->binding, 1u);

  // The push-constant block is a single vec2 -- 8 bytes.
  EXPECT_EQ(module.push_constant_size(), 8u);
}

TEST_F(ShaderTest, ResourcelessShaderReflectsEmpty) {
  // The hello-triangle vertex shader binds no descriptors and declares no push
  // constants, so reflection reports an empty interface -- with the stage still
  // recovered from the SPIR-V.
  vg::ShaderModule module = vg_test::load_module(device(), "triangle.vert.spv");
  ASSERT_TRUE(module.valid());
  EXPECT_EQ(module.stage(), VK_SHADER_STAGE_VERTEX_BIT);
  EXPECT_TRUE(module.resources().empty());
  EXPECT_EQ(module.push_constant_size(), 0u);
}

TEST_F(ShaderTest, MoveLeavesSourceEmpty) {
  vg::ShaderModule source = vg_test::load_module(device(), "triangle.vert.spv");
  ASSERT_TRUE(source.valid());

  vg::ShaderModule moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.handle(), VK_NULL_HANDLE);
}

TEST_F(ShaderTest, MoveAssignOverLiveLeavesSourceEmpty) {
  vg::ShaderModule dst = vg_test::load_module(device(), "triangle.vert.spv");
  vg::ShaderModule src = vg_test::load_module(device(), "triangle.vert.spv");

  dst = std::move(src);  // runs dst's deleter once, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(ShaderTest, SelfMoveAssignIsSafe) {
  vg::ShaderModule module = vg_test::load_module(device(), "triangle.vert.spv");

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the
  // this != &other guard must keep the module intact and not run its deleter.
  vg::ShaderModule* alias = &module;
  module = std::move(*alias);
  EXPECT_TRUE(module.valid());
}

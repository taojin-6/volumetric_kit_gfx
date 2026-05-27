// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/shader.hpp"
#include "vulkan_test_fixture.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

// Reads a .spv file into 32-bit words -- SPIR-V's natural unit and the
// alignment vkCreateShaderModule requires. Returns empty on any read failure.
// The read is capped to the word-aligned buffer size, so a stray non-SPIR-V
// file can't overrun it.
std::vector<uint32_t> load_spirv(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return {};
  }
  const std::streampos end = file.tellg();
  if (end < 0) {  // tellg() failed; -1 would wrap to a huge allocation below.
    return {};
  }
  const auto size = static_cast<size_t>(end);
  file.seekg(0);
  std::vector<uint32_t> words(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(words.data()),
            static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
  if (!file) {
    return {};
  }
  return words;
}

std::string triangle_vert_spv() {
  return std::string(VG_SHADER_DIR) + "/triangle.vert.spv";
}

// Creating a real module needs a device, so these tests use the shared
// VulkanDeviceTest fixture (skips when the runner has none). Modules are local
// to each test, so they tear down before the fixture's device.
class ShaderTest : public VulkanDeviceTest {
 protected:
  // Loads + creates the triangle vertex module; aborts via value() only if the
  // compiled shader is genuinely missing or rejected (the build depends on it).
  vg::ShaderModule load_triangle_vert() {
    std::vector<uint32_t> code = load_spirv(triangle_vert_spv());
    EXPECT_FALSE(code.empty()) << "missing/empty " << triangle_vert_spv();
    auto module = vg::ShaderModule::create(device(), code.data(),
                                           code.size() * sizeof(uint32_t));
    EXPECT_TRUE(module.ok()) << module.status().message();
    return std::move(module).value();
  }
};

}  // namespace

// --- Validation rejects: no device needed (checked before the Vulkan call) ---

TEST(ShaderModuleTest, NullCodeRejected) {
  auto module = vg::ShaderModule::create(VK_NULL_HANDLE, nullptr, 4);
  ASSERT_FALSE(module.ok());
  EXPECT_EQ(module.status().code(), VK_ERROR_INITIALIZATION_FAILED);
}

TEST(ShaderModuleTest, ZeroSizeRejected) {
  const uint32_t word = 0;
  auto module = vg::ShaderModule::create(VK_NULL_HANDLE, &word, 0);
  ASSERT_FALSE(module.ok());
  EXPECT_EQ(module.status().code(), VK_ERROR_INITIALIZATION_FAILED);
}

TEST(ShaderModuleTest, MisalignedSizeRejected) {
  const uint32_t word = 0;
  // 6 bytes is not a whole number of SPIR-V words -- reject before Vulkan.
  auto module = vg::ShaderModule::create(VK_NULL_HANDLE, &word, 6);
  ASSERT_FALSE(module.ok());
  EXPECT_EQ(module.status().code(), VK_ERROR_INITIALIZATION_FAILED);
}

TEST(ShaderModuleTest, DefaultConstructedIsEmpty) {
  vg::ShaderModule module;
  EXPECT_FALSE(module.valid());
  EXPECT_EQ(module.handle(), VK_NULL_HANDLE);
}

// --- Real module creation + move semantics: needs a device ------------------

TEST_F(ShaderTest, LoadsTriangleVertexShader) {
  vg::ShaderModule module = load_triangle_vert();
  EXPECT_TRUE(module.valid());
  EXPECT_NE(module.handle(), VK_NULL_HANDLE);
}

TEST_F(ShaderTest, MoveLeavesSourceEmpty) {
  vg::ShaderModule source = load_triangle_vert();
  ASSERT_TRUE(source.valid());

  vg::ShaderModule moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.handle(), VK_NULL_HANDLE);
}

TEST_F(ShaderTest, MoveAssignOverLiveLeavesSourceEmpty) {
  vg::ShaderModule dst = load_triangle_vert();
  vg::ShaderModule src = load_triangle_vert();

  dst = std::move(src);  // runs dst's deleter once, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(ShaderTest, SelfMoveAssignIsSafe) {
  vg::ShaderModule module = load_triangle_vert();

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the
  // this != &other guard must keep the module intact and not run its deleter.
  vg::ShaderModule* alias = &module;
  module = std::move(*alias);
  EXPECT_TRUE(module.valid());
}

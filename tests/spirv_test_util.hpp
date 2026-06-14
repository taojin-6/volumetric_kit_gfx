// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file spirv_test_util.hpp
/// Loads compiled SPIR-V blobs for the GPU-touching tests. The shaders are
/// compiled next to the test binary by vg_compile_shaders(); VG_SHADER_DIR (a
/// per-target compile definition) names the directory they land in.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/shader.hpp"

// VG_SHADER_DIR is a per-target compile definition (see tests/CMakeLists.txt:
// vg_compile_shaders + target_compile_definitions). Fail loudly with an
// actionable message if a target includes this header without defining it,
// rather than emitting a confusing "undeclared identifier" inside spirv_path().
#ifndef VG_SHADER_DIR
#error \
    "VG_SHADER_DIR must be defined to use spirv_test_util.hpp (add the test to vg_core_test or replicate its vg_compile_shaders + VG_SHADER_DIR wiring)"
#endif

namespace vg_test {

// Reads a .spv file into 32-bit words -- SPIR-V's natural unit and the
// alignment vkCreateShaderModule requires. Returns empty on any read failure.
// The read is capped to the word-aligned buffer size, so a stray non-SPIR-V
// file can't overrun it.
inline std::vector<uint32_t> load_spirv(const std::string& path) {
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

// Absolute path to a shader compiled next to the test binary, e.g.
// spirv_path("triangle.vert.spv").
inline std::string spirv_path(const char* name) {
  return std::string(VG_SHADER_DIR) + "/" + name;
}

// Loads a compiled SPIR-V shader next to the test binary and creates a module
// from it on `device`. Fails the current test (via EXPECT) on a missing blob or
// a rejected module, naming the full searched path so a misconfigured
// VG_SHADER_DIR is obvious from the failure message.
inline volumetric_kit::gfx::ShaderModule load_module(VkDevice device,
                                                     const char* name) {
  const std::string path = spirv_path(name);
  std::vector<uint32_t> code = load_spirv(path);
  EXPECT_FALSE(code.empty()) << "missing/empty " << path;
  auto module = volumetric_kit::gfx::ShaderModule::create(
      device, code.data(), code.size() * sizeof(uint32_t));
  EXPECT_TRUE(module.ok()) << module.status().message();
  return std::move(module).value();
}

}  // namespace vg_test

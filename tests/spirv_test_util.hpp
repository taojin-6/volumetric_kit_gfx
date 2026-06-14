// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file spirv_test_util.hpp
/// Loads compiled SPIR-V blobs for the GPU-touching tests. The shaders are
/// compiled next to the test binary by vg_compile_shaders(); VG_SHADER_DIR (a
/// per-target compile definition) names the directory they land in.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

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

}  // namespace vg_test

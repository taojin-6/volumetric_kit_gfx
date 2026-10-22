// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file box_glb.hpp
/// A box shaped as the Khronos glTF sample "Box" is, written as a .glb by the
/// test that loads it: 24 vertices (four a face, each face its own normal), 36
/// indices, one red material, and a root node turning -90 degrees about X over
/// the node that draws the mesh.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace vg_test {

/// Writes the box to `path`; false if the file could not be written.
inline bool write_box_glb(const std::string& path) {
  // The binary chunk: 24 positions, 24 normals (VEC3 floats), 36 indices
  // (unsigned shorts).
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<std::uint16_t> indices;
  const float corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  for (int axis = 0; axis < 3; ++axis) {
    for (const float sign : {-1.0f, 1.0f}) {
      const auto first = static_cast<std::uint16_t>(positions.size() / 3);
      for (const auto& corner : corners) {
        float position[3] = {};
        position[axis] = 0.5f * sign;
        position[(axis + 1) % 3] = 0.5f * corner[0];
        position[(axis + 2) % 3] = 0.5f * corner[1];
        float normal[3] = {};
        normal[axis] = sign;
        positions.insert(positions.end(), position, position + 3);
        normals.insert(normals.end(), normal, normal + 3);
      }
      for (const int corner : {0, 1, 2, 0, 2, 3}) {
        indices.push_back(static_cast<std::uint16_t>(first + corner));
      }
    }
  }
  const std::size_t vec3_bytes = positions.size() * sizeof(float);  // 288
  const std::size_t index_bytes = indices.size() * sizeof(std::uint16_t);
  std::vector<char> bin(2 * vec3_bytes + index_bytes);  // 648, 4-aligned
  std::memcpy(bin.data(), positions.data(), vec3_bytes);
  std::memcpy(bin.data() + vec3_bytes, normals.data(), vec3_bytes);
  std::memcpy(bin.data() + 2 * vec3_bytes, indices.data(), index_bytes);

  std::string json =
      R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],)"
      R"("nodes":[{"children":[1],)"
      R"("matrix":[1,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,1]},{"mesh":0}],)"
      R"("meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1},)"
      R"("indices":2,"material":0}]}],)"
      R"("materials":[{"pbrMetallicRoughness":)"
      R"({"baseColorFactor":[0.8,0,0,1]}}],)"
      R"("buffers":[{"byteLength":648}],)"
      R"("bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":288},)"
      R"({"buffer":0,"byteOffset":288,"byteLength":288},)"
      R"({"buffer":0,"byteOffset":576,"byteLength":72}],)"
      R"("accessors":[{"bufferView":0,"componentType":5126,"count":24,)"
      R"("type":"VEC3","min":[-0.5,-0.5,-0.5],"max":[0.5,0.5,0.5]},)"
      R"({"bufferView":1,"componentType":5126,"count":24,"type":"VEC3"},)"
      R"({"bufferView":2,"componentType":5123,"count":36,"type":"SCALAR"}]})";
  json.resize((json.size() + 3) / 4 * 4, ' ');

  // A 12-byte header, then the JSON and the binary chunk, each with its
  // length and type; every word little-endian.
  std::ofstream out(path, std::ios::binary);
  const auto word = [&out](std::uint32_t value) {
    const char bytes[4] = {static_cast<char>(value & 0xFF),
                           static_cast<char>((value >> 8) & 0xFF),
                           static_cast<char>((value >> 16) & 0xFF),
                           static_cast<char>((value >> 24) & 0xFF)};
    out.write(bytes, 4);
  };
  word(0x46546C67);  // "glTF"
  word(2);
  word(static_cast<std::uint32_t>(12 + 8 + json.size() + 8 + bin.size()));
  word(static_cast<std::uint32_t>(json.size()));
  word(0x4E4F534A);  // "JSON"
  out.write(json.data(), static_cast<std::streamsize>(json.size()));
  word(static_cast<std::uint32_t>(bin.size()));
  word(0x004E4942);  // "BIN"
  out.write(bin.data(), static_cast<std::streamsize>(bin.size()));
  return static_cast<bool>(out);
}

}  // namespace vg_test

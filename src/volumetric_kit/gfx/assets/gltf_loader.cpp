// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/assets/gltf_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// tinygltf is compiled once in tinygltf_impl.cpp; here it is a
// declarations-only header. Keep stb out of this TU's macro state to match that
// split.
#include <tiny_gltf.h>

namespace volumetric_kit::gfx::assets {
namespace {

// Lower-case file extension including the dot (e.g. ".glb"); empty if none.
std::string extension_of(std::string_view path) {
  const std::size_t dot = path.rfind('.');
  if (dot == std::string_view::npos) return {};
  std::string ext(path.substr(dot));
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return ext;
}

// Number of float components in a glTF accessor element (VEC3 -> 3, etc.).
int component_count(int type) {
  switch (type) {
    case TINYGLTF_TYPE_SCALAR:
      return 1;
    case TINYGLTF_TYPE_VEC2:
      return 2;
    case TINYGLTF_TYPE_VEC3:
      return 3;
    case TINYGLTF_TYPE_VEC4:
      return 4;
    default:
      return 0;
  }
}

// Read accessor `index` as float vectors into `out` (count * comps floats,
// element-major). Handles the byte stride and the common component types,
// normalizing integer colors to [0,1]. Returns false on an unsupported layout.
bool read_float_accessor(const tinygltf::Model& gltf, int index,
                         std::vector<float>& out, int& comps) {
  if (index < 0 || index >= static_cast<int>(gltf.accessors.size())) {
    return false;
  }
  const tinygltf::Accessor& acc =
      gltf.accessors[static_cast<std::size_t>(index)];
  comps = component_count(acc.type);
  if (comps == 0) return false;
  if (acc.bufferView < 0 ||
      acc.bufferView >= static_cast<int>(gltf.bufferViews.size())) {
    return false;
  }
  const tinygltf::BufferView& view =
      gltf.bufferViews[static_cast<std::size_t>(acc.bufferView)];
  const tinygltf::Buffer& buffer =
      gltf.buffers[static_cast<std::size_t>(view.buffer)];
  const unsigned char* base =
      buffer.data.data() + view.byteOffset + acc.byteOffset;

  const int comp_type = acc.componentType;
  std::size_t comp_size = 0;
  switch (comp_type) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
      comp_size = 4;
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      comp_size = 1;
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      comp_size = 2;
      break;
    default:
      return false;  // unsupported attribute component type
  }
  const std::size_t element_size = comp_size * static_cast<std::size_t>(comps);
  const std::size_t stride =
      view.byteStride != 0 ? view.byteStride : element_size;

  out.resize(acc.count * static_cast<std::size_t>(comps));
  for (std::size_t i = 0; i < acc.count; ++i) {
    const unsigned char* element = base + i * stride;
    for (int c = 0; c < comps; ++c) {
      const unsigned char* p =
          element + static_cast<std::size_t>(c) * comp_size;
      float value = 0.0f;
      switch (comp_type) {
        case TINYGLTF_COMPONENT_TYPE_FLOAT: {
          std::memcpy(&value, p, sizeof(float));
          break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
          value = acc.normalized ? static_cast<float>(*p) / 255.0f
                                 : static_cast<float>(*p);
          break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
          std::uint16_t raw = 0;
          std::memcpy(&raw, p, sizeof(std::uint16_t));
          value = acc.normalized ? static_cast<float>(raw) / 65535.0f
                                 : static_cast<float>(raw);
          break;
        }
        default:
          return false;
      }
      out[i * static_cast<std::size_t>(comps) + static_cast<std::size_t>(c)] =
          value;
    }
  }
  return true;
}

// Append `primitive`'s indices (offset by base_vertex) to `out` as uint32. When
// the primitive is non-indexed, emit a sequential 0..vertex_count list.
bool read_indices(const tinygltf::Model& gltf,
                  const tinygltf::Primitive& primitive,
                  std::uint32_t base_vertex, std::size_t vertex_count,
                  std::vector<std::uint32_t>& out) {
  if (primitive.indices < 0) {
    for (std::size_t i = 0; i < vertex_count; ++i) {
      out.push_back(base_vertex + static_cast<std::uint32_t>(i));
    }
    return true;
  }
  const tinygltf::Accessor& acc =
      gltf.accessors[static_cast<std::size_t>(primitive.indices)];
  if (acc.bufferView < 0) return false;
  const tinygltf::BufferView& view =
      gltf.bufferViews[static_cast<std::size_t>(acc.bufferView)];
  const tinygltf::Buffer& buffer =
      gltf.buffers[static_cast<std::size_t>(view.buffer)];
  const unsigned char* base =
      buffer.data.data() + view.byteOffset + acc.byteOffset;

  out.reserve(out.size() + acc.count);
  for (std::size_t i = 0; i < acc.count; ++i) {
    std::uint32_t value = 0;
    switch (acc.componentType) {
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        value = base[i];
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        std::uint16_t raw = 0;
        std::memcpy(&raw, base + i * sizeof(std::uint16_t), sizeof(raw));
        value = raw;
        break;
      }
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        std::uint32_t raw = 0;
        std::memcpy(&raw, base + i * sizeof(std::uint32_t), sizeof(raw));
        value = raw;
        break;
      }
      default:
        return false;
    }
    out.push_back(base_vertex + value);
  }
  return true;
}

glm::mat4 node_local_transform(const tinygltf::Node& node) {
  if (node.matrix.size() == 16) {
    glm::mat4 m(1.0f);
    for (int col = 0; col < 4; ++col) {
      for (int row = 0; row < 4; ++row) {
        m[col][row] = static_cast<float>(
            node.matrix[static_cast<std::size_t>(col * 4 + row)]);
      }
    }
    return m;
  }
  glm::mat4 t(1.0f);
  if (node.translation.size() == 3) {
    t[3][0] = static_cast<float>(node.translation[0]);
    t[3][1] = static_cast<float>(node.translation[1]);
    t[3][2] = static_cast<float>(node.translation[2]);
  }
  glm::mat4 r(1.0f);
  if (node.rotation.size() == 4) {
    const float x = static_cast<float>(node.rotation[0]);
    const float y = static_cast<float>(node.rotation[1]);
    const float z = static_cast<float>(node.rotation[2]);
    const float w = static_cast<float>(node.rotation[3]);
    // Column-major rotation from a unit quaternion (glTF stores [x,y,z,w]).
    r[0][0] = 1.0f - 2.0f * (y * y + z * z);
    r[0][1] = 2.0f * (x * y + z * w);
    r[0][2] = 2.0f * (x * z - y * w);
    r[1][0] = 2.0f * (x * y - z * w);
    r[1][1] = 1.0f - 2.0f * (x * x + z * z);
    r[1][2] = 2.0f * (y * z + x * w);
    r[2][0] = 2.0f * (x * z + y * w);
    r[2][1] = 2.0f * (y * z - x * w);
    r[2][2] = 1.0f - 2.0f * (x * x + y * y);
  }
  glm::mat4 s(1.0f);
  if (node.scale.size() == 3) {
    s[0][0] = static_cast<float>(node.scale[0]);
    s[1][1] = static_cast<float>(node.scale[1]);
    s[2][2] = static_cast<float>(node.scale[2]);
  }
  return t * r * s;  // glTF: M = T * R * S
}

AlphaMode parse_alpha_mode(const std::string& mode) {
  if (mode == "MASK") return AlphaMode::Mask;
  if (mode == "BLEND") return AlphaMode::Blend;
  return AlphaMode::Opaque;
}

std::uint32_t texture_image_index(const tinygltf::Model& gltf, int texture) {
  if (texture < 0 || texture >= static_cast<int>(gltf.textures.size())) {
    return kNoTexture;
  }
  const int source = gltf.textures[static_cast<std::size_t>(texture)].source;
  if (source < 0 || source >= static_cast<int>(gltf.images.size())) {
    return kNoTexture;
  }
  return static_cast<std::uint32_t>(source);
}

Material convert_material(const tinygltf::Model& gltf,
                          const tinygltf::Material& src) {
  Material m;
  m.name = src.name;

  const tinygltf::PbrMetallicRoughness& pbr = src.pbrMetallicRoughness;
  if (pbr.baseColorFactor.size() == 4) {
    m.base_color_factor = {static_cast<float>(pbr.baseColorFactor[0]),
                           static_cast<float>(pbr.baseColorFactor[1]),
                           static_cast<float>(pbr.baseColorFactor[2]),
                           static_cast<float>(pbr.baseColorFactor[3])};
  }
  m.metallic_factor = static_cast<float>(pbr.metallicFactor);
  m.roughness_factor = static_cast<float>(pbr.roughnessFactor);
  m.base_color_texture = texture_image_index(gltf, pbr.baseColorTexture.index);
  m.metallic_roughness_texture =
      texture_image_index(gltf, pbr.metallicRoughnessTexture.index);

  m.normal_texture = texture_image_index(gltf, src.normalTexture.index);
  m.normal_scale = static_cast<float>(src.normalTexture.scale);
  m.occlusion_texture = texture_image_index(gltf, src.occlusionTexture.index);
  m.occlusion_strength = static_cast<float>(src.occlusionTexture.strength);

  if (src.emissiveFactor.size() == 3) {
    m.emissive_factor = {static_cast<float>(src.emissiveFactor[0]),
                         static_cast<float>(src.emissiveFactor[1]),
                         static_cast<float>(src.emissiveFactor[2])};
  }
  m.emissive_texture = texture_image_index(gltf, src.emissiveTexture.index);

  m.alpha_mode = parse_alpha_mode(src.alphaMode);
  m.alpha_cutoff = static_cast<float>(src.alphaCutoff);
  m.double_sided = src.doubleSided;
  return m;
}

Image convert_image(const tinygltf::Image& src) {
  Image img;
  img.name = src.name.empty() ? src.uri : src.name;
  img.width = static_cast<std::uint32_t>(std::max(src.width, 0));
  img.height = static_cast<std::uint32_t>(std::max(src.height, 0));
  img.channels = static_cast<std::uint32_t>(std::max(src.component, 0));
  img.pixels.assign(src.image.begin(), src.image.end());
  return img;
}

// Convert one primitive into a Mesh. Returns false if it has no POSITION or an
// unsupported layout. Skips non-triangle primitives (mode != TRIANGLES).
bool convert_primitive(const tinygltf::Model& gltf,
                       const tinygltf::Primitive& primitive,
                       const std::string& name, const glm::mat4& transform,
                       Mesh& out) {
  if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != -1) {
    return false;  // TODO: line/point primitive modes
  }
  const auto pos_it = primitive.attributes.find("POSITION");
  if (pos_it == primitive.attributes.end()) return false;

  std::vector<float> positions;
  int pos_comps = 0;
  if (!read_float_accessor(gltf, pos_it->second, positions, pos_comps) ||
      pos_comps != 3) {
    return false;
  }
  const std::size_t vertex_count = positions.size() / 3;

  auto channel = [&](const char* attr, std::vector<float>& dst,
                     int& comps) -> bool {
    const auto it = primitive.attributes.find(attr);
    if (it == primitive.attributes.end()) return false;
    return read_float_accessor(gltf, it->second, dst, comps);
  };

  std::vector<float> normals, tangents, uvs, colors;
  int n_comps = 0, t_comps = 0, uv_comps = 0, c_comps = 0;
  const bool has_normals = channel("NORMAL", normals, n_comps) && n_comps == 3;
  const bool has_tangents =
      channel("TANGENT", tangents, t_comps) && t_comps == 4;
  const bool has_uvs = channel("TEXCOORD_0", uvs, uv_comps) && uv_comps == 2;
  const bool has_colors =
      channel("COLOR_0", colors, c_comps) && (c_comps == 3 || c_comps == 4);

  out.name = name;
  out.transform = transform;
  out.vertices.resize(vertex_count);
  for (std::size_t i = 0; i < vertex_count; ++i) {
    Vertex& v = out.vertices[i];
    v.position = {positions[i * 3 + 0], positions[i * 3 + 1],
                  positions[i * 3 + 2]};
    if (has_normals) {
      v.normal = {normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]};
    }
    if (has_tangents) {
      v.tangent = {tangents[i * 4 + 0], tangents[i * 4 + 1],
                   tangents[i * 4 + 2], tangents[i * 4 + 3]};
    }
    if (has_uvs) {
      v.uv0 = {uvs[i * 2 + 0], uvs[i * 2 + 1]};
    }
    if (has_colors) {
      v.color = {colors[i * c_comps + 0], colors[i * c_comps + 1],
                 colors[i * c_comps + 2],
                 c_comps == 4 ? colors[i * c_comps + 3] : 1.0f};
    }
  }

  if (!read_indices(gltf, primitive, /*base_vertex=*/0, vertex_count,
                    out.indices)) {
    return false;
  }
  out.material = primitive.material >= 0
                     ? static_cast<std::uint32_t>(primitive.material)
                     : Mesh::kNoMaterial;
  return true;
}

}  // namespace

std::optional<Model> load_gltf(std::string_view path, std::string* error) {
  const std::string ext = extension_of(path);
  const std::string filename(path);

  tinygltf::TinyGLTF loader;
  tinygltf::Model gltf;
  std::string err;
  std::string warn;

  bool ok = false;
  if (ext == ".glb") {
    ok = loader.LoadBinaryFromFile(&gltf, &err, &warn, filename);
  } else if (ext == ".gltf") {
    ok = loader.LoadASCIIFromFile(&gltf, &err, &warn, filename);
  } else {
    if (error != nullptr) {
      *error = "unsupported glTF extension '" + ext +
               "' (expected .gltf or .glb): " + filename;
    }
    return std::nullopt;
  }
  if (!ok) {
    if (error != nullptr) {
      *error = err.empty() ? ("failed to load glTF: " + filename) : err;
    }
    return std::nullopt;
  }

  Model model;

  model.images.reserve(gltf.images.size());
  for (const tinygltf::Image& img : gltf.images) {
    model.images.push_back(convert_image(img));
  }

  model.materials.reserve(gltf.materials.size());
  for (const tinygltf::Material& mat : gltf.materials) {
    model.materials.push_back(convert_material(gltf, mat));
  }

  // Flatten every mesh's primitives. Each primitive becomes a Mesh; the glTF
  // mesh index alone is not enough to address a primitive, so meshes here are
  // primitive-granular. A node referencing glTF mesh M points (in our flat
  // node) at the first primitive's Mesh index; mesh_first_primitive maps that.
  std::vector<std::uint32_t> mesh_first_primitive(gltf.meshes.size(),
                                                  Node::kNoMesh);
  for (std::size_t mi = 0; mi < gltf.meshes.size(); ++mi) {
    const tinygltf::Mesh& gmesh = gltf.meshes[mi];
    for (std::size_t pi = 0; pi < gmesh.primitives.size(); ++pi) {
      Mesh mesh;
      if (!convert_primitive(gltf, gmesh.primitives[pi], gmesh.name,
                             glm::mat4(1.0f), mesh)) {
        continue;  // skip empty / unsupported primitive
      }
      if (mesh_first_primitive[mi] == Node::kNoMesh) {
        mesh_first_primitive[mi] =
            static_cast<std::uint32_t>(model.meshes.size());
      }
      model.meshes.push_back(std::move(mesh));
    }
  }

  // Flatten the node tree (data only). Child/mesh references stay as indices.
  model.scene.nodes.reserve(gltf.nodes.size());
  for (const tinygltf::Node& gnode : gltf.nodes) {
    Node node;
    node.name = gnode.name;
    node.transform = node_local_transform(gnode);
    if (gnode.mesh >= 0 &&
        gnode.mesh < static_cast<int>(mesh_first_primitive.size())) {
      node.mesh = mesh_first_primitive[static_cast<std::size_t>(gnode.mesh)];
    }
    for (int child : gnode.children) {
      if (child >= 0)
        node.children.push_back(static_cast<std::uint32_t>(child));
    }
    model.scene.nodes.push_back(std::move(node));
  }

  const int scene_index = gltf.defaultScene >= 0
                              ? gltf.defaultScene
                              : (gltf.scenes.empty() ? -1 : 0);
  if (scene_index >= 0 && scene_index < static_cast<int>(gltf.scenes.size())) {
    const tinygltf::Scene& gscene =
        gltf.scenes[static_cast<std::size_t>(scene_index)];
    model.scene.name = gscene.name;
    for (int root : gscene.nodes) {
      if (root >= 0)
        model.scene.roots.push_back(static_cast<std::uint32_t>(root));
    }
  }

  return model;
}

}  // namespace volumetric_kit::gfx::assets

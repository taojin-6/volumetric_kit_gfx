// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/io/gltf_loader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>  // translate, scale
#include <glm/gtc/quaternion.hpp>        // quat, mat4_cast
#include <glm/gtc/type_ptr.hpp>          // make_mat4

// tinygltf is compiled once in tinygltf_impl.cpp; here it is a
// declarations-only header. Keep stb out of this TU's macro state to match that
// split.
#include <tiny_gltf.h>

namespace volumetric_kit::gfx::io {

// The loaders fill the gfx_assets value types; bring them in unqualified so the
// conversion helpers below read naturally (Mesh, Material, Image, Node, ...).
using namespace volumetric_kit::gfx::assets;

namespace {

// Lower-case file extension including the dot (e.g. ".glb"); empty if none.
std::string extension_of(std::string_view path) {
  // Search only the final path component, so a dot in a parent directory (e.g.
  // "/home/user.v2/model") is not mistaken for the file's extension.
  const std::size_t sep = path.find_last_of("/\\");
  const std::size_t dot = path.rfind('.');
  if (dot == std::string_view::npos ||
      (sep != std::string_view::npos && dot < sep)) {
    return {};
  }
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

// True when the half-open byte range [offset, offset + span) lies within
// `buffer`. Written without `offset + span` so a forged offset/span cannot
// overflow std::size_t past the check.
bool range_in_buffer(const tinygltf::Buffer& buffer, std::size_t offset,
                     std::size_t span) {
  return offset <= buffer.data.size() && span <= buffer.data.size() - offset;
}

// Resolve an accessor's backing bytes, validating every file-supplied index and
// offset so a malformed glTF cannot drive an out-of-bounds read: the bufferView
// index, the buffer index (which defaults to -1), and the full strided extent
// of `acc.count` elements against the buffer size. On success `base` points at
// the first element and `stride` is the element-to-element byte step. Returns
// false on any inconsistency.
bool resolve_accessor(const tinygltf::Model& gltf,
                      const tinygltf::Accessor& acc, std::size_t element_size,
                      const unsigned char*& base, std::size_t& stride) {
  if (acc.bufferView < 0 ||
      acc.bufferView >= static_cast<int>(gltf.bufferViews.size())) {
    return false;
  }
  const tinygltf::BufferView& view =
      gltf.bufferViews[static_cast<std::size_t>(acc.bufferView)];
  if (view.buffer < 0 || view.buffer >= static_cast<int>(gltf.buffers.size())) {
    return false;
  }
  const tinygltf::Buffer& buffer =
      gltf.buffers[static_cast<std::size_t>(view.buffer)];
  stride = view.byteStride != 0 ? static_cast<std::size_t>(view.byteStride)
                                : element_size;
  const std::size_t start = view.byteOffset + acc.byteOffset;
  if (acc.count > 0) {
    // Highest byte touched = start + (count-1)*stride + element_size.
    const std::size_t span = (acc.count - 1) * stride + element_size;
    if (!range_in_buffer(buffer, start, span)) return false;
  }
  base = buffer.data.data() + start;
  return true;
}

// Read accessor `index` as float vectors into `out` (count * comps floats,
// element-major). Validates the layout against the backing buffer, then handles
// the byte stride and the common component types. Integer components are scaled
// to [0,1] (unsigned) / [-1,1] (signed) when the accessor sets `normalized` or
// when `force_normalize_int` is set (glTF requires COLOR_0 integers normalized
// regardless of the flag). Returns false on an unsupported / out-of-bounds
// layout.
bool read_float_accessor(const tinygltf::Model& gltf, int index,
                         std::vector<float>& out, int& comps,
                         bool force_normalize_int = false) {
  if (index < 0 || index >= static_cast<int>(gltf.accessors.size())) {
    return false;
  }
  const tinygltf::Accessor& acc =
      gltf.accessors[static_cast<std::size_t>(index)];
  if (acc.sparse.isSparse) {
    return false;  // TODO: apply sparse accessor substitutions
  }
  comps = component_count(acc.type);
  if (comps == 0) return false;

  const int comp_type = acc.componentType;
  std::size_t comp_size = 0;
  switch (comp_type) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
      comp_size = 4;
      break;
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      comp_size = 1;
      break;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      comp_size = 2;
      break;
    default:
      return false;  // unsupported attribute component type
  }
  const std::size_t element_size = comp_size * static_cast<std::size_t>(comps);

  const unsigned char* base = nullptr;
  std::size_t stride = 0;
  if (!resolve_accessor(gltf, acc, element_size, base, stride)) return false;

  const bool normalize = acc.normalized || force_normalize_int;
  out.resize(acc.count * static_cast<std::size_t>(comps));
  for (std::size_t i = 0; i < acc.count; ++i) {
    const unsigned char* element = base + i * stride;
    for (int c = 0; c < comps; ++c) {
      const unsigned char* p =
          element + static_cast<std::size_t>(c) * comp_size;
      float value = 0.0f;
      switch (comp_type) {
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
          std::memcpy(&value, p, sizeof(float));
          break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
          value = normalize ? static_cast<float>(*p) / 255.0f
                            : static_cast<float>(*p);
          break;
        case TINYGLTF_COMPONENT_TYPE_BYTE: {
          std::int8_t raw = 0;
          std::memcpy(&raw, p, sizeof(raw));
          value = normalize ? std::max(static_cast<float>(raw) / 127.0f, -1.0f)
                            : static_cast<float>(raw);
          break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
          std::uint16_t raw = 0;
          std::memcpy(&raw, p, sizeof(raw));
          value = normalize ? static_cast<float>(raw) / 65535.0f
                            : static_cast<float>(raw);
          break;
        }
        case TINYGLTF_COMPONENT_TYPE_SHORT: {
          std::int16_t raw = 0;
          std::memcpy(&raw, p, sizeof(raw));
          value = normalize
                      ? std::max(static_cast<float>(raw) / 32767.0f, -1.0f)
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

// Append `primitive`'s indices to `out` as uint32. When the primitive is
// non-indexed, emit a sequential 0..vertex_count list. Every emitted index is
// validated to address one of the `vertex_count` vertices, and the source bytes
// are bounds-checked, so a malformed accessor is rejected rather than producing
// out-of-range indices or reading past the buffer. Returns false on an
// unsupported / out-of-bounds layout.
bool read_indices(const tinygltf::Model& gltf,
                  const tinygltf::Primitive& primitive,
                  std::size_t vertex_count, std::vector<std::uint32_t>& out) {
  if (primitive.indices < 0) {
    out.reserve(out.size() + vertex_count);
    for (std::size_t i = 0; i < vertex_count; ++i) {
      out.push_back(static_cast<std::uint32_t>(i));
    }
    return true;
  }
  if (primitive.indices >= static_cast<int>(gltf.accessors.size())) {
    return false;
  }
  const tinygltf::Accessor& acc =
      gltf.accessors[static_cast<std::size_t>(primitive.indices)];
  if (acc.sparse.isSparse) return false;  // TODO: sparse index accessors

  std::size_t comp_size = 0;
  switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      comp_size = 1;
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      comp_size = 2;
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      comp_size = 4;
      break;
    default:
      return false;
  }
  const unsigned char* base = nullptr;
  std::size_t stride = 0;
  if (!resolve_accessor(gltf, acc, comp_size, base, stride)) return false;

  out.reserve(out.size() + acc.count);
  for (std::size_t i = 0; i < acc.count; ++i) {
    const unsigned char* p = base + i * stride;
    std::uint32_t value = 0;
    switch (acc.componentType) {
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        value = *p;
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        std::uint16_t raw = 0;
        std::memcpy(&raw, p, sizeof(raw));
        value = raw;
        break;
      }
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        std::uint32_t raw = 0;
        std::memcpy(&raw, p, sizeof(raw));
        value = raw;
        break;
      }
      default:
        return false;
    }
    if (value >= vertex_count) return false;  // index addresses no vertex
    out.push_back(value);
  }
  return true;
}

glm::mat4 node_local_transform(const tinygltf::Node& node) {
  // glTF node.matrix is 16 doubles in column-major order -- the same layout
  // glm::make_mat4 expects, so it maps across directly.
  if (node.matrix.size() == 16) {
    std::array<float, 16> m{};
    for (std::size_t i = 0; i < 16; ++i) {
      m[i] = static_cast<float>(node.matrix[i]);
    }
    return glm::make_mat4(m.data());
  }
  glm::mat4 t(1.0f);
  if (node.translation.size() == 3) {
    t = glm::translate(t, glm::vec3(static_cast<float>(node.translation[0]),
                                    static_cast<float>(node.translation[1]),
                                    static_cast<float>(node.translation[2])));
  }
  glm::mat4 r(1.0f);
  if (node.rotation.size() == 4) {
    // glTF stores the quaternion as [x,y,z,w]; glm::quat takes (w,x,y,z).
    r = glm::mat4_cast(glm::quat(static_cast<float>(node.rotation[3]),
                                 static_cast<float>(node.rotation[0]),
                                 static_cast<float>(node.rotation[1]),
                                 static_cast<float>(node.rotation[2])));
  }
  glm::mat4 s(1.0f);
  if (node.scale.size() == 3) {
    s = glm::scale(s, glm::vec3(static_cast<float>(node.scale[0]),
                                static_cast<float>(node.scale[1]),
                                static_cast<float>(node.scale[2])));
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

// `src` is taken by non-const ref so its decoded pixel buffer can be moved out
// rather than copied (textures are megabytes); `gltf` is a local in load_gltf
// and is discarded afterwards, so mutating it is safe.
Image convert_image(tinygltf::Image& src) {
  Image img;
  // Only adopt the pixels when they are dimensionally consistent (8-bit,
  // tightly packed). A decode failure or an unsupported depth (e.g. 16-bit)
  // leaves an empty, default Image -- valid() == false -- instead of one whose
  // width/height/channels lie about a too-small buffer.
  if (src.width <= 0 || src.height <= 0 || src.component <= 0 ||
      src.bits != 8) {
    return img;
  }
  const std::size_t expected = static_cast<std::size_t>(src.width) *
                               static_cast<std::size_t>(src.height) *
                               static_cast<std::size_t>(src.component);
  if (src.image.size() != expected) return img;

  img.name = !src.name.empty() ? std::move(src.name) : std::move(src.uri);
  img.width = static_cast<std::uint32_t>(src.width);
  img.height = static_cast<std::uint32_t>(src.height);
  img.channels = static_cast<std::uint32_t>(src.component);
  img.pixels = std::move(src.image);
  return img;
}

// Convert one primitive into a Mesh. Returns false if it cannot be
// represented -- no POSITION, an unsupported accessor layout or primitive
// mode, or indices that do not form whole triangles -- pointing `reason` at a
// static string naming the category for the caller's warning.
bool convert_primitive(const tinygltf::Model& gltf,
                       const tinygltf::Primitive& primitive,
                       const std::string& name, Mesh& out,
                       const char*& reason) {
  if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != -1) {
    // TODO: line/point primitive modes
    reason = "unsupported primitive mode (only TRIANGLES)";
    return false;
  }
  const auto pos_it = primitive.attributes.find("POSITION");
  if (pos_it == primitive.attributes.end()) {
    reason = "no POSITION attribute";
    return false;
  }

  std::vector<float> positions;
  int pos_comps = 0;
  if (!read_float_accessor(gltf, pos_it->second, positions, pos_comps) ||
      pos_comps != 3) {
    reason =
        "unsupported POSITION accessor (sparse, bufferView-less, non-VEC3, or "
        "out-of-bounds layout)";
    return false;
  }
  const std::size_t vertex_count = positions.size() / 3;

  // Read an attribute channel and accept it only when it has the expected
  // component count *and* one element per vertex. A glTF whose attribute
  // accessor is shorter than POSITION would otherwise overrun the channel
  // buffer in the de-interleave loop below.
  auto channel = [&](const char* attr, std::vector<float>& dst, int want_comps,
                     bool force_norm = false) -> bool {
    const auto it = primitive.attributes.find(attr);
    if (it == primitive.attributes.end()) return false;
    int comps = 0;
    if (!read_float_accessor(gltf, it->second, dst, comps, force_norm)) {
      return false;
    }
    return comps == want_comps &&
           dst.size() == vertex_count * static_cast<std::size_t>(want_comps);
  };

  std::vector<float> normals, tangents, uvs, colors3, colors4;
  const bool has_normals = channel("NORMAL", normals, 3);
  const bool has_tangents = channel("TANGENT", tangents, 4);
  // TODO: synthesize tangents (MikkTSpace or a UV-derived frame) when
  // has_tangents is false, so a normal-mapped primitive lacking a TANGENT
  // attribute shades in a correct frame; today such a mesh keeps the Vertex
  // (1,0,0,1) default (consumers must guard the resulting degenerate TBN).
  const bool has_uvs = channel("TEXCOORD_0", uvs, 2);
  // COLOR_0 is VEC3 or VEC4; glTF requires its integer forms normalized.
  const bool has_colors3 = channel("COLOR_0", colors3, 3, /*force_norm=*/true);
  const bool has_colors4 =
      !has_colors3 && channel("COLOR_0", colors4, 4, /*force_norm=*/true);

  out.name = name;
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
    if (has_colors3) {
      v.color = {colors3[i * 3 + 0], colors3[i * 3 + 1], colors3[i * 3 + 2],
                 1.0f};
    } else if (has_colors4) {
      v.color = {colors4[i * 4 + 0], colors4[i * 4 + 1], colors4[i * 4 + 2],
                 colors4[i * 4 + 3]};
    }
  }

  if (!read_indices(gltf, primitive, vertex_count, out.indices)) {
    reason =
        "unsupported index accessor (sparse, bufferView-less, out-of-bounds "
        "layout, or an index addressing no vertex)";
    return false;
  }
  if (out.indices.size() % 3 != 0) {
    reason = "index count is not a whole number of triangles";
    return false;
  }
  out.material = (primitive.material >= 0 &&
                  primitive.material < static_cast<int>(gltf.materials.size()))
                     ? static_cast<std::uint32_t>(primitive.material)
                     : Mesh::kNoMaterial;
  return true;
}

}  // namespace

std::optional<Model> load_gltf(std::string_view path, std::string* error,
                               std::vector<std::string>* warnings) {
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
  for (tinygltf::Image& img : gltf.images) {
    model.images.push_back(convert_image(img));
  }

  model.materials.reserve(gltf.materials.size());
  for (const tinygltf::Material& mat : gltf.materials) {
    model.materials.push_back(convert_material(gltf, mat));
  }

  // Flatten every glTF mesh's primitives: each primitive becomes one Mesh, and
  // a glTF mesh maps to the contiguous range [first, first + count) it produced
  // (count 0 when every primitive was skipped). A node references that whole
  // range, so a multi-primitive mesh stays fully reachable from the scene.
  struct MeshRange {
    std::uint32_t first = Node::kNoMesh;
    std::uint32_t count = 0;
  };
  std::vector<MeshRange> mesh_ranges(gltf.meshes.size());
  for (std::size_t mi = 0; mi < gltf.meshes.size(); ++mi) {
    const tinygltf::Mesh& gmesh = gltf.meshes[mi];
    const std::uint32_t first = static_cast<std::uint32_t>(model.meshes.size());
    std::uint32_t count = 0;
    for (std::size_t pi = 0; pi < gmesh.primitives.size(); ++pi) {
      Mesh mesh;
      const char* reason = "unsupported primitive";
      if (!convert_primitive(gltf, gmesh.primitives[pi], gmesh.name, mesh,
                             reason)) {
        // A spec-valid file can land here (see the sparse-accessor @note on
        // load_gltf) and the load still succeeds, so report every dropped
        // primitive -- otherwise the model silently loses geometry.
        if (warnings != nullptr) {
          std::string what = "mesh " + std::to_string(mi);
          if (!gmesh.name.empty()) what += " ('" + gmesh.name + "')";
          what += " primitive " + std::to_string(pi) + " dropped: " + reason;
          warnings->push_back(std::move(what));
        }
        continue;
      }
      model.meshes.push_back(std::move(mesh));
      ++count;
    }
    if (count > 0) mesh_ranges[mi] = {first, count};
  }

  // Flatten the node tree (data only). Child/mesh references stay as indices,
  // each validated against the array it addresses so no dangling index escapes.
  model.scene.nodes.reserve(gltf.nodes.size());
  for (const tinygltf::Node& gnode : gltf.nodes) {
    Node node;
    node.name = gnode.name;
    node.transform = node_local_transform(gnode);
    if (gnode.mesh >= 0 && gnode.mesh < static_cast<int>(mesh_ranges.size())) {
      const MeshRange& range =
          mesh_ranges[static_cast<std::size_t>(gnode.mesh)];
      node.mesh = range.first;
      node.mesh_count = range.count;
    }
    for (int child : gnode.children) {
      if (child >= 0 && child < static_cast<int>(gltf.nodes.size())) {
        node.children.push_back(static_cast<std::uint32_t>(child));
      }
    }
    model.scene.nodes.push_back(std::move(node));
  }

  // Pick the default scene, falling back to scene 0 when defaultScene is unset
  // or out of range -- a positive-but-invalid value must not yield an empty
  // scene when valid scenes exist.
  int scene_index = gltf.scenes.empty() ? -1 : 0;
  if (gltf.defaultScene >= 0 &&
      gltf.defaultScene < static_cast<int>(gltf.scenes.size())) {
    scene_index = gltf.defaultScene;
  }
  if (scene_index >= 0) {
    const tinygltf::Scene& gscene =
        gltf.scenes[static_cast<std::size_t>(scene_index)];
    model.scene.name = gscene.name;
    for (int root : gscene.nodes) {
      if (root >= 0 && root < static_cast<int>(gltf.nodes.size())) {
        model.scene.roots.push_back(static_cast<std::uint32_t>(root));
      }
    }
  }

  return model;
}

}  // namespace volumetric_kit::gfx::io

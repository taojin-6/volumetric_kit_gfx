// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/pbr_model.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/assets/model.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"

namespace volumetric_kit::gfx::pipelines {

namespace {

// One mesh instance flattened out of the scene tree: a Model::meshes index
// under a composed world transform (a glTF mesh may be instanced by several
// nodes, so the transform lives on the instance, not the mesh).
struct DrawItem {
  uint32_t mesh = 0;
  glm::mat4 world{1.0f};
};

// Walk the scene tree, composing each node's transform down to world space,
// and emit one DrawItem per (instanced) mesh.
void collect_node(const assets::Model& model, uint32_t node_index,
                  const glm::mat4& parent, std::vector<DrawItem>& out,
                  std::vector<bool>& visited) {
  // Guard a malformed node graph: glTF requires a strict forest, but an
  // arbitrary file may not be conformant. An out-of-range or already-visited
  // index (a cycle) would otherwise recurse until the stack overflows; skip it
  // instead.
  if (node_index >= model.scene.nodes.size() || visited[node_index]) {
    return;
  }
  visited[node_index] = true;
  const assets::Node& node = model.scene.nodes[node_index];
  const glm::mat4 world = parent * node.transform;
  if (node.mesh != assets::Node::kNoMesh) {
    for (uint32_t k = 0; k < node.mesh_count; ++k) {
      const uint32_t mesh_index = node.mesh + k;
      if (mesh_index >= model.meshes.size()) {
        break;  // malformed [mesh, mesh + mesh_count) range: keep what fits
      }
      out.push_back({mesh_index, world});
    }
  }
  for (uint32_t child : node.children) {
    collect_node(model, child, world, out, visited);
  }
}

std::vector<DrawItem> collect_draws(const assets::Model& model) {
  std::vector<DrawItem> draws;
  std::vector<bool> visited(model.scene.nodes.size(), false);
  for (uint32_t root : model.scene.roots) {
    collect_node(model, root, glm::mat4(1.0f), draws, visited);
  }
  // Some files carry meshes but no scene graph: draw every mesh at the origin.
  if (draws.empty()) {
    for (uint32_t i = 0; i < model.meshes.size(); ++i) {
      draws.push_back({i, glm::mat4(1.0f)});
    }
  }
  return draws;
}

// Expand a decoded CPU image to tightly-packed RGBA8 (the layout an image
// upload takes): pass 4-channel through, replicate 1/2-channel luminance into
// RGB, and pad 3-channel with opaque alpha -- most GPUs do not sample
// 3-channel 8-bit.
std::vector<uint8_t> to_rgba8(const assets::Image& img) {
  const size_t texels = static_cast<size_t>(img.width) * img.height;
  std::vector<uint8_t> out(texels * 4);
  const uint32_t c = img.channels;
  for (size_t i = 0; i < texels; ++i) {
    const uint8_t* src = img.pixels.data() + i * c;
    out[i * 4 + 0] = src[0];
    out[i * 4 + 1] = c >= 3 ? src[1] : src[0];
    out[i * 4 + 2] = c >= 3 ? src[2] : src[0];
    out[i * 4 + 3] = c == 4 ? src[3] : (c == 2 ? src[1] : 255);
  }
  return out;
}

}  // namespace

PbrMaterialDesc pbr_material_desc(const assets::Material& material) {
  PbrMaterialDesc desc;
  desc.base_color_factor = material.base_color_factor;
  desc.emissive_factor = material.emissive_factor;
  desc.metallic_factor = material.metallic_factor;
  desc.roughness_factor = material.roughness_factor;
  desc.normal_scale = material.normal_scale;
  desc.occlusion_strength = material.occlusion_strength;
  // TODO: wire alpha_mode/alpha_cutoff/double_sided through PbrPipeline -- the
  // pipeline has no alpha-mask/blend or per-material cull state yet, so those
  // assets::Material fields are not consumed here.
  return desc;
}

Result<PbrModel> PbrModel::create(const Device& device, Allocator& allocator,
                                  const PbrPipeline& pipeline,
                                  const assets::Model& model) {
  if (device.handle() == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "PbrModel::create: device must hold a live VkDevice");
  }
  if (!pipeline.valid()) {
    return Status::invalid_argument("PbrModel::create: pipeline must be valid");
  }

  PbrModel out;

  // Shared sampler for every material map.
  VG_ASSIGN(Sampler sampler, Sampler::create(device.handle()));
  out.sampler_.emplace(std::move(sampler));

  // One batch for the whole model -- every mesh's vertex/index buffers, the
  // fallback textures, and every material map record into a single submit,
  // finished before the descriptor sets are built. A failure below returns
  // without finishing: the batch (and its pending copies into any dropped
  // resources) is discarded, never submitted.
  VG_ASSIGN(UploadBatch batch, UploadBatch::begin(device, allocator));

  // Upload every non-empty mesh; the vector is parallel to model.meshes so a
  // DrawItem's mesh index addresses it directly (empty meshes stay a default,
  // skipped GpuMesh).
  out.meshes_.resize(model.meshes.size());
  for (size_t i = 0; i < model.meshes.size(); ++i) {
    const assets::Mesh& mesh = model.meshes[i];
    if (mesh.vertices.empty() || mesh.indices.empty()) {
      continue;
    }
    VG_ASSIGN(GpuMesh gpu, upload_mesh(batch, mesh));
    out.meshes_[i] = std::move(gpu);
  }

  // Fallback textures: white (samples 1.0 for any non-normal slot, so the
  // factor alone applies) and flat-normal (0.5,0.5,1 -> (0,0,1): no
  // perturbation). add() copies the pixels into staging immediately, so the
  // locals need not outlive this scope.
  const uint8_t white_px[4] = {255, 255, 255, 255};
  const uint8_t flat_px[4] = {128, 128, 255, 255};
  ImageUploadDesc fallback_desc;
  fallback_desc.extent = {1, 1};
  fallback_desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  fallback_desc.pixels = white_px;
  fallback_desc.size = sizeof(white_px);
  VG_ASSIGN(Texture white_tex, batch.add(fallback_desc));
  fallback_desc.pixels = flat_px;
  VG_ASSIGN(Texture flat_tex, batch.add(fallback_desc));
  const size_t white = out.textures_.size();
  out.textures_.push_back(std::move(white_tex));
  const size_t flat = out.textures_.size();
  out.textures_.push_back(std::move(flat_tex));

  // Color space per image follows its slot: base-color + emissive are sRGB,
  // the rest linear. A glTF image fills one role in practice; if shared, sRGB
  // wins.
  std::vector<bool> srgb(model.images.size(), false);
  for (const assets::Material& m : model.materials) {
    if (m.base_color_texture < srgb.size()) {
      srgb[m.base_color_texture] = true;
    }
    if (m.emissive_texture < srgb.size()) {
      srgb[m.emissive_texture] = true;
    }
  }

  // Upload each image once; image_tex maps a model.images index into
  // textures_ (-1 = absent/invalid, resolved to a fallback below).
  std::vector<int> image_tex(model.images.size(), -1);
  for (size_t i = 0; i < model.images.size(); ++i) {
    const assets::Image& img = model.images[i];
    if (!img.valid()) {
      continue;
    }
    const std::vector<uint8_t> rgba = to_rgba8(img);
    ImageUploadDesc desc;
    desc.extent = {img.width, img.height};
    desc.format = srgb[i] ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    desc.pixels = rgba.data();
    desc.size = rgba.size();
    desc.generate_mips = true;
    VG_ASSIGN(Texture tex, batch.add(desc));
    image_tex[i] = static_cast<int>(out.textures_.size());
    out.textures_.push_back(std::move(tex));
  }

  // Submit every queued upload at once; the meshes are draw-ready and the
  // textures the materials below bind are sampled-ready when this returns.
  VG_TRY(batch.finish());

  // Texture index for a slot, or the given fallback (assets::kNoTexture is out
  // of image_tex's range, so it resolves to the fallback).
  const auto tex_for = [&image_tex](uint32_t slot, size_t fallback) -> size_t {
    return (slot < image_tex.size() && image_tex[slot] >= 0)
               ? static_cast<size_t>(image_tex[slot])
               : fallback;
  };

  const VkDescriptorSetLayout material_layout =
      pipeline.descriptor_set_layout(1);

  // One material (set 1) per source material: factors via pbr_material_desc,
  // maps resolved to the uploaded image for the slot or a fallback.
  out.materials_.reserve(model.materials.size() + 1);
  for (const assets::Material& m : model.materials) {
    PbrMaterialDesc desc = pbr_material_desc(m);
    desc.base_color =
        out.textures_[tex_for(m.base_color_texture, white)].view();
    desc.metallic_roughness =
        out.textures_[tex_for(m.metallic_roughness_texture, white)].view();
    desc.normal = out.textures_[tex_for(m.normal_texture, flat)].view();
    desc.occlusion = out.textures_[tex_for(m.occlusion_texture, white)].view();
    desc.emissive = out.textures_[tex_for(m.emissive_texture, white)].view();
    desc.sampler = out.sampler_->handle();
    VG_ASSIGN(
        PbrMaterial material,
        PbrMaterial::create(device.handle(), allocator, material_layout, desc));
    out.materials_.push_back(std::move(material));
  }

  // Fallback material for meshes with no material: a matte white dielectric.
  // Appended last so materials_[i] stays parallel to model.materials.
  {
    PbrMaterialDesc desc;
    desc.base_color_factor = glm::vec4(1.0f);
    desc.emissive_factor = glm::vec3(0.0f);
    desc.metallic_factor = 0.0f;
    desc.roughness_factor = 1.0f;
    desc.base_color = out.textures_[white].view();
    desc.metallic_roughness = out.textures_[white].view();
    desc.normal = out.textures_[flat].view();
    desc.occlusion = out.textures_[white].view();
    desc.emissive = out.textures_[white].view();
    desc.sampler = out.sampler_->handle();
    VG_ASSIGN(
        PbrMaterial fallback,
        PbrMaterial::create(device.handle(), allocator, material_layout, desc));
    out.materials_.push_back(std::move(fallback));
  }

  // Resolve each flattened instance into a PbrDraw: its GPU mesh, world
  // transform, and the material its mesh names (or the fallback). The
  // pointers reach into out's own vectors -- stable across moves, since a
  // vector move keeps its elements' addresses.
  const std::vector<DrawItem> items = collect_draws(model);
  const size_t source_materials = model.materials.size();
  out.draws_.reserve(items.size());
  for (const DrawItem& item : items) {
    const uint32_t mat = model.meshes[item.mesh].material;
    const PbrMaterial* material =
        (mat != assets::Mesh::kNoMaterial && mat < source_materials)
            ? &out.materials_[mat]
            : &out.materials_.back();
    out.draws_.push_back({&out.meshes_[item.mesh], item.world, material});
  }
  return out;
}

}  // namespace volumetric_kit::gfx::pipelines

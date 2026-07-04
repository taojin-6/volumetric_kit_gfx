// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/03_model: load a glTF model and fly around it. Parses a
// `.gltf`/`.glb` with the io tier into a CPU assets::Model, uploads each mesh
// (vertex + index buffer) and its material maps, and draws it depth-tested with
// glTF metallic-roughness PBR: a per-draw model/MVP push constant, a per-frame
// scene set (set 0: camera), and a per-material set (set 1: factor UBO + the
// five maps) -- all reflected automatically into the pipeline layout. The
// camera auto-frames the model's bounds, so any model fills the view. IBL
// (image-based ambient) is the next spine step; today the ambient is a flat
// fill.
//
// Usage:
//   example_03_model                       # built-in cube, in a window
//   example_03_model --model Helmet.glb    # any glTF-Sample-Assets model
//   example_03_model --frames 3            # render N frames, then exit
//   example_03_model --model m.glb --screenshot out.ppm   # headless still
//
// Controls (windowed, interactive by default): left-drag orbits, right/middle-
// drag pans, wheel zooms, WASDQE flies (Q/E down/up), Shift moves faster.
//
// Two render paths share the model load + upload + draw recording:
//  * Windowed (default): Surface + Swapchain + FrameLoop, driven by mouse +
//    keyboard (a deterministic turntable instead under --frames). The
//    swapchain owns a depth attachment per image
//    (SwapchainConfig::depth_format) and rebuilds it on resize, so each frame
//    renders straight into the loop's depth-capable target at two frames in
//    flight.
//  * --screenshot: no window -- renders one frame into an OffscreenTarget
//    (color + depth + readback) and writes a binary PPM. Headless, so it works
//    where no display / screen-capture is available.

#include "volumetric_kit/gfx/core/vulkan.hpp"  // before GLFW, so glfw3.h sees
// Vulkan and declares its helpers

#include <GLFW/glfw3.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/common.hpp>     // glm::min / glm::max (component-wise)
#include <glm/geometric.hpp>  // glm::length
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>  // glm::inverse
#include <glm/packing.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/assets/model.hpp"
#include "volumetric_kit/gfx/camera/camera_rig.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/io/gltf_loader.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_material.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_scene.hpp"
#include "volumetric_kit/gfx/windowing.hpp"

namespace vg = volumetric_kit::gfx;
namespace win = volumetric_kit::gfx::windowing;
namespace assets = volumetric_kit::gfx::assets;
namespace camera = volumetric_kit::gfx::camera;
namespace pipelines = volumetric_kit::gfx::pipelines;

namespace {

constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr float kFovY = 1.0471976f;  // 60 degrees
constexpr float kPi = 3.14159265359f;

// One thing to draw: a GPU mesh under a world transform (a glTF mesh may be
// instanced by several nodes, so the transform lives on the draw, not the
// mesh).
struct DrawItem {
  uint32_t mesh = 0;
  glm::mat4 world{1.0f};
};

// TODO: load_spirv + framebuffer_extent are duplicated across examples
// 01/02/03; hoist the shared pieces into an examples/common helper in a focused
// cleanup.
std::vector<uint32_t> load_spirv(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return {};
  }
  const std::streamsize size = file.tellg();
  if (size <= 0 || size % 4 != 0) {
    return {};
  }
  std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(code.data()), size);
  return file ? code : std::vector<uint32_t>{};
}

VkExtent2D framebuffer_extent(GLFWwindow* window) {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

// A unit cube with per-face normals (24 vertices, 12 triangles), so the example
// renders something without an external asset. One node at the origin, so it
// flows through the same scene walk as a loaded model.
assets::Model make_cube() {
  const glm::vec3 normals[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
  const glm::vec3 axis_u[6] = {{0, 0, -1}, {0, 0, 1}, {1, 0, 0},
                               {1, 0, 0},  {1, 0, 0}, {-1, 0, 0}};
  const glm::vec3 axis_v[6] = {{0, 1, 0}, {0, 1, 0}, {0, 0, -1},
                               {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
  assets::Mesh mesh;
  mesh.name = "cube";
  for (int f = 0; f < 6; ++f) {
    const auto base = static_cast<uint32_t>(mesh.vertices.size());
    for (int corner = 0; corner < 4; ++corner) {
      const float su = (corner == 1 || corner == 2) ? 0.5f : -0.5f;
      const float sv = (corner >= 2) ? 0.5f : -0.5f;
      assets::Vertex v;
      v.position = normals[f] * 0.5f + axis_u[f] * su + axis_v[f] * sv;
      v.normal = normals[f];
      v.tangent = glm::vec4(axis_u[f], 1.0f);  // w=+1: bitangent = n x t
      mesh.vertices.push_back(v);
    }
    const uint32_t quad[6] = {base, base + 1, base + 2,
                              base, base + 2, base + 3};
    mesh.indices.insert(mesh.indices.end(), quad, quad + 6);
  }

  assets::Model model;
  model.meshes.push_back(std::move(mesh));
  assets::Node node;
  node.mesh = 0;
  node.mesh_count = 1;
  model.scene.nodes.push_back(std::move(node));
  model.scene.roots.push_back(0);
  return model;
}

// Load `model_path` (or the built-in cube when null); null `*ok` on failure.
assets::Model load_model_or_cube(const char* model_path, bool* ok) {
  if (model_path == nullptr) {
    std::printf("03_model: no --model given; showing the built-in cube\n");
    return make_cube();
  }
  std::string error;
  auto loaded = vg::io::load_gltf(model_path, &error);
  if (!loaded) {
    std::fprintf(stderr, "load_gltf(%s): %s\n", model_path, error.c_str());
    *ok = false;
    return {};
  }
  std::printf("03_model: loaded '%s' (%zu mesh(es))\n", model_path,
              loaded->meshes.size());
  return std::move(*loaded);
}

// Walk the scene tree, composing each node's transform down to world space, and
// emit one DrawItem per (instanced) mesh primitive.
void collect_node(const assets::Model& model, uint32_t node_index,
                  const glm::mat4& parent, std::vector<DrawItem>& out,
                  std::vector<bool>& visited) {
  // Guard a malformed node graph: glTF requires a strict forest, but an
  // arbitrary --model file may not be conformant. An out-of-range or
  // already-visited index (a cycle) would otherwise recurse until the stack
  // overflows; skip it instead.
  if (node_index >= model.scene.nodes.size() || visited[node_index]) {
    return;
  }
  visited[node_index] = true;
  const assets::Node& node = model.scene.nodes[node_index];
  const glm::mat4 world = parent * node.transform;
  if (node.mesh != assets::Node::kNoMesh) {
    for (uint32_t k = 0; k < node.mesh_count; ++k) {
      out.push_back({node.mesh + k, world});
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

// World-space axis-aligned bounds over every drawn vertex, for camera framing.
struct Bounds {
  glm::vec3 min{0.0f};
  glm::vec3 max{0.0f};
  glm::vec3 center() const { return (min + max) * 0.5f; }
  float radius() const { return glm::length(max - min) * 0.5f; }
};

Bounds compute_bounds(const assets::Model& model,
                      const std::vector<DrawItem>& draws) {
  bool any = false;
  Bounds b;
  for (const DrawItem& draw : draws) {
    const assets::Mesh& mesh = model.meshes[draw.mesh];
    for (const assets::Vertex& v : mesh.vertices) {
      const glm::vec3 p = glm::vec3(draw.world * glm::vec4(v.position, 1.0f));
      if (!any) {
        b.min = p;
        b.max = p;
        any = true;
      } else {
        b.min = glm::min(b.min, p);
        b.max = glm::max(b.max, p);
      }
    }
  }
  if (!any) {  // no vertices anywhere: a unit box so framing stays finite
    b.min = glm::vec3(-0.5f);
    b.max = glm::vec3(0.5f);
  }
  return b;
}

// Frame the model's bounds with a three-quarter starting view: the rig looks
// slightly down at the bounds center from a distance that fits the bounding
// sphere. Near/far are fit separately by fit_clip (refit per frame while
// navigating); interactive input (or the --frames turntable) takes over from
// this pose.
void frame_camera(camera::CameraRig& rig, const Bounds& bounds) {
  const float radius = std::fmax(bounds.radius(), 1e-3f);
  const float distance = radius / std::sin(kFovY * 0.5f) * 1.3f;  // fit sphere
  // Eye offset from the center: a slight yaw for a three-quarter view, tilted
  // up so the camera looks down on the model.
  constexpr float kAzimuth = 0.7f;
  constexpr float kElevation = 0.35f;
  const float cos_e = std::cos(kElevation);
  const glm::vec3 offset(cos_e * std::sin(kAzimuth), std::sin(kElevation),
                         cos_e * std::cos(kAzimuth));
  rig.set_position(bounds.center() + offset * distance);
  rig.set_focus(bounds.center());  // aim at center; sets focus distance to it
}

// Near/far planes fitting the bounds sphere as seen from `eye`, recomputed as
// the camera moves so zoom/fly keep the model in view. Conservative: the
// Euclidean eye->center distance +/- the bounding radius (extra far slack),
// clamped so z_near stays positive when the eye is at or inside the sphere.
std::pair<float, float> fit_clip(const Bounds& bounds, const glm::vec3& eye) {
  const float radius = std::fmax(bounds.radius(), 1e-3f);
  const float distance = glm::length(eye - bounds.center());
  const float z_far = distance + radius * 4.0f;
  const float z_near = std::fmax(distance - radius, radius * 0.02f);
  return {z_near, z_far};
}

// Upload every non-empty mesh through the pipelines tier; the returned vector
// is parallel to model.meshes so a DrawItem's mesh index addresses it directly
// (empty primitives stay a default, skipped GpuMesh).
std::vector<pipelines::GpuMesh> upload_meshes(vg::Allocator& allocator,
                                              const assets::Model& model,
                                              bool* ok) {
  std::vector<pipelines::GpuMesh> gpu(model.meshes.size());
  for (size_t i = 0; i < model.meshes.size() && *ok; ++i) {
    const assets::Mesh& mesh = model.meshes[i];
    if (mesh.vertices.empty() || mesh.indices.empty()) {
      continue;
    }
    auto uploaded = pipelines::upload_mesh(allocator, mesh);
    if (!uploaded.ok()) {
      std::fprintf(stderr, "upload: %s\n", uploaded.status().message().c_str());
      *ok = false;
      return gpu;
    }
    gpu[i] = std::move(uploaded).value();
  }
  return gpu;
}

// Load + create one shader module from VG_EXAMPLE_SHADER_DIR by .spv name; null
// *ok on failure.
vg::ShaderModule load_shader(VkDevice device, const char* spv_name, bool* ok) {
  const std::vector<uint32_t> code =
      load_spirv((std::string(VG_EXAMPLE_SHADER_DIR "/") + spv_name).c_str());
  if (code.empty()) {
    std::fprintf(stderr, "missing compiled shader %s\n", spv_name);
    *ok = false;
    return {};
  }
  auto module = vg::ShaderModule::create(device, code.data(),
                                         code.size() * sizeof(uint32_t));
  if (!module.ok()) {
    std::fprintf(stderr, "shader module %s: %s\n", spv_name,
                 module.status().message().c_str());
    *ok = false;
    return {};
  }
  return std::move(module).value();
}

// Set a viewport + scissor covering the whole target (both are dynamic state).
// Shared by the skybox and model passes.
void set_full_viewport(VkCommandBuffer cmd, VkExtent2D extent) {
  VkViewport viewport{};
  viewport.width = static_cast<float>(extent.width);
  viewport.height = static_cast<float>(extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  VkRect2D scissor{};
  scissor.extent = extent;
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

VkClearColorValue background() {
  VkClearColorValue c{};
  c.float32[0] = 0.02f;
  c.float32[1] = 0.02f;
  c.float32[2] = 0.05f;
  c.float32[3] = 1.0f;
  return c;
}

// Write RGBA8 readback pixels out as a binary PPM (P6, RGB -- alpha dropped).
bool write_ppm(const char* path, const uint8_t* rgba, uint32_t width,
               uint32_t height) {
  std::FILE* file = std::fopen(path, "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "screenshot: cannot open '%s'\n", path);
    return false;
  }
  std::fprintf(file, "P6\n%u %u\n255\n", width, height);
  std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
  for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
    rgb[i * 3 + 0] = rgba[i * 4 + 0];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  const size_t wrote = std::fwrite(rgb.data(), 1, rgb.size(), file);
  std::fclose(file);
  return wrote == rgb.size();
}

void barrier_image(VkCommandBuffer cmd, VkImage image,
                   VkImageAspectFlags aspect, VkImageLayout new_layout,
                   VkAccessFlags dst_access, VkPipelineStageFlags dst_stage) {
  VkImageMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.dstAccessMask = dst_access;
  b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  b.newLayout = new_layout;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {aspect, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &b);
}

// Expand a decoded CPU image to tightly-packed RGBA8 (the layout upload_texture
// takes): pass 4-channel through, replicate 1/2-channel luminance into RGB, and
// pad 3-channel with opaque alpha -- most GPUs do not sample 3-channel 8-bit.
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

// Precomputed image-based-lighting textures, convolved on the CPU from the
// analytic sky (see make_ibl) and bound into the model's scene set (set 0): a
// diffuse irradiance cube, a roughness-prefiltered specular cube (mipped), and
// the BRDF integration LUT. Outlives the draw loop.
struct Ibl {
  std::optional<vg::Sampler> sampler;  // CLAMP_TO_EDGE, trilinear (mipped cube)
  vg::Texture irradiance;              // diffuse, small single-mip cube
  vg::Texture prefilter;               // specular, mipped cube
  vg::Texture brdf_lut;                // 2D RG integration LUT
  float prefilter_max_lod = 0.0f;      // prefilter mip count - 1
};

// All PBR GPU state built on the pipelines tier, outliving the draw loop: a
// shared sampler; every uploaded map plus 1x1 white / flat-normal fallbacks;
// the per-frame scene (set 0: camera + IBL) and one material (set 1) per glTF
// material, plus a fallback for material-less meshes. The textures back the
// materials' descriptors, so they are declared first (destroyed last).
struct PbrResources {
  std::vector<vg::Texture> textures;   // owns every uploaded map + fallbacks
  std::optional<vg::Sampler> sampler;  // filters the material maps
  pipelines::PbrScene scene;           // set 0 (camera + IBL), per frame
  std::vector<pipelines::PbrMaterial>
      materials;                             // set 1, parallel to materials
  pipelines::PbrMaterial fallback_material;  // set 1, material-less meshes
};

// Upload every material map (each image once, in the color space its slot
// needs), then build the pipelines-tier PbrScene (set 0) from the IBL and one
// PbrMaterial (set 1) per glTF material (plus a fallback for material-less
// meshes). The set 0 / set 1 layouts are reflected from the pipeline's shaders.
PbrResources setup_pbr(const vg::Device& device, vg::Allocator& alloc,
                       const pipelines::PbrPipeline& pipeline,
                       const assets::Model& model, const Ibl& ibl,
                       uint32_t frames_in_flight, bool* ok) {
  PbrResources r;
  *ok = true;  // output flag; cleared on the first failure below

  auto sampler = vg::Sampler::create(device.handle());
  if (!sampler.ok()) {
    std::fprintf(stderr, "sampler: %s\n", sampler.status().message().c_str());
    *ok = false;
    return r;
  }
  r.sampler = std::move(sampler).value();

  // Upload a tightly-packed RGBA8 image; returns its index into r.textures, or
  // -1 on failure.
  auto upload = [&](VkExtent2D ext, VkFormat fmt, const uint8_t* px, size_t sz,
                    bool mips) -> int {
    vg::ImageUploadDesc d;
    d.extent = ext;
    d.format = fmt;
    d.pixels = px;
    d.size = sz;
    d.generate_mips = mips;
    auto tex = vg::upload_texture(device, alloc, d);
    if (!tex.ok()) {
      std::fprintf(stderr, "texture upload: %s\n",
                   tex.status().message().c_str());
      return -1;
    }
    r.textures.push_back(std::move(tex).value());
    return static_cast<int>(r.textures.size()) - 1;
  };

  // Fallbacks: white (samples 1.0 for any non-normal slot, so the factor alone
  // applies) and flat-normal (0.5,0.5,1 -> (0,0,1): no perturbation).
  const uint8_t white_px[4] = {255, 255, 255, 255};
  const uint8_t flat_px[4] = {128, 128, 255, 255};
  const int white =
      upload({1, 1}, VK_FORMAT_R8G8B8A8_UNORM, white_px, 4, false);
  const int flat = upload({1, 1}, VK_FORMAT_R8G8B8A8_UNORM, flat_px, 4, false);
  if (white < 0 || flat < 0) {
    *ok = false;
    return r;
  }

  // Color space per image follows its slot: base-color + emissive are sRGB, the
  // rest linear. A glTF image fills one role in practice; if shared, sRGB wins.
  std::vector<bool> srgb(model.images.size(), false);
  for (const assets::Material& m : model.materials) {
    if (m.base_color_texture < srgb.size()) {
      srgb[m.base_color_texture] = true;
    }
    if (m.emissive_texture < srgb.size()) {
      srgb[m.emissive_texture] = true;
    }
  }

  // Upload each image once; image_tex maps a model.images index to r.textures.
  std::vector<int> image_tex(model.images.size(), -1);
  for (size_t i = 0; i < model.images.size(); ++i) {
    const assets::Image& img = model.images[i];
    if (!img.valid()) {
      continue;
    }
    const std::vector<uint8_t> rgba = to_rgba8(img);
    const VkFormat fmt =
        srgb[i] ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    image_tex[i] =
        upload({img.width, img.height}, fmt, rgba.data(), rgba.size(), true);
    if (image_tex[i] < 0) {
      *ok = false;
      return r;
    }
  }

  // Scene (set 0): the per-frame camera + the IBL maps, one UBO ring slot per
  // frame in flight. The caller refreshes the acquired slot's camera each
  // frame through r.scene.set_camera(slot, ...).
  pipelines::PbrSceneDesc scene_desc;
  scene_desc.irradiance = ibl.irradiance.view();
  scene_desc.prefilter = ibl.prefilter.view();
  scene_desc.brdf_lut = ibl.brdf_lut.view();
  scene_desc.sampler = ibl.sampler->handle();
  auto scene = pipelines::PbrScene::create(device.handle(), alloc,
                                           pipeline.descriptor_set_layout(0),
                                           scene_desc, frames_in_flight);
  if (!scene.ok()) {
    std::fprintf(stderr, "scene set: %s\n", scene.status().message().c_str());
    *ok = false;
    return r;
  }
  r.scene = std::move(scene).value();

  // Texture index for a slot, or the given fallback (kNoTexture is out of range
  // of image_tex, so it resolves to the fallback).
  auto tex_for = [&](uint32_t slot, int fallback) {
    return (slot < image_tex.size() && image_tex[slot] >= 0) ? image_tex[slot]
                                                             : fallback;
  };

  const VkDescriptorSetLayout material_layout =
      pipeline.descriptor_set_layout(1);

  // Build one PbrMaterial (set 1) from a fully-populated desc.
  auto make_material = [&](const pipelines::PbrMaterialDesc& desc,
                           bool* mat_ok) -> pipelines::PbrMaterial {
    auto mat = pipelines::PbrMaterial::create(device.handle(), alloc,
                                              material_layout, desc);
    if (!mat.ok()) {
      std::fprintf(stderr, "material: %s\n", mat.status().message().c_str());
      *mat_ok = false;
      return {};
    }
    return std::move(mat).value();
  };

  // Fallback material for meshes with no material: a matte white dielectric.
  {
    pipelines::PbrMaterialDesc d;
    d.base_color_factor = glm::vec4(1.0f);
    d.emissive_factor = glm::vec3(0.0f);
    d.metallic_factor = 0.0f;
    d.roughness_factor = 1.0f;
    d.base_color = r.textures[white].view();
    d.metallic_roughness = r.textures[white].view();
    d.normal = r.textures[flat].view();
    d.occlusion = r.textures[white].view();
    d.emissive = r.textures[white].view();
    d.sampler = r.sampler->handle();
    r.fallback_material = make_material(d, ok);
    if (!*ok) {
      return r;
    }
  }

  r.materials.reserve(model.materials.size());
  for (const assets::Material& m : model.materials) {
    pipelines::PbrMaterialDesc d;
    d.base_color_factor = m.base_color_factor;
    d.emissive_factor = m.emissive_factor;
    d.metallic_factor = m.metallic_factor;
    d.roughness_factor = m.roughness_factor;
    d.normal_scale = m.normal_scale;
    d.occlusion_strength = m.occlusion_strength;
    d.base_color = r.textures[tex_for(m.base_color_texture, white)].view();
    d.metallic_roughness =
        r.textures[tex_for(m.metallic_roughness_texture, white)].view();
    d.normal = r.textures[tex_for(m.normal_texture, flat)].view();
    d.occlusion = r.textures[tex_for(m.occlusion_texture, white)].view();
    d.emissive = r.textures[tex_for(m.emissive_texture, white)].view();
    d.sampler = r.sampler->handle();
    r.materials.push_back(make_material(d, ok));
    if (!*ok) {
      return r;
    }
  }

  return r;
}

// Resolve each collected DrawItem into a PbrDraw: its GPU mesh, world
// transform, and the material its mesh uses (or the fallback). The returned
// draws borrow `meshes` and `pbr`, which must outlive them.
std::vector<pipelines::PbrDraw> build_pbr_draws(
    const assets::Model& model, const std::vector<pipelines::GpuMesh>& meshes,
    const PbrResources& pbr, const std::vector<DrawItem>& draws) {
  std::vector<pipelines::PbrDraw> out;
  out.reserve(draws.size());
  for (const DrawItem& d : draws) {
    const uint32_t mat = model.meshes[d.mesh].material;
    const pipelines::PbrMaterial* material =
        (mat != assets::Mesh::kNoMaterial && mat < pbr.materials.size())
            ? &pbr.materials[mat]
            : &pbr.fallback_material;
    out.push_back({&meshes[d.mesh], d.world, material});
  }
  return out;
}

// --- Skybox: a procedural environment cubemap drawn behind the model --------

// Analytic sky in linear HDR RGB: a zenith->horizon->ground vertical gradient
// plus a tight, bright (> 1) sun toward the key-light direction -- the same
// environment the IBL bake convolves its ambient from. Tone-mapped on output.
glm::vec3 sky_color(const glm::vec3& dir) {
  const glm::vec3 sun_dir = glm::normalize(glm::vec3(0.5f, 0.8f, 0.6f));
  const glm::vec3 zenith(0.12f, 0.22f, 0.42f);
  const glm::vec3 horizon(0.52f, 0.58f, 0.66f);
  const glm::vec3 ground(0.10f, 0.09f, 0.08f);
  const float t = glm::clamp(dir.y, -1.0f, 1.0f);
  const glm::vec3 base = t >= 0.0f
                             ? glm::mix(horizon, zenith, std::pow(t, 0.5f))
                             : glm::mix(horizon, ground, std::pow(-t, 0.4f));
  // A tight, bright HDR sun reads as a highlight and drives crisp specular
  // reflections through the IBL prefilter; tone mapping pulls it back in range.
  const float sun = std::pow(std::fmax(glm::dot(dir, sun_dir), 0.0f), 200.0f);
  return base + glm::vec3(1.0f, 0.95f, 0.85f) * (sun * 20.0f);
}

// World direction for cube face `f` (Vulkan layer order +X,-X,+Y,-Y,+Z,-Z) at
// face coordinates u, v in [-1, 1].
glm::vec3 cube_dir(int f, float u, float v) {
  switch (f) {
    case 0:
      return glm::normalize(glm::vec3(1.0f, -v, -u));
    case 1:
      return glm::normalize(glm::vec3(-1.0f, -v, u));
    case 2:
      return glm::normalize(glm::vec3(u, 1.0f, v));
    case 3:
      return glm::normalize(glm::vec3(u, -1.0f, -v));
    case 4:
      return glm::normalize(glm::vec3(u, -v, 1.0f));
    default:
      return glm::normalize(glm::vec3(-u, -v, -1.0f));
  }
}

// Bake the analytic sky into a sampled-ready cubemap: generate the six faces on
// the CPU, stage them, and copy all six layers in one submit. Stores linear HDR
// color in a float cube (the skybox shader tone-maps it on output).
vg::Texture make_sky_cube(const vg::Device& device, vg::Allocator& alloc,
                          uint32_t size, bool* ok) {
  // RGBA16F (half) staging: 16-bit float filters on the broad device set (incl.
  // MoltenVK/Metal); RGBA32F linear filtering is an optional feature many GPUs
  // lack. Unclamped HDR (the skybox tone-maps on output); 2 uint32/texel.
  std::vector<uint32_t> pixels;
  pixels.reserve(static_cast<size_t>(size) * size * 6 * 2);
  for (int f = 0; f < 6; ++f) {
    for (uint32_t y = 0; y < size; ++y) {
      for (uint32_t x = 0; x < size; ++x) {
        const float u = (static_cast<float>(x) + 0.5f) / size * 2.0f - 1.0f;
        const float v = (static_cast<float>(y) + 0.5f) / size * 2.0f - 1.0f;
        const glm::vec3 c = sky_color(cube_dir(f, u, v));
        pixels.push_back(glm::packHalf2x16(glm::vec2(c.x, c.y)));
        pixels.push_back(glm::packHalf2x16(glm::vec2(c.z, 1.0f)));
      }
    }
  }

  vg::BufferDesc sd;
  sd.size = pixels.size() * sizeof(uint32_t);
  sd.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  sd.memory = vg::MemoryUsage::HostVisible;
  sd.mapped = true;
  sd.host_access = vg::HostAccess::SequentialWrite;
  auto staging = alloc.create_buffer(sd);
  if (!staging.ok()) {
    std::fprintf(stderr, "sky staging: %s\n",
                 staging.status().message().c_str());
    *ok = false;
    return {};
  }
  std::memcpy(staging.value().mapped(), pixels.data(),
              pixels.size() * sizeof(uint32_t));

  vg::TextureDesc td;
  td.extent = {size, size};
  td.format = VK_FORMAT_R16G16B16A16_SFLOAT;  // linear HDR, broadly filterable
  td.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  td.array_layers = 6;
  td.cube = true;
  auto cube = alloc.create_image(td);
  if (!cube.ok()) {
    std::fprintf(stderr, "sky cube: %s\n", cube.status().message().c_str());
    *ok = false;
    return {};
  }

  // Upload all six faces in one submit. Hand-rolled because the core
  // upload_texture helper is single-layer only.
  // TODO: extend upload_texture/ImageUploadDesc with array_layers (and a
  // layer_count on cmd_image_barrier) so cube/array uploads reuse one path.
  const VkImage image = cube.value().image();
  const VkBuffer src = staging.value().handle();
  const vg::Status copied =
      device.submit_single_time([image, src, size](VkCommandBuffer cmd) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &barrier);

        VkBufferImageCopy copies[6]{};
        for (uint32_t f = 0; f < 6; ++f) {
          copies[f].bufferOffset =
              VkDeviceSize{f} * size * size * 2 * sizeof(uint32_t);
          copies[f].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, f, 1};
          copies[f].imageExtent = {size, size, 1};
        }
        vkCmdCopyBufferToImage(cmd, src, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, copies);

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
      });
  if (!copied.ok()) {
    std::fprintf(stderr, "sky upload: %s\n", copied.message().c_str());
    *ok = false;
    return {};
  }
  return std::move(cube).value();
}

// Per-frame skybox transform (fragment push constant); mirrors skybox.frag.
struct SkyboxPush {
  glm::mat4 inv_view_proj;
  glm::vec4 camera_pos;
};

// The environment cubemap plus the pipeline/descriptor that draws it.
struct Skybox {
  std::optional<vg::Sampler> sampler;  // no public default ctor (see #47)
  vg::Texture cube;
  vg::GraphicsPipeline pipeline;
  vg::DescriptorPool pool;
  vg::DescriptorSet set;  // set 0: the samplerCube
};

vg::Result<vg::GraphicsPipeline> build_skybox_pipeline(
    VkDevice device, const vg::ShaderModule& vert, const vg::ShaderModule& frag,
    const vg::RenderTargetLayout& layout) {
  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = &vert;
  desc.fragment_shader = &frag;
  desc.layout = layout;
  // Procedural full-screen triangle (no vertex input). depth_test/depth_write
  // default to false, which is exactly what the skybox wants: it is drawn first
  // and fills the whole frame, then the model (depth-tested) overdraws it.
  return vg::GraphicsPipeline::create(device, desc);
}

// Bake the environment cube + build the skybox pipeline and its descriptor set.
Skybox setup_skybox(const vg::Device& device, vg::Allocator& alloc,
                    const vg::RenderTargetLayout& layout, bool* ok) {
  *ok = true;  // output flag; cleared on the first failure below
  Skybox s;

  vg::SamplerDesc sampler_desc;
  sampler_desc.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_desc.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_desc.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  auto sampler = vg::Sampler::create(device.handle(), sampler_desc);
  if (!sampler.ok()) {
    std::fprintf(stderr, "sky sampler: %s\n",
                 sampler.status().message().c_str());
    *ok = false;
    return s;
  }
  s.sampler = std::move(sampler).value();

  s.cube = make_sky_cube(device, alloc, 128, ok);
  if (!*ok) {
    return s;
  }

  const vg::ShaderModule vert =
      load_shader(device.handle(), "skybox.vert.spv", ok);
  const vg::ShaderModule frag =
      load_shader(device.handle(), "skybox.frag.spv", ok);
  if (!*ok) {
    return s;
  }
  auto pipeline = build_skybox_pipeline(device.handle(), vert, frag, layout);
  if (!pipeline.ok()) {
    std::fprintf(stderr, "skybox pipeline: %s\n",
                 pipeline.status().message().c_str());
    *ok = false;
    return s;
  }
  s.pipeline = std::move(pipeline).value();

  const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  auto pool = vg::DescriptorPool::create(device.handle(), &size, 1, 1);
  if (!pool.ok()) {
    std::fprintf(stderr, "skybox pool: %s\n", pool.status().message().c_str());
    *ok = false;
    return s;
  }
  s.pool = std::move(pool).value();
  auto set = s.pool.allocate(s.pipeline.descriptor_set_layout(0));
  if (!set.ok()) {
    std::fprintf(stderr, "skybox set: %s\n", set.status().message().c_str());
    *ok = false;
    return s;
  }
  s.set = std::move(set).value();
  s.set.write_combined_image_sampler(0, s.cube.view(), s.sampler->handle(),
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  return s;
}

// Draw the skybox: a full-screen triangle sampling the environment along the
// per-pixel view ray. Call inside the render scope, before the model.
void record_skybox(VkCommandBuffer cmd, VkExtent2D extent, const Skybox& skybox,
                   const glm::mat4& view_proj, const glm::vec3& camera_pos) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    skybox.pipeline.handle());
  set_full_viewport(cmd, extent);

  SkyboxPush push;
  push.inv_view_proj = glm::inverse(view_proj);
  push.camera_pos = glm::vec4(camera_pos, 1.0f);
  vkCmdPushConstants(cmd, skybox.pipeline.layout(),
                     VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
  const VkDescriptorSet set = skybox.set.handle();
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          skybox.pipeline.layout(), 0, 1, &set, 0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
}

// --- IBL: convolve the analytic sky into diffuse/specular/BRDF textures ------

// Hammersley low-discrepancy 2D sample (van der Corput radical inverse).
glm::vec2 hammersley(uint32_t i, uint32_t n) {
  uint32_t bits = i;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  const float radical = static_cast<float>(bits) * 2.3283064365386963e-10f;
  return glm::vec2(static_cast<float>(i) / static_cast<float>(n), radical);
}

// GGX-importance-sampled half-vector around `n` for `roughness`.
glm::vec3 importance_sample_ggx(glm::vec2 xi, const glm::vec3& n,
                                float roughness) {
  const float a = roughness * roughness;
  const float phi = 2.0f * kPi * xi.x;
  const float cos_t = std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
  const float sin_t = std::sqrt(1.0f - cos_t * cos_t);
  const glm::vec3 h(std::cos(phi) * sin_t, std::sin(phi) * sin_t, cos_t);
  const glm::vec3 up =
      std::abs(n.z) < 0.999f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
  const glm::vec3 tangent = glm::normalize(glm::cross(up, n));
  const glm::vec3 bitangent = glm::cross(n, tangent);
  return glm::normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

// Smith geometry with the IBL roughness remap (k = roughness^2 / 2 -- the
// material roughness, per Karis; not the squared GGX alpha).
float geometry_smith_ibl(float n_dot_v, float n_dot_l, float roughness) {
  const float k = roughness * roughness / 2.0f;
  const float gv = n_dot_v / (n_dot_v * (1.0f - k) + k);
  const float gl = n_dot_l / (n_dot_l * (1.0f - k) + k);
  return gv * gl;
}

// Cosine-weighted hemisphere integral of the analytic sky around `n`: the
// diffuse irradiance for that normal. PI is folded in (LearnOpenGL form), so
// the shader's diffuse term is just irradiance * albedo.
glm::vec4 irradiance_at(const glm::vec3& n) {
  glm::vec3 up =
      std::abs(n.y) < 0.999f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
  const glm::vec3 right = glm::normalize(glm::cross(up, n));
  up = glm::cross(n, right);
  glm::vec3 sum(0.0f);
  int samples = 0;
  // ~0.1 rad hemisphere step (~63 x 16 taps); coarse vs the tight HDR sun (mild
  // diffuse banding), but adequate for the low-frequency diffuse fill.
  for (float phi = 0.0f; phi < 2.0f * kPi; phi += 0.1f) {
    for (float theta = 0.0f; theta < 0.5f * kPi; theta += 0.1f) {
      const float st = std::sin(theta);
      const glm::vec3 tan_sample(st * std::cos(phi), st * std::sin(phi),
                                 std::cos(theta));
      const glm::vec3 world =
          right * tan_sample.x + up * tan_sample.y + n * tan_sample.z;
      sum += sky_color(world) * std::cos(theta) * st;
      ++samples;
    }
  }
  return glm::vec4(kPi * sum / static_cast<float>(samples), 1.0f);
}

// GGX-prefiltered specular radiance from `r` at `roughness`: the environment
// blurred for that gloss level (the split sum's L_i term).
glm::vec4 prefilter_at(const glm::vec3& r, float roughness, uint32_t samples) {
  const glm::vec3 n = r;
  const glm::vec3 v = r;
  glm::vec3 sum(0.0f);
  float weight = 0.0f;
  for (uint32_t i = 0; i < samples; ++i) {
    const glm::vec3 h =
        importance_sample_ggx(hammersley(i, samples), n, roughness);
    const glm::vec3 l = glm::normalize(2.0f * glm::dot(v, h) * h - v);
    const float n_dot_l = glm::dot(n, l);
    if (n_dot_l > 0.0f) {
      sum += sky_color(l) * n_dot_l;
      weight += n_dot_l;
    }
  }
  return glm::vec4(weight > 0.0f ? sum / weight : sky_color(r), 1.0f);
}

// Environment-BRDF integration (scale, bias) for the split sum, per (NdotV,
// roughness). Independent of the environment -- the standard BRDF LUT.
glm::vec2 brdf_integrate(float n_dot_v, float roughness, uint32_t samples) {
  const glm::vec3 v(std::sqrt(1.0f - n_dot_v * n_dot_v), 0.0f, n_dot_v);
  const glm::vec3 n(0.0f, 0.0f, 1.0f);
  float a = 0.0f;
  float b = 0.0f;
  for (uint32_t i = 0; i < samples; ++i) {
    const glm::vec3 h =
        importance_sample_ggx(hammersley(i, samples), n, roughness);
    const glm::vec3 l = glm::normalize(2.0f * glm::dot(v, h) * h - v);
    const float n_dot_l = std::fmax(l.z, 0.0f);
    const float n_dot_h = std::fmax(h.z, 0.0f);
    const float v_dot_h = std::fmax(glm::dot(v, h), 0.0f);
    if (n_dot_l > 0.0f) {
      const float g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
      const float g_vis = (g * v_dot_h) / std::fmax(n_dot_h * n_dot_v, 1e-6f);
      const float fc = std::pow(1.0f - v_dot_h, 5.0f);
      a += (1.0f - fc) * g_vis;
      b += fc * g_vis;
    }
  }
  return glm::vec2(a / static_cast<float>(samples),
                   b / static_cast<float>(samples));
}

// Create a (possibly mipped) float cube and fill every (mip, face) from `gen`,
// which returns that subresource's RGBA pixels; upload all subresources in one
// staged submit.
template <class Gen>
vg::Texture upload_cube(const vg::Device& device, vg::Allocator& alloc,
                        uint32_t base_size, uint32_t mips, VkFormat format,
                        Gen gen, bool* ok) {
  struct Region {
    VkDeviceSize offset;
    uint32_t mip;
    uint32_t face;
    uint32_t size;
  };
  // Pack gen()'s float pixels to RGBA16F (2 uint32 = 8 bytes/texel). 16-bit
  // float cubes filter on the broad device set (incl. MoltenVK/Metal); RGBA32F
  // linear filtering is an optional feature many GPUs lack. Each face starts on
  // an 8-byte (texel-block) aligned offset since every texel is 2 uint32.
  std::vector<Region> regions;
  std::vector<uint32_t> data;
  for (uint32_t m = 0; m < mips; ++m) {
    const uint32_t size = (base_size >> m) > 0 ? (base_size >> m) : 1u;
    for (uint32_t f = 0; f < 6; ++f) {
      const std::vector<glm::vec4> face = gen(m, static_cast<int>(f), size);
      regions.push_back(
          {VkDeviceSize{data.size()} * sizeof(uint32_t), m, f, size});
      for (const glm::vec4& px : face) {
        data.push_back(glm::packHalf2x16(glm::vec2(px.x, px.y)));
        data.push_back(glm::packHalf2x16(glm::vec2(px.z, px.w)));
      }
    }
  }
  const VkDeviceSize total = VkDeviceSize{data.size()} * sizeof(uint32_t);

  vg::BufferDesc sd;
  sd.size = total;
  sd.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  sd.memory = vg::MemoryUsage::HostVisible;
  sd.mapped = true;
  sd.host_access = vg::HostAccess::SequentialWrite;
  auto staging = alloc.create_buffer(sd);
  if (!staging.ok()) {
    std::fprintf(stderr, "ibl staging: %s\n",
                 staging.status().message().c_str());
    *ok = false;
    return {};
  }
  std::memcpy(staging.value().mapped(), data.data(), total);

  vg::TextureDesc td;
  td.extent = {base_size, base_size};
  td.format = format;
  td.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  td.array_layers = 6;
  td.cube = true;
  td.mip_levels = mips;
  auto cube = alloc.create_image(td);
  if (!cube.ok()) {
    std::fprintf(stderr, "ibl cube: %s\n", cube.status().message().c_str());
    *ok = false;
    return {};
  }

  const VkImage image = cube.value().image();
  const VkBuffer src = staging.value().handle();
  const vg::Status copied = device.submit_single_time([&](VkCommandBuffer cmd) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 6};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    std::vector<VkBufferImageCopy> copies;
    copies.reserve(regions.size());
    for (const Region& region : regions) {
      VkBufferImageCopy copy{};
      copy.bufferOffset = region.offset;
      copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, region.mip,
                               region.face, 1};
      copy.imageExtent = {region.size, region.size, 1};
      copies.push_back(copy);
    }
    vkCmdCopyBufferToImage(cmd, src, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(copies.size()), copies.data());

    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
  });
  if (!copied.ok()) {
    std::fprintf(stderr, "ibl upload: %s\n", copied.message().c_str());
    *ok = false;
    return {};
  }
  return std::move(cube).value();
}

// Convolve the analytic sky into the IBL texture set. CPU-side because the
// environment is analytic; a loaded HDR environment would convolve on the GPU.
Ibl make_ibl(const vg::Device& device, vg::Allocator& alloc, bool* ok) {
  Ibl ibl;
  *ok = true;

  vg::SamplerDesc sd;
  sd.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sd.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sd.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  auto sampler = vg::Sampler::create(device.handle(), sd);
  if (!sampler.ok()) {
    std::fprintf(stderr, "ibl sampler: %s\n",
                 sampler.status().message().c_str());
    *ok = false;
    return ibl;
  }
  ibl.sampler = std::move(sampler).value();

  // Diffuse irradiance: 16x16 single-mip cube -- ample for the low-frequency,
  // heavily-blurred cosine convolution.
  ibl.irradiance = upload_cube(
      device, alloc, 16, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
      [](uint32_t, int face, uint32_t size) {
        std::vector<glm::vec4> px(static_cast<size_t>(size) * size);
        for (uint32_t y = 0; y < size; ++y) {
          for (uint32_t x = 0; x < size; ++x) {
            const float u = (x + 0.5f) / size * 2.0f - 1.0f;
            const float v = (y + 0.5f) / size * 2.0f - 1.0f;
            px[y * size + x] = irradiance_at(cube_dir(face, u, v));
          }
        }
        return px;
      },
      ok);
  if (!*ok) {
    return ibl;
  }

  // Prefiltered specular: a 64x64 base over kPrefilterMips mips maps mip ->
  // roughness 0..1; 64 GGX samples/texel (the tight HDR sun can alias on low
  // mips -- accepted for the example).
  constexpr uint32_t kPrefilterMips = 5;
  ibl.prefilter = upload_cube(
      device, alloc, 64, kPrefilterMips, VK_FORMAT_R16G16B16A16_SFLOAT,
      [](uint32_t mip, int face, uint32_t size) {
        const float roughness =
            static_cast<float>(mip) / static_cast<float>(kPrefilterMips - 1);
        std::vector<glm::vec4> px(static_cast<size_t>(size) * size);
        for (uint32_t y = 0; y < size; ++y) {
          for (uint32_t x = 0; x < size; ++x) {
            const float u = (x + 0.5f) / size * 2.0f - 1.0f;
            const float v = (y + 0.5f) / size * 2.0f - 1.0f;
            px[y * size + x] =
                prefilter_at(cube_dir(face, u, v), roughness, 64u);
          }
        }
        return px;
      },
      ok);
  if (!*ok) {
    return ibl;
  }
  ibl.prefilter_max_lod = static_cast<float>(kPrefilterMips - 1);

  // BRDF integration LUT (2D RG16F), environment-independent. 128x128 with 256
  // importance samples/texel is the standard split-sum table resolution.
  // TODO: this table never changes -- bake it to an asset rather than
  // recomputing it (~4M importance samples) on every launch.
  constexpr uint32_t kLutSize = 128;
  std::vector<uint32_t> lut(static_cast<size_t>(kLutSize) * kLutSize);
  for (uint32_t y = 0; y < kLutSize; ++y) {
    for (uint32_t x = 0; x < kLutSize; ++x) {
      lut[y * kLutSize + x] = glm::packHalf2x16(
          brdf_integrate((x + 0.5f) / kLutSize, (y + 0.5f) / kLutSize, 256u));
    }
  }
  vg::ImageUploadDesc lut_desc;
  lut_desc.extent = {kLutSize, kLutSize};
  lut_desc.format = VK_FORMAT_R16G16_SFLOAT;
  lut_desc.pixels = lut.data();
  lut_desc.size = lut.size() * sizeof(uint32_t);
  auto brdf = vg::upload_texture(device, alloc, lut_desc);
  if (!brdf.ok()) {
    std::fprintf(stderr, "brdf lut: %s\n", brdf.status().message().c_str());
    *ok = false;
    return ibl;
  }
  ibl.brdf_lut = std::move(brdf).value();
  return ibl;
}

// --- Headless path: render one frame into an OffscreenTarget, write a PPM
// -----

int run_screenshot(const char* model_path, const char* out_path, uint32_t width,
                   uint32_t height) {
  vg::InstanceConfig instance_config;
  instance_config.app_name = "03_model";
  instance_config.enable_validation = true;  // a no-op when the layer is absent
  auto instance = vg::Instance::create(instance_config);
  if (!instance.ok()) {
    std::fprintf(stderr, "instance: %s\n", instance.status().message().c_str());
    return 1;
  }
  auto physical = instance.value().select_physical_device();  // no surface
  if (!physical.ok()) {
    std::fprintf(stderr, "device: %s\n", physical.status().message().c_str());
    return 1;
  }
  auto device = vg::Device::create(instance.value().handle(), physical.value(),
                                   vg::DeviceConfig{});  // headless, no present
  if (!device.ok()) {
    std::fprintf(stderr, "device: %s\n", device.status().message().c_str());
    return 1;
  }
  auto allocator =
      vg::Allocator::create(instance.value().handle(), device.value());
  if (!allocator.ok()) {
    std::fprintf(stderr, "allocator: %s\n",
                 allocator.status().message().c_str());
    return 1;
  }

  bool ok = true;
  const assets::Model model = load_model_or_cube(model_path, &ok);
  if (!ok) {
    return 1;
  }
  const std::vector<DrawItem> draws = collect_draws(model);
  const std::vector<pipelines::GpuMesh> meshes =
      upload_meshes(allocator.value(), model, &ok);
  if (!ok) {
    return 1;
  }

  const Bounds bounds = compute_bounds(model, draws);
  camera::CameraRig rig;
  frame_camera(rig, bounds);
  const std::pair<float, float> clip = fit_clip(bounds, rig.position());

  // sRGB color so the encoded readback matches the windowed (sRGB) look.
  vg::OffscreenTargetDesc target_desc;
  target_desc.extent = {width, height};
  target_desc.color_format = VK_FORMAT_R8G8B8A8_SRGB;
  target_desc.depth_format = kDepthFormat;
  auto target = vg::OffscreenTarget::create(allocator.value(), target_desc);
  if (!target.ok()) {
    std::fprintf(stderr, "offscreen: %s\n", target.status().message().c_str());
    return 1;
  }

  auto pipeline = pipelines::PbrPipeline::create(device.value().handle(),
                                                 target.value().layout());
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  Ibl ibl = make_ibl(device.value(), allocator.value(), &ok);
  if (!ok) {
    return 1;
  }
  PbrResources pbr =
      setup_pbr(device.value(), allocator.value(), pipeline.value(), model, ibl,
                /*frames_in_flight=*/1, &ok);
  if (!ok) {
    return 1;
  }

  // Fixed camera for the still: write the eye into the scene set once.
  pbr.scene.set_camera(0, rig.position(), ibl.prefilter_max_lod);
  const std::vector<pipelines::PbrDraw> pbr_draws =
      build_pbr_draws(model, meshes, pbr, draws);

  Skybox skybox = setup_skybox(device.value(), allocator.value(),
                               target.value().layout(), &ok);
  if (!ok) {
    return 1;
  }

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const glm::mat4 view_proj =
      rig.to_camera(kFovY, aspect, clip.first, clip.second).view_proj();

  const vg::Status recorded =
      device.value().submit_single_time([&](VkCommandBuffer cmd) {
        barrier_image(cmd, target.value().color_image(),
                      VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        barrier_image(cmd, target.value().depth_image(),
                      VK_IMAGE_ASPECT_DEPTH_BIT,
                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

        vg::RenderTargetBeginInfo begin;
        begin.clear_color = background();
        const vg::RenderTarget rt = target.value().target();
        rt.begin(cmd, begin);
        record_skybox(cmd, {width, height}, skybox, view_proj, rig.position());
        pipelines::PbrFrame frame;
        frame.extent = {width, height};
        frame.view_proj = view_proj;
        frame.scene = &pbr.scene;
        frame.draws = pbr_draws.data();
        frame.draw_count = static_cast<uint32_t>(pbr_draws.size());
        pipeline.value().submit(cmd, frame);
        rt.end(cmd);
        target.value().record_readback(cmd);
      });
  if (!recorded.ok()) {
    std::fprintf(stderr, "render: %s\n", recorded.message().c_str());
    return 1;
  }

  if (!write_ppm(out_path, static_cast<const uint8_t*>(target.value().pixels()),
                 width, height)) {
    return 1;
  }
  std::printf("03_model: wrote %ux%u screenshot to '%s'\n", width, height,
              out_path);
  return 0;
}

// --- Interactive camera input: map mouse + keyboard onto CameraRig verbs -----

// Per-window input state, registered as the GLFW user pointer so the scroll
// callback (the one event GLFW has no per-frame poll for) can accumulate into
// it. Everything else is polled each frame in apply_input.
struct InputState {
  double last_x = 0.0;
  double last_y = 0.0;
  bool has_cursor = false;  // seed the delta on the first frame, so no jump
  double scroll = 0.0;      // wheel notches accumulated since the last apply
  double last_time = 0.0;   // for a per-frame dt
};

void scroll_callback(GLFWwindow* window, double /*x_offset*/, double y_offset) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input != nullptr) {
    input->scroll += y_offset;
  }
}

// Drive the rig from this frame's input: left-drag orbits, right/middle-drag
// pans (scaled by focus distance so it tracks the cursor at any zoom), the
// wheel zooms, and WASDQE flies (Q/E = down/up, Shift = faster). move_speed
// scales the fly speed to the model size so it feels right for any model.
void apply_input(GLFWwindow* window, InputState& input, camera::CameraRig& rig,
                 float move_speed) {
  const double now = glfwGetTime();
  float dt =
      input.last_time > 0.0 ? static_cast<float>(now - input.last_time) : 0.0f;
  input.last_time = now;
  dt = std::fmin(dt, 0.1f);  // clamp a long stall (e.g. dragging the title bar)

  double x = 0.0;
  double y = 0.0;
  glfwGetCursorPos(window, &x, &y);
  if (!input.has_cursor) {  // first frame: no delta, just latch the position
    input.last_x = x;
    input.last_y = y;
    input.has_cursor = true;
  }
  const float dx = static_cast<float>(x - input.last_x);
  const float dy = static_cast<float>(y - input.last_y);
  input.last_x = x;
  input.last_y = y;

  constexpr float kOrbitSpeed = 0.005f;  // radians / pixel
  constexpr float kPanSpeed = 0.0015f;   // world units / pixel, per focus unit
  constexpr float kZoomStep = 0.9f;      // multiplier / wheel notch

  const bool left =
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  const bool panning =
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  if (left) {
    rig.orbit(-dx * kOrbitSpeed, -dy * kOrbitSpeed);
  } else if (panning) {
    const float scale = kPanSpeed * rig.focus_distance();
    rig.pan(-dx * scale, dy * scale);
  }

  if (input.scroll != 0.0) {
    rig.zoom(std::pow(kZoomStep, static_cast<float>(input.scroll)));
    input.scroll = 0.0;
  }

  glm::vec3 move(0.0f);
  move.x += glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ? 1.0f : 0.0f;
  move.x -= glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ? 1.0f : 0.0f;
  move.y += glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS ? 1.0f : 0.0f;
  move.y -= glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS ? 1.0f : 0.0f;
  move.z += glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ? 1.0f : 0.0f;
  move.z -= glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ? 1.0f : 0.0f;
  if (glm::dot(move, move) > 0.0f) {  // a movement key is held
    const bool fast = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    rig.move_local(glm::normalize(move) * move_speed * (fast ? 4.0f : 1.0f) *
                   dt);
  }
}

// --- Windowed path: Surface + Swapchain + FrameLoop --------------------------

// Owns all Vulkan/windowing state for one window; everything is destroyed when
// this returns, before main() tears GLFW down. Interactive by default; with
// max_frames >= 0 it runs a deterministic turntable and exits (for CI).
int run_windowed(GLFWwindow* window, const char* model_path, int max_frames) {
  uint32_t glfw_ext_count = 0;
  const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  vg::InstanceConfig instance_config;
  instance_config.app_name = "03_model";
  instance_config.enable_validation = true;  // a no-op when the layer is absent
  instance_config.extra_instance_extensions.assign(glfw_exts,
                                                   glfw_exts + glfw_ext_count);
  auto instance = vg::Instance::create(instance_config);
  if (!instance.ok()) {
    std::fprintf(stderr, "instance: %s\n", instance.status().message().c_str());
    return 1;
  }

  VkSurfaceKHR raw_surface = VK_NULL_HANDLE;
  if (glfwCreateWindowSurface(instance.value().handle(), window, nullptr,
                              &raw_surface) != VK_SUCCESS) {
    std::fprintf(stderr, "glfwCreateWindowSurface failed\n");
    return 1;
  }
  win::Surface surface(instance.value().handle(), raw_surface);

  auto physical = instance.value().select_physical_device(surface.handle());
  if (!physical.ok()) {
    std::fprintf(stderr, "device: %s\n", physical.status().message().c_str());
    return 1;
  }

  vg::DeviceConfig device_config;
  device_config.needs_present = true;
  auto device = vg::Device::create(instance.value().handle(), physical.value(),
                                   device_config, surface.handle());
  if (!device.ok()) {
    std::fprintf(stderr, "device: %s\n", device.status().message().c_str());
    return 1;
  }

  auto allocator =
      vg::Allocator::create(instance.value().handle(), device.value());
  if (!allocator.ok()) {
    std::fprintf(stderr, "allocator: %s\n",
                 allocator.status().message().c_str());
    return 1;
  }

  bool ok = true;
  const assets::Model model = load_model_or_cube(model_path, &ok);
  if (!ok) {
    return 1;
  }
  const std::vector<DrawItem> draws = collect_draws(model);
  const std::vector<pipelines::GpuMesh> meshes =
      upload_meshes(allocator.value(), model, &ok);
  if (!ok) {
    return 1;
  }

  const Bounds bounds = compute_bounds(model, draws);
  camera::CameraRig rig;
  frame_camera(rig, bounds);
  // Interactive by default; a deterministic turntable when --frames is given,
  // so CI renders a reproducible sequence. Fly speed scales to the model size.
  const bool interactive = max_frames < 0;
  const float move_speed = std::fmax(bounds.radius(), 1e-3f) * 1.5f;
  InputState input;
  if (interactive) {
    glfwSetWindowUserPointer(window, &input);
    glfwSetScrollCallback(window, scroll_callback);
  }

  // The swapchain owns a depth attachment per image (rebuilt with the chain on
  // resize), so its render targets are depth-capable and frames in flight never
  // share a depth image. The allocator (created above) is borrowed and must
  // outlive the swapchain.
  win::SwapchainConfig swapchain_config;
  swapchain_config.extent = framebuffer_extent(window);
  swapchain_config.depth_format = kDepthFormat;
  auto swapchain = win::Swapchain::create(device.value(), surface.handle(),
                                          swapchain_config, &allocator.value());
  if (!swapchain.ok()) {
    std::fprintf(stderr, "swapchain: %s\n",
                 swapchain.status().message().c_str());
    return 1;
  }

  // The pipeline is built for the swapchain's layout (color + depth formats).
  // Size-independent (dynamic viewport), so it survives resizes without a
  // rebuild.
  auto pipeline = pipelines::PbrPipeline::create(device.value().handle(),
                                                 swapchain.value().layout());
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  // Two frames in flight: the swapchain keeps depth per image and the scene
  // UBO rings per slot, so nothing is shared across in-flight frames.
  constexpr uint32_t kFramesInFlight = 2;
  Ibl ibl = make_ibl(device.value(), allocator.value(), &ok);
  if (!ok) {
    return 1;
  }
  PbrResources pbr =
      setup_pbr(device.value(), allocator.value(), pipeline.value(), model, ibl,
                kFramesInFlight, &ok);
  if (!ok) {
    return 1;
  }
  const std::vector<pipelines::PbrDraw> pbr_draws =
      build_pbr_draws(model, meshes, pbr, draws);

  Skybox skybox = setup_skybox(device.value(), allocator.value(),
                               swapchain.value().layout(), &ok);
  if (!ok) {
    return 1;
  }

  // CPU-ahead depth shared by the loop and the profiler driving it (declared
  // above, before setup_pbr, so the scene UBO ring matches).
  vg::ProfilerConfig profiler_config;
  profiler_config.frames_in_flight = kFramesInFlight;
  auto profiler = vg::Profiler::create(device.value(), profiler_config);
  if (!profiler.ok()) {
    std::fprintf(stderr, "profiler: %s\n", profiler.status().message().c_str());
    return 1;
  }

  auto loop = win::FrameLoop::create(device.value(), swapchain.value(),
                                     kFramesInFlight);
  if (!loop.ok()) {
    std::fprintf(stderr, "frame loop: %s\n", loop.status().message().c_str());
    return 1;
  }
  // Turnkey: the loop now calls profiler.begin_frame/end_frame for us, so the
  // render loop only opens a scope around each pass. Resizes need no hook —
  // the swapchain rebuilds its own depth attachments with the chain.
  loop.value().set_profiler(&profiler.value());

  int rendered = 0;
  while (!glfwWindowShouldClose(window)) {
    if (max_frames >= 0 && rendered >= max_frames) {
      break;
    }
    glfwPollEvents();

    // The loop owns the staleness protocol: it rebuilds the swapchain (which
    // rebuilds its per-image depth attachments) after a resize, and skips
    // ticks while the window is minimized.
    auto frame = loop.value().begin_frame(framebuffer_extent(window));
    if (!frame.ok()) {
      std::fprintf(stderr, "begin_frame: %s\n",
                   frame.status().message().c_str());
      return 1;  // ~FrameLoop drains any in-flight submission before teardown
    }
    if (!frame.value().has_value()) {
      // Paused: minimized, or the surface is still settling after a rebuild.
      // Idle briefly rather than block outright, so a settling surface retries
      // even when the compositor sends no further event.
      glfwWaitEventsTimeout(0.1);
      continue;
    }
    const win::Frame& f = *frame.value();

    const VkExtent2D extent = swapchain.value().extent();
    const VkCommandBuffer cmd = f.cmd;

    // The acquired image's target already pairs its color view with its own
    // depth attachment; the default load op clears both.
    vg::RenderTargetBeginInfo begin;
    begin.clear_color = background();
    f.target->begin(cmd, begin);

    // Advance the camera: interactive input, or a deterministic per-frame
    // turntable step under --frames (reproducible for CI).
    if (interactive) {
      apply_input(window, input, rig, move_speed);
    } else {
      rig.orbit(0.0075f, 0.0f);
    }
    // Refit near/far to the model from the camera's new position, so zoom/fly
    // keep it within the frustum (orbit holds its distance, so this stays
    // constant under --frames).
    const std::pair<float, float> clip = fit_clip(bounds, rig.position());
    const float aspect =
        static_cast<float>(extent.width) /
        static_cast<float>(extent.height == 0 ? 1 : extent.height);
    const glm::mat4 view_proj =
        rig.to_camera(kFovY, aspect, clip.first, clip.second).view_proj();
    // The camera moves each frame, so refresh the acquired slot's camera UBO
    // before drawing: begin_frame waited that slot's fence, so the GPU is not
    // reading it.
    pbr.scene.set_camera(f.slot, rig.position(), ibl.prefilter_max_lod);
    {
      // Per-pass GPU stages: a timestamp pair + a VK_EXT_debug_utils label
      // around each, resolved into the metrics printed below.
      vg::Profiler::Scope skybox_scope =
          profiler.value().gpu_scope(cmd, "skybox");
      record_skybox(cmd, extent, skybox, view_proj, rig.position());
    }
    pipelines::PbrFrame frame_info;
    frame_info.extent = extent;
    frame_info.view_proj = view_proj;
    frame_info.scene = &pbr.scene;
    frame_info.slot = f.slot;
    frame_info.draws = pbr_draws.data();
    frame_info.draw_count = static_cast<uint32_t>(pbr_draws.size());
    {
      vg::Profiler::Scope pbr_scope = profiler.value().gpu_scope(cmd, "pbr");
      pipeline.value().submit(cmd, frame_info);
    }

    f.target->end(cmd);

    const vg::Status present = loop.value().end_frame(f);
    if (!present.ok() && !win::swapchain_stale(present)) {
      std::fprintf(stderr, "end_frame: %s\n", present.message().c_str());
      return 1;  // ~FrameLoop drains the submitted frame before teardown
    }

    // Periodically dump the resolved per-pass timings. A slot's GPU times
    // resolve when the slot recurs, so the snapshot trails the current frame
    // by kFramesInFlight. On a device without timestamp support (MoltenVK) the
    // GPU column reads n/a; CPU times remain.
    if (rendered % 30 == 29) {
      const vg::FrameMetrics& metrics = profiler.value().metrics();
      std::printf("03_model frame %d: %.1f fps, %.2f ms/frame\n",
                  rendered - static_cast<int>(kFramesInFlight), metrics.fps,
                  metrics.cpu_frame_ms);
      for (const vg::FrameMetrics::Section& s : metrics.sections) {
        if (s.has_gpu) {
          std::printf("  %-7s cpu %6.3f ms  gpu %6.3f ms\n", s.name, s.cpu_ms,
                      s.gpu_ms);
        } else {
          std::printf("  %-7s cpu %6.3f ms  gpu     n/a\n", s.name, s.cpu_ms);
        }
      }
    }
    ++rendered;
  }

  std::printf("03_model: rendered %d frame(s)\n", rendered);
  return 0;
}

bool parse_uint(const char* arg, uint32_t* out) {
  char* end = nullptr;
  const long value = std::strtol(arg, &end, 10);
  if (*end != '\0' || value <= 0 || value > 16384) {
    return false;
  }
  *out = static_cast<uint32_t>(value);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int max_frames = -1;  // < 0 means run until the window is closed
  const char* model_path = nullptr;
  const char* screenshot_path = nullptr;
  uint32_t width = 1024;
  uint32_t height = 720;
  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (std::strcmp(argv[i], "--model") == 0 && has_value) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--screenshot") == 0 && has_value) {
      screenshot_path = argv[++i];
    } else if (std::strcmp(argv[i], "--width") == 0 && has_value) {
      if (!parse_uint(argv[++i], &width)) {
        std::fprintf(stderr, "--width: invalid value '%s'\n", argv[i]);
        return 2;
      }
    } else if (std::strcmp(argv[i], "--height") == 0 && has_value) {
      if (!parse_uint(argv[++i], &height)) {
        std::fprintf(stderr, "--height: invalid value '%s'\n", argv[i]);
        return 2;
      }
    } else if (std::strcmp(argv[i], "--frames") == 0 && has_value) {
      char* end = nullptr;
      const long value = std::strtol(argv[++i], &end, 10);
      if (*end != '\0' || value < 0 || value > INT_MAX) {
        std::fprintf(stderr, "--frames: invalid value '%s'\n", argv[i]);
        return 2;
      }
      max_frames = static_cast<int>(value);
    } else {
      std::fprintf(
          stderr,
          "usage: %s [--model <file.gltf|.glb>] [--frames N]\n"
          "          [--screenshot <out.ppm> [--width W] [--height H]]\n",
          argv[0]);
      return 2;
    }
  }

  // Headless still: no window, no GLFW -- render one frame and write a PPM.
  if (screenshot_path != nullptr) {
    return run_screenshot(model_path, screenshot_path, width, height);
  }

  if (glfwInit() != GLFW_TRUE) {
    std::fprintf(stderr, "glfwInit failed\n");
    return 1;
  }
  if (glfwVulkanSupported() != GLFW_TRUE) {
    std::fprintf(stderr, "GLFW reports no Vulkan loader available\n");
    glfwTerminate();
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Vulkan, not OpenGL
  GLFWwindow* window = glfwCreateWindow(
      static_cast<int>(width), static_cast<int>(height),
      "volumetric_kit_gfx \xE2\x80\x94 03_model", nullptr, nullptr);
  if (window == nullptr) {
    std::fprintf(stderr, "glfwCreateWindow failed\n");
    glfwTerminate();
    return 1;
  }

  const int rc = run_windowed(window, model_path, max_frames);

  glfwDestroyWindow(window);
  glfwTerminate();
  return rc;
}

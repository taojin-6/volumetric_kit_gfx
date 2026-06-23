// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/03_model: load a glTF model and look at it. Parses a `.gltf`/`.glb`
// with the io tier into a CPU assets::Model, uploads each mesh (vertex + index
// buffer) and its material maps, and draws it depth-tested with glTF
// metallic-roughness PBR: a per-draw model/MVP push constant, a per-frame scene
// set (set 0: camera), and a per-material set (set 1: factor UBO + the five
// maps) -- all reflected automatically into the pipeline layout. The camera
// auto-frames the model's bounds, so any model fills the view. IBL (image-based
// ambient) is the next spine step; today the ambient is a flat fill.
//
// Usage:
//   example_03_model                       # built-in cube, in a window
//   example_03_model --model Helmet.glb    # any glTF-Sample-Assets model
//   example_03_model --frames 3            # render N frames, then exit
//   example_03_model --model m.glb --screenshot out.ppm   # headless still
//
// Two render paths share the model load + upload + draw recording:
//  * Windowed (default): Surface + Swapchain + FrameLoop, slowly orbiting. The
//    swapchain is color-only, so this example owns the depth image and pairs it
//    with the swapchain's color view (Swapchain::image_view) into its own
//    RenderTarget. One frame in flight keeps that single depth image free of
//    cross-frame hazards; a per-slot depth ring would be the throughput
//    upgrade.
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
#include "volumetric_kit/gfx/camera/orbit_camera.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/io/gltf_loader.hpp"
#include "volumetric_kit/gfx/windowing.hpp"

namespace vg = volumetric_kit::gfx;
namespace win = volumetric_kit::gfx::windowing;
namespace assets = volumetric_kit::gfx::assets;
namespace camera = volumetric_kit::gfx::camera;

namespace {

constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr float kFovY = 1.0471976f;  // 60 degrees
constexpr float kPi = 3.14159265359f;

// Per-draw transform fed to model.vert as a push constant (128 bytes -- the
// guaranteed minimum maxPushConstantsSize). Layout matches the shader's block.
struct PushConstants {
  glm::mat4 mvp;    // projection * view * world
  glm::mat4 model;  // world (for the world-space normal)
};

// One mesh uploaded to the GPU. index_count == 0 marks a skipped (empty) mesh.
struct GpuMesh {
  vg::Buffer vertices;
  vg::Buffer indices;
  uint32_t index_count = 0;
};

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

// Point the orbit camera at the model's bounds and return a fitting near/far.
std::pair<float, float> frame_camera(camera::OrbitCamera& orbit,
                                     const Bounds& bounds) {
  const float radius = std::fmax(bounds.radius(), 1e-3f);
  orbit.set_target(bounds.center());
  orbit.set_distance(radius / std::sin(kFovY * 0.5f) * 1.3f);  // fit the sphere
  orbit.set_elevation(0.35f);  // look slightly down on it
  const float z_far = orbit.distance() + radius * 4.0f;
  const float z_near = std::fmax(orbit.distance() - radius, radius * 0.02f);
  return {z_near, z_far};
}

vg::Buffer upload_buffer(vg::Allocator& allocator, const void* data,
                         size_t size, VkBufferUsageFlags usage, bool* ok) {
  vg::BufferDesc desc;
  desc.size = size;
  desc.usage = usage;
  desc.memory = vg::MemoryUsage::HostVisible;
  desc.mapped = true;
  desc.host_access = vg::HostAccess::SequentialWrite;
  auto buffer = allocator.create_buffer(desc);
  if (!buffer.ok()) {
    std::fprintf(stderr, "upload: %s\n", buffer.status().message().c_str());
    *ok = false;
    return {};
  }
  std::memcpy(buffer.value().mapped(), data, size);
  return std::move(buffer).value();
}

// Upload every non-empty mesh; the returned vector is parallel to model.meshes
// so a DrawItem's mesh index addresses it directly (empty meshes stay
// index_count == 0 and are skipped at draw time).
std::vector<GpuMesh> upload_meshes(vg::Allocator& allocator,
                                   const assets::Model& model, bool* ok) {
  std::vector<GpuMesh> gpu(model.meshes.size());
  for (size_t i = 0; i < model.meshes.size() && *ok; ++i) {
    const assets::Mesh& mesh = model.meshes[i];
    if (mesh.vertices.empty() || mesh.indices.empty()) {
      continue;
    }
    gpu[i].vertices =
        upload_buffer(allocator, mesh.vertices.data(),
                      mesh.vertices.size() * sizeof(assets::Vertex),
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, ok);
    gpu[i].indices = upload_buffer(allocator, mesh.indices.data(),
                                   mesh.indices.size() * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ok);
    // Only mark the mesh drawable once both buffers uploaded: a failed upload
    // leaves index_count 0 (skipped), keeping "index_count > 0 => valid
    // buffers".
    if (*ok) {
      gpu[i].index_count = static_cast<uint32_t>(mesh.indices.size());
    }
  }
  return gpu;
}

VkVertexInputBindingDescription mesh_binding() {
  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = sizeof(assets::Vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  return binding;
}

// model.vert reads position (location 0), normal (1), the primary UV (2), and
// the tangent (3) out of the interleaved assets::Vertex; vertex color is
// unused.
void mesh_attributes(VkVertexInputAttributeDescription attrs[4]) {
  attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
              offsetof(assets::Vertex, position)};
  attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,
              offsetof(assets::Vertex, normal)};
  attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(assets::Vertex, uv0)};
  attrs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
              offsetof(assets::Vertex, tangent)};
}

vg::Result<vg::GraphicsPipeline> build_pipeline(
    VkDevice device, const vg::ShaderModule& vert, const vg::ShaderModule& frag,
    const vg::RenderTargetLayout& layout,
    const VkVertexInputBindingDescription* binding,
    const VkVertexInputAttributeDescription* attrs) {
  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = &vert;
  desc.fragment_shader = &frag;
  desc.layout = layout;
  desc.vertex_bindings = binding;
  desc.vertex_binding_count = 1;
  desc.vertex_attributes = attrs;
  desc.vertex_attribute_count = 4;
  desc.depth_test = true;
  desc.depth_write = true;
  return vg::GraphicsPipeline::create(device, desc);
}

struct Shaders {
  vg::ShaderModule vert;
  vg::ShaderModule frag;
};

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

// Load + create both model shader modules from VG_EXAMPLE_SHADER_DIR; null *ok
// on failure. Shared by both render paths.
Shaders load_shaders(VkDevice device, bool* ok) {
  vg::ShaderModule vert = load_shader(device, "model.vert.spv", ok);
  vg::ShaderModule frag = load_shader(device, "model.frag.spv", ok);
  if (!*ok) {
    return {};
  }
  return {std::move(vert), std::move(frag)};
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

// Bind the pipeline, set a full-target viewport/scissor, and draw every item
// with its per-draw MVP push constant. Shared by both render paths; the caller
// owns the surrounding dynamic-rendering scope.
void record_scene(VkCommandBuffer cmd, VkExtent2D extent,
                  const vg::GraphicsPipeline& pipeline,
                  const glm::mat4& view_proj,
                  const std::vector<DrawItem>& draws,
                  const std::vector<GpuMesh>& meshes, VkDescriptorSet scene_set,
                  const std::vector<VkDescriptorSet>& mesh_sets) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
  // Per-frame scene data (set 0: camera position) binds once for all draws.
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline.layout(), 0, 1, &scene_set, 0, nullptr);
  set_full_viewport(cmd, extent);

  for (const DrawItem& draw : draws) {
    const GpuMesh& gpu = meshes[draw.mesh];
    if (gpu.index_count == 0) {
      continue;  // empty/skipped mesh
    }
    PushConstants pc;
    pc.mvp = view_proj * draw.world;
    pc.model = draw.world;
    vkCmdPushConstants(cmd, pipeline.layout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(pc), &pc);
    // Bind this mesh's material (set 1: factor UBO + the five maps). Empty
    // meshes were skipped above, so every bound set is fully written.
    const VkDescriptorSet material_set = mesh_sets[draw.mesh];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.layout(), 1, 1, &material_set, 0, nullptr);
    const VkDeviceSize offset = 0;
    const VkBuffer vbuf = gpu.vertices.handle();
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &offset);
    vkCmdBindIndexBuffer(cmd, gpu.indices.handle(), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, gpu.index_count, 1, 0, 0, 0);
  }
}

VkClearColorValue background() {
  VkClearColorValue c{};
  c.float32[0] = 0.02f;
  c.float32[1] = 0.02f;
  c.float32[2] = 0.05f;
  c.float32[3] = 1.0f;
  return c;
}

// Allocate the depth image and transition it UNDEFINED -> DEPTH_ATTACHMENT for
// the dynamic-rendering scope. Recreated (with the swapchain) on resize.
vg::Texture create_depth(vg::Allocator& allocator, const vg::Device& device,
                         VkExtent2D extent, bool* ok) {
  vg::TextureDesc desc;
  desc.extent = extent;
  desc.format = kDepthFormat;
  desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  auto depth = allocator.create_image(desc);
  if (!depth.ok()) {
    std::fprintf(stderr, "depth image: %s\n", depth.status().message().c_str());
    *ok = false;
    return {};
  }
  const VkImage image = depth.value().image();
  const vg::Status transition =
      device.submit_single_time([image](VkCommandBuffer cmd) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &b);
      });
  if (!transition.ok()) {
    std::fprintf(stderr, "depth transition: %s\n",
                 transition.message().c_str());
    *ok = false;
    return {};
  }
  return std::move(depth).value();
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

// std140 per-material parameters; mirrors the Material UBO in model.frag.
struct MaterialUbo {
  glm::vec4 base_color_factor;
  glm::vec4 emissive_factor;  // .rgb used
  float metallic_factor;
  float roughness_factor;
  float normal_scale;
  float occlusion_strength;
};
static_assert(sizeof(MaterialUbo) == 48,
              "MaterialUbo must match the std140 Material block");

// std140 per-frame parameters; mirrors the Scene UBO in model.frag.
struct SceneUbo {
  glm::vec4 camera_pos;  // .xyz world-space eye, .w = prefilter max LOD
};

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

// All PBR GPU state, outliving the draw loop: a shared sampler; every uploaded
// map plus 1x1 white / flat-normal fallbacks; per-material factor UBOs and the
// per-frame scene UBO; and the descriptor sets. set 0 (scene) binds once per
// frame; set 1 (material) per draw. mesh_sets[i] is the set-1 for
// model.meshes[i].
struct PbrResources {
  std::optional<vg::Sampler> sampler;     // no public default ctor
  std::vector<vg::Texture> textures;      // owns every uploaded map + fallbacks
  std::vector<vg::Buffer> material_ubos;  // owns the per-material factor UBOs
  vg::Buffer scene_ubo;                   // per-frame; host-mapped
  vg::DescriptorPool pool;
  vg::DescriptorSet scene_set;                   // set 0
  std::vector<vg::DescriptorSet> material_sets;  // set 1, parallel to materials
  vg::DescriptorSet fallback_material_set;       // set 1, material-less meshes
  std::vector<VkDescriptorSet> mesh_sets;        // set 1 resolved per mesh
};

// Upload every material map (each image once, in the color space its slot
// needs), build a factor UBO + descriptor set per material (plus a default
// fallback) and the per-frame scene set, and resolve the set each mesh binds.
// The set 0 / set 1 layouts are reflected from the shaders.
PbrResources setup_pbr(const vg::Device& device, vg::Allocator& alloc,
                       const vg::GraphicsPipeline& pipeline,
                       const assets::Model& model, const Ibl& ibl, bool* ok) {
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

  // Pool: a material set per material + a fallback, plus the scene set.
  // Material sets each hold a UBO + 5 samplers; the scene set holds a UBO + the
  // 3 IBL textures.
  const uint32_t material_count =
      static_cast<uint32_t>(model.materials.size()) + 1;
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, material_count + 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, material_count * 5 + 3}};
  auto pool =
      vg::DescriptorPool::create(device.handle(), sizes, 2, material_count + 1);
  if (!pool.ok()) {
    std::fprintf(stderr, "descriptor pool: %s\n",
                 pool.status().message().c_str());
    *ok = false;
    return r;
  }
  r.pool = std::move(pool).value();

  const VkDescriptorSetLayout scene_layout = pipeline.descriptor_set_layout(0);
  const VkDescriptorSetLayout material_layout =
      pipeline.descriptor_set_layout(1);

  // Host-mapped uniform buffer of `size` bytes (empty on failure).
  auto make_ubo = [&](size_t size, bool* ubo_ok) {
    *ubo_ok = true;  // sink: set here, cleared on failure below
    vg::BufferDesc bd;
    bd.size = size;
    bd.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bd.memory = vg::MemoryUsage::HostVisible;
    bd.mapped = true;
    auto buf = alloc.create_buffer(bd);
    if (!buf.ok()) {
      std::fprintf(stderr, "uniform buffer: %s\n",
                   buf.status().message().c_str());
      *ubo_ok = false;
      return vg::Buffer{};
    }
    return std::move(buf).value();
  };

  // Scene set (set 0): the per-frame camera UBO, refreshed each frame by the
  // caller through scene_ubo.mapped().
  r.scene_ubo = make_ubo(sizeof(SceneUbo), ok);
  if (!*ok) {
    return r;
  }
  {
    auto set = r.pool.allocate(scene_layout);
    if (!set.ok()) {
      std::fprintf(stderr, "scene set: %s\n", set.status().message().c_str());
      *ok = false;
      return r;
    }
    r.scene_set = std::move(set).value();
    r.scene_set.write_uniform_buffer(0, r.scene_ubo.handle(), 0,
                                     sizeof(SceneUbo));
    // The IBL textures are per-frame-constant, so they live in the scene set
    // alongside the camera (bindings 1-3).
    r.scene_set.write_combined_image_sampler(
        1, ibl.irradiance.view(), ibl.sampler->handle(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    r.scene_set.write_combined_image_sampler(
        2, ibl.prefilter.view(), ibl.sampler->handle(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    r.scene_set.write_combined_image_sampler(
        3, ibl.brdf_lut.view(), ibl.sampler->handle(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  r.material_ubos.reserve(material_count);

  // Build one material set: its factor UBO at binding 0, then the five maps.
  auto make_material_set = [&](const MaterialUbo& ubo, int base, int mr,
                               int normal, int occ, int emissive,
                               bool* set_ok) -> vg::DescriptorSet {
    vg::Buffer buf = make_ubo(sizeof(MaterialUbo), set_ok);
    if (!*set_ok) {
      return {};
    }
    std::memcpy(buf.mapped(), &ubo, sizeof(ubo));
    r.material_ubos.push_back(std::move(buf));
    auto set = r.pool.allocate(material_layout);
    if (!set.ok()) {
      std::fprintf(stderr, "material set: %s\n",
                   set.status().message().c_str());
      *set_ok = false;
      return {};
    }
    vg::DescriptorSet ds = std::move(set).value();
    ds.write_uniform_buffer(0, r.material_ubos.back().handle(), 0,
                            sizeof(MaterialUbo));
    const int maps[5] = {base, mr, normal, occ, emissive};
    for (uint32_t b = 0; b < 5; ++b) {
      ds.write_combined_image_sampler(b + 1, r.textures[maps[b]].view(),
                                      r.sampler->handle(),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    return ds;
  };

  // Texture index for a slot, or the given fallback (kNoTexture is out of range
  // of image_tex, so it resolves to the fallback).
  auto tex_for = [&](uint32_t slot, int fallback) {
    return (slot < image_tex.size() && image_tex[slot] >= 0) ? image_tex[slot]
                                                             : fallback;
  };

  // Fallback material for meshes with no material: a matte white dielectric.
  {
    MaterialUbo def{};
    def.base_color_factor = glm::vec4(1.0f);
    def.emissive_factor = glm::vec4(0.0f);
    def.metallic_factor = 0.0f;
    def.roughness_factor = 1.0f;
    def.normal_scale = 1.0f;
    def.occlusion_strength = 1.0f;
    r.fallback_material_set =
        make_material_set(def, white, white, flat, white, white, ok);
    if (!*ok) {
      return r;
    }
  }

  r.material_sets.reserve(model.materials.size());
  for (const assets::Material& m : model.materials) {
    MaterialUbo ubo{};
    ubo.base_color_factor = m.base_color_factor;
    ubo.emissive_factor = glm::vec4(m.emissive_factor, 0.0f);
    ubo.metallic_factor = m.metallic_factor;
    ubo.roughness_factor = m.roughness_factor;
    ubo.normal_scale = m.normal_scale;
    ubo.occlusion_strength = m.occlusion_strength;
    r.material_sets.push_back(make_material_set(
        ubo, tex_for(m.base_color_texture, white),
        tex_for(m.metallic_roughness_texture, white),
        tex_for(m.normal_texture, flat), tex_for(m.occlusion_texture, white),
        tex_for(m.emissive_texture, white), ok));
    if (!*ok) {
      return r;
    }
  }

  // Resolve the set each mesh binds: its material's set, or the fallback.
  r.mesh_sets.resize(model.meshes.size());
  for (size_t i = 0; i < model.meshes.size(); ++i) {
    const uint32_t mat = model.meshes[i].material;
    r.mesh_sets[i] =
        (mat != assets::Mesh::kNoMaterial && mat < r.material_sets.size())
            ? r.material_sets[mat].handle()
            : r.fallback_material_set.handle();
  }
  return r;
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
  const std::vector<GpuMesh> meshes =
      upload_meshes(allocator.value(), model, &ok);
  if (!ok) {
    return 1;
  }

  camera::OrbitCamera orbit;
  const std::pair<float, float> clip =
      frame_camera(orbit, compute_bounds(model, draws));
  orbit.set_azimuth(0.7f);  // a fixed three-quarter view for the still

  Shaders shaders = load_shaders(device.value().handle(), &ok);
  if (!ok) {
    return 1;
  }

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

  const VkVertexInputBindingDescription binding = mesh_binding();
  VkVertexInputAttributeDescription attrs[4];
  mesh_attributes(attrs);
  auto pipeline =
      build_pipeline(device.value().handle(), shaders.vert, shaders.frag,
                     target.value().layout(), &binding, attrs);
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  Ibl ibl = make_ibl(device.value(), allocator.value(), &ok);
  if (!ok) {
    return 1;
  }
  PbrResources pbr = setup_pbr(device.value(), allocator.value(),
                               pipeline.value(), model, ibl, &ok);
  if (!ok) {
    return 1;
  }

  // Fixed camera for the still: write the eye into the scene UBO once.
  *static_cast<SceneUbo*>(pbr.scene_ubo.mapped()) =
      SceneUbo{glm::vec4(orbit.eye(), ibl.prefilter_max_lod)};

  Skybox skybox = setup_skybox(device.value(), allocator.value(),
                               target.value().layout(), &ok);
  if (!ok) {
    return 1;
  }

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const glm::mat4 view_proj =
      orbit.to_camera(kFovY, aspect, clip.first, clip.second).view_proj();

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
        record_skybox(cmd, {width, height}, skybox, view_proj, orbit.eye());
        record_scene(cmd, {width, height}, pipeline.value(), view_proj, draws,
                     meshes, pbr.scene_set.handle(), pbr.mesh_sets);
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

// --- Windowed path: Surface + Swapchain + FrameLoop, slowly orbiting
// ----------

// Owns all Vulkan/windowing state for one window; everything is destroyed when
// this returns, before main() tears GLFW down.
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
  const std::vector<GpuMesh> meshes =
      upload_meshes(allocator.value(), model, &ok);
  if (!ok) {
    return 1;
  }

  camera::OrbitCamera orbit;
  const std::pair<float, float> clip =
      frame_camera(orbit, compute_bounds(model, draws));

  Shaders shaders = load_shaders(device.value().handle(), &ok);
  if (!ok) {
    return 1;
  }

  win::SwapchainConfig swapchain_config;
  swapchain_config.extent = framebuffer_extent(window);
  auto swapchain = win::Swapchain::create(device.value(), surface.handle(),
                                          swapchain_config);
  if (!swapchain.ok()) {
    std::fprintf(stderr, "swapchain: %s\n",
                 swapchain.status().message().c_str());
    return 1;
  }

  vg::Texture depth = create_depth(allocator.value(), device.value(),
                                   swapchain.value().extent(), &ok);
  if (!ok) {
    return 1;
  }

  // Pipeline layout = swapchain color + our depth format. Size-independent
  // (dynamic viewport), so it survives resizes without a rebuild.
  vg::RenderTargetLayout pipeline_layout;
  pipeline_layout.color_formats[0] = swapchain.value().format();
  pipeline_layout.color_count = 1;
  pipeline_layout.depth_format = kDepthFormat;
  const VkVertexInputBindingDescription binding = mesh_binding();
  VkVertexInputAttributeDescription attrs[4];
  mesh_attributes(attrs);
  auto pipeline =
      build_pipeline(device.value().handle(), shaders.vert, shaders.frag,
                     pipeline_layout, &binding, attrs);
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  // One frame in flight: the single depth image is then never written by two
  // frames at once (see the file header).
  // TODO: promote depth into the windowing tier (a per-slot depth ring on the
  // swapchain's RenderTarget) so consumers get a depth-capable target and can
  // run more frames in flight, instead of the example owning a single depth.
  Ibl ibl = make_ibl(device.value(), allocator.value(), &ok);
  if (!ok) {
    return 1;
  }
  PbrResources pbr = setup_pbr(device.value(), allocator.value(),
                               pipeline.value(), model, ibl, &ok);
  if (!ok) {
    return 1;
  }

  Skybox skybox =
      setup_skybox(device.value(), allocator.value(), pipeline_layout, &ok);
  if (!ok) {
    return 1;
  }

  auto loop = win::FrameLoop::create(device.value(), swapchain.value(), 1);
  if (!loop.ok()) {
    std::fprintf(stderr, "frame loop: %s\n", loop.status().message().c_str());
    return 1;
  }

  int rendered = 0;
  while (!glfwWindowShouldClose(window)) {
    if (max_frames >= 0 && rendered >= max_frames) {
      break;
    }
    glfwPollEvents();

    // Pause while minimized (zero framebuffer): a zero-extent swapchain/depth
    // recreate would fail. Block until the window is restored or closed.
    VkExtent2D fb = framebuffer_extent(window);
    while ((fb.width == 0 || fb.height == 0) &&
           !glfwWindowShouldClose(window)) {
      glfwWaitEvents();
      fb = framebuffer_extent(window);
    }
    if (glfwWindowShouldClose(window)) {
      break;
    }

    auto frame = loop.value().begin_frame();
    if (!frame.ok()) {
      if (frame.status().code() == VK_ERROR_OUT_OF_DATE_KHR) {
        if (!swapchain.value().recreate(framebuffer_extent(window)).ok()) {
          break;
        }
        depth = create_depth(allocator.value(), device.value(),
                             swapchain.value().extent(), &ok);
        if (!ok) {
          return 1;
        }
        continue;
      }
      std::fprintf(stderr, "begin_frame: %s\n",
                   frame.status().message().c_str());
      return 1;
    }

    const VkExtent2D extent = swapchain.value().extent();
    const uint32_t image_index = frame.value().image_index;
    const VkCommandBuffer cmd = frame.value().cmd;

    // Assemble this image's color view + our depth into a render target.
    const vg::RenderTargetAttachment color{
        swapchain.value().image(image_index),
        swapchain.value().image_view(image_index), swapchain.value().format()};
    const vg::RenderTargetAttachment depth_att{depth.image(), depth.view(),
                                               kDepthFormat};
    const vg::RenderTarget rt(extent, &color, 1, VK_SAMPLE_COUNT_1_BIT,
                              &depth_att);

    vg::RenderTargetBeginInfo begin;
    begin.clear_color = background();
    rt.begin(cmd, begin);

    // Slow turntable, advanced per rendered frame (deterministic for --frames).
    orbit.set_azimuth(static_cast<float>(rendered) * 0.0075f);
    const float aspect =
        static_cast<float>(extent.width) /
        static_cast<float>(extent.height == 0 ? 1 : extent.height);
    const glm::mat4 view_proj =
        orbit.to_camera(kFovY, aspect, clip.first, clip.second).view_proj();
    // The camera orbits each frame, so refresh the scene UBO before drawing.
    // Safe with one frame in flight: begin_frame waited the previous frame's
    // fence above, so the GPU has finished reading this single shared UBO.
    // Raising frames_in_flight > 1 would need a per-slot scene UBO.
    *static_cast<SceneUbo*>(pbr.scene_ubo.mapped()) =
        SceneUbo{glm::vec4(orbit.eye(), ibl.prefilter_max_lod)};
    record_skybox(cmd, extent, skybox, view_proj, orbit.eye());
    record_scene(cmd, extent, pipeline.value(), view_proj, draws, meshes,
                 pbr.scene_set.handle(), pbr.mesh_sets);

    rt.end(cmd);

    const vg::Status present = loop.value().end_frame(frame.value());
    if (!present.ok()) {
      if (present.code() == VK_ERROR_OUT_OF_DATE_KHR ||
          present.code() == VK_SUBOPTIMAL_KHR) {
        if (!swapchain.value().recreate(framebuffer_extent(window)).ok()) {
          break;
        }
        depth = create_depth(allocator.value(), device.value(),
                             swapchain.value().extent(), &ok);
        if (!ok) {
          return 1;
        }
      } else {
        std::fprintf(stderr, "end_frame: %s\n", present.message().c_str());
        return 1;
      }
    }
    ++rendered;
  }

  vkDeviceWaitIdle(device.value().handle());
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

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/03_model: load a glTF model and look at it. Parses a `.gltf`/`.glb`
// with the io tier into a CPU assets::Model, uploads each mesh to a vertex +
// index buffer, and draws it depth-tested with a per-draw model/MVP push
// constant (reflected automatically into the pipeline layout -- no descriptor
// set). The camera auto-frames the model's bounds, so any model fills the view.
// Untextured normal shading: the geometry's form, not its materials (PBR +
// material textures arrive with the IBL spine); this is the first look.
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
#include <string>
#include <utility>
#include <vector>

#include <glm/common.hpp>     // glm::min / glm::max (component-wise)
#include <glm/geometric.hpp>  // glm::length
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/assets/model.hpp"
#include "volumetric_kit/gfx/camera/orbit_camera.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/io/gltf_loader.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace vg = volumetric_kit::gfx;
namespace win = volumetric_kit::gfx::windowing;
namespace assets = volumetric_kit::gfx::assets;
namespace camera = volumetric_kit::gfx::camera;

namespace {

constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr float kFovY = 1.0471976f;  // 60 degrees

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
                  const glm::mat4& parent, std::vector<DrawItem>& out) {
  const assets::Node& node = model.scene.nodes[node_index];
  const glm::mat4 world = parent * node.transform;
  if (node.mesh != assets::Node::kNoMesh) {
    for (uint32_t k = 0; k < node.mesh_count; ++k) {
      out.push_back({node.mesh + k, world});
    }
  }
  for (uint32_t child : node.children) {
    collect_node(model, child, world, out);
  }
}

std::vector<DrawItem> collect_draws(const assets::Model& model) {
  std::vector<DrawItem> draws;
  for (uint32_t root : model.scene.roots) {
    collect_node(model, root, glm::mat4(1.0f), draws);
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
    gpu[i].index_count = static_cast<uint32_t>(mesh.indices.size());
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

// model.vert reads position (location 0) + normal (location 1) out of the
// interleaved assets::Vertex; the rest of the stride is ignored by this shader.
void mesh_attributes(VkVertexInputAttributeDescription attrs[2]) {
  attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
              offsetof(assets::Vertex, position)};
  attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,
              offsetof(assets::Vertex, normal)};
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
  desc.vertex_attribute_count = 2;
  desc.depth_test = true;
  desc.depth_write = true;
  return vg::GraphicsPipeline::create(device, desc);
}

// Bind the pipeline, set a full-target viewport/scissor, and draw every item
// with its per-draw MVP push constant. Shared by both render paths; the caller
// owns the surrounding dynamic-rendering scope.
void record_scene(VkCommandBuffer cmd, VkExtent2D extent,
                  const vg::GraphicsPipeline& pipeline,
                  const glm::mat4& view_proj,
                  const std::vector<DrawItem>& draws,
                  const std::vector<GpuMesh>& meshes) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
  VkViewport viewport{};
  viewport.width = static_cast<float>(extent.width);
  viewport.height = static_cast<float>(extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  VkRect2D scissor{};
  scissor.extent = extent;
  vkCmdSetScissor(cmd, 0, 1, &scissor);

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

  const std::vector<uint32_t> vert_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/model.vert.spv");
  const std::vector<uint32_t> frag_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/model.frag.spv");
  if (vert_code.empty() || frag_code.empty()) {
    std::fprintf(stderr, "missing compiled shaders in %s\n",
                 VG_EXAMPLE_SHADER_DIR);
    return 1;
  }
  auto vert =
      vg::ShaderModule::create(device.value().handle(), vert_code.data(),
                               vert_code.size() * sizeof(uint32_t));
  auto frag =
      vg::ShaderModule::create(device.value().handle(), frag_code.data(),
                               frag_code.size() * sizeof(uint32_t));
  if (!vert.ok() || !frag.ok()) {
    std::fprintf(stderr, "shader module creation failed\n");
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
  VkVertexInputAttributeDescription attrs[2];
  mesh_attributes(attrs);
  auto pipeline =
      build_pipeline(device.value().handle(), vert.value(), frag.value(),
                     target.value().layout(), &binding, attrs);
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
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
        record_scene(cmd, {width, height}, pipeline.value(), view_proj, draws,
                     meshes);
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

  const std::vector<uint32_t> vert_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/model.vert.spv");
  const std::vector<uint32_t> frag_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/model.frag.spv");
  if (vert_code.empty() || frag_code.empty()) {
    std::fprintf(stderr, "missing compiled shaders in %s\n",
                 VG_EXAMPLE_SHADER_DIR);
    return 1;
  }
  auto vert =
      vg::ShaderModule::create(device.value().handle(), vert_code.data(),
                               vert_code.size() * sizeof(uint32_t));
  auto frag =
      vg::ShaderModule::create(device.value().handle(), frag_code.data(),
                               frag_code.size() * sizeof(uint32_t));
  if (!vert.ok() || !frag.ok()) {
    std::fprintf(stderr, "shader module creation failed\n");
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
  VkVertexInputAttributeDescription attrs[2];
  mesh_attributes(attrs);
  auto pipeline =
      build_pipeline(device.value().handle(), vert.value(), frag.value(),
                     pipeline_layout, &binding, attrs);
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  // One frame in flight: the single depth image is then never written by two
  // frames at once (see the file header).
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
    record_scene(cmd, extent, pipeline.value(), view_proj, draws, meshes);

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

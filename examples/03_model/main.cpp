// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/03_model: load a glTF model and fly around it. Parses a
// `.gltf`/`.glb` with the io tier into a CPU assets::Model, bridges it to the
// GPU with pipelines::PbrModel (meshes + material maps uploaded in one submit,
// materials built, the scene flattened into a draw list), and draws it
// depth-tested with glTF metallic-roughness PBR: a per-draw model/MVP push
// constant, a per-frame scene set (set 0: camera + IBL), and a per-material
// set (set 1: factor UBO + the five maps) -- all reflected automatically into
// the pipeline layout. The ambient comes from a CPU-baked IBL of the analytic
// sky, drawn as a skybox behind the model. The camera auto-frames the model's
// bounds, so any model fills the view.
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
#include "volumetric_kit/gfx/pipelines/pbr_model.hpp"
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

// World-space axis-aligned bounds over every drawn vertex, for camera framing.
struct Bounds {
  glm::vec3 min{0.0f};
  glm::vec3 max{0.0f};
  glm::vec3 center() const { return (min + max) * 0.5f; }
  float radius() const { return glm::length(max - min) * 0.5f; }
};

// Grow `b` by one mesh's vertices under `world`.
void accumulate_bounds(const assets::Mesh& mesh, const glm::mat4& world,
                       Bounds& b, bool& any) {
  for (const assets::Vertex& v : mesh.vertices) {
    const glm::vec3 p = glm::vec3(world * glm::vec4(v.position, 1.0f));
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

// Bounds over exactly what PbrModel draws: it flattens the scene the same way
// (pipelines::flatten_scene, shared so the two never diverge), accumulated over
// each instance's CPU-side vertices -- framing needs positions, which the
// uploaded GPU meshes no longer expose -- so the camera frames precisely the
// drawn geometry. flatten_scene guarantees every instance.mesh is in range and
// handles the no-scene-graph fallback (every mesh at the origin).
Bounds compute_bounds(const assets::Model& model) {
  bool any = false;
  Bounds b;
  for (const pipelines::MeshInstance& inst : pipelines::flatten_scene(model)) {
    accumulate_bounds(model.meshes[inst.mesh], inst.world, b, any);
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

// Build the pipelines-tier PbrScene (set 0: camera + IBL) against the
// pipeline's reflected set-0 layout, one camera UBO ring slot per frame in
// flight. The caller refreshes the acquired slot's camera each frame through
// scene.set_camera(slot, ...). Everything else GPU-side for the model -- the
// meshes, material maps, materials (set 1), and draw list -- comes from
// pipelines::PbrModel::create.
pipelines::PbrScene make_pbr_scene(const vg::Device& device,
                                   vg::Allocator& alloc,
                                   const pipelines::PbrPipeline& pipeline,
                                   const Ibl& ibl, uint32_t frames_in_flight,
                                   bool* ok) {
  pipelines::PbrSceneDesc desc;
  desc.irradiance = ibl.irradiance.view();
  desc.prefilter = ibl.prefilter.view();
  desc.brdf_lut = ibl.brdf_lut.view();
  desc.sampler = ibl.sampler->handle();
  auto scene = pipelines::PbrScene::create(device.handle(), alloc,
                                           pipeline.descriptor_set_layout(0),
                                           desc, frames_in_flight);
  if (!scene.ok()) {
    std::fprintf(stderr, "scene set: %s\n", scene.status().message().c_str());
    *ok = false;
    return {};
  }
  return std::move(scene).value();
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
// the CPU and upload them through the core cube-upload path in one submit.
// Stores linear HDR color in a float cube (the skybox shader tone-maps it on
// output).
vg::Texture make_sky_cube(const vg::Device& device, vg::Allocator& alloc,
                          uint32_t size, bool* ok) {
  // RGBA16F (half) pixels: 16-bit float filters on the broad device set (incl.
  // MoltenVK/Metal); RGBA32F linear filtering is an optional feature many GPUs
  // lack. Unclamped HDR (the skybox tone-maps on output); 2 uint32/texel,
  // packed face-major -- the single-mip case of ImageUploadDesc's layout.
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

  vg::ImageUploadDesc desc;
  desc.extent = {size, size};
  desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;  // linear HDR, filterable
  desc.pixels = pixels.data();
  desc.size = pixels.size() * sizeof(uint32_t);
  desc.array_layers = 6;
  desc.cube = true;
  auto cube = vg::upload_texture(device, alloc, desc);
  if (!cube.ok()) {
    std::fprintf(stderr, "sky cube: %s\n", cube.status().message().c_str());
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

// Pack per-(mip, face) RGBA float pixels from `gen` into tightly packed
// RGBA16F (2 uint32 = 8 bytes/texel), mip-major then face -- ImageUploadDesc's
// layout. 16-bit float cubes filter on the broad device set (incl.
// MoltenVK/Metal); RGBA32F linear filtering is an optional feature many GPUs
// lack.
template <class Gen>
std::vector<uint32_t> pack_cube_rgba16f(uint32_t base_size, uint32_t mips,
                                        Gen gen) {
  std::vector<uint32_t> data;
  for (uint32_t m = 0; m < mips; ++m) {
    const uint32_t size = (base_size >> m) > 0 ? (base_size >> m) : 1u;
    for (uint32_t f = 0; f < 6; ++f) {
      const std::vector<glm::vec4> face = gen(m, static_cast<int>(f), size);
      for (const glm::vec4& px : face) {
        data.push_back(glm::packHalf2x16(glm::vec2(px.x, px.y)));
        data.push_back(glm::packHalf2x16(glm::vec2(px.z, px.w)));
      }
    }
  }
  return data;
}

// Convolve the analytic sky into the IBL texture set. CPU-side because the
// environment is analytic; a loaded HDR environment would convolve on the GPU.
// All three textures upload through one UploadBatch: one submit instead
// of a blocking round trip each.
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

  auto batch = vg::UploadBatch::begin(device, alloc);
  if (!batch.ok()) {
    std::fprintf(stderr, "ibl batch: %s\n", batch.status().message().c_str());
    *ok = false;
    return ibl;
  }

  // Queue an RGBA16F cube upload (sampled-ready once the batch finishes);
  // returns an empty texture and clears *ok on failure.
  auto add_cube = [&](uint32_t base_size, uint32_t mips,
                      const std::vector<uint32_t>& data) -> vg::Texture {
    vg::ImageUploadDesc d;
    d.extent = {base_size, base_size};
    d.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    d.pixels = data.data();
    d.size = data.size() * sizeof(uint32_t);
    d.array_layers = 6;
    d.cube = true;
    d.mip_levels = mips;
    auto tex = batch.value().add(d);
    if (!tex.ok()) {
      std::fprintf(stderr, "ibl cube: %s\n", tex.status().message().c_str());
      *ok = false;
      return {};
    }
    return std::move(tex).value();
  };

  // Diffuse irradiance: 16x16 single-mip cube -- ample for the low-frequency,
  // heavily-blurred cosine convolution.
  const std::vector<uint32_t> irradiance =
      pack_cube_rgba16f(16, 1, [](uint32_t, int face, uint32_t size) {
        std::vector<glm::vec4> px(static_cast<size_t>(size) * size);
        for (uint32_t y = 0; y < size; ++y) {
          for (uint32_t x = 0; x < size; ++x) {
            const float u = (x + 0.5f) / size * 2.0f - 1.0f;
            const float v = (y + 0.5f) / size * 2.0f - 1.0f;
            px[y * size + x] = irradiance_at(cube_dir(face, u, v));
          }
        }
        return px;
      });
  ibl.irradiance = add_cube(16, 1, irradiance);
  if (!*ok) {
    return ibl;
  }

  // Prefiltered specular: a 64x64 base over kPrefilterMips mips maps mip ->
  // roughness 0..1; 64 GGX samples/texel (the tight HDR sun can alias on low
  // mips -- accepted for the example).
  constexpr uint32_t kPrefilterMips = 5;
  const std::vector<uint32_t> prefilter = pack_cube_rgba16f(
      64, kPrefilterMips, [](uint32_t mip, int face, uint32_t size) {
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
      });
  ibl.prefilter = add_cube(64, kPrefilterMips, prefilter);
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
  auto brdf = batch.value().add(lut_desc);
  if (!brdf.ok()) {
    std::fprintf(stderr, "brdf lut: %s\n", brdf.status().message().c_str());
    *ok = false;
    return ibl;
  }
  ibl.brdf_lut = std::move(brdf).value();

  // One submit + fence wait for all three IBL textures.
  const vg::Status finished = batch.value().finish();
  if (!finished.ok()) {
    std::fprintf(stderr, "ibl upload: %s\n", finished.message().c_str());
    *ok = false;
    return ibl;
  }
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

  const Bounds bounds = compute_bounds(model);
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

  // The whole assets::Model -> GPU bridge in one call: meshes + material maps
  // uploaded (a single submit), materials built, the scene flattened into a
  // draw list ready for PbrFrame.
  auto gpu_model = pipelines::PbrModel::create(
      device.value(), allocator.value(), pipeline.value(), model);
  if (!gpu_model.ok()) {
    std::fprintf(stderr, "model: %s\n", gpu_model.status().message().c_str());
    return 1;
  }

  Ibl ibl = make_ibl(device.value(), allocator.value(), &ok);
  if (!ok) {
    return 1;
  }
  pipelines::PbrScene scene =
      make_pbr_scene(device.value(), allocator.value(), pipeline.value(), ibl,
                     /*frames_in_flight=*/1, &ok);
  if (!ok) {
    return 1;
  }

  // Fixed camera for the still: write the eye into the scene set once.
  scene.set_camera(0, rig.position(), ibl.prefilter_max_lod);

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
        target.value().prepare(cmd);  // color + depth -> attachment layouts

        vg::RenderTargetBeginInfo begin;
        begin.clear_color = background();
        const vg::RenderTarget rt = target.value().target();
        rt.begin(cmd, begin);
        record_skybox(cmd, {width, height}, skybox, view_proj, rig.position());
        pipelines::PbrFrame frame;
        frame.extent = {width, height};
        frame.view_proj = view_proj;
        frame.scene = &scene;
        frame.draws = gpu_model.value().draws().data();
        frame.draw_count =
            static_cast<uint32_t>(gpu_model.value().draws().size());
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

  const Bounds bounds = compute_bounds(model);
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

  // The whole assets::Model -> GPU bridge in one call: meshes + material maps
  // uploaded (a single submit), materials built, the scene flattened into a
  // draw list ready for PbrFrame.
  auto gpu_model = pipelines::PbrModel::create(
      device.value(), allocator.value(), pipeline.value(), model);
  if (!gpu_model.ok()) {
    std::fprintf(stderr, "model: %s\n", gpu_model.status().message().c_str());
    return 1;
  }

  // Two frames in flight: the swapchain keeps depth per image and the scene
  // UBO rings per slot, so nothing is shared across in-flight frames.
  constexpr uint32_t kFramesInFlight = 2;
  Ibl ibl = make_ibl(device.value(), allocator.value(), &ok);
  if (!ok) {
    return 1;
  }
  pipelines::PbrScene scene =
      make_pbr_scene(device.value(), allocator.value(), pipeline.value(), ibl,
                     kFramesInFlight, &ok);
  if (!ok) {
    return 1;
  }

  Skybox skybox = setup_skybox(device.value(), allocator.value(),
                               swapchain.value().layout(), &ok);
  if (!ok) {
    return 1;
  }

  // CPU-ahead depth shared by the loop and the profiler driving it (declared
  // above, before make_pbr_scene, so the scene UBO ring matches).
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
    scene.set_camera(f.slot, rig.position(), ibl.prefilter_max_lod);
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
    frame_info.scene = &scene;
    frame_info.slot = f.slot;
    frame_info.draws = gpu_model.value().draws().data();
    frame_info.draw_count =
        static_cast<uint32_t>(gpu_model.value().draws().size());
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

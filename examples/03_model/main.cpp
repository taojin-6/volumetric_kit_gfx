// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/03_model: load a glTF model and fly around it. Parses a
// `.gltf`/`.glb` with the io tier into a CPU assets::Model, bridges it to the
// GPU with pipelines::PbrModel (meshes + material maps uploaded in one submit,
// materials built, the scene flattened into a draw list), and draws it
// depth-tested with glTF metallic-roughness PBR: a per-draw model/MVP push
// constant, a per-frame scene set (set 0: camera + IBL), and a per-material
// set (set 1: factor UBO + the five maps) -- all reflected automatically into
// the pipeline layout. The ambient comes from pipelines::bake_ibl convolving
// the analytic sky, which is also drawn as a skybox behind the model. The
// camera auto-frames the model's bounds, so any model fills the view.
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
// Two render paths share the model load + upload + draw recording, each with
// its bring-up collapsed into one app-tier call:
//  * Windowed (default): app::WindowedApp (instance -> surface -> device ->
//    swapchain -> frame loop), driven by mouse + keyboard (a deterministic
//    turntable instead under --frames). The swapchain owns a depth attachment
//    per image (SwapchainConfig::depth_format) and rebuilds it on resize, so
//    each frame renders straight into the loop's depth-capable target at two
//    frames in flight.
//  * --screenshot: app::HeadlessApp (no window, no surface) -- renders one
//    frame into an OffscreenTarget (color + depth + readback) and writes a
//    binary PPM. Headless, so it works where no display / screen-capture is
//    available.

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

#include "common/glfw_surface.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "volumetric_kit/gfx/app/headless_app.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/assets/model.hpp"
#include "volumetric_kit/gfx/camera/camera_rig.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/io/gltf_loader.hpp"
#include "volumetric_kit/gfx/pipelines/ibl.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_model.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_scene.hpp"
#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"
#include "volumetric_kit/gfx/ui/metrics_panel.hpp"
#include "volumetric_kit/gfx/windowing.hpp"

namespace vg = volumetric_kit::gfx;
namespace win = volumetric_kit::gfx::windowing;
namespace assets = volumetric_kit::gfx::assets;
namespace camera = volumetric_kit::gfx::camera;
namespace pipelines = volumetric_kit::gfx::pipelines;

namespace {

constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr float kFovY = 1.0471976f;  // 60 degrees

// TODO: load_spirv + framebuffer_extent are still duplicated across examples
// 01/02/03; hoist them into examples/common/ too (the GLFW surface factory
// already lives there) in a focused cleanup.
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
  // up so the camera looks down on the model. Mirrors
  // camera::spherical_direction (in camera/impl/orientation.hpp, internal to
  // the tier so the example can't include it) -- keep the two in step.
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

// Build the pipelines-tier PbrScene (set 0: camera + IBL) against the
// pipeline's reflected set-0 layout, one camera UBO ring slot per frame in
// flight. The caller refreshes the acquired slot's camera each frame through
// scene.set_camera(slot, ...). Everything else GPU-side for the model -- the
// meshes, material maps, materials (set 1), and draw list -- comes from
// pipelines::PbrModel::create.
pipelines::PbrScene make_pbr_scene(const vg::Device& device,
                                   vg::Allocator& alloc,
                                   const pipelines::PbrPipeline& pipeline,
                                   const pipelines::IblMaps& ibl,
                                   uint32_t frames_in_flight, bool* ok) {
  auto scene = pipelines::PbrScene::create(device.handle(), alloc,
                                           pipeline.descriptor_set_layout(0),
                                           ibl.scene_desc(), frames_in_flight);
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

// Bake the analytic sky into a sampled-ready cubemap: generate the six faces
// on the CPU (pipelines::cube_face_direction keeps the orientation in lockstep
// with the IBL bake) and upload them through the core cube-upload path in one
// submit. Stores linear HDR color in a float cube (the skybox shader tone-maps
// it on output).
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
        const glm::vec3 c = sky_color(pipelines::cube_face_direction(f, u, v));
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

// --- Headless path: render one frame into an OffscreenTarget, write a PPM
// -----

int run_screenshot(const char* model_path, const char* out_path, uint32_t width,
                   uint32_t height) {
  // The whole headless bring-up (instance -> device -> allocator, no surface,
  // no present queue) in one call.
  vg::app::HeadlessAppConfig app_config;
  app_config.app_name = "03_model";
  app_config.enable_validation = true;  // a no-op when the layer is absent
  auto created = vg::app::HeadlessApp::create(app_config);
  if (!created.ok()) {
    std::fprintf(stderr, "app: %s\n", created.status().message().c_str());
    return 1;
  }
  vg::app::HeadlessApp app = std::move(created).value();

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
  auto target = vg::OffscreenTarget::create(app.allocator(), target_desc);
  if (!target.ok()) {
    std::fprintf(stderr, "offscreen: %s\n", target.status().message().c_str());
    return 1;
  }

  auto pipeline = pipelines::PbrPipeline::create(app.device().handle(),
                                                 target.value().layout());
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  // The whole assets::Model -> GPU bridge in one call: meshes + material maps
  // uploaded (a single submit), materials built, the scene flattened into a
  // draw list ready for PbrFrame.
  auto gpu_model = pipelines::PbrModel::create(app.device(), app.allocator(),
                                               pipeline.value(), model);
  if (!gpu_model.ok()) {
    std::fprintf(stderr, "model: %s\n", gpu_model.status().message().c_str());
    return 1;
  }

  // Convolve the analytic sky into the IBL maps (irradiance + prefiltered
  // specular + BRDF LUT); sky_color is pure, as the concurrent bake requires.
  auto ibl = pipelines::bake_ibl(app.device(), app.allocator(), sky_color);
  if (!ibl.ok()) {
    std::fprintf(stderr, "ibl: %s\n", ibl.status().message().c_str());
    return 1;
  }
  pipelines::PbrScene scene =
      make_pbr_scene(app.device(), app.allocator(), pipeline.value(),
                     ibl.value(), /*frames_in_flight=*/1, &ok);
  if (!ok) {
    return 1;
  }

  // Fixed camera for the still: write the eye into the scene set once.
  scene.set_camera(0, rig.position(), ibl.value().prefilter_max_lod);

  Skybox skybox =
      setup_skybox(app.device(), app.allocator(), target.value().layout(), &ok);
  if (!ok) {
    return 1;
  }

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const glm::mat4 view_proj =
      rig.to_camera(kFovY, aspect, clip.first, clip.second).view_proj();

  const vg::Status recorded =
      app.device().submit_single_time([&](VkCommandBuffer cmd) {
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
                 float move_speed, bool capture_mouse, bool capture_keyboard) {
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

  // The ImGui overlay claims the cursor/wheel when hovered (capture_mouse) and
  // the keys when a widget is focused (capture_keyboard); skip the matching
  // camera verbs so dragging or scrolling the panel does not move the camera.
  const bool left =
      !capture_mouse &&
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  const bool panning =
      !capture_mouse &&
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
       glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  if (left) {
    rig.orbit(-dx * kOrbitSpeed, -dy * kOrbitSpeed);
  } else if (panning) {
    const float scale = kPanSpeed * rig.focus_distance();
    rig.pan(-dx * scale, dy * scale);
  }

  if (capture_mouse) {
    input.scroll = 0.0;  // the wheel scrolled the panel, not the camera
  } else if (input.scroll != 0.0) {
    rig.zoom(std::pow(kZoomStep, static_cast<float>(input.scroll)));
    input.scroll = 0.0;
  }

  glm::vec3 move(0.0f);
  if (!capture_keyboard) {
    move.x += glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ? 1.0f : 0.0f;
    move.x -= glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ? 1.0f : 0.0f;
    move.y += glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS ? 1.0f : 0.0f;
    move.y -= glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS ? 1.0f : 0.0f;
    move.z += glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ? 1.0f : 0.0f;
    move.z -= glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ? 1.0f : 0.0f;
  }
  if (glm::dot(move, move) > 0.0f) {  // a movement key is held
    const bool fast = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    rig.move_local(glm::normalize(move) * move_speed * (fast ? 4.0f : 1.0f) *
                   dt);
  }
}

// --- Windowed path: app::WindowedApp -----------------------------------------

// Owns all Vulkan/windowing state for one window; everything is destroyed when
// this returns, before main() tears GLFW down. Interactive by default; with
// max_frames >= 0 it runs a deterministic turntable and exits (for CI).
int run_windowed(GLFWwindow* window, const char* model_path, int max_frames) {
  uint32_t glfw_ext_count = 0;
  const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  // Two frames in flight: the swapchain keeps depth per image and the scene
  // UBO rings per slot, so nothing is shared across in-flight frames.
  constexpr uint32_t kFramesInFlight = 2;

  // The whole bring-up chain in one call; the lambda supplies the GLFW
  // surface. SwapchainConfig::depth_format makes the swapchain own a depth
  // attachment per image (rebuilt with the chain on resize), so its render
  // targets are depth-capable.
  vg::app::WindowedAppConfig app_config;
  app_config.app_name = "03_model";
  app_config.enable_validation = true;  // a no-op when the layer is absent
  app_config.instance_extensions.assign(glfw_exts, glfw_exts + glfw_ext_count);
  app_config.swapchain.extent = framebuffer_extent(window);
  app_config.swapchain.depth_format = kDepthFormat;
  app_config.frames_in_flight = kFramesInFlight;
  auto created = vg::app::WindowedApp::create(
      app_config, example::glfw_surface_factory(window));
  if (!created.ok()) {
    std::fprintf(stderr, "app: %s\n", created.status().message().c_str());
    return 1;
  }
  vg::app::WindowedApp app = std::move(created).value();

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

  // The pipeline is built for the swapchain's layout (color + depth formats).
  // Size-independent (dynamic viewport), so it survives resizes without a
  // rebuild.
  auto pipeline = pipelines::PbrPipeline::create(app.device().handle(),
                                                 app.swapchain().layout());
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  // The whole assets::Model -> GPU bridge in one call: meshes + material maps
  // uploaded (a single submit), materials built, the scene flattened into a
  // draw list ready for PbrFrame.
  auto gpu_model = pipelines::PbrModel::create(app.device(), app.allocator(),
                                               pipeline.value(), model);
  if (!gpu_model.ok()) {
    std::fprintf(stderr, "model: %s\n", gpu_model.status().message().c_str());
    return 1;
  }

  // Convolve the analytic sky into the IBL maps (irradiance + prefiltered
  // specular + BRDF LUT); sky_color is pure, as the concurrent bake requires.
  auto ibl = pipelines::bake_ibl(app.device(), app.allocator(), sky_color);
  if (!ibl.ok()) {
    std::fprintf(stderr, "ibl: %s\n", ibl.status().message().c_str());
    return 1;
  }
  pipelines::PbrScene scene =
      make_pbr_scene(app.device(), app.allocator(), pipeline.value(),
                     ibl.value(), kFramesInFlight, &ok);
  if (!ok) {
    return 1;
  }

  Skybox skybox = setup_skybox(app.device(), app.allocator(),
                               app.swapchain().layout(), &ok);
  if (!ok) {
    return 1;
  }

  // CPU-ahead depth shared by the app's loop and the profiler driving it.
  // Borrowed by the loop (set_profiler below) and detached again before
  // teardown.
  vg::ProfilerConfig profiler_config;
  profiler_config.frames_in_flight = kFramesInFlight;
  auto profiler = vg::Profiler::create(app.device(), profiler_config);
  if (!profiler.ok()) {
    std::fprintf(stderr, "profiler: %s\n", profiler.status().message().c_str());
    return 1;
  }

  // Turnkey: the app's loop now calls profiler.begin_frame/end_frame for us,
  // so the render loop only opens a scope around each pass. Resizes need no
  // hook — the swapchain rebuilds its own depth attachments with the chain.
  app.set_profiler(&profiler.value());

  // Debug overlay (ui tier): a Dear ImGui panel of the profiler's per-pass
  // metrics, composited on top of the scene. Its pipeline bakes the swapchain
  // layout's color + depth formats, so it draws into the same depth-capable
  // target; the swapchain holds its format + image count stable across
  // recreate, so the overlay survives resizes without rebuilding.
  vg::ui::ImGuiOverlayConfig overlay_config;
  overlay_config.layout = app.swapchain().layout();
  overlay_config.min_image_count = app.swapchain().image_count();
  overlay_config.image_count = app.swapchain().image_count();
  auto overlay = vg::ui::ImGuiOverlay::create(
      app.device(), app.instance().handle(), overlay_config);
  if (!overlay.ok()) {
    std::fprintf(stderr, "overlay: %s\n", overlay.status().message().c_str());
    return 1;
  }
  // Platform backend (this example's half): bind it to the overlay's context,
  // then let it feed input + io.DisplaySize each frame. Initialized last, after
  // the fallible setup above, so no earlier error return leaves a live
  // ImGui_ImplGlfw backend without its paired Shutdown at teardown; it chains
  // onto the scroll callback registered above (interactive), so the wheel still
  // reaches the camera.
  ImGui::SetCurrentContext(overlay.value().context());
  if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
    std::fprintf(stderr, "ImGui_ImplGlfw_InitForVulkan failed\n");
    return 1;
  }

  // A hard error inside the loop breaks out to the shared teardown below
  // (wait_idle + set_profiler(nullptr) + ImGui_ImplGlfw_Shutdown) rather than
  // returning straight away, so in-flight frames are drained before the
  // after-app resources (pipeline, model, scene, skybox, profiler, overlay)
  // destruct.
  int exit_code = 0;
  int rendered = 0;
  while (!glfwWindowShouldClose(window)) {
    if (max_frames >= 0 && rendered >= max_frames) {
      break;
    }
    glfwPollEvents();

    // The app's loop owns the staleness protocol: it rebuilds the swapchain
    // (which rebuilds its per-image depth attachments) after a resize, and
    // skips ticks while the window is minimized.
    auto frame = app.begin_frame(framebuffer_extent(window));
    if (!frame.ok()) {
      std::fprintf(stderr, "begin_frame: %s\n",
                   frame.status().message().c_str());
      exit_code = 1;
      break;
    }
    if (!frame.value().has_value()) {
      // Paused: minimized, or the surface is still settling after a rebuild.
      // Idle briefly rather than block outright, so a settling surface retries
      // even when the compositor sends no further event.
      glfwWaitEventsTimeout(0.1);
      continue;
    }
    const win::Frame& f = *frame.value();

    const VkExtent2D extent = app.swapchain().extent();
    const VkCommandBuffer cmd = f.cmd;

    // Open the ImGui frame and build the metrics panel before recording:
    // new_frame computes io.WantCaptureMouse/Keyboard, which gate the camera
    // input below so the panel takes the cursor/keys when the user is on it.
    ImGui_ImplGlfw_NewFrame();
    overlay.value().new_frame();
    const ImGuiIO& io = ImGui::GetIO();
    vg::ui::draw_metrics_panel(profiler.value().metrics());

    // The acquired image's target already pairs its color view with its own
    // depth attachment; the default load op clears both.
    vg::RenderTargetBeginInfo begin;
    begin.clear_color = background();
    f.target->begin(cmd, begin);

    // Advance the camera: interactive input, or a deterministic per-frame
    // turntable step under --frames (reproducible for CI).
    if (interactive) {
      apply_input(window, input, rig, move_speed, io.WantCaptureMouse,
                  io.WantCaptureKeyboard);
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
    scene.set_camera(f.slot, rig.position(), ibl.value().prefilter_max_lod);
    {
      // Per-pass GPU stages: a timestamp pair + a VK_EXT_debug_utils label
      // around each, resolved into the metrics the overlay panel shows.
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
    // Composite the debug overlay on top of the scene, within the same
    // dynamic-rendering scope.
    overlay.value().render(cmd);

    f.target->end(cmd);

    const vg::Status present = app.end_frame(f);
    if (!present.ok() && !win::swapchain_stale(present)) {
      std::fprintf(stderr, "end_frame: %s\n", present.message().c_str());
      exit_code = 1;
      break;
    }

    // The resolved per-pass timings now show live in the overlay panel (a
    // slot's GPU times resolve when the slot recurs, so they trail the current
    // frame by kFramesInFlight; on a device without timestamp support the GPU
    // column reads n/a and CPU times remain).
    ++rendered;
  }

  // Everything above (pipeline, model, scene, skybox, profiler, overlay) was
  // created after the app, so it destructs before it — while the app's frame
  // loop may still have frames in flight referencing it (including after an
  // error break above). Idle the device first so that teardown is safe, and
  // detach the borrowed profiler from the loop before it goes out of scope.
  app.wait_idle();
  app.set_profiler(nullptr);
  // Tear the platform backend down while the ImGui context is still alive; the
  // overlay's destructor then shuts the renderer backend down and destroys it.
  ImGui::SetCurrentContext(overlay.value().context());
  ImGui_ImplGlfw_Shutdown();
  std::printf("03_model: rendered %d frame(s)\n", rendered);
  return exit_code;
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

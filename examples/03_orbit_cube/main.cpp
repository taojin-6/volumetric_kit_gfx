// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/03_orbit_cube: an interactive RGB cube viewed through an orbit
// camera, drawn into the swapchain via the windowing tier. Drag with the left
// mouse button to orbit, scroll to zoom, and press W to toggle wireframe. This
// is the first example that ties the camera + core descriptor/MVP path
// together: each frame the OrbitCamera bakes an MVP that is written into a
// per-slot uniform buffer and bound as a descriptor set the pipeline reflected
// from mesh_mvp.vert.
//
// The cube needs no depth buffer: it is convex, so backface culling (front
// faces counter-clockwise, matching the camera's Y-flip) draws it correctly.
// Wireframe disables culling and needs the device's fillModeNonSolid feature --
// the toggle is a no-op when the device lacks it.
//
// Run with `--frames N` to render N frames and exit, so CI can drive the real
// window -> swapchain -> present path headlessly under Xvfb.

#include "volumetric_kit/gfx/core/vulkan.hpp"  // before GLFW, so it sees Vulkan

#include <GLFW/glfw3.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/trigonometric.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "volumetric_kit/gfx/camera/orbit_camera.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace vg = volumetric_kit::gfx;
namespace cam = volumetric_kit::gfx::camera;
namespace win = volumetric_kit::gfx::windowing;

namespace {

struct Vertex {
  float pos[3];
  float color[3];
};

// The classic RGB cube: 8 corners of a unit cube, each tinted by its position
// (so opposite corners are complementary colors). 36 indices wind every face
// counter-clockwise when seen from outside, so backface culling keeps it solid.
constexpr Vertex kCubeVertices[8] = {
    {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}},
};
constexpr uint32_t kCubeIndices[36] = {
    4, 5, 6, 4, 6, 7,  // +Z
    1, 0, 3, 1, 3, 2,  // -Z
    5, 1, 2, 5, 2, 6,  // +X
    0, 4, 7, 0, 7, 3,  // -X
    7, 6, 2, 7, 2, 3,  // +Y
    0, 1, 5, 0, 5, 4,  // -Y
};

// Interactive state hung off the GLFW window so the input callbacks can reach
// it.
struct AppState {
  cam::OrbitCamera orbit;
  bool dragging = false;
  double last_x = 0.0;
  double last_y = 0.0;
  bool wireframe = false;
  bool wireframe_available = false;
};

AppState& app(GLFWwindow* window) {
  return *static_cast<AppState*>(glfwGetWindowUserPointer(window));
}

void on_cursor_pos(GLFWwindow* window, double x, double y) {
  AppState& state = app(window);
  if (state.dragging) {
    const float dx = static_cast<float>(x - state.last_x);
    const float dy = static_cast<float>(y - state.last_y);
    // Drag right -> the cube turns to follow the cursor; drag up -> tilt up.
    state.orbit.orbit(-dx * 0.01f, -dy * 0.01f);
  }
  state.last_x = x;
  state.last_y = y;
}

void on_mouse_button(GLFWwindow* window, int button, int action, int /*mods*/) {
  if (button != GLFW_MOUSE_BUTTON_LEFT) {
    return;
  }
  AppState& state = app(window);
  state.dragging = action == GLFW_PRESS;
  glfwGetCursorPos(window, &state.last_x, &state.last_y);
}

void on_scroll(GLFWwindow* window, double /*x*/, double y) {
  // Wheel up zooms in (shrinks the orbit radius), wheel down zooms out.
  app(window).orbit.zoom(y > 0.0 ? 0.9f : 1.0f / 0.9f);
}

void on_key(GLFWwindow* window, int key, int /*sc*/, int action, int /*mods*/) {
  AppState& state = app(window);
  if (action != GLFW_PRESS) {
    return;
  }
  if (key == GLFW_KEY_W && state.wireframe_available) {
    state.wireframe = !state.wireframe;
  } else if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

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

vg::GraphicsPipeline build_cube_pipeline(
    VkDevice device, const vg::ShaderModule& vert, const vg::ShaderModule& frag,
    const vg::RenderTargetLayout& layout,
    const VkVertexInputBindingDescription& binding,
    const VkVertexInputAttributeDescription* attrs, bool wireframe) {
  vg::GraphicsPipelineDesc desc;
  desc.vertex_shader = &vert;
  desc.fragment_shader = &frag;
  desc.layout = layout;
  desc.vertex_bindings = &binding;
  desc.vertex_binding_count = 1;
  desc.vertex_attributes = attrs;
  desc.vertex_attribute_count = 2;
  // Solid: cull back faces so the convex cube needs no depth buffer. Wireframe:
  // line fill, no culling, so every edge shows through.
  desc.polygon_mode = wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
  desc.cull_mode = wireframe ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
  auto pipeline = vg::GraphicsPipeline::create(device, desc);
  return pipeline.ok() ? std::move(pipeline).value() : vg::GraphicsPipeline{};
}

// Owns all Vulkan/windowing state for one window; everything is destroyed when
// this returns, before main() tears GLFW down.
int run(GLFWwindow* window, AppState& state, int max_frames) {
  uint32_t glfw_ext_count = 0;
  const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  vg::InstanceConfig instance_config;
  instance_config.app_name = "03_orbit_cube";
  instance_config.enable_validation = true;
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

  // Wireframe needs fillModeNonSolid; enable it only when the device offers it,
  // and gate the W toggle on the same flag.
  VkPhysicalDeviceFeatures supported{};
  vkGetPhysicalDeviceFeatures(physical.value(), &supported);
  state.wireframe_available = supported.fillModeNonSolid == VK_TRUE;

  vg::DeviceConfig device_config;
  device_config.needs_present = true;
  device_config.features.fillModeNonSolid = supported.fillModeNonSolid;
  auto device = vg::Device::create(instance.value().handle(), physical.value(),
                                   device_config, surface.handle());
  if (!device.ok()) {
    std::fprintf(stderr, "device: %s\n", device.status().message().c_str());
    return 1;
  }
  const VkDevice dev = device.value().handle();

  std::vector<uint32_t> vert_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/mesh_mvp.vert.spv");
  std::vector<uint32_t> frag_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/mesh.frag.spv");
  if (vert_code.empty() || frag_code.empty()) {
    std::fprintf(stderr, "missing compiled shaders in %s\n",
                 VG_EXAMPLE_SHADER_DIR);
    return 1;
  }
  auto vert = vg::ShaderModule::create(dev, vert_code.data(),
                                       vert_code.size() * sizeof(uint32_t));
  auto frag = vg::ShaderModule::create(dev, frag_code.data(),
                                       frag_code.size() * sizeof(uint32_t));
  if (!vert.ok() || !frag.ok()) {
    std::fprintf(stderr, "shader module creation failed\n");
    return 1;
  }

  auto allocator =
      vg::Allocator::create(instance.value().handle(), device.value());
  if (!allocator.ok()) {
    std::fprintf(stderr, "allocator: %s\n",
                 allocator.status().message().c_str());
    return 1;
  }

  // Host-visible geometry buffers, filled once. Static geometry would normally
  // live in device-local memory uploaded via a staging buffer; host-visible
  // keeps this example short.
  auto make_filled = [&](const void* data, size_t size,
                         VkBufferUsageFlags usage) -> vg::Buffer {
    vg::BufferDesc d;
    d.size = size;
    d.usage = usage;
    d.memory = vg::MemoryUsage::HostVisible;
    d.mapped = true;
    auto buf = allocator.value().create_buffer(d);
    if (!buf.ok()) {
      return vg::Buffer{};
    }
    vg::Buffer out = std::move(buf).value();
    std::memcpy(out.mapped(), data, size);
    return out;
  };
  vg::Buffer vbuf = make_filled(kCubeVertices, sizeof(kCubeVertices),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  vg::Buffer ibuf = make_filled(kCubeIndices, sizeof(kCubeIndices),
                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  if (!vbuf.valid() || !ibuf.valid()) {
    std::fprintf(stderr, "geometry buffer allocation failed\n");
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

  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = sizeof(Vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  VkVertexInputAttributeDescription attrs[2]{};
  attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
  attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};

  vg::GraphicsPipeline solid =
      build_cube_pipeline(dev, vert.value(), frag.value(),
                          swapchain.value().layout(), binding, attrs, false);
  vg::GraphicsPipeline wire;
  if (state.wireframe_available) {
    wire =
        build_cube_pipeline(dev, vert.value(), frag.value(),
                            swapchain.value().layout(), binding, attrs, true);
  }
  if (!solid.valid() || (state.wireframe_available && !wire.valid())) {
    std::fprintf(stderr, "pipeline creation failed\n");
    return 1;
  }

  // One uniform buffer + descriptor set per in-flight slot, so updating the MVP
  // for the next frame never races the GPU still reading the previous one. Both
  // pipelines share the reflected set-0 layout, so one set works for either.
  constexpr uint32_t kSlots = 2;
  const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       kSlots};
  auto pool = vg::DescriptorPool::create(dev, &pool_size, 1, kSlots);
  if (!pool.ok()) {
    std::fprintf(stderr, "descriptor pool: %s\n",
                 pool.status().message().c_str());
    return 1;
  }
  std::vector<vg::Buffer> ubos;
  std::vector<vg::DescriptorSet> sets;
  for (uint32_t i = 0; i < kSlots; ++i) {
    vg::BufferDesc ud;
    ud.size = sizeof(glm::mat4);
    ud.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ud.memory = vg::MemoryUsage::HostVisible;
    ud.mapped = true;
    auto ubo = allocator.value().create_buffer(ud);
    auto set = pool.value().allocate(solid.descriptor_set_layout(0));
    if (!ubo.ok() || !set.ok()) {
      std::fprintf(stderr, "uniform/descriptor allocation failed\n");
      return 1;
    }
    set.value().write_uniform_buffer(0, ubo.value().handle(), 0,
                                     sizeof(glm::mat4));
    ubos.push_back(std::move(ubo).value());
    sets.push_back(set.value());
  }

  auto loop = win::FrameLoop::create(device.value(), swapchain.value(), kSlots);
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
        continue;
      }
      std::fprintf(stderr, "begin_frame: %s\n",
                   frame.status().message().c_str());
      return 1;
    }

    const VkExtent2D extent = swapchain.value().extent();
    const float aspect = extent.height == 0
                             ? 1.0f
                             : static_cast<float>(extent.width) /
                                   static_cast<float>(extent.height);
    const glm::mat4 mvp =
        state.orbit.to_camera(glm::radians(60.0f), aspect, 0.1f, 100.0f)
            .view_proj();
    const uint32_t slot = frame.value().slot;
    std::memcpy(ubos[slot].mapped(), glm::value_ptr(mvp), sizeof(glm::mat4));

    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[0] = 0.02f;
    begin.clear_color.float32[1] = 0.02f;
    begin.clear_color.float32[2] = 0.05f;
    begin.clear_color.float32[3] = 1.0f;
    frame.value().target->begin(frame.value().cmd, begin);

    const vg::GraphicsPipeline& active =
        (state.wireframe && state.wireframe_available) ? wire : solid;
    vkCmdBindPipeline(frame.value().cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      active.handle());
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.value().cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(frame.value().cmd, 0, 1, &scissor);

    const VkDeviceSize offset = 0;
    const VkBuffer vb = vbuf.handle();
    vkCmdBindVertexBuffers(frame.value().cmd, 0, 1, &vb, &offset);
    vkCmdBindIndexBuffer(frame.value().cmd, ibuf.handle(), 0,
                         VK_INDEX_TYPE_UINT32);
    const VkDescriptorSet ds = sets[slot].handle();
    vkCmdBindDescriptorSets(frame.value().cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            active.layout(), 0, 1, &ds, 0, nullptr);
    vkCmdDrawIndexed(frame.value().cmd, 36, 1, 0, 0, 0);

    frame.value().target->end(frame.value().cmd);

    const vg::Status present = loop.value().end_frame(frame.value());
    if (!present.ok()) {
      if (present.code() == VK_ERROR_OUT_OF_DATE_KHR ||
          present.code() == VK_SUBOPTIMAL_KHR) {
        if (!swapchain.value().recreate(framebuffer_extent(window)).ok()) {
          break;
        }
      } else {
        std::fprintf(stderr, "end_frame: %s\n", present.message().c_str());
        return 1;
      }
    }
    ++rendered;
  }

  vkDeviceWaitIdle(dev);
  std::printf("03_orbit_cube: rendered %d frame(s)%s\n", rendered,
              state.wireframe_available ? ""
                                        : " (no wireframe: device lacks "
                                          "fillModeNonSolid)");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int max_frames = -1;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--frames") == 0) {
      max_frames = std::atoi(argv[i + 1]);
    }
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
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(
      800, 600, "volumetric_kit_gfx \xE2\x80\x94 03_orbit_cube", nullptr,
      nullptr);
  if (window == nullptr) {
    std::fprintf(stderr, "glfwCreateWindow failed\n");
    glfwTerminate();
    return 1;
  }

  AppState state;
  state.orbit.set_target({0.0f, 0.0f, 0.0f});
  state.orbit.set_distance(3.0f);
  state.orbit.set_azimuth(0.6f);
  state.orbit.set_elevation(0.5f);
  glfwSetWindowUserPointer(window, &state);
  glfwSetCursorPosCallback(window, on_cursor_pos);
  glfwSetMouseButtonCallback(window, on_mouse_button);
  glfwSetScrollCallback(window, on_scroll);
  glfwSetKeyCallback(window, on_key);

  const int rc = run(window, state, max_frames);

  glfwDestroyWindow(window);
  glfwTerminate();
  return rc;
}

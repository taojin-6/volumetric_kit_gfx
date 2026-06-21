// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/01_triangle: open a window and draw the hello-triangle into the
// swapchain through the windowing tier (Surface + Swapchain + FrameLoop) on
// dynamic rendering. Run with `--frames N` to render N frames and exit — CI
// drives that under Xvfb to validate the real window -> surface -> swapchain ->
// present path headlessly.

#include "volumetric_kit/gfx/core/vulkan.hpp"  // before GLFW, so glfw3.h sees
// Vulkan and declares its helpers

#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace vg = volumetric_kit::gfx;
namespace win = volumetric_kit::gfx::windowing;

namespace {

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

// Owns all Vulkan/windowing state for one window; everything is destroyed when
// this returns, before main() tears GLFW down.
int run(GLFWwindow* window, int max_frames) {
  uint32_t glfw_ext_count = 0;
  const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  vg::InstanceConfig instance_config;
  instance_config.app_name = "01_triangle";
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

  std::vector<uint32_t> vert_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/triangle.vert.spv");
  std::vector<uint32_t> frag_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/triangle.frag.spv");
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

  vg::GraphicsPipelineDesc pipeline_desc;
  pipeline_desc.vertex_shader = &vert.value();
  pipeline_desc.fragment_shader = &frag.value();
  pipeline_desc.layout = swapchain.value().layout();
  auto pipeline =
      vg::GraphicsPipeline::create(device.value().handle(), pipeline_desc);
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  auto loop = win::FrameLoop::create(device.value(), swapchain.value(), 2);
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

    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[0] = 0.02f;
    begin.clear_color.float32[1] = 0.02f;
    begin.clear_color.float32[2] = 0.05f;
    begin.clear_color.float32[3] = 1.0f;
    frame.value().target->begin(frame.value().cmd, begin);

    vkCmdBindPipeline(frame.value().cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.value().handle());
    const VkExtent2D extent = swapchain.value().extent();
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.value().cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(frame.value().cmd, 0, 1, &scissor);
    vkCmdDraw(frame.value().cmd, 3, 1, 0, 0);

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

  vkDeviceWaitIdle(device.value().handle());
  std::printf("01_triangle: rendered %d frame(s)\n", rendered);
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
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Vulkan, not OpenGL
  GLFWwindow* window =
      glfwCreateWindow(800, 600, "volumetric_kit_gfx \xE2\x80\x94 01_triangle",
                       nullptr, nullptr);
  if (window == nullptr) {
    std::fprintf(stderr, "glfwCreateWindow failed\n");
    glfwTerminate();
    return 1;
  }

  const int rc = run(window, max_frames);

  glfwDestroyWindow(window);
  glfwTerminate();
  return rc;
}

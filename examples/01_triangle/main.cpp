// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/01_triangle: open a window and draw the hello-triangle into the
// swapchain on dynamic rendering. The whole instance -> surface -> device ->
// swapchain -> frame-loop bring-up is one app::WindowedApp::create call; the
// example keeps only what is its own — the GLFW window, the shaders/pipeline,
// and the render loop. Run with `--frames N` to render N frames and exit — CI
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

#include "common/glfw_surface.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/windowing.hpp"

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

  // The whole bring-up chain in one call; example::glfw_surface_factory
  // supplies the GLFW surface, keeping the library tier window-system-free.
  vg::app::WindowedAppConfig config;
  config.app_name = "01_triangle";
  config.enable_validation = true;  // a no-op when the layer is absent
  config.instance_extensions.assign(glfw_exts, glfw_exts + glfw_ext_count);
  config.swapchain.extent = framebuffer_extent(window);
  auto created = vg::app::WindowedApp::create(
      config, example::glfw_surface_factory(window));
  if (!created.ok()) {
    std::fprintf(stderr, "app: %s\n", created.status().message().c_str());
    return 1;
  }
  vg::app::WindowedApp app = std::move(created).value();

  std::vector<uint32_t> vert_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/triangle.vert.spv");
  std::vector<uint32_t> frag_code =
      load_spirv(VG_EXAMPLE_SHADER_DIR "/triangle.frag.spv");
  if (vert_code.empty() || frag_code.empty()) {
    std::fprintf(stderr, "missing compiled shaders in %s\n",
                 VG_EXAMPLE_SHADER_DIR);
    return 1;
  }
  auto vert = vg::ShaderModule::create(app.device().handle(), vert_code.data(),
                                       vert_code.size() * sizeof(uint32_t));
  auto frag = vg::ShaderModule::create(app.device().handle(), frag_code.data(),
                                       frag_code.size() * sizeof(uint32_t));
  if (!vert.ok() || !frag.ok()) {
    std::fprintf(stderr, "shader module creation failed\n");
    return 1;
  }

  vg::GraphicsPipelineDesc pipeline_desc;
  pipeline_desc.vertex_shader = &vert.value();
  pipeline_desc.fragment_shader = &frag.value();
  pipeline_desc.layout = app.swapchain().layout();
  auto pipeline =
      vg::GraphicsPipeline::create(app.device().handle(), pipeline_desc);
  if (!pipeline.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", pipeline.status().message().c_str());
    return 1;
  }

  // A hard error inside the loop breaks out to the shared wait_idle() teardown
  // below rather than returning straight away, so any in-flight frame is
  // drained before the after-app resources (pipeline, shaders) destruct.
  int exit_code = 0;
  int rendered = 0;
  while (!glfwWindowShouldClose(window)) {
    if (max_frames >= 0 && rendered >= max_frames) {
      break;
    }
    glfwPollEvents();

    // The app's loop owns the staleness protocol: it rebuilds the swapchain
    // after a resize / out-of-date result and skips ticks while the window is
    // minimized, so only hard failures surface here.
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

    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[0] = 0.02f;
    begin.clear_color.float32[1] = 0.02f;
    begin.clear_color.float32[2] = 0.05f;
    begin.clear_color.float32[3] = 1.0f;
    f.target->begin(f.cmd, begin);

    vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.value().handle());
    const VkExtent2D extent = app.swapchain().extent();
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(f.cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(f.cmd, 0, 1, &scissor);
    vkCmdDraw(f.cmd, 3, 1, 0, 0);

    f.target->end(f.cmd);

    const vg::Status present = app.end_frame(f);
    if (!present.ok() && !win::swapchain_stale(present)) {
      std::fprintf(stderr, "end_frame: %s\n", present.message().c_str());
      exit_code = 1;
      break;
    }
    ++rendered;
  }

  // The shaders + pipeline were created after the app, so they destruct before
  // it — while its frame loop may still have frames in flight that reference
  // them (including after an error break above). Idle the device first so their
  // destruction is safe.
  app.wait_idle();
  std::printf("01_triangle: rendered %d frame(s)\n", rendered);
  return exit_code;
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

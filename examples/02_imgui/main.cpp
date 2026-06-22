// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/02_imgui: a Dear ImGui debug overlay drawn into the swapchain
// through the ui tier (ImGuiOverlay) on dynamic rendering. Shows the split the
// ui tier is built around: this example owns the *platform* backend
// (imgui_impl_glfw — input + window sizing), while ImGuiOverlay wraps only the
// *renderer* backend (imgui_impl_vulkan), so the library tier stays GLFW-free
// like windowing. Run with `--frames N` to render N frames and exit — CI drives
// that under Xvfb with validation enabled to exercise the path headlessly.

#include "volumetric_kit/gfx/core/vulkan.hpp"  // before GLFW, so glfw3.h sees
// Vulkan and declares its helpers

#include <GLFW/glfw3.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"
#include "volumetric_kit/gfx/windowing.hpp"

namespace vg = volumetric_kit::gfx;
namespace win = volumetric_kit::gfx::windowing;

namespace {

VkExtent2D framebuffer_extent(GLFWwindow* window) {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

// Build this frame's UI. A real consumer would surface render/camera/material
// knobs here; the demo window stands in as a non-trivial draw list.
void build_ui(int frame_index) {
  ImGui::ShowDemoWindow();
  ImGui::Begin("volumetric_kit_gfx");
  ImGui::Text("ui tier: ImGuiOverlay over dynamic rendering");
  ImGui::Text("frame %d", frame_index);
  ImGui::End();
}

// Owns all Vulkan/windowing/UI state for one window; everything is destroyed
// when this returns, before main() tears GLFW down.
int run(GLFWwindow* window, int max_frames) {
  uint32_t glfw_ext_count = 0;
  const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  vg::InstanceConfig instance_config;
  instance_config.app_name = "02_imgui";
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

  win::SwapchainConfig swapchain_config;
  swapchain_config.extent = framebuffer_extent(window);
  auto swapchain = win::Swapchain::create(device.value(), surface.handle(),
                                          swapchain_config);
  if (!swapchain.ok()) {
    std::fprintf(stderr, "swapchain: %s\n",
                 swapchain.status().message().c_str());
    return 1;
  }

  auto loop = win::FrameLoop::create(device.value(), swapchain.value(), 2);
  if (!loop.ok()) {
    std::fprintf(stderr, "frame loop: %s\n", loop.status().message().c_str());
    return 1;
  }

  // The overlay's pipeline is built for the swapchain's layout; the swapchain
  // holds its format AND image count stable across recreate, so the overlay
  // survives resizes without rebuilding.
  // TODO: a swapchain that changed its image count on recreate would need the
  // overlay's backend updated (ImGui_ImplVulkan_SetMinImageCount, not yet
  // exposed by the ui tier); the kit's swapchain keeps it stable today.
  vg::ui::ImGuiOverlayConfig overlay_config;
  overlay_config.layout = swapchain.value().layout();
  overlay_config.min_image_count = swapchain.value().image_count();
  overlay_config.image_count = swapchain.value().image_count();
  auto overlay = vg::ui::ImGuiOverlay::create(
      device.value(), instance.value().handle(), overlay_config);
  if (!overlay.ok()) {
    std::fprintf(stderr, "overlay: %s\n", overlay.status().message().c_str());
    return 1;
  }

  // Platform backend (this example's half): bind it to the overlay's context,
  // then let it feed input + io.DisplaySize each frame.
  ImGui::SetCurrentContext(overlay.value().context());
  if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
    std::fprintf(stderr, "ImGui_ImplGlfw_InitForVulkan failed\n");
    return 1;
  }

  int rendered = 0;
  while (!glfwWindowShouldClose(window)) {
    if (max_frames >= 0 && rendered >= max_frames) {
      break;
    }
    glfwPollEvents();

    ImGui_ImplGlfw_NewFrame();    // platform: sets io.DisplaySize + input
    overlay.value().new_frame();  // renderer: begins the ImGui frame
    build_ui(rendered);

    auto frame = loop.value().begin_frame();
    if (!frame.ok()) {
      if (frame.status().code() == VK_ERROR_OUT_OF_DATE_KHR) {
        ImGui::EndFrame();  // discard the frame we began; no render() will pair
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
    // A consumer would record its scene here first; the overlay composes on top
    // within the same dynamic-rendering scope.
    overlay.value().render(frame.value().cmd);
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

  if (vkDeviceWaitIdle(device.value().handle()) != VK_SUCCESS) {
    std::fprintf(stderr, "vkDeviceWaitIdle failed at teardown\n");
  }
  // Tear the platform backend down while the ImGui context is still alive; the
  // overlay's destructor shuts the renderer backend down and destroys it.
  ImGui::SetCurrentContext(overlay.value().context());
  ImGui_ImplGlfw_Shutdown();
  std::printf("02_imgui: rendered %d frame(s)\n", rendered);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int max_frames = -1;  // < 0 means run until the window is closed
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--frames") != 0) {
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "--frames needs a non-negative integer value\n");
      return 2;
    }
    char* end = nullptr;
    const long value = std::strtol(argv[++i], &end, 10);
    if (*end != '\0' || value < 0 || value > INT_MAX) {
      std::fprintf(stderr, "--frames: invalid value '%s'\n", argv[i]);
      return 2;
    }
    max_frames = static_cast<int>(value);
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
      1024, 720, "volumetric_kit_gfx \xE2\x80\x94 02_imgui", nullptr, nullptr);
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

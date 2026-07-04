// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/02_imgui: a Dear ImGui debug overlay drawn into the swapchain
// through the ui tier (ImGuiOverlay) on dynamic rendering. The Vulkan bring-up
// is one app::WindowedApp::create call; the example shows the split the ui
// tier is built around: it owns the *platform* backend (imgui_impl_glfw —
// input + window sizing), while ImGuiOverlay wraps only the *renderer* backend
// (imgui_impl_vulkan), so the library tier stays GLFW-free like windowing. Run
// with `--frames N` to render N frames and exit — CI drives that under Xvfb
// with validation enabled to exercise the path headlessly.

#include "volumetric_kit/gfx/core/vulkan.hpp"  // before GLFW, so glfw3.h sees
// Vulkan and declares its helpers

#include <GLFW/glfw3.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"
#include "volumetric_kit/gfx/ui/metrics_panel.hpp"
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

  // CPU-ahead depth shared by the app's frame loop and the profiler driving it.
  constexpr uint32_t kFramesInFlight = 2;

  // The whole bring-up chain in one call; the lambda supplies the GLFW
  // surface, keeping the library tier window-system-free.
  vg::app::WindowedAppConfig config;
  config.app_name = "02_imgui";
  config.enable_validation = true;  // a no-op when the layer is absent
  config.instance_extensions.assign(glfw_exts, glfw_exts + glfw_ext_count);
  config.swapchain.extent = framebuffer_extent(window);
  config.frames_in_flight = kFramesInFlight;
  auto created = vg::app::WindowedApp::create(
      config, [window](VkInstance instance) -> vg::Result<VkSurfaceKHR> {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        const VkResult result =
            glfwCreateWindowSurface(instance, window, nullptr, &surface);
        if (result != VK_SUCCESS) {
          return vg::vk_error(result, "glfwCreateWindowSurface");
        }
        return surface;
      });
  if (!created.ok()) {
    std::fprintf(stderr, "app: %s\n", created.status().message().c_str());
    return 1;
  }
  vg::app::WindowedApp app = std::move(created).value();

  // On MoltenVK (zero timestamp valid bits) GPU scopes fall back to CPU-only
  // timing; the panel shows whatever resolved. Borrowed by the app's loop
  // (set_profiler below) and detached again before teardown.
  vg::ProfilerConfig profiler_config;
  profiler_config.frames_in_flight = kFramesInFlight;
  auto profiler = vg::Profiler::create(app.device(), profiler_config);
  if (!profiler.ok()) {
    std::fprintf(stderr, "profiler: %s\n", profiler.status().message().c_str());
    return 1;
  }

  // The overlay's pipeline is built for the swapchain's layout; the swapchain
  // holds its format AND image count stable across recreate, so the overlay
  // survives resizes without rebuilding.
  // TODO: a swapchain that changed its image count on recreate would need the
  // overlay's backend updated (ImGui_ImplVulkan_SetMinImageCount, not yet
  // exposed by the ui tier); the kit's swapchain keeps it stable today.
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

  // Turnkey: the loop now calls profiler.begin_frame/end_frame for us, so the
  // render loop below only opens a scope around its work.
  app.set_profiler(&profiler.value());

  // Platform backend (this example's half): bind it to the overlay's context,
  // then let it feed input + io.DisplaySize each frame. Initialized last, after
  // the app's fallible bring-up and set_profiler, so no earlier error return
  // leaves a live ImGui_ImplGlfw backend without its paired Shutdown at
  // teardown.
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

    // Begin the Vulkan frame *before* the ImGui frame: the loop skips ticks
    // while minimized / rebuilding, so a skipped tick never opens an ImGui
    // frame that would then need an EndFrame() discard to stay paired.
    auto frame = app.begin_frame(framebuffer_extent(window));
    if (!frame.ok()) {
      std::fprintf(stderr, "begin_frame: %s\n",
                   frame.status().message().c_str());
      return 1;
    }
    if (!frame.value().has_value()) {
      // Paused: minimized, or the surface is still settling after a rebuild.
      // Idle briefly rather than block outright, so a settling surface retries
      // even when the compositor sends no further event.
      glfwWaitEventsTimeout(0.1);
      continue;
    }
    const win::Frame& f = *frame.value();

    ImGui_ImplGlfw_NewFrame();    // platform: sets io.DisplaySize + input
    overlay.value().new_frame();  // renderer: begins the ImGui frame
    build_ui(rendered);
    // The live profiler view; the resolved metrics lag the in-flight depth, so
    // it is empty for the first couple of frames, then fills in.
    vg::ui::draw_metrics_panel(profiler.value().metrics());

    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[0] = 0.02f;
    begin.clear_color.float32[1] = 0.02f;
    begin.clear_color.float32[2] = 0.05f;
    begin.clear_color.float32[3] = 1.0f;
    {
      // A GPU-timed, debug-labelled stage around the frame's rendering; the
      // profiler resolves it into the metrics the panel above displays.
      vg::Profiler::Scope scope = profiler.value().gpu_scope(f.cmd, "overlay");
      f.target->begin(f.cmd, begin);
      // A consumer would record its scene here first; the overlay composes on
      // top within the same dynamic-rendering scope.
      overlay.value().render(f.cmd);
      f.target->end(f.cmd);
    }

    const vg::Status present = app.end_frame(f);
    if (!present.ok() && !win::swapchain_stale(present)) {
      std::fprintf(stderr, "end_frame: %s\n", present.message().c_str());
      return 1;
    }
    ++rendered;
  }

  // The overlay + profiler were created after the app, so they destruct before
  // it — while its frame loop may still have frames in flight that reference
  // them. Idle the device first so their destruction is safe, and detach the
  // borrowed profiler from the loop before it goes out of scope.
  app.wait_idle();
  app.set_profiler(nullptr);

  // Tear the platform backend down while the ImGui context is still alive; the
  // overlay's destructor then shuts the renderer backend down and destroys it.
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

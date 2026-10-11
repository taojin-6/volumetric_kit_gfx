// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/02_imgui: a Dear ImGui debug overlay drawn into the swapchain
// through the ui tier (ImGuiOverlay) on dynamic rendering. The Vulkan bring-up
// is one app::WindowedApp::create call; the overlay is two halves: the
// *renderer* backend (imgui_impl_vulkan) in ui::ImGuiOverlay, and the
// *platform* backend (imgui_impl_glfw — input + window sizing) in
// glfw::ImGuiGlfw. Run with `--frames N` to render N frames and exit — CI
// drives that under Xvfb with validation enabled to exercise the path
// headlessly.

// Before GLFW, so glfw3.h sees Vulkan and declares its helpers.
#include "volumetric_kit/core/vulkan/vulkan.hpp"

#include <GLFW/glfw3.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <memory>

#include "imgui.h"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/glfw/imgui_glfw.hpp"
#include "volumetric_kit/gfx/glfw/window.hpp"
#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"
#include "volumetric_kit/gfx/ui/metrics_panel.hpp"
#include "volumetric_kit/gfx/windowing.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;
namespace win = volumetric_kit::gfx::windowing;
namespace glfw = volumetric_kit::gfx::glfw;

namespace {

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
  auto extensions = glfw::instance_extensions();
  if (!extensions.ok()) {
    std::fprintf(stderr, "%s\n", extensions.status().message().c_str());
    return 1;
  }

  // CPU-ahead depth shared by the app's frame loop and the profiler driving it.
  constexpr uint32_t kFramesInFlight = 2;

  // The whole bring-up chain in one call; glfw::glfw_surface_factory supplies
  // the GLFW surface.
  vg::app::WindowedAppConfig config;
  config.app_name = "02_imgui";
  config.enable_validation = true;  // a no-op when the layer is absent
  config.instance_extensions = extensions.value();
  config.swapchain.extent = glfw::framebuffer_extent(window);
  config.frames_in_flight = kFramesInFlight;
  auto created =
      vg::app::WindowedApp::create(config, glfw::glfw_surface_factory(window));
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

  // Platform backend: bound to the overlay's context, it feeds input +
  // io.DisplaySize each frame. Declared after the overlay, so it shuts down
  // first, while the context it binds is alive.
  auto created_platform =
      glfw::ImGuiGlfw::create(window, overlay.value().context());
  if (!created_platform.ok()) {
    std::fprintf(stderr, "ImGuiGlfw: %s\n",
                 created_platform.status().message().c_str());
    return 1;
  }
  const std::unique_ptr<glfw::ImGuiGlfw> platform =
      std::move(created_platform).value();

  // Turnkey: the loop now calls profiler.begin_frame/end_frame for us, so the
  // render loop below only opens a scope around its work.
  app.set_profiler(&profiler.value());

  // A hard error inside the loop breaks out to the shared teardown below
  // (wait_idle + set_profiler(nullptr)) rather than returning straight away,
  // so in-flight frames are drained and the profiler is detached before the
  // after-app overlay + profiler destruct.
  int exit_code = 0;
  int rendered = 0;
  while (!glfwWindowShouldClose(window)) {
    if (max_frames >= 0 && rendered >= max_frames) {
      break;
    }
    glfwPollEvents();

    // Begin the Vulkan frame *before* the ImGui frame: the loop skips ticks
    // while minimized / rebuilding, so a skipped tick never opens an ImGui
    // frame that would then need an EndFrame() discard to stay paired.
    auto frame = app.begin_frame(glfw::framebuffer_extent(window));
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

    platform->new_frame();        // platform: sets io.DisplaySize + input
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

    const vkc::Status present = app.end_frame(f);
    if (!present.ok() && !win::swapchain_stale(present)) {
      std::fprintf(stderr, "end_frame: %s\n", present.message().c_str());
      exit_code = 1;
      break;
    }
    ++rendered;
  }

  // The overlay + profiler were created after the app, so they destruct before
  // it — while its frame loop may still have frames in flight that reference
  // them (including after an error break above). Idle the device first so their
  // destruction is safe, and detach the borrowed profiler from the loop before
  // it goes out of scope.
  if (const vkc::Status idle = app.wait_idle(); !idle) {
    std::fprintf(stderr, "wait_idle: %s\n", idle.message().c_str());
    exit_code = 1;
  }
  app.set_profiler(nullptr);
  std::printf("02_imgui: rendered %d frame(s)\n", rendered);
  return exit_code;
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

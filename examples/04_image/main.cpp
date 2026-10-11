// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// examples/04_image: a 2D image in a window through pipelines::ImagePipeline,
// panned and zoomed with camera::ImageView2D. The image is a zone plate --
// rings whose frequency rises to one cycle per two texels at the edge -- the
// classic test of minification: drawn smaller than its texels without a mip
// chain it breaks into moire, and with one it fades evenly to the mid grey of
// its average. Resize the window, scroll to zoom about the cursor, drag to
// pan, press F to fit. The picture is re-converted every frame from a device
// buffer, as a live camera stream would be, so the per-frame copy -> convert
// -> mip path runs under the frame loop. Set `run.frames` in the JSON config to
// render a fixed frame count and exit, as CI does under Xvfb with validation
// enabled.

// Before GLFW, so glfw3.h sees Vulkan and declares its helpers.
#include "volumetric_kit/core/vulkan/vulkan.hpp"

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/config/config.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/glfw/window.hpp"
#include "volumetric_kit/gfx/pipelines/image_pipeline.hpp"
#include "volumetric_kit/gfx/windowing.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;
namespace win = volumetric_kit::gfx::windowing;
namespace glfw = volumetric_kit::gfx::glfw;
namespace pipelines = volumetric_kit::gfx::pipelines;

namespace {

constexpr uint32_t kSide = 2048;

// Scroll arrives through a callback; the loop drains it once per frame.
double g_scroll = 0.0;

void on_scroll(GLFWwindow* /*window*/, double /*x*/, double y) {
  g_scroll += y;
}

// A zone plate: cos(pi r^2 / side), whose local frequency r / side reaches
// 0.5 cycles per texel at the edge, as sRGB-encoded grey bytes.
std::vector<uint8_t> zone_plate() {
  std::vector<uint8_t> grey(static_cast<size_t>(kSide) * kSide);
  const float c = 0.5f * static_cast<float>(kSide);
  const float k = 3.14159265f / static_cast<float>(kSide);
  for (uint32_t y = 0; y < kSide; ++y) {
    for (uint32_t x = 0; x < kSide; ++x) {
      const float dx = static_cast<float>(x) + 0.5f - c;
      const float dy = static_cast<float>(y) + 0.5f - c;
      const float v = 0.5f + 0.5f * std::cos(k * (dx * dx + dy * dy));
      grey[static_cast<size_t>(y) * kSide + x] =
          static_cast<uint8_t>(std::lround(v * 255.0f));
    }
  }
  return grey;
}

int run(GLFWwindow* window, const vkc::config::Config& settings) {
  const int max_frames = settings.run.frames;
  auto extensions = glfw::instance_extensions();
  if (!extensions.ok()) {
    std::fprintf(stderr, "%s\n", extensions.status().message().c_str());
    return 1;
  }

  vg::app::WindowedAppConfig config;
  config.app_name = "04_image";
  config.enable_validation = settings.window.validation;
  config.instance_extensions = extensions.value();
  config.swapchain.extent = glfw::framebuffer_extent(window);
  auto created =
      vg::app::WindowedApp::create(config, glfw::glfw_surface_factory(window));
  if (!created.ok()) {
    std::fprintf(stderr, "app: %s\n", created.status().message().c_str());
    return 1;
  }
  vg::app::WindowedApp app = std::move(created).value();

  // The swapchain keeps its format across resizes, so the pipeline built for
  // its layout survives them.
  auto images =
      pipelines::ImagePipeline::create(app.device(), app.swapchain().layout());
  if (!images.ok()) {
    std::fprintf(stderr, "pipeline: %s\n", images.status().message().c_str());
    return 1;
  }

  // The source: a device buffer a stream's producer would refill.
  const std::vector<uint8_t> grey = zone_plate();
  vg::BufferUploadDesc upload;
  upload.data = grey.data();
  upload.size = grey.size();
  upload.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  auto source = vg::upload_buffer(app.device(), app.allocator(), upload);
  if (!source.ok()) {
    std::fprintf(stderr, "source: %s\n", source.status().message().c_str());
    return 1;
  }

  pipelines::ImageTextureDesc desc;
  desc.extent = {kSide, kSide};
  desc.mapping = pipelines::ImageMapping::Grey;
  desc.format = VK_FORMAT_R8_UNORM;
  auto texture =
      pipelines::ImageTexture::create(images.value(), app.allocator(), desc);
  if (!texture.ok()) {
    std::fprintf(stderr, "texture: %s\n", texture.status().message().c_str());
    return 1;
  }
  pipelines::ImageUpdate update;
  update.planes[0].buffer = &source.value();

  const VkExtent2D start = glfw::framebuffer_extent(window);
  vg::camera::ImageView2D view(
      {static_cast<float>(kSide), static_cast<float>(kSide)}, {0.0f, 0.0f},
      {static_cast<float>(start.width), static_cast<float>(start.height)});
  glfwSetScrollCallback(window, on_scroll);
  bool dragging = false;
  glm::vec2 drag_from{0.0f, 0.0f};

  int exit_code = 0;
  int rendered = 0;
  while (!glfwWindowShouldClose(window)) {
    if (max_frames >= 0 && rendered >= max_frames) {
      break;
    }
    glfwPollEvents();

    auto frame = app.begin_frame(glfw::framebuffer_size(window));
    if (!frame.ok()) {
      std::fprintf(stderr, "begin_frame: %s\n",
                   frame.status().message().c_str());
      exit_code = 1;
      break;
    }
    if (!frame.value().has_value()) {
      glfwWaitEventsTimeout(0.1);  // minimized, or the surface is settling
      continue;
    }
    const win::Frame& f = *frame.value();
    const VkExtent2D extent = f.target->extent();

    // Input, in framebuffer pixels. The view follows the window: a resize
    // keeps what is shown and scales it, so the image minifies as the window
    // shrinks and the mip level follows.
    view.set_viewport({0.0f, 0.0f}, {static_cast<float>(extent.width),
                                     static_cast<float>(extent.height)});
    const glm::vec2 cursor = glfw::cursor_in_framebuffer(window);
    if (g_scroll != 0.0) {
      view.zoom_about(cursor, static_cast<float>(std::pow(1.1, g_scroll)));
      g_scroll = 0.0;
    }
    const bool down =
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (down && dragging) {
      view.pan(cursor - drag_from);
    }
    dragging = down;
    drag_from = cursor;
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
      view.fit();
    }

    // Re-convert the picture, as for a stream's new frame, before rendering.
    const vkc::Status updated =
        images.value().record_update(f.cmd, texture.value(), update);
    if (!updated.ok()) {
      std::fprintf(stderr, "record_update: %s\n", updated.message().c_str());
      // The frame is begun, so it must still be ended.
    }

    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[0] = 0.02f;
    begin.clear_color.float32[1] = 0.02f;
    begin.clear_color.float32[2] = 0.05f;
    begin.clear_color.float32[3] = 1.0f;
    f.target->begin(f.cmd, begin);
    pipelines::ImageDraw draw;
    draw.texture = &texture.value();
    draw.viewport = {{0, 0}, extent};
    draw.origin = view.origin();
    draw.scale = view.scale();
    images.value().submit(f.cmd, {extent, &draw, 1});
    f.target->end(f.cmd);

    const vkc::Status present = app.end_frame(f);
    if (!present.ok() && !win::swapchain_stale(present)) {
      std::fprintf(stderr, "end_frame: %s\n", present.message().c_str());
      exit_code = 1;
      break;
    }
    if (!updated.ok()) {
      exit_code = 1;
      break;
    }

    char title[96];
    std::snprintf(title, sizeof(title),
                  "volumetric_kit_gfx \xE2\x80\x94 04_image (%.3f px/texel)",
                  view.scale());
    glfwSetWindowTitle(window, title);
    ++rendered;
  }

  // The pipeline, texture and source were created after the app, so they
  // destruct before it -- while frames may still be in flight. Idle first.
  if (const vkc::Status idle = app.wait_idle(); !idle) {
    std::fprintf(stderr, "wait_idle: %s\n", idle.message().c_str());
    exit_code = 1;
  }
  std::printf("04_image: rendered %d frame(s)\n", rendered);
  return exit_code;
}

}  // namespace

int main(int argc, char** argv) {
  const auto args = vkc::config::parse_args(argc, argv);
  if (!args.ok()) {
    std::fprintf(stderr, "%s\n", args.status().message().c_str());
    return 2;
  }
  if (args.value().help) {
    std::printf("%s", vkc::config::usage());
    return 0;
  }
  const auto loaded = vkc::config::load(args.value().config_path,
                                        vkc::config::Application::GfxImage);
  if (!loaded.ok()) {
    std::fprintf(stderr, "%s\n", loaded.status().message().c_str());
    return 2;
  }
  const vkc::config::Config& settings = loaded.value();

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
      settings.window.width, settings.window.height,
      "volumetric_kit_gfx \xE2\x80\x94 04_image", nullptr, nullptr);
  if (window == nullptr) {
    std::fprintf(stderr, "glfwCreateWindow failed\n");
    glfwTerminate();
    return 1;
  }

  const int rc = run(window, settings);

  glfwDestroyWindow(window);
  glfwTerminate();
  return rc;
}

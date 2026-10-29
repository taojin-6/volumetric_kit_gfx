// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The pieces together, as an app assembles them: a mesh in the scene's
// rectangle beside an ImagePanel of its texture's pictures, drawn by
// HybridMeshTextureView through a PicturePyramid, with the panel's chrome
// over them -- on the panel's design screens, one tall picture and two wide
// ones, and on a desktop at fit, 1:1 and 4:1. Each frame checks that the
// mesh stays in the scene, the pictures in their places and nothing else in
// the panel. With VG_TEST_DUMP_DIR set, each frame is also written there as a
// PPM, to look at.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/trigonometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "gfx_test_support.hpp"
#include "image_test_support.hpp"
#include "imgui.h"
#include "imgui_internal.h"  // FindWindowByName, for the chrome's rects
#include "texture_view_test_support.hpp"
#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/camera/camera.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_set.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_view.hpp"
#include "volumetric_kit/gfx/ui/image_panel.hpp"
#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"
#include "volumetric_kit/gfx/ui/imgui_touch.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;
namespace assets = volumetric_kit::gfx::assets;
namespace ui = volumetric_kit::gfx::ui;

constexpr VkFormat kColor = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepth = VK_FORMAT_D32_SFLOAT;
constexpr std::array<uint8_t, 3> kClear{24, 24, 32};
constexpr const char* kTitle = "Texture map";

vg::RenderTargetLayout layout() {
  vg::RenderTargetLayout l;
  l.color_formats[0] = kColor;
  l.color_count = 1;
  l.depth_format = kDepth;
  return l;
}

struct Screen {
  const char* name;
  glm::vec2 points;
  float scale;
  ui::EdgeInsets safe;
  float share;
  bool touch;
  VkExtent2D pixels() const {
    return {static_cast<uint32_t>(points.x * scale),
            static_cast<uint32_t>(points.y * scale)};
  }
};

const Screen kScreens[] = {
    {"macbook_pro_16", {1728, 1117}, 2.0f, {}, 0.4f, false},
    {"linux_4k", {3840, 2160}, 1.0f, {}, 0.4f, false},
    {"iphone_portrait", {440, 956}, 3.0f, {62, 0, 34, 0}, 0.45f, true},
    {"iphone_landscape", {956, 440}, 3.0f, {0, 62, 21, 62}, 0.45f, true},
    {"ipad_landscape", {1210, 834}, 2.0f, {24, 0, 20, 0}, 0.4f, true},
    {"ipad_portrait", {834, 1210}, 2.0f, {24, 0, 20, 0}, 0.4f, true},
};

// One textured quad a picture, side by side and turned a little, its uv0
// over that picture.
assets::Mesh make_scene(uint32_t count, glm::vec2 picture) {
  assets::Mesh mesh;
  const float aspect = picture.x / picture.y;
  const float h = count == 1 ? 1.6f : 1.0f;
  const float w = h * aspect;
  for (uint32_t p = 0; p < count; ++p) {
    const float x0 = (static_cast<float>(p) - 0.5f * count) * (w + 0.1f);
    const auto base = static_cast<uint32_t>(mesh.vertices.size());
    for (const auto& c :
         {glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1)}) {
      assets::Vertex v;
      v.position = {x0 + c.x * w, (0.5f - c.y) * h, -0.3f * c.x};
      v.normal = {0.0f, 0.0f, 1.0f};
      v.uv0 = {c.x, static_cast<float>(p) + c.y};
      v.color = {1.0f, 1.0f, 1.0f, 1.0f};
      mesh.vertices.push_back(v);
    }
    for (const uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) {
      mesh.indices.push_back(base + i);
    }
  }
  return mesh;
}

void write_ppm(const std::string& path, const uint8_t* rgba, VkExtent2D e) {
  std::ofstream out(path, std::ios::binary);
  out << "P6\n" << e.width << " " << e.height << "\n255\n";
  for (size_t i = 0; i < size_t{e.width} * e.height; ++i) {
    out.write(reinterpret_cast<const char*>(rgba + 4 * i), 3);
  }
}

class TextureMapScreensTest : public vg_test::RendererDeviceTest,
                              protected vg_test::ImageHelpers {
 protected:
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::ShaderAccesses;
  }
  vkc::Device& image_device() override { return device(); }
  vkc::Allocator& image_allocator() override { return allocator(); }

  void SetUp() override {
    vg_test::RendererDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto mesh =
        pipelines::HybridMeshPipeline::create(device(), allocator(), layout());
    ASSERT_TRUE(mesh.ok()) << mesh.status().message();
    mesh_.emplace(std::move(mesh).value());
    auto view =
        pipelines::HybridMeshTextureView::create(device(), *mesh_, layout());
    ASSERT_TRUE(view.ok()) << view.status().message();
    view_ = std::move(view).value();
  }

  // `count` cards of `size` as Y'CbCr pictures in one set, its pyramid and a
  // mesh of them.
  struct Texture {
    std::vector<vkc::Image> planes;
    std::unique_ptr<pipelines::HybridMeshTextureSet> set;
    std::unique_ptr<pipelines::PicturePyramid> pyramid;
    std::optional<pipelines::GpuMesh> mesh;
  };
  std::unique_ptr<Texture> make_texture(VkExtent2D size, uint32_t count) {
    auto t = std::make_unique<Texture>();
    t->planes.reserve(2 * count);
    std::vector<pipelines::YcbcrPicture> pictures;
    for (uint32_t p = 0; p < count; ++p) {
      const vg_test::PlaneBytes bytes = vg_test::planes_of(
          vg_test::make_card(size.width, size.height, 5 * p), size);
      t->planes.push_back(
          source_image(VK_FORMAT_R8_UNORM, bytes.luma_size, bytes.luma));
      t->planes.push_back(
          source_image(VK_FORMAT_R8G8_UNORM, bytes.chroma_size, bytes.chroma));
      pipelines::YcbcrPicture picture;
      picture.luma = &t->planes[2 * p];
      picture.chroma = &t->planes[2 * p + 1];
      picture.extent = size;
      picture.coding = vg_test::bt709_full();
      pictures.push_back(picture);
    }
    auto set = pipelines::HybridMeshTextureSet::create(
        device(), allocator(), *mesh_, pictures.data(), count);
    EXPECT_TRUE(set.ok()) << set.status().message();
    auto pyramid =
        pipelines::PicturePyramid::create(*view_, allocator(), size, count);
    EXPECT_TRUE(pyramid.ok()) << pyramid.status().message();
    auto mesh = pipelines::upload_mesh(
        device(), allocator(),
        make_scene(count, {static_cast<float>(size.width),
                           static_cast<float>(size.height)}));
    EXPECT_TRUE(mesh.ok()) << mesh.status().message();
    if (!set.ok() || !pyramid.ok() || !mesh.ok()) {
      return nullptr;
    }
    t->set = std::move(set).value();
    t->pyramid = std::move(pyramid).value();
    t->mesh.emplace(std::move(mesh).value());
    return t;
  }

  // Two frames of `panel` beside `texture`'s mesh on `screen`, the view set
  // by `zoom` (on the panel, after its first layout) before the second, which
  // is checked and, with VG_TEST_DUMP_DIR set, written as `name`.ppm.
  void show(const Screen& screen, ui::ImagePanel& panel, Texture& texture,
            VkExtent2D size, uint32_t count, const std::string& name,
            const std::function<void(ui::ImagePanel&)>& zoom = nullptr) {
    ui::ImGuiOverlayConfig config;
    config.layout = layout();
    config.scale = screen.touch ? 1.3f : 1.0f;
    config.min_frame_height = screen.touch ? 44.0f : 0.0f;
    auto overlay =
        ui::ImGuiOverlay::create(device(), instance().handle(), config);
    ASSERT_TRUE(overlay.ok()) << overlay.status().message();
    ImGui::SetCurrentContext(overlay.value().context());
    ImGui::GetIO().IniFilename = nullptr;
    auto touch = ui::ImGuiTouch::create(overlay.value().context());
    ASSERT_TRUE(touch.ok()) << touch.status().message();
    const VkExtent2D pixels = screen.pixels();
    panel.set_pictures(
        {static_cast<float>(size.width), static_cast<float>(size.height)},
        count);

    vg::OffscreenTargetDesc desc;
    desc.extent = pixels;
    desc.color_format = kColor;
    desc.depth_format = kDepth;
    auto target = vg::OffscreenTarget::create(allocator(), desc);
    ASSERT_TRUE(target.ok()) << target.status().message();

    ui::PanelLayout laid;
    for (int f = 0; f < 2; ++f) {
      touch.value()->new_frame(
          {pixels, screen.scale, screen.safe, (f + 1) / 60.0});
      overlay.value().new_frame();
      laid = panel.layout(pixels);
      if (f == 1 && zoom) {
        zoom(panel);
      }
      panel.draw_chrome(kTitle);
      const vkc::Status s = submit([&](VkCommandBuffer cmd) {
        texture.pyramid->record(cmd, texture.set->set());
        target.value().prepare(cmd);
        vg::RenderTargetBeginInfo begin;
        for (int c = 0; c < 3; ++c) {
          begin.clear_color.float32[c] = kClear[c] / 255.0f;
        }
        begin.clear_color.float32[3] = 1.0f;
        const vg::RenderTarget rt = target.value().target();
        rt.begin(cmd, begin);
        const auto aspect = static_cast<float>(laid.scene.extent.width) /
                            static_cast<float>(laid.scene.extent.height);
        const vg::camera::Camera camera =
            vg::camera::Camera::look_at_perspective(
                {0.6f, 0.4f, count == 1 ? 3.4f : 5.6f}, {0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f}, glm::radians(45.0f), aspect, 0.1f, 20.0f);
        const pipelines::HybridMeshDraw draw{&*texture.mesh};
        pipelines::HybridMeshFrame frame;
        frame.extent = pixels;
        frame.viewport = laid.scene;
        frame.view_proj = camera.view_proj();
        frame.flags = 0;
        frame.atlas = texture.set->set();
        frame.draws = &draw;
        frame.draw_count = 1;
        mesh_->submit(cmd, frame);
        std::vector<pipelines::TextureViewDraw> draws;
        for (uint32_t p = 0; panel.open() && p < count; ++p) {
          draws.push_back(pipelines::texture_view_draw(
              texture.set->set(), texture.pyramid.get(), panel.view(),
              panel.picture_origin(p),
              {static_cast<float>(size.width), static_cast<float>(size.height)},
              {0.0f, static_cast<float>(p), 1.0f, p + 1.0f}));
        }
        view_->submit(cmd, pixels, draws.data(),
                      static_cast<uint32_t>(draws.size()));
        overlay.value().render(cmd);
        rt.end(cmd);
        target.value().record_readback(cmd);
      });
      ASSERT_TRUE(s.ok()) << s.message();
    }

    // The panel's pixels: the pictures where they are drawn, the chrome's
    // windows, and the clear elsewhere; the mesh in the scene alone.
    const auto* px = static_cast<const uint8_t*>(target.value().pixels());
    ASSERT_NE(px, nullptr);
    const ImVec2 s = ImGui::GetIO().DisplayFramebufferScale;
    std::vector<ImRect> chrome;
    for (const char* suffix : {"##header", "##handle"}) {
      const ImGuiWindow* window =
          ImGui::FindWindowByName((std::string(kTitle) + suffix).c_str());
      ASSERT_NE(window, nullptr) << suffix;
      chrome.emplace_back(
          window->Rect().Min.x * s.x - 1.0f, window->Rect().Min.y * s.y - 1.0f,
          window->Rect().Max.x * s.x + 1.0f, window->Rect().Max.y * s.y + 1.0f);
    }
    const auto in = [](const VkRect2D& r, float x, float y) {
      return x >= r.offset.x && y >= r.offset.y &&
             x < r.offset.x + static_cast<float>(r.extent.width) &&
             y < r.offset.y + static_cast<float>(r.extent.height);
    };
    size_t stray = 0;
    size_t mesh = 0;
    size_t picture = 0;
    size_t bare = 0;
    for (uint32_t y = 0; y < pixels.height; ++y) {
      for (uint32_t x = 0; x < pixels.width; ++x) {
        const float cx = x + 0.5f;
        const float cy = y + 0.5f;
        const uint8_t* p = px + (size_t{y} * pixels.width + x) * 4;
        const bool clear = std::abs(p[0] - kClear[0]) <= 1 &&
                           std::abs(p[1] - kClear[1]) <= 1 &&
                           std::abs(p[2] - kClear[2]) <= 1;
        if (in(laid.scene, cx, cy)) {
          mesh += !clear;
          continue;
        }
        bool covered = false;
        for (const ImRect& r : chrome) {
          covered = covered || r.Contains(ImVec2(cx, cy));
        }
        for (uint32_t i = 0; i < count && !covered; ++i) {
          const glm::vec2 lo =
              panel.view().image_to_target(panel.picture_origin(i));
          const glm::vec2 hi = panel.view().image_to_target(
              panel.picture_origin(i) +
              glm::vec2(static_cast<float>(size.width),
                        static_cast<float>(size.height)));
          // A pixel the picture's edge cuts may be either.
          if (cx > lo.x - 1 && cy > lo.y - 1 && cx < hi.x + 1 &&
              cy < hi.y + 1 && in(laid.pictures, cx, cy)) {
            covered = true;
            const bool whole = cx > lo.x + 1 && cy > lo.y + 1 &&
                               cx < hi.x - 1 && cy < hi.y - 1;
            picture += whole && clear;  // a picture pixel left clear
          }
        }
        if (!covered) {
          bare += 1;
          stray += !clear;
        }
      }
    }
    EXPECT_GT(mesh, size_t{pixels.width} * pixels.height / 50) << "no mesh";
    EXPECT_EQ(picture, 0u) << "picture pixels left clear";
    EXPECT_EQ(stray, 0u) << "of " << bare << " bare panel pixels";

    if (const char* dir = std::getenv("VG_TEST_DUMP_DIR")) {
      write_ppm(std::string(dir) + "/" + name + ".ppm", px, pixels);
    }
    vkDeviceWaitIdle(device().handle());
  }

  std::optional<pipelines::HybridMeshPipeline> mesh_;
  std::unique_ptr<pipelines::HybridMeshTextureView> view_;
};

// The design's screens, with one tall picture and with two wide ones.
TEST_F(TextureMapScreensTest, OnTheDesignsScreens) {
  struct Pictures {
    const char* name;
    VkExtent2D size;
    uint32_t count;
  };
  for (const Pictures& pictures : {Pictures{"tall", {1024, 2048}, 1},
                                   Pictures{"two_wide", {2048, 1088}, 2}}) {
    std::unique_ptr<Texture> texture =
        make_texture(pictures.size, pictures.count);
    ASSERT_NE(texture, nullptr);
    for (const Screen& screen : kScreens) {
      SCOPED_TRACE(::testing::Message()
                   << screen.name << ", " << pictures.name);
      ui::ImagePanelConfig config;
      config.state = {true, screen.share, screen.share};
      ui::ImagePanel panel(config);
      show(screen, panel, *texture, pictures.size, pictures.count,
           std::string(screen.name) + "_" + pictures.name);
    }
  }
}

// One desktop at fit (minified through the pyramid), at 1:1 and at 4:1
// (sharp squares), each about a point on the pictures.
TEST_F(TextureMapScreensTest, FitOneToOneAndMagnified) {
  constexpr VkExtent2D kSize{2048, 1088};
  std::unique_ptr<Texture> texture = make_texture(kSize, 2);
  ASSERT_NE(texture, nullptr);
  const Screen& screen = kScreens[0];
  for (const float scale : {0.0f, 1.0f, 4.0f}) {
    SCOPED_TRACE(::testing::Message() << "scale " << scale);
    ui::ImagePanelConfig config;
    config.state = {true, 0.4f, 0.4f};
    ui::ImagePanel panel(config);
    const std::string name = scale == 0.0f   ? "zoom_fit"
                             : scale == 1.0f ? "zoom_1to1"
                                             : "zoom_4to1";
    show(screen, panel, *texture, kSize, 2, name, [&](ui::ImagePanel& p) {
      if (scale > 0.0f) {
        // About the point of the first picture's checker, near its middle.
        const glm::vec2 at = p.view().image_to_target({1400.0f, 400.0f});
        p.view().zoom_about(at, scale / p.view().scale());
      }
    });
  }
}

}  // namespace

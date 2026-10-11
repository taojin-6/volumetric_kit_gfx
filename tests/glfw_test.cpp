// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// gfx_glfw: the instance extensions, the surface factory, the framebuffer
// extent and cursor, the shared-device config and device, and ImGui's GLFW
// platform backend. Windows are made on GLFW's null platform (GLFW 3.4+), so
// no display is needed; with an older GLFW those cases skip. The surface and
// the shared device also need a Vulkan device with VK_EXT_headless_surface,
// which the null platform presents through.

// Before GLFW, so glfw3.h sees Vulkan and declares its helpers.
#include "volumetric_kit/core/vulkan/vulkan.hpp"

#include <GLFW/glfw3.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gfx_test_support.hpp"
#include "imgui.h"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/core/vulkan/shared_device.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/glfw/imgui_glfw.hpp"
#include "volumetric_kit/gfx/glfw/shared_device.hpp"
#include "volumetric_kit/gfx/glfw/window.hpp"

namespace glfw = volumetric_kit::gfx::glfw;

namespace {

constexpr int kWidth = 320;
constexpr int kHeight = 240;

// GLFW on its null platform, with one window, for this object's lifetime.
// `ready()` is false, with `reason()`, where the platform is unavailable.
class NullWindow {
 public:
  NullWindow() {
#ifdef GLFW_PLATFORM_NULL
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
    // The loader this test links, rather than one GLFW searches for.
    glfwInitVulkanLoader(vkGetInstanceProcAddr);
    if (glfwInit() != GLFW_TRUE) {
      const char* description = nullptr;
      glfwGetError(&description);
      reason_ = std::string("glfwInit: ") +
                (description != nullptr ? description : "failed");
      return;
    }
    initialized_ = true;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(kWidth, kHeight, "glfw_test", nullptr, nullptr);
    if (window_ == nullptr) {
      reason_ = "glfwCreateWindow failed";
    }
#else
    reason_ = "GLFW " + std::to_string(GLFW_VERSION_MAJOR) + "." +
              std::to_string(GLFW_VERSION_MINOR) + " has no null platform";
#endif
  }
  ~NullWindow() {
    if (window_ != nullptr) {
      glfwDestroyWindow(window_);
    }
    if (initialized_) {
      glfwTerminate();
    }
  }
  NullWindow(const NullWindow&) = delete;
  NullWindow& operator=(const NullWindow&) = delete;

  bool ready() const { return window_ != nullptr; }
  const std::string& reason() const { return reason_; }
  GLFWwindow* get() const { return window_; }

 private:
  bool initialized_ = false;
  GLFWwindow* window_ = nullptr;
  std::string reason_;
};

// A test with a null-platform window.
class GlfwWindowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    window_ = std::make_unique<NullWindow>();
    if (!window_->ready()) {
      GTEST_SKIP() << window_->reason();
    }
  }
  void TearDown() override { window_.reset(); }

  GLFWwindow* window() const { return window_->get(); }

 private:
  std::unique_ptr<NullWindow> window_;
};

// An ImGui context that writes no imgui.ini, leaving the current one as it
// was.
ImGuiContext* make_context() {
  ImGuiContext* previous = ImGui::GetCurrentContext();
  ImGuiContext* context = ImGui::CreateContext();
  ImGui::SetCurrentContext(context);
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::SetCurrentContext(previous);
  return context;
}

// The context's IO, leaving the current context as it was.
const ImGuiIO& io_of(ImGuiContext* context) {
  ImGuiContext* previous = ImGui::GetCurrentContext();
  ImGui::SetCurrentContext(context);
  const ImGuiIO& io = ImGui::GetIO();
  ImGui::SetCurrentContext(previous);
  return io;
}

bool has_extension(const std::vector<const char*>& names, const char* name) {
  for (const char* listed : names) {
    if (std::strcmp(listed, name) == 0) {
      return true;
    }
  }
  return false;
}

// --- instance_extensions ----------------------------------------------------

TEST(GlfwInstanceExtensionsTest, UninitializedGlfwIsAnError) {
  const vkc::Result<std::vector<const char*>> extensions =
      glfw::instance_extensions();
  ASSERT_FALSE(extensions.ok());
  EXPECT_EQ(extensions.status().domain(), vkc::Status::Code::Unsupported);
  EXPECT_NE(extensions.status().message().find("not initialized"),
            std::string::npos)
      << extensions.status().message();
}

TEST_F(GlfwWindowTest, InstanceExtensionsNameTheSurface) {
  if (!vg_test::has_headless_surface()) {
    GTEST_SKIP() << "VK_EXT_headless_surface unavailable";
  }
  const vkc::Result<std::vector<const char*>> extensions =
      glfw::instance_extensions();
  ASSERT_TRUE(extensions.ok()) << extensions.status().message();
  EXPECT_TRUE(has_extension(extensions.value(), VK_KHR_SURFACE_EXTENSION_NAME));
  EXPECT_TRUE(has_extension(extensions.value(),
                            VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME));
}

// --- framebuffer_extent / cursor_in_framebuffer -----------------------------

TEST_F(GlfwWindowTest, FramebufferExtentIsTheWindowsAndNeverZero) {
  const VkExtent2D extent = glfw::framebuffer_extent(window());
  EXPECT_EQ(extent.width, static_cast<uint32_t>(kWidth));
  EXPECT_EQ(extent.height, static_cast<uint32_t>(kHeight));

  // What a minimized window may report.
  glfwSetWindowSize(window(), 0, 0);
  int width = -1;
  int height = -1;
  glfwGetFramebufferSize(window(), &width, &height);
  ASSERT_EQ(width, 0);
  ASSERT_EQ(height, 0);
  const VkExtent2D clamped = glfw::framebuffer_extent(window());
  EXPECT_EQ(clamped.width, 1u);
  EXPECT_EQ(clamped.height, 1u);
}

TEST_F(GlfwWindowTest, CursorIsInFramebufferPixels) {
  // The null platform keeps whole-pixel positions, at a scale of 1.
  glfwSetCursorPos(window(), 12.0, 7.0);
  const glm::vec2 cursor = glfw::cursor_in_framebuffer(window());
  EXPECT_EQ(cursor.x, 12.0f);
  EXPECT_EQ(cursor.y, 7.0f);

  // With no size there is no scale to apply.
  glfwSetWindowSize(window(), 0, 0);
  const glm::vec2 unscaled = glfw::cursor_in_framebuffer(window());
  EXPECT_EQ(unscaled.x, 12.0f);
  EXPECT_EQ(unscaled.y, 7.0f);
}

// --- shared_device_config ---------------------------------------------------

TEST(GlfwSharedDeviceConfigTest, IsThePureUnionOfItsInputs) {
  glfw::SharedDeviceConfig config;
  config.instance.app_name = "viewer";
  config.instance.enable_validation = true;
  config.instance.extensions = {VK_KHR_SURFACE_EXTENSION_NAME};
  config.graphics.features.fillModeNonSolid = VK_TRUE;
  vkc::DeviceRequirements compute;
  compute.scalar_block_layout = true;
  compute.extensions = {"VK_KHR_external_memory_fd"};
  const std::vector<const char*> surface = {
      VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};

  const vkc::SharedDeviceConfig shared =
      glfw::shared_device_config(surface, compute, config);

  EXPECT_EQ(shared.instance.app_name, "viewer");
  EXPECT_TRUE(shared.instance.enable_validation);
  // The embedder's extensions first, then GLFW's, each once.
  ASSERT_EQ(shared.instance.extensions.size(), 2u);
  EXPECT_STREQ(shared.instance.extensions[0], VK_KHR_SURFACE_EXTENSION_NAME);
  EXPECT_STREQ(shared.instance.extensions[1],
               VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
  // The compute library's requirements pass through as given.
  EXPECT_EQ(shared.compute.queue_flags, compute.queue_flags);
  EXPECT_TRUE(shared.compute.scalar_block_layout);
  EXPECT_FALSE(shared.compute.needs_present);
  EXPECT_EQ(shared.compute.extensions, compute.extensions);
  // The renderer's, with presentation added.
  EXPECT_TRUE(shared.graphics.needs_present);
  EXPECT_EQ(shared.graphics.api_version, VK_API_VERSION_1_3);
  EXPECT_TRUE(shared.graphics.dynamic_rendering);
  EXPECT_EQ(shared.graphics.features.fillModeNonSolid, VK_TRUE);
  // make_shared_device sets the surface maker.
  EXPECT_FALSE(static_cast<bool>(shared.make_surface));

  // Nothing it was given changed.
  EXPECT_FALSE(config.graphics.needs_present);
  ASSERT_EQ(config.instance.extensions.size(), 1u);
  EXPECT_FALSE(compute.needs_present);
  EXPECT_EQ(surface.size(), 2u);
}

// --- glfw_surface_factory / make_shared_device ------------------------------

TEST(GlfwSharedDeviceTest, NullWindowIsRefused) {
  const auto shared = glfw::make_shared_device(nullptr, {}, {});
  ASSERT_FALSE(shared.ok());
  EXPECT_EQ(shared.status().domain(), vkc::Status::Code::InvalidArgument);
}

// A Vulkan device under the core fixtures' policy, and a null-platform window
// that presents through VK_EXT_headless_surface.
class GlfwDeviceTest : public vg_test::RendererTest {
 protected:
  void SetUp() override {
    vg_test::RendererTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    if (!vg_test::has_headless_surface()) {
      GTEST_SKIP() << "VK_EXT_headless_surface unavailable";
    }
    window_ = std::make_unique<NullWindow>();
    if (!window_->ready()) {
      GTEST_SKIP() << window_->reason();
    }
  }
  void TearDown() override {
    window_.reset();
    vg_test::RendererTest::TearDown();
  }

  GLFWwindow* window() const { return window_->get(); }

 private:
  std::unique_ptr<NullWindow> window_;
};

TEST_F(GlfwDeviceTest, SurfaceFactoryMakesTheWindowsSurface) {
  vkc::InstanceConfig config = vg_test::headless_instance_config();
  auto instance = vkc::Instance::create(config);
  ASSERT_TRUE(instance.ok()) << instance.status().message();

  const vkc::Result<VkSurfaceKHR> surface =
      glfw::glfw_surface_factory(window())(instance.value().handle());
  ASSERT_TRUE(surface.ok()) << surface.status().message();
  EXPECT_NE(surface.value(), VK_NULL_HANDLE);
  vkDestroySurfaceKHR(instance.value().handle(), surface.value(), nullptr);
}

TEST_F(GlfwDeviceTest, SurfaceFactoryReportsGlfwsVkResult) {
  // No surface extensions, and no layer between the loader and the call.
  vkc::InstanceConfig config;
  config.app_name = "glfw_test";
  auto instance = vkc::Instance::create(config);
  ASSERT_TRUE(instance.ok()) << instance.status().message();

  const vkc::Result<VkSurfaceKHR> surface =
      glfw::glfw_surface_factory(window())(instance.value().handle());
  ASSERT_FALSE(surface.ok());
  EXPECT_EQ(vkc::vk_result(surface.status()),
            std::optional<VkResult>(VK_ERROR_EXTENSION_NOT_PRESENT))
      << surface.status().message();
}

TEST_F(GlfwDeviceTest, MakeSharedDeviceBuildsAPresentingDeviceForTheWindow) {
  glfw::SharedDeviceConfig config;
  config.instance = vkc::test::instance_config(active_validation());
  vkc::DeviceRequirements compute;  // a compute queue, timeline semaphores

  auto made = glfw::make_shared_device(window(), compute, config);
  ASSERT_TRUE(made.ok()) << made.status().message();
  const std::unique_ptr<vkc::SharedDevice> shared = std::move(made).value();
  EXPECT_NE(shared->surface(), VK_NULL_HANDLE);
  EXPECT_TRUE(shared->graphics_payload().has_present);
  EXPECT_FALSE(shared->compute_payload().has_present);

  // The renderer adopts it, taking the window's surface, and sizes its
  // swapchain to the window.
  vg::app::WindowedAppConfig app_config;
  app_config.swapchain.extent = glfw::framebuffer_extent(window());
  auto app = vg::app::WindowedApp::adopt(
      shared->graphics_payload(), app_config,
      [&shared](VkInstance) -> vkc::Result<VkSurfaceKHR> {
        return shared->release_surface();
      });
  ASSERT_TRUE(app.ok()) << app.status().message();
  const VkExtent2D extent = app.value().swapchain().extent();
  EXPECT_EQ(extent.width, static_cast<uint32_t>(kWidth));
  EXPECT_EQ(extent.height, static_cast<uint32_t>(kHeight));
}

// --- ImGuiGlfw --------------------------------------------------------------

TEST(GlfwImGuiTest, NullWindowIsRefused) {
  ImGuiContext* context = make_context();
  auto platform = glfw::ImGuiGlfw::create(nullptr, context);
  ASSERT_FALSE(platform.ok());
  EXPECT_EQ(platform.status().domain(), vkc::Status::Code::InvalidArgument);
  ImGui::DestroyContext(context);
}

TEST_F(GlfwWindowTest, ImGuiGlfwRefusesANullContext) {
  auto platform = glfw::ImGuiGlfw::create(window(), nullptr);
  ASSERT_FALSE(platform.ok());
  EXPECT_EQ(platform.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(GlfwWindowTest, ImGuiGlfwBindsItsContextUntilDestroyed) {
  ImGuiContext* context = make_context();
  ImGuiContext* other = make_context();
  ImGui::SetCurrentContext(other);
  {
    auto platform = glfw::ImGuiGlfw::create(window(), context);
    ASSERT_TRUE(platform.ok()) << platform.status().message();
    EXPECT_EQ(platform.value()->context(), context);
    EXPECT_EQ(platform.value()->window(), window());
    EXPECT_NE(io_of(context).BackendPlatformUserData, nullptr);
    EXPECT_EQ(io_of(other).BackendPlatformUserData, nullptr);
    EXPECT_EQ(ImGui::GetCurrentContext(), other);
  }
  // Shut down on its own context, though another is current, which stays so.
  EXPECT_EQ(io_of(context).BackendPlatformUserData, nullptr);
  EXPECT_EQ(io_of(context).BackendPlatformName, nullptr);
  EXPECT_EQ(ImGui::GetCurrentContext(), other);
  ImGui::DestroyContext(other);
  ImGui::DestroyContext(context);
}

TEST_F(GlfwWindowTest, ImGuiGlfwRefusesASecondBackendOnAContext) {
  ImGuiContext* context = make_context();
  {
    auto first = glfw::ImGuiGlfw::create(window(), context);
    ASSERT_TRUE(first.ok()) << first.status().message();
    auto second = glfw::ImGuiGlfw::create(window(), context);
    ASSERT_FALSE(second.ok());
    EXPECT_EQ(second.status().domain(), vkc::Status::Code::InvalidArgument);
    EXPECT_NE(io_of(context).BackendPlatformUserData, nullptr);
  }
  ImGui::DestroyContext(context);
}

int g_scrolls = 0;
void count_scroll(GLFWwindow* /*window*/, double /*x*/, double /*y*/) {
  ++g_scrolls;
}

TEST_F(GlfwWindowTest, ImGuiGlfwChainsAndRestoresTheWindowsCallbacks) {
  g_scrolls = 0;
  glfwSetScrollCallback(window(), count_scroll);
  ImGuiContext* context = make_context();
  {
    auto platform = glfw::ImGuiGlfw::create(window(), context);
    ASSERT_TRUE(platform.ok()) << platform.status().message();
    // The backend's callback is installed, and forwards to the window's own.
    const GLFWscrollfun installed =
        glfwSetScrollCallback(window(), count_scroll);
    ASSERT_NE(installed, nullptr);
    ASSERT_NE(installed, &count_scroll);
    glfwSetScrollCallback(window(), installed);
    installed(window(), 0.0, 1.0);
    EXPECT_EQ(g_scrolls, 1);
  }
  // Destroyed: the window's own callback is back.
  EXPECT_EQ(glfwSetScrollCallback(window(), nullptr), &count_scroll);
  ImGui::DestroyContext(context);
}

TEST_F(GlfwWindowTest, ImGuiGlfwNewFrameFeedsItsOwnContext) {
  ImGuiContext* context = make_context();
  ImGuiContext* other = make_context();
  const ImVec2 other_size = io_of(other).DisplaySize;
  {
    auto platform = glfw::ImGuiGlfw::create(window(), context);
    ASSERT_TRUE(platform.ok()) << platform.status().message();
    ImGui::SetCurrentContext(other);
    platform.value()->new_frame();
    EXPECT_EQ(ImGui::GetCurrentContext(), context);
    EXPECT_EQ(io_of(context).DisplaySize.x, static_cast<float>(kWidth));
    EXPECT_EQ(io_of(context).DisplaySize.y, static_cast<float>(kHeight));
    EXPECT_EQ(io_of(other).DisplaySize.x, other_size.x);
    EXPECT_EQ(io_of(other).DisplaySize.y, other_size.y);
  }
  ImGui::DestroyContext(other);
  ImGui::DestroyContext(context);
}

}  // namespace

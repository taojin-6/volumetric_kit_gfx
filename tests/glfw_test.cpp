// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// gfx_glfw: the instance extensions, the surface factory, the framebuffer
// size and cursor, the shared device, and ImGui's GLFW platform backend.
// Windows are made on GLFW's null platform (GLFW 3.4+), so no display is
// needed; with an older GLFW those cases skip. The surface and the shared
// device need more: GLFW 3.5.1+, whose null platform presents through
// VK_EXT_headless_surface (3.4's makes no surface), and a Vulkan device with
// that extension. One case iconifies a window on the native window system,
// and skips where there is no display.

// Before GLFW, so glfw3.h sees Vulkan and declares its helpers.
#include "volumetric_kit/core/vulkan/vulkan.hpp"

#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gfx_test_support.hpp"
#include "imgui.h"
#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/core/vulkan/shared_device.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/app/shared_device_config.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/glfw/imgui_glfw.hpp"
#include "volumetric_kit/gfx/glfw/impl/framebuffer_pixels.hpp"
#include "volumetric_kit/gfx/glfw/shared_device.hpp"
#include "volumetric_kit/gfx/glfw/window.hpp"

namespace glfw = volumetric_kit::gfx::glfw;

namespace {

constexpr int kWidth = 320;
constexpr int kHeight = 240;

// GLFW's last error, prefixed by the call that failed.
std::string glfw_failure(const char* call) {
  const char* description = nullptr;
  glfwGetError(&description);
  return std::string(call) + ": " +
         (description != nullptr ? description : "failed");
}

// GLFW with one window, for this object's lifetime: on the null platform,
// which needs no display, or on the native window system, which does. `ready()`
// is false, with `reason()`, where the platform is unavailable.
class TestWindow {
 public:
  enum class Platform { Null, Native };

  explicit TestWindow(Platform platform) {
    if (platform == Platform::Null) {
#ifdef GLFW_PLATFORM_NULL
      glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
#else
      reason_ = "GLFW " + std::to_string(GLFW_VERSION_MAJOR) + "." +
                std::to_string(GLFW_VERSION_MINOR) + " has no null platform";
      return;
#endif
    } else {
#ifdef GLFW_ANY_PLATFORM
      glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
#endif
#ifndef __APPLE__
      if (std::getenv("DISPLAY") == nullptr &&
          std::getenv("WAYLAND_DISPLAY") == nullptr) {
        reason_ = "no display: neither DISPLAY nor WAYLAND_DISPLAY is set";
        return;
      }
#endif
    }
#ifdef GLFW_PLATFORM_NULL  // GLFW 3.4+
    // The loader this test links, rather than one GLFW searches for.
    glfwInitVulkanLoader(vkGetInstanceProcAddr);
#endif
    if (glfwInit() != GLFW_TRUE) {
      reason_ = glfw_failure("glfwInit");
      return;
    }
    initialized_ = true;
    if (platform == Platform::Native && glfwGetPrimaryMonitor() == nullptr) {
      reason_ = "no display: GLFW finds no monitor";
      return;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (platform == Platform::Native) {
      // Shown, as only a shown window iconifies, but taking no focus from
      // the desktop it appears on.
      glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
      glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
    }
    window_ = glfwCreateWindow(kWidth, kHeight, "glfw_test", nullptr, nullptr);
    if (window_ == nullptr) {
      reason_ = glfw_failure("glfwCreateWindow");
    }
  }
  ~TestWindow() {
    if (window_ != nullptr) {
      glfwDestroyWindow(window_);
    }
    if (initialized_) {
      glfwTerminate();
    }
  }
  TestWindow(const TestWindow&) = delete;
  TestWindow& operator=(const TestWindow&) = delete;

  bool ready() const { return window_ != nullptr; }
  const std::string& reason() const { return reason_; }
  GLFWwindow* get() const { return window_; }

 private:
  bool initialized_ = false;
  GLFWwindow* window_ = nullptr;
  std::string reason_;
};

// Whether the window's GLFW_ICONIFIED reads `iconified` within five seconds,
// handling events meanwhile: a window manager or compositor acts on an iconify
// or restore request in its own time, if at all.
bool settles_iconified(GLFWwindow* window, bool iconified) {
  const int wanted = iconified ? GLFW_TRUE : GLFW_FALSE;
  const double deadline = glfwGetTime() + 5.0;
  while (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != wanted) {
    if (glfwGetTime() >= deadline) {
      return false;
    }
    glfwWaitEventsTimeout(0.05);
  }
  return true;
}

// Why the null platform cannot make a Vulkan surface, or empty if it can. Its
// surface arrived in GLFW 3.5.1: 3.4 offers no extensions and refuses
// glfwCreateWindowSurface. The version, not a call to the code under test,
// decides, so a regression there still fails.
std::string null_surface_unavailable() {
  int major = 0;
  int minor = 0;
  int revision = 0;
  glfwGetVersion(&major, &minor, &revision);
  if (major == 3 && (minor < 5 || (minor == 5 && revision < 1))) {
    return "GLFW " + std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(revision) +
           "'s null platform makes no Vulkan surface (3.5.1+ does)";
  }
  if (!vg_test::has_headless_surface()) {
    return "VK_EXT_headless_surface unavailable";
  }
  return {};
}

// A test with a null-platform window.
class GlfwWindowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    window_ = std::make_unique<TestWindow>(TestWindow::Platform::Null);
    if (!window_->ready()) {
      GTEST_SKIP() << window_->reason();
    }
  }
  void TearDown() override { window_.reset(); }

  GLFWwindow* window() const { return window_->get(); }

 private:
  std::unique_ptr<TestWindow> window_;
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
  if (const std::string reason = null_surface_unavailable(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  const vkc::Result<std::vector<const char*>> extensions =
      glfw::instance_extensions();
  ASSERT_TRUE(extensions.ok()) << extensions.status().message();
  EXPECT_TRUE(has_extension(extensions.value(), VK_KHR_SURFACE_EXTENSION_NAME));
  EXPECT_TRUE(has_extension(extensions.value(),
                            VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME));
}

// --- framebuffer_size / framebuffer_extent / cursor_in_framebuffer ----------

TEST_F(GlfwWindowTest, FramebufferSizeIsTheWindowsAndZeroWhenMinimized) {
  const VkExtent2D size = glfw::framebuffer_size(window());
  EXPECT_EQ(size.width, static_cast<uint32_t>(kWidth));
  EXPECT_EQ(size.height, static_cast<uint32_t>(kHeight));

  // Iconified, the window keeps its size, as on macOS and X11; reading none
  // instead, begin_frame skips the tick rather than draw it.
  glfwIconifyWindow(window());
  ASSERT_EQ(glfwGetWindowAttrib(window(), GLFW_ICONIFIED), GLFW_TRUE);
  int width = -1;
  int height = -1;
  glfwGetFramebufferSize(window(), &width, &height);
  ASSERT_EQ(width, kWidth);
  ASSERT_EQ(height, kHeight);
  const VkExtent2D iconified = glfw::framebuffer_size(window());
  EXPECT_EQ(iconified.width, 0u);
  EXPECT_EQ(iconified.height, 0u);
  glfwRestoreWindow(window());
  const VkExtent2D restored = glfw::framebuffer_size(window());
  EXPECT_EQ(restored.width, static_cast<uint32_t>(kWidth));
  EXPECT_EQ(restored.height, static_cast<uint32_t>(kHeight));

  // A zero size, which a minimized window may report instead, passed on:
  // begin_frame skips the tick on it, where a 1 x 1 extent would rebuild the
  // swapchain.
  glfwSetWindowSize(window(), 0, 0);
  glfwGetFramebufferSize(window(), &width, &height);
  ASSERT_EQ(width, 0);
  ASSERT_EQ(height, 0);
  const VkExtent2D minimized = glfw::framebuffer_size(window());
  EXPECT_EQ(minimized.width, 0u);
  EXPECT_EQ(minimized.height, 0u);
}

// The decision framebuffer_size makes, apart from GLFW, so that it is tested
// with a GLFW that has no null platform too.
TEST(GlfwFramebufferSizeTest, IsNoneWhileIconified) {
  const VkExtent2D shown = glfw::detail::to_framebuffer_size({320, 240}, false);
  EXPECT_EQ(shown.width, 320u);
  EXPECT_EQ(shown.height, 240u);
  const VkExtent2D iconified =
      glfw::detail::to_framebuffer_size({320, 240}, true);
  EXPECT_EQ(iconified.width, 0u);
  EXPECT_EQ(iconified.height, 0u);
}

// On the native window system, where there is a display.
TEST(GlfwNativeWindowTest, FramebufferSizeIsZeroWhileIconified) {
  const TestWindow window(TestWindow::Platform::Native);
  if (!window.ready()) {
    GTEST_SKIP() << window.reason();
  }
#ifdef GLFW_PLATFORM_WAYLAND
  if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
    GTEST_SKIP() << "Wayland: xdg-shell reports no iconified state";
  }
#endif
  const VkExtent2D shown = glfw::framebuffer_size(window.get());
  ASSERT_GT(shown.width, 0u);
  ASSERT_GT(shown.height, 0u);

  glfwIconifyWindow(window.get());
  if (!settles_iconified(window.get(), true)) {
    GTEST_SKIP() << "the window was not iconified within 5 s (no window "
                    "manager?)";
  }
  const VkExtent2D iconified = glfw::framebuffer_size(window.get());
  EXPECT_EQ(iconified.width, 0u);
  EXPECT_EQ(iconified.height, 0u);

  glfwRestoreWindow(window.get());
  ASSERT_TRUE(settles_iconified(window.get(), false))
      << "iconified, but not restored within 5 s";
  const VkExtent2D restored = glfw::framebuffer_size(window.get());
  EXPECT_EQ(restored.width, shown.width);
  EXPECT_EQ(restored.height, shown.height);
}

TEST_F(GlfwWindowTest, FramebufferExtentIsTheWindowsAndNeverZero) {
  const VkExtent2D extent = glfw::framebuffer_extent(window());
  EXPECT_EQ(extent.width, static_cast<uint32_t>(kWidth));
  EXPECT_EQ(extent.height, static_cast<uint32_t>(kHeight));

  // Minimized: iconified, or a zero size.
  glfwIconifyWindow(window());
  const VkExtent2D iconified = glfw::framebuffer_extent(window());
  EXPECT_EQ(iconified.width, 1u);
  EXPECT_EQ(iconified.height, 1u);
  glfwRestoreWindow(window());
  glfwSetWindowSize(window(), 0, 0);
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

// The null platform's scale is always 1, so the high-DPI arithmetic is tested
// on its own.
TEST(GlfwFramebufferPixelsTest, ScalesWindowCoordinatesPerAxis) {
  // A uniform scale of 2, as on a Retina display.
  const glm::vec2 retina =
      glfw::detail::to_framebuffer_pixels({10.0, 10.0}, {100, 50}, {200, 100});
  EXPECT_EQ(retina, glm::vec2(20.0f, 20.0f));
  // Each axis by its own ratio: 2 across, 3 down.
  const glm::vec2 uneven =
      glfw::detail::to_framebuffer_pixels({10.0, 10.0}, {100, 50}, {200, 150});
  EXPECT_EQ(uneven, glm::vec2(20.0f, 30.0f));
  // A sub-pixel window position keeps its fraction.
  const glm::vec2 fraction =
      glfw::detail::to_framebuffer_pixels({0.25, 1.5}, {100, 50}, {200, 100});
  EXPECT_EQ(fraction, glm::vec2(0.5f, 3.0f));
}

TEST(GlfwFramebufferPixelsTest, LeavesAnAxisWithNoWindowSizeUnscaled) {
  EXPECT_EQ(glfw::detail::to_framebuffer_pixels({10.0, 10.0}, {0, 0}, {0, 0}),
            glm::vec2(10.0f, 10.0f));
  EXPECT_EQ(
      glfw::detail::to_framebuffer_pixels({10.0, 10.0}, {0, 50}, {200, 100}),
      glm::vec2(10.0f, 20.0f));
}

// The window functions check their precondition, a non-null window, rather
// than hand null to GLFW, which only asserts it in debug builds.
TEST(GlfwWindowDeathTest, NullWindowIsAProgrammerError) {
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});  // route the abort message to stderr
        glfw::framebuffer_size(nullptr);
      },
      "framebuffer_size needs a window");
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});
        glfw::framebuffer_extent(nullptr);
      },
      "framebuffer_extent needs a window");
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});
        glfw::cursor_in_framebuffer(nullptr);
      },
      "cursor_in_framebuffer needs a window");
}

// --- glfw_surface_factory / make_shared_device ------------------------------

TEST(GlfwSharedDeviceTest, NullWindowIsRefused) {
  const auto shared = glfw::make_shared_device(nullptr, {}, {});
  ASSERT_FALSE(shared.ok());
  EXPECT_EQ(shared.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST(GlfwSurfaceFactoryTest, NullWindowIsRefused) {
  // Refused before GLFW or the instance is touched.
  const vkc::Result<VkSurfaceKHR> surface =
      glfw::glfw_surface_factory(nullptr)(VK_NULL_HANDLE);
  ASSERT_FALSE(surface.ok());
  EXPECT_EQ(surface.status().domain(), vkc::Status::Code::InvalidArgument);
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
    if (const std::string reason = null_surface_unavailable();
        !reason.empty()) {
      GTEST_SKIP() << reason;
    }
    window_ = std::make_unique<TestWindow>(TestWindow::Platform::Null);
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
  std::unique_ptr<TestWindow> window_;
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
  vg::app::WindowDeviceConfig config;
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

// A frame loop given framebuffer_size skips an iconified window's ticks, and
// draws again on restore without rebuilding its swapchain.
TEST_F(GlfwDeviceTest, AnIconifiedWindowsTicksAreSkippedWithoutARebuild) {
  const vkc::Result<std::vector<const char*>> extensions =
      glfw::instance_extensions();
  ASSERT_TRUE(extensions.ok()) << extensions.status().message();
  vg::app::WindowedAppConfig config;
  config.enable_validation =
      vkc::test::instance_config(active_validation()).enable_validation;
  config.instance_extensions = extensions.value();
  config.swapchain.extent = glfw::framebuffer_extent(window());
  auto app = vg::app::WindowedApp::create(config,
                                          glfw::glfw_surface_factory(window()));
  ASSERT_TRUE(app.ok()) << app.status().message();
  int rebuilds = 0;
  app.value().set_recreate_callback([&rebuilds](VkExtent2D) {
    ++rebuilds;
    return vkc::Status{};
  });

  glfwIconifyWindow(window());
  auto skipped = app.value().begin_frame(glfw::framebuffer_size(window()));
  ASSERT_TRUE(skipped.ok()) << skipped.status().message();
  ASSERT_FALSE(skipped.value().has_value());

  glfwRestoreWindow(window());
  auto frame = app.value().begin_frame(glfw::framebuffer_size(window()));
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  ASSERT_TRUE(frame.value().has_value());
  vg::RenderTargetBeginInfo begin;
  begin.clear_color.float32[3] = 1.0f;
  frame.value()->target->begin(frame.value()->cmd, begin);
  frame.value()->target->end(frame.value()->cmd);
  EXPECT_TRUE(app.value().end_frame(*frame.value()).ok());
  EXPECT_EQ(rebuilds, 0);
  EXPECT_TRUE(app.value().wait_idle().ok());
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

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// App tier: the WindowedApp/HeadlessApp bring-up facades. WindowedApp is
// driven through a VK_EXT_headless_surface factory — the same vehicle as
// windowing_test — so the full instance -> surface -> device -> allocator ->
// swapchain -> frame-loop chain runs on Linux CI (lavapipe) without a display;
// the suite skips wholesale where that surface or a present-capable device is
// unavailable (e.g. MoltenVK). HeadlessApp needs no surface at all, so its
// cases run wherever a Vulkan device exists. Validation is enabled in every
// config (a no-op when the layer is absent); the facades own their instances,
// so unlike windowing_test no external error-capturing messenger can outlive
// the chain — the tests are behavior-level instead.

#include <gtest/gtest.h>

#include <cstring>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/app/headless_app.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"

namespace vg = volumetric_kit::gfx;
namespace win = volumetric_kit::gfx::windowing;

namespace {

bool instance_has_headless_surface() {
  uint32_t count = 0;
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) !=
      VK_SUCCESS) {
    return false;
  }
  std::vector<VkExtensionProperties> props(count);
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, props.data()) !=
      VK_SUCCESS) {
    return false;
  }
  for (const VkExtensionProperties& p : props) {
    if (std::strcmp(p.extensionName, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) ==
        0) {
      return true;
    }
  }
  return false;
}

// The SurfaceFactory the tests hand to WindowedApp::create: a raw
// VK_EXT_headless_surface the app adopts (and later destroys) — the headless
// stand-in for a window system's glfwCreateWindowSurface.
vg::Result<VkSurfaceKHR> create_headless_surface(VkInstance instance) {
  auto create = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT"));
  if (create == nullptr) {
    return vg::Status::unsupported("vkCreateHeadlessSurfaceEXT unavailable");
  }
  VkHeadlessSurfaceCreateInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  const VkResult result = create(instance, &info, nullptr, &surface);
  if (result != VK_SUCCESS) {
    return vg::vk_error(result, "vkCreateHeadlessSurfaceEXT");
  }
  return surface;
}

vg::app::WindowedAppConfig windowed_config(
    VkFormat depth_format = VK_FORMAT_UNDEFINED) {
  vg::app::WindowedAppConfig config;
  config.app_name = "vg_app_test";
  config.enable_validation = true;
  config.instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                                VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
  config.swapchain.extent = {256, 256};
  config.swapchain.depth_format = depth_format;
  return config;
}

vg::app::HeadlessAppConfig headless_config() {
  vg::app::HeadlessAppConfig config;
  config.app_name = "vg_app_test";
  config.enable_validation = true;
  return config;
}

class WindowedAppTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Probe the pre-facade steps with the core/windowing APIs so the skip
    // conditions stay as precise as windowing_test's; the probe is torn down
    // again and every test then asserts on the facade itself. Validation is
    // enabled to match the configs below, so a runner whose validated
    // instance cannot be created skips here instead of failing later.
    if (!instance_has_headless_surface()) {
      GTEST_SKIP() << "VK_EXT_headless_surface unavailable (e.g. MoltenVK)";
    }
    vg::InstanceConfig icfg;
    icfg.enable_validation = true;
    icfg.extra_instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                                      VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
    auto instance = vg::Instance::create(icfg);
    if (!instance.ok()) {
      GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
    }
    auto surface = win::Surface::headless(instance.value().handle());
    if (!surface.ok()) {
      GTEST_SKIP() << "headless surface: " << surface.status().message();
    }
    auto physical =
        instance.value().select_physical_device(surface.value().handle());
    if (!physical.ok()) {
      GTEST_SKIP() << "no present-capable device: "
                   << physical.status().message();
    }
  }

  vg::app::WindowedApp make_app(const vg::app::WindowedAppConfig& config) {
    auto app = vg::app::WindowedApp::create(config, create_headless_surface);
    EXPECT_TRUE(app.ok()) << app.status().message();
    // Return empty on failure rather than aborting via Result::value()
    // (VG_CHECK): callers assert on validity, so the test fails cleanly.
    return app.ok() ? std::move(app).value() : vg::app::WindowedApp{};
  }

  // Drive `count` clear-only frames through the app's begin/end passthroughs.
  // The fixed headless extent matches the built swapchain, so no tick is
  // skipped and any non-OK status is a real failure.
  vg::Status run_frames(vg::app::WindowedApp& app, int count) {
    for (int i = 0; i < count; ++i) {
      auto frame = app.begin_frame(app.swapchain().extent());
      if (!frame.ok()) {
        return frame.status();
      }
      if (!frame.value().has_value()) {
        return vg::Status::invalid_argument("unexpected skipped tick");
      }
      vg::RenderTargetBeginInfo begin;
      begin.clear_color.float32[0] = 0.1f;
      begin.clear_color.float32[3] = 1.0f;
      frame.value()->target->begin(frame.value()->cmd, begin);
      frame.value()->target->end(frame.value()->cmd);
      vg::Status end = app.end_frame(*frame.value());
      if (!end.ok()) {
        return end;
      }
    }
    return {};
  }
};

// WindowedApp::adopt runs the same chain on a VkDevice the caller created --
// the shared-device interop case, windowed. Stands in for an embedder by
// building instance + present-capable device the way another library would,
// then handing over the raw handles.
TEST_F(WindowedAppTest, AdoptBuildsChainOnBorrowedDeviceAndRendersFrames) {
  vg::InstanceConfig icfg;
  icfg.enable_validation = true;
  icfg.extra_instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                                    VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
  auto instance = vg::Instance::create(icfg);
  ASSERT_TRUE(instance.ok()) << instance.status().message();
  // Selection needs a surface; the app builds its own below, so this one only
  // serves the embedder's device creation.
  auto probe = win::Surface::headless(instance.value().handle());
  ASSERT_TRUE(probe.ok()) << probe.status().message();
  auto physical =
      instance.value().select_physical_device(probe.value().handle());
  ASSERT_TRUE(physical.ok()) << physical.status().message();
  vg::DeviceConfig dcfg;
  dcfg.needs_present = true;
  auto owner = vg::Device::create(instance.value().handle(), physical.value(),
                                  dcfg, probe.value().handle());
  ASSERT_TRUE(owner.ok()) << owner.status().message();

  // A present-capable device enables the swapchain extension; adopt verifies
  // the declaration against what the renderer needs.
  const char* const kEnabled[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  vg::AdoptedDevice adopted;
  adopted.instance = instance.value().handle();
  adopted.physical_device = physical.value();
  adopted.device = owner.value().handle();
  adopted.graphics_family = owner.value().graphics_family();
  adopted.graphics_queue = owner.value().graphics_queue();
  adopted.has_present = true;
  adopted.present_family = owner.value().present_family();
  adopted.present_queue = owner.value().present_queue();
  adopted.enabled_device_extensions = kEnabled;
  adopted.enabled_device_extension_count = 1;

  {
    auto app = vg::app::WindowedApp::adopt(adopted, windowed_config(),
                                           create_headless_surface);
    ASSERT_TRUE(app.ok()) << app.status().message();
    ASSERT_TRUE(app.value().valid());

    // Borrowed, not built: the same VkDevice and the embedder's instance.
    EXPECT_FALSE(app.value().device().owns_device());
    EXPECT_EQ(app.value().device().handle(), owner.value().handle());
    EXPECT_EQ(app.value().instance_handle(), instance.value().handle());

    // Everything downstream of the device is still the app's own, and live:
    // frames render end to end through the borrowed device's queues.
    EXPECT_EQ(app.value().swapchain().extent().width, 256u);
    vg::Status frames = run_frames(app.value(), 3);
    EXPECT_TRUE(frames.ok()) << frames.message();
  }  // the app destructs here -- it must NOT destroy the borrowed device

  // The owner's device survived the app's teardown: a submit proves the
  // adopted app left it intact (a double-free trips the sanitizer job).
  vg::Status after = owner.value().submit_single_time([](VkCommandBuffer) {});
  EXPECT_TRUE(after.ok()) << after.message();
}

// A windowed app must present, so a compute-only share is refused up front
// rather than failing deeper as a missing present queue.
TEST_F(WindowedAppTest, AdoptRejectsShareWithoutPresentQueue) {
  vg::AdoptedDevice adopted;
  adopted.instance = reinterpret_cast<VkInstance>(0x1);  // never dereferenced
  adopted.has_present = false;
  auto app = vg::app::WindowedApp::adopt(adopted, windowed_config(),
                                         create_headless_surface);
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vg::Status::Code::InvalidArgument);
}

// One create() call yields the whole chain: every accessor hands back a live
// object, and the begin/end passthroughs render real frames through it.
TEST_F(WindowedAppTest, CreateBuildsFullChainAndRendersFrames) {
  vg::app::WindowedApp app = make_app(windowed_config());
  ASSERT_TRUE(app.valid());
  EXPECT_NE(app.instance().handle(), VK_NULL_HANDLE);
  EXPECT_NE(app.device().handle(), VK_NULL_HANDLE);
  // The one-call chain threaded the surface through device selection and
  // creation, so the device really is present-capable.
  EXPECT_TRUE(app.device().has_present());
  EXPECT_GE(app.allocator().memory_stats().heap_count, 1u);
  EXPECT_TRUE(app.swapchain().valid());
  EXPECT_TRUE(app.frame_loop().valid());
  EXPECT_EQ(app.frame_loop().frames_in_flight(), 2u);

  const vg::Status status = run_frames(app, 3);
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(app.wait_idle().ok());
}

// swapchain.depth_format flows through create(): the app passes its allocator
// to the swapchain, whose targets come out depth-capable, and frames render.
TEST_F(WindowedAppTest, DepthConfigBuildsDepthCapableSwapchain) {
  vg::app::WindowedApp app = make_app(windowed_config(VK_FORMAT_D32_SFLOAT));
  ASSERT_TRUE(app.valid());
  EXPECT_EQ(app.swapchain().layout().depth_format, VK_FORMAT_D32_SFLOAT);
  for (uint32_t i = 0; i < app.swapchain().image_count(); ++i) {
    EXPECT_EQ(app.swapchain().render_target(i).layout().depth_format,
              VK_FORMAT_D32_SFLOAT);
  }

  // run_frames' begin info clears depth too: load_op defaults to CLEAR, so
  // every frame writes the depth attachments create() wired in.
  const vg::Status status = run_frames(app, 3);
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(app.wait_idle().ok());
}

// set_recreate_callback forwards to the loop: a frame at the built extent runs
// nothing, and a changed extent rebuilds the swapchain and runs the callback
// with the rebuilt extent.
TEST_F(WindowedAppTest, RecreateCallbackForwardsToLoop) {
  vg::app::WindowedApp app = make_app(windowed_config());
  ASSERT_TRUE(app.valid());

  int callback_runs = 0;
  VkExtent2D callback_extent{};
  app.set_recreate_callback([&](VkExtent2D extent) {
    ++callback_runs;
    callback_extent = extent;
    return vg::Status{};
  });

  // A frame at the current extent: no rebuild, the callback stays quiet.
  ASSERT_TRUE(run_frames(app, 1).ok());
  EXPECT_EQ(callback_runs, 0);

  // A changed extent: the loop rebuilds the swapchain and runs the callback
  // once with the rebuilt extent (the headless surface honors the request).
  auto resized = app.begin_frame(VkExtent2D{320, 240});
  ASSERT_TRUE(resized.ok()) << resized.status().message();
  ASSERT_TRUE(resized.value().has_value());
  EXPECT_EQ(callback_runs, 1);
  EXPECT_EQ(callback_extent.width, app.swapchain().extent().width);
  EXPECT_EQ(callback_extent.height, app.swapchain().extent().height);
  EXPECT_EQ(app.swapchain().extent().width, 320u);
  vg::RenderTargetBeginInfo begin;
  begin.clear_color.float32[3] = 1.0f;
  resized.value()->target->begin(resized.value()->cmd, begin);
  resized.value()->target->end(resized.value()->cmd);
  EXPECT_TRUE(app.end_frame(*resized.value()).ok());
  EXPECT_TRUE(app.wait_idle().ok());
}

// A zero frames_in_flight reaches FrameLoop::create, whose own rejection
// propagates out of the facade unchanged.
TEST_F(WindowedAppTest, ZeroFramesInFlightPropagatesLoopError) {
  vg::app::WindowedAppConfig config = windowed_config();
  config.frames_in_flight = 0;
  auto app = vg::app::WindowedApp::create(config, create_headless_surface);
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vg::Status::Code::InvalidArgument);
}

// A factory that reports failure has its exact Status surfaced by create().
TEST_F(WindowedAppTest, SurfaceFactoryFailurePropagates) {
  auto app = vg::app::WindowedApp::create(
      windowed_config(), [](VkInstance) -> vg::Result<VkSurfaceKHR> {
        return vg::Status::io_error("test: window system failed");
      });
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vg::Status::Code::IoError);
}

// A factory that "succeeds" with a null handle is rejected up front instead of
// being handed to the swapchain.
TEST_F(WindowedAppTest, NullSurfaceFromFactoryIsRejected) {
  auto app = vg::app::WindowedApp::create(
      windowed_config(),
      [](VkInstance) -> vg::Result<VkSurfaceKHR> { return VK_NULL_HANDLE; });
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(WindowedAppTest, MoveLeavesSourceEmpty) {
  vg::app::WindowedApp src = make_app(windowed_config());
  ASSERT_TRUE(src.valid());

  vg::app::WindowedApp moved(std::move(src));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)

  // The moved-to app still renders: the loop's borrowed swapchain/device
  // addresses survived the move (the chain lives behind one stable pointer).
  EXPECT_TRUE(run_frames(moved, 2).ok());
  EXPECT_TRUE(moved.wait_idle().ok());

  // Frame calls on the emptied source fail cleanly rather than crashing.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  auto frame = src.begin_frame(VkExtent2D{256, 256});
  ASSERT_FALSE(frame.ok());
  EXPECT_EQ(frame.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(WindowedAppTest, MoveAssignOverLiveLeavesSourceEmpty) {
  // Two complete chains (each app owns its own instance and surface); the
  // assignment must tear dst's chain down in reverse order before adopting
  // src's — ASan/LSan turns a wrong order into a detected failure.
  vg::app::WindowedApp src = make_app(windowed_config());
  vg::app::WindowedApp dst = make_app(windowed_config());
  ASSERT_TRUE(src.valid());
  ASSERT_TRUE(dst.valid());

  dst = std::move(src);
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(run_frames(dst, 2).ok());
  EXPECT_TRUE(dst.wait_idle().ok());
}

TEST_F(WindowedAppTest, SelfMoveAssignIsSafe) {
  vg::app::WindowedApp app = make_app(windowed_config());
  ASSERT_TRUE(app.valid());

  vg::app::WindowedApp* alias = &app;
  app = std::move(*alias);  // laundered so -Wself-move stays quiet
  EXPECT_TRUE(app.valid());
  EXPECT_TRUE(run_frames(app, 2).ok());
  EXPECT_TRUE(app.wait_idle().ok());
}

// Frame calls on a default-constructed (never-created) app fail cleanly.
// Needs no instance/device, so it runs everywhere.
TEST(WindowedAppEmpty, OperationsFailCleanly) {
  vg::app::WindowedApp app;
  EXPECT_FALSE(app.valid());
  auto frame = app.begin_frame(VkExtent2D{256, 256});
  ASSERT_FALSE(frame.ok());
  EXPECT_EQ(frame.status().domain(), vg::Status::Code::InvalidArgument);
  const vg::Status end = app.end_frame(win::Frame{});
  ASSERT_FALSE(end.ok());
  EXPECT_EQ(end.domain(), vg::Status::Code::InvalidArgument);
  const vg::Status idle = app.wait_idle();
  ASSERT_FALSE(idle.ok());
  EXPECT_EQ(idle.domain(), vg::Status::Code::InvalidArgument);
}

// A null surface factory is rejected before any Vulkan call, so this runs
// everywhere too.
TEST(WindowedAppValidation, NullSurfaceFactoryIsRejected) {
  auto app = vg::app::WindowedApp::create(
      vg::app::WindowedAppConfig{}, vg::app::WindowedApp::SurfaceFactory{});
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vg::Status::Code::InvalidArgument);
}

// HeadlessApp needs no surface extension and no present queue, so this fixture
// guards only on a device existing — it runs wherever windowing cannot (e.g.
// MoltenVK).
class HeadlessAppTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto app = vg::app::HeadlessApp::create(headless_config());
    if (!app.ok()) {
      GTEST_SKIP() << "no Vulkan device: " << app.status().message();
    }
    app_ = std::move(app).value();
  }

  vg::app::HeadlessApp app_;
};

TEST_F(HeadlessAppTest, CreateBuildsInstanceDeviceAllocator) {
  ASSERT_TRUE(app_.valid());
  EXPECT_NE(app_.instance().handle(), VK_NULL_HANDLE);
  EXPECT_NE(app_.device().handle(), VK_NULL_HANDLE);
  EXPECT_FALSE(app_.device().has_present());  // headless: no present queue
  EXPECT_GE(app_.allocator().memory_stats().heap_count, 1u);
}

TEST_F(HeadlessAppTest, MoveLeavesSourceEmpty) {
  vg::app::HeadlessApp moved(std::move(app_));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(app_.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_NE(moved.device().handle(), VK_NULL_HANDLE);
  EXPECT_GE(moved.allocator().memory_stats().heap_count, 1u);
}

TEST_F(HeadlessAppTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto other = vg::app::HeadlessApp::create(headless_config());
  ASSERT_TRUE(other.ok()) << other.status().message();
  vg::app::HeadlessApp dst = std::move(other).value();
  ASSERT_TRUE(dst.valid());

  // Destroys dst's chain in reverse order (allocator, device, instance), then
  // adopts app_'s; ASan/LSan turns a wrong order into a detected failure.
  dst = std::move(app_);
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(app_.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_NE(dst.device().handle(), VK_NULL_HANDLE);
}

TEST_F(HeadlessAppTest, SelfMoveAssignIsSafe) {
  vg::app::HeadlessApp* alias = &app_;
  app_ = std::move(*alias);  // laundered so -Wself-move stays quiet
  EXPECT_TRUE(app_.valid());
  EXPECT_NE(app_.device().handle(), VK_NULL_HANDLE);
}

}  // namespace

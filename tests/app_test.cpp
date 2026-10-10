// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// App tier: the WindowedApp/HeadlessApp bring-up facades. WindowedApp is
// driven through a VK_EXT_headless_surface factory -- the same vehicle as
// windowing_test -- so the full instance -> surface -> device -> allocator ->
// swapchain -> frame-loop chain runs without a display; the suite skips where
// that surface or a present-capable device is unavailable. HeadlessApp needs
// no surface at all. The facades own their instances, validated as the
// environment asks, and the fixtures fail a test on any error their layer
// reports.

#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/app/headless_app.hpp"
#include "volumetric_kit/gfx/app/windowed_app.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"

namespace win = volumetric_kit::gfx::windowing;

namespace {

// The SurfaceFactory the tests hand to WindowedApp::create: a raw
// VK_EXT_headless_surface the app adopts (and later destroys) — the headless
// stand-in for a window system's glfwCreateWindowSurface.
vkc::Result<VkSurfaceKHR> create_headless_surface(VkInstance instance) {
  auto create = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT"));
  if (create == nullptr) {
    return vkc::Status::unsupported("vkCreateHeadlessSurfaceEXT unavailable");
  }
  VkHeadlessSurfaceCreateInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  const VkResult result = create(instance, &info, nullptr, &surface);
  if (result != VK_SUCCESS) {
    return vkc::vk_error(result, "vkCreateHeadlessSurfaceEXT");
  }
  return surface;
}

vg::app::WindowedAppConfig windowed_config(
    VkFormat depth_format = VK_FORMAT_UNDEFINED) {
  vg::app::WindowedAppConfig config;
  config.app_name = "vg_app_test";
  config.enable_validation = vkc::test::instance_config().enable_validation;
  config.instance_extensions = vg_test::headless_surface_extensions();
  config.swapchain.extent = {256, 256};
  config.swapchain.depth_format = depth_format;
  return config;
}

// A share whose handles are all non-null and whose instance version is set, so
// a test can knock out exactly one field and watch that check fire. Never
// dereferenced: every case built on this is rejected on its arguments, before
// a handle reaches Vulkan.
vkc::AdoptedDevice placeholder_share() {
  vkc::AdoptedDevice adopted;
  adopted.instance = reinterpret_cast<VkInstance>(0x1);
  adopted.instance_api_version = VK_API_VERSION_1_3;
  adopted.physical_device = reinterpret_cast<VkPhysicalDevice>(0x2);
  adopted.device = reinterpret_cast<VkDevice>(0x3);
  adopted.queue = reinterpret_cast<VkQueue>(0x4);
  adopted.has_present = true;
  adopted.present_queue = reinterpret_cast<VkQueue>(0x5);
  return adopted;
}

vg::app::HeadlessAppConfig headless_config() {
  vg::app::HeadlessAppConfig config;
  config.app_name = "vg_app_test";
  config.enable_validation = vkc::test::instance_config().enable_validation;
  return config;
}

class WindowedAppTest : public vg_test::RendererTest {
 protected:
  void SetUp() override {
    // Probe the pre-facade steps with the core/windowing APIs so the skip
    // conditions stay as precise as windowing_test's; the probe is torn down
    // again and every test then asserts on the facade itself.
    RendererTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    if (!vg_test::has_headless_surface()) {
      GTEST_SKIP() << "VK_EXT_headless_surface unavailable";
    }
    auto instance = vkc::Instance::create(vg_test::headless_instance_config());
    ASSERT_TRUE(instance.ok()) << instance.status().message();
    auto surface = win::Surface::headless(instance.value().handle());
    if (!surface.ok()) {
      GTEST_SKIP() << "headless surface: " << surface.status().message();
    }
    vkc::DeviceRequirements reqs = vg::device_requirements();
    reqs.needs_present = true;
    auto physical =
        instance.value().select_physical_device(reqs, surface.value().handle());
    if (!physical.ok()) {
      GTEST_SKIP() << "no present-capable device: "
                   << physical.status().message();
    }
  }

  vg::app::WindowedApp make_app(const vg::app::WindowedAppConfig& config) {
    auto app = vg::app::WindowedApp::create(config, create_headless_surface);
    EXPECT_TRUE(app.ok()) << app.status().message();
    // Return empty on failure rather than aborting via Result::value()
    // (VKC_CHECK): callers assert on validity, so the test fails cleanly.
    return app.ok() ? std::move(app).value() : vg::app::WindowedApp{};
  }

  // Drive `count` clear-only frames through the app's begin/end passthroughs.
  // The fixed headless extent matches the built swapchain, so no tick is
  // skipped and any non-OK status is a real failure.
  vkc::Status run_frames(vg::app::WindowedApp& app, int count) {
    for (int i = 0; i < count; ++i) {
      auto frame = app.begin_frame(app.swapchain().extent());
      if (!frame.ok()) {
        return frame.status();
      }
      if (!frame.value().has_value()) {
        return vkc::Status::invalid_argument("unexpected skipped tick");
      }
      vg::RenderTargetBeginInfo begin;
      begin.clear_color.float32[0] = 0.1f;
      begin.clear_color.float32[3] = 1.0f;
      frame.value()->target->begin(frame.value()->cmd, begin);
      frame.value()->target->end(frame.value()->cmd);
      vkc::Status end = app.end_frame(*frame.value());
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
  auto instance = vkc::Instance::create(vg_test::headless_instance_config());
  ASSERT_TRUE(instance.ok()) << instance.status().message();
  // Selection needs a surface; the app builds its own below, so this one only
  // serves the embedder's device creation.
  auto probe = win::Surface::headless(instance.value().handle());
  ASSERT_TRUE(probe.ok()) << probe.status().message();
  vkc::DeviceRequirements reqs = vg::device_requirements();
  reqs.needs_present = true;
  auto physical =
      instance.value().select_physical_device(reqs, probe.value().handle());
  ASSERT_TRUE(physical.ok()) << physical.status().message();
  auto owner = vkc::Device::create(instance.value(), physical.value(), reqs,
                                   probe.value().handle());
  ASSERT_TRUE(owner.ok()) << owner.status().message();

  // A present-capable device enables the swapchain extension; adopt verifies
  // the declaration against what the renderer needs.
  const char* const kEnabled[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  vkc::AdoptedDevice adopted;
  adopted.instance = instance.value().handle();
  adopted.instance_api_version = instance.value().api_version();
  adopted.physical_device = physical.value().handle();
  adopted.device = owner.value().handle();
  adopted.queue_family = owner.value().queue_family();
  adopted.queue = owner.value().queue();
  adopted.submit_mutex = owner.value().submit_mutex();
  adopted.has_present = true;
  adopted.present_family = owner.value().present_family();
  adopted.present_queue = owner.value().present_queue();
  adopted.enabled_extensions = kEnabled;
  adopted.enabled_extension_count = 1;
  // Device::create enabled both version-core bits on `owner`; adopt verifies
  // this declaration rather than mere physical-device support.
  adopted.enabled_features.timeline_semaphore = true;
  adopted.enabled_features.dynamic_rendering = true;
  adopted.enabled_debug_utils = instance.value().debug_utils_enabled();

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
    vkc::Status frames = run_frames(app.value(), 3);
    EXPECT_TRUE(frames.ok()) << frames.message();
  }  // the app destructs here -- it must NOT destroy the borrowed device

  // The owner's device survived the app's teardown: a submit proves the
  // adopted app left it intact (a double-free trips the sanitizer job).
  vkc::Status after = owner.value().submit_single_time([](VkCommandBuffer) {});
  EXPECT_TRUE(after.ok()) << after.message();

  // The share is held to the renderer's floor even when config.device starts
  // from bare requirements, which name no dynamic rendering: a share that
  // never enabled it is refused rather than rendered on.
  vg::app::WindowedAppConfig bare = windowed_config();
  bare.device = vkc::DeviceRequirements{};
  adopted.enabled_features.dynamic_rendering = false;
  auto refused =
      vg::app::WindowedApp::adopt(adopted, bare, create_headless_surface);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().domain(), vkc::Status::Code::Unsupported)
      << refused.status().message();
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
  EXPECT_EQ(app.frame_loop().frames_in_flight(), 2u);

  const vkc::Status status = run_frames(app, 3);
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(app.wait_idle().ok());
}

// config.device can only add to the renderer's floor: bare requirements, which
// name no dynamic rendering, still build a chain every pass can render on.
TEST_F(WindowedAppTest, CreateMergesTheRendererFloorIn) {
  vg::app::WindowedAppConfig config = windowed_config();
  config.device = vkc::DeviceRequirements{};
  vg::app::WindowedApp app = make_app(config);
  ASSERT_TRUE(app.valid());
  const vkc::Status enabled =
      app.device().check_enabled(vg::device_requirements());
  EXPECT_TRUE(enabled.ok()) << enabled.message();
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
  const vkc::Status status = run_frames(app, 3);
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
    return vkc::Status{};
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
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument);
}

// A factory that reports failure has its exact Status surfaced by create().
TEST_F(WindowedAppTest, SurfaceFactoryFailurePropagates) {
  auto app = vg::app::WindowedApp::create(
      windowed_config(), [](VkInstance) -> vkc::Result<VkSurfaceKHR> {
        return vkc::Status::io_error("test: window system failed");
      });
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::IoError);
}

// A factory that "succeeds" with a null handle is rejected up front instead of
// being handed to the swapchain.
TEST_F(WindowedAppTest, NullSurfaceFromFactoryIsRejected) {
  auto app = vg::app::WindowedApp::create(
      windowed_config(),
      [](VkInstance) -> vkc::Result<VkSurfaceKHR> { return VK_NULL_HANDLE; });
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument);
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
  EXPECT_EQ(frame.status().domain(), vkc::Status::Code::InvalidArgument);
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
  EXPECT_EQ(frame.status().domain(), vkc::Status::Code::InvalidArgument);
  const vkc::Status end = app.end_frame(win::Frame{});
  ASSERT_FALSE(end.ok());
  EXPECT_EQ(end.domain(), vkc::Status::Code::InvalidArgument);
  const vkc::Status idle = app.wait_idle();
  ASSERT_FALSE(idle.ok());
  EXPECT_EQ(idle.domain(), vkc::Status::Code::InvalidArgument);
}

// A null surface factory is rejected before any Vulkan call, so this runs
// everywhere too.
TEST(WindowedAppValidation, NullSurfaceFactoryIsRejected) {
  auto app = vg::app::WindowedApp::create(
      vg::app::WindowedAppConfig{}, vg::app::WindowedApp::SurfaceFactory{});
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument);
}

// adopt()'s argument checks all reject before a handle reaches Vulkan, so they
// belong here rather than behind WindowedAppTest -- they need no device and no
// headless surface, and so run on MoltenVK too.

// A windowed app must present, so a compute-only share is refused up front
// rather than failing deeper as a missing present queue.
TEST(WindowedAppValidation, AdoptRejectsShareWithoutPresentQueue) {
  vkc::AdoptedDevice adopted = placeholder_share();
  adopted.has_present = false;
  adopted.present_queue = VK_NULL_HANDLE;
  auto app = vg::app::WindowedApp::adopt(adopted, windowed_config(),
                                         create_headless_surface);
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument);
}

// Every handle is vetted before the factory runs: a share that was never going
// to work must not first cost the embedder a created-then-destroyed surface
// (a real window surface, in the non-headless case).
TEST(WindowedAppValidation, AdoptRejectsNullHandlesWithoutRunningFactory) {
  bool factory_ran = false;
  auto factory = [&factory_ran](VkInstance) -> vkc::Result<VkSurfaceKHR> {
    factory_ran = true;
    return vkc::Status::unsupported("the factory must not run");
  };
  auto expect_rejected = [&factory](const vkc::AdoptedDevice& adopted,
                                    const char* which) {
    auto app = vg::app::WindowedApp::adopt(adopted, windowed_config(), factory);
    ASSERT_FALSE(app.ok()) << which;
    EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument)
        << which;
  };

  vkc::AdoptedDevice adopted = placeholder_share();
  adopted.instance = VK_NULL_HANDLE;
  expect_rejected(adopted, "null instance");

  adopted = placeholder_share();
  adopted.physical_device = VK_NULL_HANDLE;
  expect_rejected(adopted, "null physical device");

  adopted = placeholder_share();
  adopted.device = VK_NULL_HANDLE;
  expect_rejected(adopted, "null device");

  adopted = placeholder_share();
  adopted.queue = VK_NULL_HANDLE;
  expect_rejected(adopted, "null queue");

  EXPECT_FALSE(factory_ran);
}

// Device::adopt refuses an unset instance version too, but only after the
// factory has made a surface; the pre-check spares the embedder that.
TEST(WindowedAppValidation,
     AdoptRejectsUnsetInstanceVersionWithoutRunningFactory) {
  bool factory_ran = false;
  auto factory = [&factory_ran](VkInstance) -> vkc::Result<VkSurfaceKHR> {
    factory_ran = true;
    return vkc::Status::unsupported("the factory must not run");
  };
  vkc::AdoptedDevice adopted = placeholder_share();
  adopted.instance_api_version = 0;
  auto app = vg::app::WindowedApp::adopt(adopted, windowed_config(), factory);
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_FALSE(factory_ran);
}

// The surface step is shared with create(), so it names the path that called
// it -- an embedder is told which entry point rejected its factory.
TEST(WindowedAppValidation, AdoptRejectsNullSurfaceFactory) {
  auto app =
      vg::app::WindowedApp::adopt(placeholder_share(), windowed_config(),
                                  vg::app::WindowedApp::SurfaceFactory{});
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_NE(app.status().message().find("WindowedApp::adopt"),
            std::string::npos)
      << app.status().message();
}

TEST(WindowedAppValidation, AdoptRejectsNullSurfaceFromFactory) {
  auto app = vg::app::WindowedApp::adopt(
      placeholder_share(), windowed_config(),
      [](VkInstance) -> vkc::Result<VkSurfaceKHR> { return VK_NULL_HANDLE; });
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument);
  EXPECT_NE(app.status().message().find("WindowedApp::adopt"),
            std::string::npos)
      << app.status().message();
}

// HeadlessApp needs no surface extension and no present queue, so this fixture
// asks only for a device fit for the renderer.
class HeadlessAppTest : public vg_test::RendererTest {
 protected:
  void SetUp() override {
    RendererTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto app = vg::app::HeadlessApp::create(headless_config());
    ASSERT_TRUE(app.ok()) << app.status().message();
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

// The device has debug labels iff the instance enabled VK_EXT_debug_utils.
// The app tier builds both, so it is the tier's job to connect them -- and
// until it did, every label and object name recorded through an app-tier
// device was a silent no-op (a capture of a validation-enabled app showed no
// pass markers, with no diagnostic saying why).
TEST_F(HeadlessAppTest, DeviceDebugLabelsMatchInstanceFlag) {
  ASSERT_TRUE(app_.valid());
  EXPECT_EQ(app_.device().debug_labels_available(),
            app_.instance().debug_utils_enabled());
}

// The requirements passthrough reaches Device::create: without it the facade
// could not enable a device feature at all (GraphicsPipeline tells consumers to
// require fillModeNonSolid, which had no app-tier route).
using HeadlessAppConfigTest = vg_test::RendererTest;

TEST_F(HeadlessAppConfigTest, PassesDeviceFeaturesThrough) {
  auto probe = vg::app::HeadlessApp::create(headless_config());
  ASSERT_TRUE(probe.ok()) << probe.status().message();
  VkPhysicalDeviceFeatures supported{};
  vkGetPhysicalDeviceFeatures(probe.value().device().physical_device(),
                              &supported);
  if (supported.fillModeNonSolid != VK_TRUE) {
    GTEST_SKIP() << "device does not support fillModeNonSolid";
  }

  vg::app::HeadlessAppConfig config = headless_config();
  config.device.features.fillModeNonSolid = VK_TRUE;
  auto app = vg::app::HeadlessApp::create(config);
  EXPECT_TRUE(app.ok()) << app.status().message();
}

// config.device can only add to the renderer's floor: requirements built from
// DeviceRequirements{} (the core's defaults, without dynamic rendering) still
// yield a device every gfx type runs on.
TEST_F(HeadlessAppConfigTest, MergesTheRendererFloorIn) {
  vg::app::HeadlessAppConfig config = headless_config();
  config.device = vkc::DeviceRequirements{};
  auto app = vg::app::HeadlessApp::create(config);
  ASSERT_TRUE(app.ok()) << app.status().message();
  const vkc::Status enabled =
      app.value().device().check_enabled(vg::device_requirements());
  EXPECT_TRUE(enabled.ok()) << enabled.message();
}

// A headless app has no surface to present to. Refused before any Vulkan call,
// so this runs everywhere.
TEST(HeadlessAppValidation, NeedsPresentIsRejected) {
  vg::app::HeadlessAppConfig config = headless_config();
  config.device.needs_present = true;
  auto app = vg::app::HeadlessApp::create(config);
  ASSERT_FALSE(app.ok());
  EXPECT_EQ(app.status().domain(), vkc::Status::Code::InvalidArgument);
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

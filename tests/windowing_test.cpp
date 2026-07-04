// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Windowing tier on a headless surface (VK_EXT_headless_surface): swapchain
// creation, the FrameLoop acquire -> render -> present choreography, recreate,
// and the move-only lifecycle. The whole suite skips when the runner has no
// headless surface (e.g. MoltenVK) or no present-capable device, so it provides
// real coverage on Linux CI (lavapipe) without needing a display.
//
// Validation is enabled AND given teeth: a second debug messenger records every
// validation error, and TearDown fails the test if any were emitted -- so a
// mis-wired barrier / semaphore is caught, not just a non-VK_SUCCESS return.

#include <gtest/gtest.h>

#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

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

// Records validation errors into the std::vector<std::string> passed as
// pUserData; never asks the driver to abort the call (returns VK_FALSE).
VKAPI_ATTR VkBool32 VKAPI_CALL record_validation_error(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0 &&
      user != nullptr) {
    auto* errors = static_cast<std::vector<std::string>*>(user);
    errors->emplace_back(data != nullptr && data->pMessage != nullptr
                             ? data->pMessage
                             : "(validation error)");
  }
  return VK_FALSE;
}

class WindowingTest : public ::testing::Test {
 protected:
  void SetUp() override {
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
    instance_.emplace(std::move(instance).value());

    // Attach an error-recording messenger so validation has teeth (no-op when
    // the instance lacks VK_EXT_debug_utils, e.g. no validation layer present).
    install_validation_capture();

    auto surface = win::Surface::headless(instance_->handle());
    if (!surface.ok()) {
      GTEST_SKIP() << "headless surface: " << surface.status().message();
    }
    surface_ = std::move(surface).value();

    auto physical = instance_->select_physical_device(surface_.handle());
    if (!physical.ok()) {
      GTEST_SKIP() << "no present-capable device: "
                   << physical.status().message();
    }

    vg::DeviceConfig dcfg;
    dcfg.needs_present = true;
    auto device = vg::Device::create(instance_->handle(), physical.value(),
                                     dcfg, surface_.handle());
    ASSERT_TRUE(device.ok()) << device.status().message();
    device_.emplace(std::move(device).value());
  }

  void TearDown() override {
    if (device_) {
      vkDeviceWaitIdle(device_->handle());
    }
    // Destroy the messenger (created on the instance) before the instance is
    // torn down with the fixture, then surface any captured validation errors.
    if (messenger_ != VK_NULL_HANDLE && destroy_messenger_ != nullptr) {
      destroy_messenger_(instance_->handle(), messenger_, nullptr);
      messenger_ = VK_NULL_HANDLE;
    }
    for (const std::string& msg : validation_errors_) {
      ADD_FAILURE() << "Vulkan validation error: " << msg;
    }
  }

  void install_validation_capture() {
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_->handle(),
                              "vkCreateDebugUtilsMessengerEXT"));
    destroy_messenger_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_->handle(),
                              "vkDestroyDebugUtilsMessengerEXT"));
    if (create == nullptr || destroy_messenger_ == nullptr) {
      return;  // VK_EXT_debug_utils not enabled; capture stays inert.
    }
    VkDebugUtilsMessengerCreateInfoEXT mcfg{};
    mcfg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    mcfg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    mcfg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    mcfg.pfnUserCallback = record_validation_error;
    mcfg.pUserData = &validation_errors_;
    if (create(instance_->handle(), &mcfg, nullptr, &messenger_) !=
        VK_SUCCESS) {
      messenger_ = VK_NULL_HANDLE;
    }
  }

  win::Surface make_headless_surface() {
    auto s = win::Surface::headless(instance_->handle());
    EXPECT_TRUE(s.ok()) << s.status().message();
    return s.ok() ? std::move(s).value() : win::Surface{};
  }

  win::Swapchain make_swapchain_on(VkSurfaceKHR surface,
                                   VkExtent2D extent = {256, 256}) {
    win::SwapchainConfig cfg;
    cfg.extent = extent;
    auto sc = win::Swapchain::create(*device_, surface, cfg);
    EXPECT_TRUE(sc.ok()) << sc.status().message();
    // Return empty on failure rather than aborting via Result::value()
    // (VG_CHECK): callers assert on validity, so the test fails cleanly.
    return sc.ok() ? std::move(sc).value() : win::Swapchain{};
  }

  win::Swapchain make_swapchain(VkExtent2D extent = {256, 256}) {
    return make_swapchain_on(surface_.handle(), extent);
  }

  // Drive `count` clear-only frames through the loop. A fixed headless extent
  // never goes out of date, so any non-OK status is a real failure.
  vg::Status run_frames(win::FrameLoop& loop, int count) {
    for (int i = 0; i < count; ++i) {
      auto frame = loop.begin_frame();
      if (!frame.ok()) {
        return frame.status();
      }
      vg::RenderTargetBeginInfo begin;
      begin.clear_color.float32[0] = 0.1f;
      begin.clear_color.float32[3] = 1.0f;
      frame.value().target->begin(frame.value().cmd, begin);
      frame.value().target->end(frame.value().cmd);
      vg::Status end = loop.end_frame(frame.value());
      if (!end.ok()) {
        return end;
      }
    }
    return {};
  }

  std::optional<vg::Instance> instance_;
  win::Surface surface_;
  std::optional<vg::Device> device_;
  std::vector<std::string> validation_errors_;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger_ = nullptr;
};

TEST_F(WindowingTest, CreatesSwapchainWithRenderTargets) {
  win::Swapchain sc = make_swapchain();
  EXPECT_TRUE(sc.valid());
  EXPECT_GE(sc.image_count(), 1u);
  EXPECT_NE(sc.format(), VK_FORMAT_UNDEFINED);
  const vg::RenderTargetLayout layout = sc.layout();
  EXPECT_EQ(layout.color_count, 1u);
  EXPECT_EQ(layout.color_formats[0], sc.format());
  EXPECT_TRUE(sc.render_target(0).valid());
  // Every image exposes a non-null VkImage + color VkImageView, so a caller can
  // assemble its own render target over it (e.g. adding a depth attachment).
  for (uint32_t i = 0; i < sc.image_count(); ++i) {
    EXPECT_NE(sc.image(i), VK_NULL_HANDLE);
    EXPECT_NE(sc.image_view(i), VK_NULL_HANDLE);
  }
  // Each index maps to a distinct image + view (no aliasing across slots), so a
  // per-image render target addresses the right one.
  for (uint32_t i = 0; i < sc.image_count(); ++i) {
    for (uint32_t j = i + 1; j < sc.image_count(); ++j) {
      EXPECT_NE(sc.image(i), sc.image(j));
      EXPECT_NE(sc.image_view(i), sc.image_view(j));
    }
  }
}

TEST_F(WindowingTest, FrameLoopRendersAndPresents) {
  win::Swapchain sc = make_swapchain();
  auto loop = win::FrameLoop::create(*device_, sc, /*frames_in_flight=*/2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();
  EXPECT_EQ(loop.value().frames_in_flight(), 2u);

  const vg::Status status = run_frames(loop.value(), /*count=*/8);
  EXPECT_TRUE(status.ok()) << status.message();

  vkDeviceWaitIdle(device_->handle());
}

// The loop drives an attached profiler's begin_frame/end_frame; the caller only
// opens a scope around the render. After enough frames a slot recurs and its
// GPU timing resolves into the published snapshot. The fixture's validation
// capture catches a mis-wired query reset / timestamp / label nesting as a test
// failure.
TEST_F(WindowingTest, FrameLoopDrivesAttachedProfiler) {
  win::Swapchain sc = make_swapchain();

  // Declare the profiler before the loop so it outlives the loop that borrows
  // it: at scope exit the loop is destroyed first, then the profiler.
  vg::ProfilerConfig pcfg;
  pcfg.frames_in_flight = 2;  // match the loop's in-flight depth
  auto profiler = vg::Profiler::create(*device_, pcfg);
  ASSERT_TRUE(profiler.ok()) << profiler.status().message();

  auto loop = win::FrameLoop::create(*device_, sc, /*frames_in_flight=*/2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();
  loop.value().set_profiler(&profiler.value());

  for (int i = 0; i < 6; ++i) {
    auto frame = loop.value().begin_frame();
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    {
      // Scope the whole render so its GPU timestamps + label sit outside the
      // render pass; it closes before end_frame ends + submits the buffer.
      vg::Profiler::Scope pass =
          profiler.value().gpu_scope(frame.value().cmd, "frame");
      vg::RenderTargetBeginInfo begin;
      begin.clear_color.float32[3] = 1.0f;
      frame.value().target->begin(frame.value().cmd, begin);
      frame.value().target->end(frame.value().cmd);
    }
    ASSERT_TRUE(loop.value().end_frame(frame.value()).ok());
  }
  vkDeviceWaitIdle(device_->handle());

  const vg::FrameMetrics& m = profiler.value().metrics();
  ASSERT_FALSE(m.sections.empty());
  EXPECT_STREQ(m.sections[0].name, "frame");
  EXPECT_GT(m.cpu_frame_ms, 0.0);  // end_frame stamped a real CPU frame time
  // has_gpu is the load-bearing wiring check: it goes true only when the loop
  // passed the frame's own cmd to the profiler's begin_frame, and the fixture's
  // validation capture fails the test on any mis-wired reset / timestamp.
  EXPECT_EQ(m.sections[0].has_gpu, profiler.value().gpu_timing());
}

// A moved FrameLoop carries its attached profiler: the moved-to loop drives
// begin/end_frame, so a stage opened on its frames still resolves. Guards the
// move pair against dropping the borrowed profiler_ (which would silently stop
// profiling after any move). `profiler` is declared before the loops so it
// outlives both.
TEST_F(WindowingTest, MovedFrameLoopKeepsDrivingProfiler) {
  win::Swapchain sc = make_swapchain();

  vg::ProfilerConfig pcfg;
  pcfg.frames_in_flight = 2;
  auto profiler = vg::Profiler::create(*device_, pcfg);
  ASSERT_TRUE(profiler.ok()) << profiler.status().message();

  auto created = win::FrameLoop::create(*device_, sc, /*frames_in_flight=*/2);
  ASSERT_TRUE(created.ok()) << created.status().message();
  created.value().set_profiler(&profiler.value());

  // Move-construct: the borrowed profiler_ must travel to `loop`.
  win::FrameLoop loop = std::move(created).value();

  for (int i = 0; i < 4; ++i) {
    auto frame = loop.begin_frame();
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    {
      vg::Profiler::Scope pass =
          profiler.value().gpu_scope(frame.value().cmd, "moved");
      vg::RenderTargetBeginInfo begin;
      begin.clear_color.float32[3] = 1.0f;
      frame.value().target->begin(frame.value().cmd, begin);
      frame.value().target->end(frame.value().cmd);
    }
    ASSERT_TRUE(loop.end_frame(frame.value()).ok());
  }
  vkDeviceWaitIdle(device_->handle());

  // The moved-to loop drove the profiler, so a "moved" stage was published.
  const vg::FrameMetrics& m = profiler.value().metrics();
  ASSERT_FALSE(m.sections.empty());
  EXPECT_STREQ(m.sections[0].name, "moved");
}

TEST_F(WindowingTest, RecreateKeepsFormatAndLayout) {
  win::Swapchain sc = make_swapchain({256, 256});
  const VkFormat format = sc.format();
  const vg::RenderTargetLayout before = sc.layout();

  ASSERT_TRUE(sc.recreate({320, 240}).ok());
  EXPECT_EQ(sc.format(), format);
  EXPECT_TRUE(sc.layout().compatible_with(before));
  // The recreate actually applied the new size (a headless surface honors the
  // requested extent), proving it was not a no-op.
  EXPECT_EQ(sc.extent().width, 320u);
  EXPECT_EQ(sc.extent().height, 240u);
  // The recreated swapchain re-exposes a non-null color view per image, so a
  // caller-owned render target can be rebuilt over the new images.
  for (uint32_t i = 0; i < sc.image_count(); ++i) {
    EXPECT_NE(sc.image_view(i), VK_NULL_HANDLE);
  }

  auto loop = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();
  EXPECT_TRUE(run_frames(loop.value(), 4).ok());
  vkDeviceWaitIdle(device_->handle());
}

// A zero-extent recreate (minimized window) must fail *without* destroying the
// current swapchain: the old chain stays valid and presentable, so the render
// loop keeps working until a restore-sized recreate succeeds. Regression test:
// recreate used to destroy-then-rebuild, leaving a null chain that the next
// begin_frame handed straight to vkAcquireNextImageKHR.
TEST_F(WindowingTest, RecreateZeroExtentLeavesSwapchainUsable) {
  win::Swapchain sc = make_swapchain({256, 256});
  ASSERT_TRUE(sc.valid());
  const VkSwapchainKHR before = sc.handle();

  const vg::Status zero = sc.recreate({0, 0});
  EXPECT_FALSE(zero.ok());
  EXPECT_EQ(zero.domain(), vg::Status::Code::InvalidArgument);
  EXPECT_TRUE(sc.valid());
  EXPECT_EQ(sc.handle(), before);  // untouched, not rebuilt
  EXPECT_EQ(sc.extent().width, 256u);
  EXPECT_EQ(sc.extent().height, 256u);

  // The untouched chain still drives frames...
  auto loop = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();
  EXPECT_TRUE(run_frames(loop.value(), 2).ok());

  // ...and a later non-zero recreate recovers normally under the same loop.
  ASSERT_TRUE(sc.recreate({320, 240}).ok());
  EXPECT_TRUE(run_frames(loop.value(), 2).ok());
  vkDeviceWaitIdle(device_->handle());
}

// begin_frame on a loop whose borrowed swapchain has been emptied (moved-from,
// as after a failed rebuild) fails cleanly instead of acquiring on a null
// handle. NOTE: this covers begin_frame's empty-swapchain guard only, NOT the
// post-acquire recover_slot path — reaching that needs a Vulkan call (fence
// wait / command begin/end / queue submit) to fail, which does not happen on a
// healthy device; see the TODO in FrameLoop::recover_slot on the missing
// fault-injection coverage for those branches.
TEST_F(WindowingTest, FrameLoopBeginFrameOnEmptiedSwapchainFailsCleanly) {
  win::Swapchain sc = make_swapchain();
  auto loop = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();
  EXPECT_TRUE(run_frames(loop.value(), 1).ok());
  vkDeviceWaitIdle(device_->handle());

  // Move the swapchain out from under the loop: the borrowed &sc now refers to
  // an empty object.
  win::Swapchain stolen = std::move(sc);
  auto frame = loop.value().begin_frame();
  ASSERT_FALSE(frame.ok());
  EXPECT_EQ(frame.status().domain(), vg::Status::Code::InvalidArgument);
  vkDeviceWaitIdle(device_->handle());
}

TEST_F(WindowingTest, FrameLoopSurvivesSwapchainRecreate) {
  win::Swapchain sc = make_swapchain({256, 256});
  auto loop = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();

  EXPECT_TRUE(run_frames(loop.value(), 3).ok());
  // Recreate under the SAME FrameLoop (the documented resize pattern): the loop
  // must resync its per-image sync objects and keep driving frames.
  ASSERT_TRUE(sc.recreate({320, 240}).ok());
  EXPECT_TRUE(run_frames(loop.value(), 3).ok());
  vkDeviceWaitIdle(device_->handle());
}

// The extent-taking begin_frame owns the loop protocol: a zero extent skips
// the tick (minimized) without acquiring, a matching extent renders normally,
// and a changed extent rebuilds the swapchain and runs the recreate callback
// (with the rebuilt extent) before delivering the next frame.
TEST_F(WindowingTest, ManagedBeginFrameSkipsAndRebuilds) {
  win::Swapchain sc = make_swapchain({256, 256});
  auto loop = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();

  int callback_runs = 0;
  VkExtent2D callback_extent{};
  loop.value().set_recreate_callback([&](VkExtent2D extent) {
    ++callback_runs;
    callback_extent = extent;
    return vg::Status{};
  });

  // Zero extent: a skipped tick — no acquire, no rebuild.
  auto skipped = loop.value().begin_frame(VkExtent2D{0, 0});
  ASSERT_TRUE(skipped.ok()) << skipped.status().message();
  EXPECT_FALSE(skipped.value().has_value());
  EXPECT_EQ(callback_runs, 0);

  // Matching extent: a normal frame, still no rebuild.
  auto frame = loop.value().begin_frame(VkExtent2D{256, 256});
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  ASSERT_TRUE(frame.value().has_value());
  {
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[3] = 1.0f;
    frame.value()->target->begin(frame.value()->cmd, begin);
    frame.value()->target->end(frame.value()->cmd);
  }
  ASSERT_TRUE(loop.value().end_frame(*frame.value()).ok());
  EXPECT_EQ(callback_runs, 0);

  // Changed extent: the loop rebuilds the swapchain, runs the callback with
  // the rebuilt extent, and still delivers a frame.
  auto resized = loop.value().begin_frame(VkExtent2D{320, 240});
  ASSERT_TRUE(resized.ok()) << resized.status().message();
  ASSERT_TRUE(resized.value().has_value());
  EXPECT_EQ(callback_runs, 1);
  EXPECT_EQ(callback_extent.width, 320u);
  EXPECT_EQ(callback_extent.height, 240u);
  EXPECT_EQ(sc.extent().width, 320u);
  {
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[3] = 1.0f;
    resized.value()->target->begin(resized.value()->cmd, begin);
    resized.value()->target->end(resized.value()->cmd);
  }
  ASSERT_TRUE(loop.value().end_frame(*resized.value()).ok());
  vkDeviceWaitIdle(device_->handle());
}

// A failing recreate callback aborts the frame as a hard error the caller must
// not retry — the loop does not swallow it into a skipped tick.
TEST_F(WindowingTest, ManagedBeginFramePropagatesCallbackFailure) {
  win::Swapchain sc = make_swapchain({256, 256});
  auto loop = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();
  loop.value().set_recreate_callback([](VkExtent2D) {
    return vg::Status::out_of_memory("test: depth rebuild failed");
  });

  auto resized = loop.value().begin_frame(VkExtent2D{320, 240});
  ASSERT_FALSE(resized.ok());
  EXPECT_EQ(resized.status().domain(), vg::Status::Code::OutOfMemory);
  vkDeviceWaitIdle(device_->handle());
}

// The recreate callback is a borrowed hook moved with the loop (like the
// profiler): a move-constructed loop still runs it, and a changed extent driven
// through the moved loop rebuilds and invokes it with the rebuilt extent.
TEST_F(WindowingTest, MovedFrameLoopKeepsRunningRecreateCallback) {
  win::Swapchain sc = make_swapchain({256, 256});
  auto created = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(created.ok()) << created.status().message();

  int callback_runs = 0;
  VkExtent2D callback_extent{};
  created.value().set_recreate_callback([&](VkExtent2D extent) {
    ++callback_runs;
    callback_extent = extent;
    return vg::Status{};
  });

  // Move-construct after the hook is set; the moved-to loop must carry it.
  win::FrameLoop loop = std::move(created.value());

  auto resized = loop.begin_frame(VkExtent2D{320, 240});
  ASSERT_TRUE(resized.ok()) << resized.status().message();
  ASSERT_TRUE(resized.value().has_value());
  EXPECT_EQ(callback_runs, 1);
  EXPECT_EQ(callback_extent.width, 320u);
  {
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[3] = 1.0f;
    resized.value()->target->begin(resized.value()->cmd, begin);
    resized.value()->target->end(resized.value()->cmd);
  }
  ASSERT_TRUE(loop.end_frame(*resized.value()).ok());
  vkDeviceWaitIdle(device_->handle());
}

// Destruction drains in-flight frames: no explicit device wait before the loop
// goes out of scope. The fixture's validation capture fails the test if the
// destructor freed command buffers / semaphores the GPU still referenced. A
// best-effort guard (a tiny clear may finish before teardown); ASan/LSan and
// the sanitizer CI job are what turn it into a reliable detector.
TEST_F(WindowingTest, DestructionDrainsInFlightFrames) {
  win::Swapchain sc = make_swapchain();
  {
    auto loop = win::FrameLoop::create(*device_, sc, 2);
    ASSERT_TRUE(loop.ok()) << loop.status().message();
    EXPECT_TRUE(run_frames(loop.value(), 3).ok());
  }
}

// Acquire / present / recreate on an empty swapchain (default-constructed,
// moved-from, or after a failed rebuild) fail with InvalidArgument instead of
// dereferencing null handles. Needs no instance/device, so it runs everywhere.
TEST(SwapchainEmpty, OperationsFailCleanly) {
  win::Swapchain sc;
  auto acquired = sc.acquire_next_image(VK_NULL_HANDLE);
  ASSERT_FALSE(acquired.ok());
  EXPECT_EQ(acquired.status().domain(), vg::Status::Code::InvalidArgument);
  const vg::Status presented = sc.present(0, VK_NULL_HANDLE);
  ASSERT_FALSE(presented.ok());
  EXPECT_EQ(presented.domain(), vg::Status::Code::InvalidArgument);
  const vg::Status recreated = sc.recreate({256, 256});
  ASSERT_FALSE(recreated.ok());
  EXPECT_EQ(recreated.domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(WindowingTest, SwapchainMoveLeavesSourceEmpty) {
  win::Swapchain src = make_swapchain();
  ASSERT_TRUE(src.valid());

  win::Swapchain moved(std::move(src));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  // Metadata is reset, not left stale, so accessors track valid().
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(src.format(), VK_FORMAT_UNDEFINED);
}

TEST_F(WindowingTest, SwapchainMoveAssignOverLiveLeavesSourceEmpty) {
  // A surface permits only one live swapchain, so dst needs its own surface.
  win::Surface surface_b = make_headless_surface();
  ASSERT_TRUE(surface_b.valid());
  win::Swapchain src = make_swapchain();
  win::Swapchain dst = make_swapchain_on(surface_b.handle());
  ASSERT_TRUE(src.valid());
  ASSERT_TRUE(dst.valid());

  dst = std::move(src);  // destroy()-then-adopt path
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(src.format(), VK_FORMAT_UNDEFINED);
  vkDeviceWaitIdle(device_->handle());
}

TEST_F(WindowingTest, SwapchainSelfMoveAssignIsSafe) {
  win::Swapchain sc = make_swapchain();
  ASSERT_TRUE(sc.valid());

  win::Swapchain* alias = &sc;
  sc = std::move(*alias);   // guarded by if (this != &other)
  EXPECT_TRUE(sc.valid());  // unchanged and still usable
  EXPECT_NE(sc.format(), VK_FORMAT_UNDEFINED);
}

TEST_F(WindowingTest, FrameLoopMoveLeavesSourceEmpty) {
  win::Swapchain sc = make_swapchain();
  auto created = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(created.ok()) << created.status().message();

  win::FrameLoop moved(std::move(created).value());
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(created.value().valid());  // NOLINT(bugprone-use-after-move)
  // The moved-to loop owns the resources and drives frames; confirm it works.
  EXPECT_TRUE(run_frames(moved, 2).ok());
  vkDeviceWaitIdle(device_->handle());
}

TEST_F(WindowingTest, FrameLoopMoveAssignOverLiveLeavesSourceEmpty) {
  // Each swapchain needs its own surface (one live swapchain per surface).
  win::Surface surface_b = make_headless_surface();
  ASSERT_TRUE(surface_b.valid());
  win::Swapchain sc_a = make_swapchain();
  win::Swapchain sc_b = make_swapchain_on(surface_b.handle());
  auto a = win::FrameLoop::create(*device_, sc_a, 2);
  auto b = win::FrameLoop::create(*device_, sc_b, 2);
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();

  // Leave b with in-flight work so the move-assign's drain() has something to
  // wait on before it frees b's command pool + buffers: freeing them while the
  // GPU still references them is a fault the validation-with-teeth fixture (and
  // ASan/LSan on the wrong free order) catches here.
  EXPECT_TRUE(run_frames(b.value(), 2).ok());
  b.value() = std::move(a.value());
  EXPECT_TRUE(b.value().valid());
  EXPECT_FALSE(a.value().valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(run_frames(b.value(), 2).ok());  // adopted loop drives sc_a
  vkDeviceWaitIdle(device_->handle());
}

TEST_F(WindowingTest, FrameLoopSelfMoveAssignIsSafe) {
  win::Swapchain sc = make_swapchain();
  auto loop = win::FrameLoop::create(*device_, sc, 2);
  ASSERT_TRUE(loop.ok()) << loop.status().message();

  win::FrameLoop* alias = &loop.value();
  loop.value() = std::move(*alias);  // guarded by if (this != &other)
  EXPECT_TRUE(loop.value().valid());
  EXPECT_TRUE(run_frames(loop.value(), 2).ok());
  vkDeviceWaitIdle(device_->handle());
}

TEST_F(WindowingTest, SurfaceMoveLeavesSourceEmpty) {
  win::Surface src = make_headless_surface();
  ASSERT_TRUE(src.valid());

  win::Surface moved(std::move(src));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(WindowingTest, SurfaceMoveAssignOverLiveLeavesSourceEmpty) {
  win::Surface dst = make_headless_surface();
  win::Surface src = make_headless_surface();
  ASSERT_TRUE(dst.valid());
  ASSERT_TRUE(src.valid());

  dst = std::move(src);  // destroy()-then-adopt path
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(WindowingTest, SurfaceSelfMoveAssignIsSafe) {
  win::Surface s = make_headless_surface();
  ASSERT_TRUE(s.valid());

  win::Surface* alias = &s;
  s = std::move(*alias);  // guarded by if (this != &other)
  EXPECT_TRUE(s.valid());
}

}  // namespace

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <utility>

// Internal helper header: the DebugUtils tests re-derive the expected
// debug-utils enable decision from the loader's reported instance extensions
// (has_extension / instance_extensions), matching how Instance::create decides.
#include "volumetric_kit/gfx/core/impl/vk_query.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// The shared instance + physical-device + headless logical-device fixture.
using DeviceTest = VulkanDeviceTest;

}  // namespace

TEST_F(DeviceTest, ExposesGraphicsQueueAndCommandPool) {
  EXPECT_NE(device_->handle(), VK_NULL_HANDLE);
  EXPECT_NE(device_->graphics_queue(), VK_NULL_HANDLE);
  EXPECT_NE(device_->command_pool(), VK_NULL_HANDLE);
  EXPECT_FALSE(device_->has_present());  // headless config
}

TEST_F(DeviceTest, SingleTimeSubmitRoundTrips) {
  // No-op recording exercises allocate / begin / end / submit / fence-wait.
  vg::Status status = device_->submit_single_time([](VkCommandBuffer) {});
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_F(DeviceTest, NeedsPresentWithoutSurfaceErrors) {
  vg::DeviceConfig config;
  config.needs_present = true;
  auto device =
      vg::Device::create(instance_->handle(), physical_, config);  // no surface
  ASSERT_FALSE(device.ok());
  EXPECT_EQ(device.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(DeviceTest, PresentQueuePathNeedsSurface) {
  // The present-queue success path — has_present(), present_family/queue, and
  // the graphics==present dedup — needs a real VkSurfaceKHR, which the core
  // tier cannot create (surfaces are the windowing tier). Document the coverage
  // gap with an explicit skip so it stays visible until a surface-backed test
  // lands downstream, rather than being silently uncovered.
  GTEST_SKIP() << "present-queue success path requires a surface (windowing "
                  "tier); covered there";
}

TEST_F(DeviceTest, SelectedDeviceMeetsVulkan13Floor) {
  // Device::create rejects a sub-1.3 device (shaders target SPIR-V 1.6 and the
  // timeline-semaphore path uses 1.2 core entry points); the fixture device was
  // created successfully, so the selected physical device must report >= 1.3.
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_, &props);
  EXPECT_GE(props.apiVersion, VK_API_VERSION_1_3);
}

TEST_F(DeviceTest, FeatureChainWithVulkan12FeaturesEnablesTimeline) {
  // The headline pNext fix: a caller passing VkPhysicalDeviceVulkan12Features
  // (which aggregates timelineSemaphore) must NOT also get our standalone
  // VkPhysicalDeviceTimelineSemaphoreFeatures linked — both in one chain
  // violates VUID-VkDeviceCreateInfo-pNext-02830. create() instead raises
  // timelineSemaphore inside the caller's struct. Verify device creation
  // succeeds and a TimelineSemaphore is usable (proving timeline was enabled).
  VkPhysicalDeviceVulkan12Features v12{};
  v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  // Left timelineSemaphore = VK_FALSE on purpose; create() must raise it.
  vg::DeviceConfig config;
  config.feature_chain = &v12;

  auto device = vg::Device::create(instance_->handle(), physical_, config);
  ASSERT_TRUE(device.ok()) << device.status().message();

  auto timeline = vg::TimelineSemaphore::create(device.value().handle());
  ASSERT_TRUE(timeline.ok()) << timeline.status().message();
  EXPECT_TRUE(timeline.value().valid());
}

TEST_F(DeviceTest, BogusExtensionFailsUnsupported) {
  vg::DeviceConfig config;
  config.extra_device_extensions = {"VK_VG_definitely_not_a_real_extension"};
  auto device = vg::Device::create(instance_->handle(), physical_, config);
  ASSERT_FALSE(device.ok());
  EXPECT_EQ(device.status().domain(), vg::Status::Code::Unsupported);
}

TEST_F(DeviceTest, KnownExtensionRequestSucceeds) {
  // Request an extension the device actually reports, so create() must accept
  // it.
  auto caps = instance_->query_physical_device(physical_);
  const char* candidate = nullptr;
  if (caps.supports_device_extension("VK_KHR_portability_subset")) {
    candidate = "VK_KHR_portability_subset";
  } else if (caps.supports_device_extension(
                 VK_KHR_MAINTENANCE2_EXTENSION_NAME)) {
    candidate = VK_KHR_MAINTENANCE2_EXTENSION_NAME;
  }
  if (candidate == nullptr) {
    GTEST_SKIP() << "device reports no extension to request";
  }
  vg::DeviceConfig config;
  config.extra_device_extensions = {candidate};
  auto device = vg::Device::create(instance_->handle(), physical_, config);
  EXPECT_TRUE(device.ok()) << device.status().message();
}

TEST_F(DeviceTest, CapsReportsSaneExtensionsAndFormats) {
  const vg::PhysicalDeviceInfo& caps = device_->caps();
  EXPECT_EQ(caps.handle(), physical_);
  EXPECT_FALSE(caps.supports_device_extension("VK_VG_not_real"));
  // R8G8B8A8_UNORM as a sampled OPTIMAL image is required of every conformant
  // implementation — true on MoltenVK and lavapipe.
  EXPECT_TRUE(caps.format_supports(VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_IMAGE_TILING_OPTIMAL,
                                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT));
  EXPECT_FALSE(caps.format_supports(VK_FORMAT_UNDEFINED,
                                    VK_IMAGE_TILING_OPTIMAL,
                                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT));
  EXPECT_GT(caps.limits().maxImageDimension2D, 0u);
}

TEST_F(DeviceTest, CapsExposesFeaturesPropertiesAndFormatProperties) {
  const vg::PhysicalDeviceInfo& caps = device_->caps();
  // properties(): a created device reports the >= 1.3 floor and a non-empty
  // name.
  EXPECT_GE(caps.properties().apiVersion, VK_API_VERSION_1_3);
  EXPECT_NE(caps.properties().deviceName[0], '\0');
  // features2(): query() sets the sType; features() forwards to its inner set.
  EXPECT_EQ(caps.features2().sType,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
  EXPECT_EQ(&caps.features(), &caps.features2().features);
  // format_properties(): the live query reports the sampled bit for a
  // ubiquitous optimal-tiling format.
  VkFormatProperties props = caps.format_properties(VK_FORMAT_R8G8B8A8_UNORM);
  EXPECT_NE(props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
            0u);
}

TEST_F(DeviceTest, GraphicsTimestampValidBitsAreConsistent) {
  // timestampValidBits is reported per queue family in [0, 64]; 0 means the
  // graphics queue cannot write timestamps (MoltenVK may report this), so don't
  // assert a specific value — lavapipe reports 64, MoltenVK may report 0. Just
  // check internal consistency, plus the ns-per-tick conversion factor the
  // capability gate pairs with.
  const uint32_t bits = device_->graphics_timestamp_valid_bits();
  EXPECT_LE(bits, 64u);
  if (bits > 0) {
    // If the graphics queue supports timestamps, timestampPeriod (the
    // tick->nanosecond factor a timing path divides by) must be positive.
    EXPECT_GT(device_->caps().limits().timestampPeriod, 0.0f);
  }
}

TEST_F(DeviceTest, InstanceCapsMatchesDeviceCaps) {
  // The same physical device, queried via the instance and via the created
  // device, agrees on extension support.
  auto via_instance = instance_->query_physical_device(physical_);
  EXPECT_EQ(via_instance.supports_device_extension("VK_KHR_swapchain"),
            device_->caps().supports_device_extension("VK_KHR_swapchain"));
}

TEST_F(DeviceTest, MoveConstructTransfersOwnership) {
  auto made =
      vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::Device source = std::move(made).value();
  ASSERT_NE(source.handle(), VK_NULL_HANDLE);

  vg::Device moved(std::move(source));
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  // If the move had not nulled the source, both the moved-from device and
  // `moved` would vkDestroyDevice the same handle at scope exit — a validation
  // error.
  EXPECT_EQ(source.handle(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
  // Metadata is zeroed too, not just the owned handles: a moved-from device
  // reports an empty physical device (the recurring "forgot a scalar" miss).
  EXPECT_EQ(source.physical_device(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(DeviceTest, MoveAssignOverLiveDeviceLeavesSourceEmpty) {
  auto a =
      vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
  auto b =
      vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Device dst = std::move(a).value();
  vg::Device src = std::move(b).value();

  dst = std::move(src);  // frees dst's original VkDevice, then adopts src's
  EXPECT_NE(dst.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(src.handle(), VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST_F(DeviceTest, SelfMoveAssignIsSafe) {
  auto made =
      vg::Device::create(instance_->handle(), physical_, vg::DeviceConfig{});
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::Device device = std::move(made).value();

  // Pointer-laundered so -Wself-move stays quiet under -Werror.
  vg::Device* alias = &device;
  device = std::move(*alias);
  EXPECT_NE(device.handle(), VK_NULL_HANDLE);
}

// Instance creation needs only the loader, so this runs without a GPU; it still
// skips on a truly Vulkan-less host.
TEST(InstanceTest, MoveConstructLeavesSourceEmpty) {
  auto instance = vg::Instance::create(vg::InstanceConfig{});
  if (!instance.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
  }
  vg::Instance source = std::move(instance).value();
  ASSERT_NE(source.handle(), VK_NULL_HANDLE);

  vg::Instance moved(std::move(source));
  EXPECT_NE(moved.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.handle(),
            VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST(InstanceTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto a = vg::Instance::create(vg::InstanceConfig{});
  if (!a.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << a.status().message();
  }
  auto b = vg::Instance::create(vg::InstanceConfig{});
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Instance dst = std::move(a).value();
  vg::Instance src = std::move(b).value();

  dst = std::move(src);  // destroys dst's original instance, then adopts src's
  EXPECT_NE(dst.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(src.handle(), VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move)
}

TEST(InstanceTest, SelfMoveAssignIsSafe) {
  auto created = vg::Instance::create(vg::InstanceConfig{});
  if (!created.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << created.status().message();
  }
  vg::Instance instance = std::move(created).value();

  vg::Instance* alias = &instance;
  instance = std::move(*alias);
  EXPECT_NE(instance.handle(), VK_NULL_HANDLE);
}

// Exercises the validation path: enabling it drives the debug-messenger
// create/destroy lifetime (proc-addr resolution, pNext chaining, teardown) and
// validation_enabled(). On the sanitizers job — which installs the validation
// layers and runs under ASan — this is where the messenger lifetime and the
// layer's own handle/lifetime checks get exercised. When the layer is absent,
// create() warns and disables, so both outcomes are valid; the test asserts the
// instance is still usable.
TEST(InstanceTest, ValidationEnabledInstanceIsUsable) {
  vg::InstanceConfig config;
  config.enable_validation = true;
  auto instance = vg::Instance::create(config);
  if (!instance.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
  }
  // validation_enabled() must be coherent: true only if the messenger was
  // created. Either way the instance must select a device (messenger active or
  // not).
  auto physical = instance.value().select_physical_device();
  if (!physical.ok()) {
    GTEST_SKIP() << "no Vulkan device: " << physical.status().message();
  }
  EXPECT_NE(physical.value(), VK_NULL_HANDLE);
}

// --- Instance::debug_utils_enabled() + InstanceConfig::enable_debug_utils
// Named with a DebugUtils prefix and kept at the file tail to localize the diff
// (a sibling branch also appends to this file).

// Default config requests neither validation nor debug-utils, so the extension
// stays off and the accessor is coherent with that: false, and distinct from
// validation_enabled() (also false here).
TEST(InstanceDebugUtilsTest, DefaultConfigLeavesDebugUtilsDisabled) {
  auto instance = vg::Instance::create(vg::InstanceConfig{});
  if (!instance.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
  }
  EXPECT_FALSE(instance.value().debug_utils_enabled());
  EXPECT_FALSE(instance.value().validation_enabled());
}

// Opt-in path: enable_debug_utils with validation OFF. When the extension is
// available the accessor reports true; when it is absent it must report false
// (and never crash). Validation stays off either way, proving the two facts are
// independent. The extension is enabled iff debug_utils_enabled() is true, so
// we re-derive the expectation from the loader's reported instance extensions
// and require an exact match — the single-decision invariant.
TEST(InstanceDebugUtilsTest, OptInEnablesDebugUtilsWithoutValidation) {
  vg::InstanceConfig config;
  config.enable_debug_utils = true;
  auto instance = vg::Instance::create(config);
  if (!instance.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
  }
  const bool ext_present = vg::has_extension(vg::instance_extensions(),
                                             VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  EXPECT_EQ(instance.value().debug_utils_enabled(), ext_present);
  // Opting into debug-utils must not turn validation on.
  EXPECT_FALSE(instance.value().validation_enabled());
}

// With validation ON, debug_utils_enabled() stays coherent with today's
// behavior via two implications that hold on every path (including the degraded
// one where the extension is enabled but the messenger fails to create):
//   * the messenger lives only alongside the extension, so validation_enabled()
//     implies debug_utils_enabled();
//   * the flag is never true without the extension actually being present.
TEST(InstanceDebugUtilsTest, ValidationKeepsDebugUtilsCoherent) {
  vg::InstanceConfig config;
  config.enable_validation = true;
  auto instance = vg::Instance::create(config);
  if (!instance.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
  }
  const bool debug_utils = instance.value().debug_utils_enabled();
  if (instance.value().validation_enabled()) {
    EXPECT_TRUE(debug_utils);
  }
  if (debug_utils) {
    EXPECT_TRUE(vg::has_extension(vg::instance_extensions(),
                                  VK_EXT_DEBUG_UTILS_EXTENSION_NAME));
  }
}

// The flag is metadata that must follow ownership: a moved-from instance
// reports false, and the destination inherits the source's state.
TEST(InstanceDebugUtilsTest, MoveTransfersDebugUtilsFlag) {
  vg::InstanceConfig config;
  config.enable_debug_utils = true;
  auto created = vg::Instance::create(config);
  if (!created.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << created.status().message();
  }
  vg::Instance source = std::move(created).value();
  const bool had_debug_utils = source.debug_utils_enabled();

  vg::Instance moved(std::move(source));
  EXPECT_EQ(moved.debug_utils_enabled(), had_debug_utils);
  EXPECT_FALSE(
      source.debug_utils_enabled());  // NOLINT(bugprone-use-after-move)
}

// Move-assignment must carry the flag too: assigning over a live instance runs
// the destroy()-then-adopt path, so the destination takes the source's state
// and the moved-from source is emptied.
TEST(InstanceDebugUtilsTest, MoveAssignTransfersDebugUtilsFlag) {
  vg::InstanceConfig config;
  config.enable_debug_utils = true;
  auto created_src = vg::Instance::create(config);
  auto created_dst = vg::Instance::create(config);
  if (!created_src.ok() || !created_dst.ok()) {
    GTEST_SKIP() << "no Vulkan instance";
  }
  vg::Instance source = std::move(created_src).value();
  vg::Instance dest = std::move(created_dst).value();
  const bool had_debug_utils = source.debug_utils_enabled();

  dest = std::move(source);
  EXPECT_EQ(dest.debug_utils_enabled(), had_debug_utils);
  EXPECT_FALSE(
      source.debug_utils_enabled());  // NOLINT(bugprone-use-after-move)
}

// Self-move is a no-op for the flag (and the instance): launder through a
// pointer so -Wself-move under -Werror doesn't reject the assignment.
TEST(InstanceDebugUtilsTest, SelfMoveKeepsDebugUtilsFlag) {
  vg::InstanceConfig config;
  config.enable_debug_utils = true;
  auto created = vg::Instance::create(config);
  if (!created.ok()) {
    GTEST_SKIP() << "no Vulkan instance: " << created.status().message();
  }
  vg::Instance instance = std::move(created).value();
  const bool had_debug_utils = instance.debug_utils_enabled();

  vg::Instance* p = &instance;
  instance = std::move(*p);
  EXPECT_EQ(instance.debug_utils_enabled(), had_debug_utils);
}

// --- Device::requirements / Device::adopt (shared-device interop) -----------

TEST(DeviceRequirementsTest, ReflectConfig) {
  // A headless default config needs graphics + the core timeline/dynamic
  // features and no device extensions.
  vg::DeviceRequirements reqs = vg::Device::requirements(vg::DeviceConfig{});
  EXPECT_EQ(reqs.api_version, VK_API_VERSION_1_3);
  EXPECT_TRUE((reqs.queue_flags & VK_QUEUE_GRAPHICS_BIT) != 0);
  EXPECT_FALSE(reqs.needs_present);
  EXPECT_TRUE(reqs.timeline_semaphore);
  EXPECT_TRUE(reqs.dynamic_rendering);
  EXPECT_TRUE(reqs.device_extensions.empty());

  // needs_present pulls in VK_KHR_swapchain as a requirement.
  vg::DeviceConfig present_config;
  present_config.needs_present = true;
  vg::DeviceRequirements present = vg::Device::requirements(present_config);
  EXPECT_TRUE(present.needs_present);
  ASSERT_EQ(present.device_extensions.size(), 1u);
  EXPECT_STREQ(present.device_extensions.front(),
               VK_KHR_SWAPCHAIN_EXTENSION_NAME);
}

TEST(DeviceAdoptTest, RejectsNullHandles) {
  auto adopted = vg::Device::adopt(vg::AdoptedDevice{}, vg::DeviceConfig{});
  ASSERT_FALSE(adopted.ok());
  EXPECT_EQ(adopted.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(DeviceTest, AdoptBorrowsSharedDeviceWithoutOwningIt) {
  // Borrow the fixture's live device by its raw handles — the shared-VkDevice
  // interop case. A default config needs no device extensions, so no
  // enabled-extension declaration is required.
  vg::AdoptedDevice adopted;
  adopted.instance = instance_->handle();
  adopted.physical_device = physical_;
  adopted.device = device_->handle();
  adopted.graphics_family = device_->graphics_family();
  adopted.graphics_queue = device_->graphics_queue();

  {
    auto borrowed = vg::Device::adopt(adopted, vg::DeviceConfig{});
    ASSERT_TRUE(borrowed.ok()) << borrowed.status().message();
    EXPECT_FALSE(borrowed.value().owns_device());
    EXPECT_EQ(borrowed.value().handle(), device_->handle());
    // It owns its own command pool on the shared device, distinct from the
    // owner's.
    EXPECT_NE(borrowed.value().command_pool(), VK_NULL_HANDLE);
    EXPECT_NE(borrowed.value().command_pool(), device_->command_pool());
    // Fully usable: records + submits on the shared queue.
    vg::Status s = borrowed.value().submit_single_time([](VkCommandBuffer) {});
    EXPECT_TRUE(s.ok()) << s.message();
  }  // borrowed destructs here — it must NOT destroy the underlying VkDevice.

  // The owner's device is still valid: a second submit proves the adopted
  // wrapper left it intact (a double-free trips the sanitizer job; a
  // use-after-free would fail this submit).
  vg::Status after = device_->submit_single_time([](VkCommandBuffer) {});
  EXPECT_TRUE(after.ok()) << after.message();
}

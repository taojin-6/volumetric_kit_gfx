// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// gfx's side of the device: the requirements the renderer brings, a device
// made from them, and the shared-device seam. The instance, physical device
// and device are volumetric_kit_core's, whose own tests cover them.

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// The shared instance + physical-device + headless logical-device fixture,
// its device made from device_requirements().
using DeviceTest = VulkanDeviceTest;

// A device made to another library's requirements -- the core's defaults:
// Vulkan 1.2 on a compute queue, without dynamic rendering -- as an embedder
// might hand gfx one made for recon.
class ForeignDeviceTest : public VulkanDeviceTest {
 protected:
  vg::DeviceRequirements requirements() const override { return {}; }
};

// Borrow a live device on its queue, declaring what Device::create enabled
// for the renderer's requirements -- the shared-VkDevice interop shape.
vg::AdoptedDevice borrow_device(const vg::Instance& instance,
                                const vg::Device& device) {
  vg::AdoptedDevice adopted;
  adopted.instance = instance.handle();
  adopted.instance_api_version = instance.api_version();
  adopted.physical_device = device.physical_device();
  adopted.device = device.handle();
  adopted.queue_family = device.queue_family();
  adopted.queue = device.queue();
  adopted.submit_mutex = device.submit_mutex();
  adopted.enabled_features.timeline_semaphore = true;
  adopted.enabled_features.dynamic_rendering = true;
  adopted.enabled_debug_utils = instance.debug_utils_enabled();
  return adopted;
}

}  // namespace

// The renderer's floor: Vulkan 1.3 on a graphics queue, with dynamic rendering
// (every pass) and timeline semaphores (core::TimelineSemaphore). No present: a
// windowed caller asks for it.
TEST(DeviceRequirementsTest, TheRendererFloor) {
  const vg::DeviceRequirements reqs = vg::device_requirements();
  EXPECT_EQ(reqs.api_version, VK_API_VERSION_1_3);
  EXPECT_NE(reqs.queue_flags & VK_QUEUE_GRAPHICS_BIT, 0U);
  EXPECT_TRUE(reqs.dynamic_rendering);
  EXPECT_TRUE(reqs.timeline_semaphore);
  EXPECT_FALSE(reqs.needs_present);
  EXPECT_TRUE(reqs.extensions.empty());
}

// An embedder sharing one device with a compute library builds it from the
// union of both libraries' requirements.
TEST(DeviceRequirementsTest, MergesWithAComputeLibrarysRequirements) {
  vg::DeviceRequirements compute;
  compute.queue_flags = VK_QUEUE_COMPUTE_BIT;
  compute.scalar_block_layout = true;
  compute.extensions = {"VK_KHR_external_memory_fd"};
  vg::DeviceRequirements renderer = vg::device_requirements();
  renderer.needs_present = true;

  auto both = vg::merge(renderer, compute);
  ASSERT_TRUE(both.ok()) << both.status().message();
  EXPECT_EQ(both.value().api_version, VK_API_VERSION_1_3);
  EXPECT_EQ(both.value().queue_flags,
            VkQueueFlags{VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT});
  EXPECT_TRUE(both.value().needs_present);
  EXPECT_TRUE(both.value().dynamic_rendering);
  EXPECT_TRUE(both.value().scalar_block_layout);
  EXPECT_EQ(both.value().extensions.size(), 1U);
}

// A device made from the renderer's requirements has what every pass needs,
// on a graphics queue.
TEST_F(DeviceTest, CreatesADeviceTheRendererCanUse) {
  EXPECT_NE(device_->handle(), VK_NULL_HANDLE);
  EXPECT_NE(device_->queue(), VK_NULL_HANDLE);
  EXPECT_NE(device_->queue_flags() & VK_QUEUE_GRAPHICS_BIT, 0U);
  EXPECT_FALSE(device_->has_present());  // headless
  EXPECT_GE(device_->caps().api_version(), VK_API_VERSION_1_3);
  const vg::Status enabled = device_->check_enabled(vg::device_requirements());
  EXPECT_TRUE(enabled.ok()) << enabled.message();
}

TEST_F(DeviceTest, SingleTimeSubmitRoundTrips) {
  // No-op recording exercises allocate / begin / end / submit / fence-wait.
  const vg::Status status = device_->submit_single_time([](VkCommandBuffer) {});
  EXPECT_TRUE(status.ok()) << status.message();
}

// The shared-device seam: the renderer adopts a device another library made,
// held to its own requirements, and leaves it alive when it goes.
TEST_F(DeviceTest, AdoptBorrowsASharedDeviceWithoutOwningIt) {
  {
    auto borrowed = vg::Device::adopt(borrow_device(*instance_, *device_),
                                      vg::device_requirements());
    ASSERT_TRUE(borrowed.ok()) << borrowed.status().message();
    EXPECT_FALSE(borrowed.value().owns_device());
    EXPECT_EQ(borrowed.value().handle(), device_->handle());
    EXPECT_EQ(borrowed.value().queue(), device_->queue());
    // Fully usable: records + submits on the shared queue, under its lock.
    const vg::Status s =
        borrowed.value().submit_single_time([](VkCommandBuffer) {});
    EXPECT_TRUE(s.ok()) << s.message();
  }  // borrowed destructs here — it must NOT destroy the underlying VkDevice.

  // The owner's device is still valid: a second submit proves the adopted
  // wrapper left it intact (a double-free trips the sanitizer job; a
  // use-after-free would fail this submit).
  const vg::Status after = device_->submit_single_time([](VkCommandBuffer) {});
  EXPECT_TRUE(after.ok()) << after.message();
}

// A share that did not enable what the renderer needs is refused, naming it,
// rather than failing later inside a pass.
TEST_F(DeviceTest, AdoptRefusesAShareWithoutDynamicRendering) {
  vg::AdoptedDevice adopted = borrow_device(*instance_, *device_);
  adopted.enabled_features.dynamic_rendering = false;
  auto borrowed = vg::Device::adopt(adopted, vg::device_requirements());
  ASSERT_FALSE(borrowed.ok());
  EXPECT_EQ(borrowed.status().domain(), vg::Status::Code::Unsupported)
      << borrowed.status().message();
}

// gfx's entry points hold a device they did not make to the renderer's floor,
// rather than record barriers its queue may not support or passes it did not
// enable dynamic rendering for.
TEST_F(ForeignDeviceTest, RendererEntryPointsRefuseIt) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto batch = vg::UploadBatch::begin(*device_, allocator.value());
  ASSERT_FALSE(batch.ok());
  EXPECT_EQ(batch.status().domain(), vg::Status::Code::Unsupported)
      << batch.status().message();

  const std::uint32_t word = 0;
  vg::BufferUploadDesc desc;
  desc.data = &word;
  desc.size = sizeof(word);
  desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  auto buffer = vg::upload_buffer(*device_, allocator.value(), desc);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().domain(), vg::Status::Code::Unsupported);

  auto profiler = vg::Profiler::create(*device_);
  ASSERT_FALSE(profiler.ok());
  EXPECT_EQ(profiler.status().domain(), vg::Status::Code::Unsupported)
      << profiler.status().message();
}

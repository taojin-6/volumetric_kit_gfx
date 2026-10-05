// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// buffer_barrier: the public header stands alone (included first, before
// anything that could mask a missing include), and cmd_buffer_barrier orders a
// copy chain through a sub-range and the whole buffer.

#include "volumetric_kit/gfx/core/buffer_barrier.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// Adds a VMA allocator on top of the shared device fixture (mirrors
// ImageBarrierTest) to create the buffers the barriers order.
class BufferBarrierTest : public VulkanDeviceTest {
 protected:
  // Records real barriers, so run under the validation layer with teeth: a
  // range past the buffer or a stage the queue lacks fails the test (on CI,
  // where the layer is present), and so does a copy the barriers fail to
  // order, under synchronization validation.
  bool wants_validation() const override { return true; }
  bool wants_sync_validation() const override { return true; }

  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;  // no device, or the base SetUp failed fatally
    }
    auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

  vkc::Buffer make_buffer(VkBufferUsageFlags usage, vkc::MemoryUsage memory,
                          vkc::HostAccess access) {
    vkc::BufferDesc desc;
    desc.size = kBytes;
    desc.usage = usage;
    desc.memory = memory;
    desc.host_access = access;
    auto buffer = allocator_->create_buffer(desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
  }

  static constexpr VkDeviceSize kBytes = 64;
  std::optional<vkc::Allocator> allocator_;
};

}  // namespace

TEST_F(BufferBarrierTest, OrdersACopyChainThroughTheGpu) {
  // Staging -> device-only -> readback, each copy ordered after the last by a
  // barrier: the first over the half the copy wrote, the others over the whole
  // buffer through the offset/size defaults.
  vkc::Buffer upload =
      make_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vkc::MemoryUsage::Staging,
                  vkc::HostAccess::SequentialWrite);
  vkc::Buffer device_only = make_buffer(
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      vkc::MemoryUsage::DeviceOnly, vkc::HostAccess::SequentialWrite);
  vkc::Buffer readback =
      make_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, vkc::MemoryUsage::Staging,
                  vkc::HostAccess::Random);
  ASSERT_TRUE(upload.valid() && device_only.valid() && readback.valid());

  std::array<std::uint8_t, kBytes> src{};
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>(i * 5 + 3);
  }
  std::memcpy(upload.mapped(), src.data(), src.size());

  const VkBuffer up = upload.handle();
  const VkBuffer mid = device_only.handle();
  const VkBuffer down = readback.handle();
  auto recorded = device_->submit_single_time([&](VkCommandBuffer cmd) {
    // Two halves, the second ordered after the first's sub-range.
    VkBufferCopy first{0, 0, kBytes / 2};
    vkCmdCopyBuffer(cmd, up, mid, 1, &first);
    vg::BufferBarrierDesc after_first;
    after_first.buffer = mid;
    after_first.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    after_first.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    after_first.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    after_first.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
    after_first.offset = 0;
    after_first.size = kBytes / 2;
    vg::cmd_buffer_barrier(cmd, after_first);
    VkBufferCopy second{kBytes / 2, kBytes / 2, kBytes / 2};
    vkCmdCopyBuffer(cmd, up, mid, 1, &second);

    // The whole buffer, then out to the host.
    vg::BufferBarrierDesc to_read;
    to_read.buffer = mid;
    to_read.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    to_read.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    to_read.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_read.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
    vg::cmd_buffer_barrier(cmd, to_read);
    VkBufferCopy whole{0, 0, kBytes};
    vkCmdCopyBuffer(cmd, mid, down, 1, &whole);

    vg::BufferBarrierDesc to_host;
    to_host.buffer = down;
    to_host.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    to_host.dst_stage = VK_PIPELINE_STAGE_HOST_BIT;
    to_host.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_host.dst_access = VK_ACCESS_HOST_READ_BIT;
    vg::cmd_buffer_barrier(cmd, to_host);
  });
  ASSERT_TRUE(recorded.ok()) << recorded.message();

  const auto* got = static_cast<const std::uint8_t*>(readback.mapped());
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(std::memcmp(got, src.data(), src.size()), 0);
}

// A null buffer is a caller bug cmd_buffer_barrier VKC_CHECKs rather than
// handing to the driver. The check fires before any Vulkan call, so no device
// is needed; the "DeathTest" suffix runs it isolated.
TEST(BufferBarrierDeathTest, RejectsNullBuffer) {
  vg::BufferBarrierDesc desc;  // buffer defaults to VK_NULL_HANDLE
  EXPECT_DEATH(
      {
        vkc::set_log_handler({});  // route the abort message to stderr
        vg::cmd_buffer_barrier(VK_NULL_HANDLE, desc);
      },
      "buffer must be set");
}

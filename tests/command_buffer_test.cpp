// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/sync.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

using CommandTest = VulkanDeviceTest;

// A graphics-family pool. value() aborts only if the device is invalid, which
// the fixture has already asserted against.
vg::CommandPool make_pool(const vg::Device& device) {
  return std::move(
             vg::CommandPool::create(device.handle(), device.graphics_family()))
      .value();
}

}  // namespace

TEST_F(CommandTest, PoolCreatesOnGraphicsFamily) {
  auto pool = vg::CommandPool::create(device(), device_->graphics_family());
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  EXPECT_TRUE(pool.value().valid());
  EXPECT_NE(pool.value().handle(), VK_NULL_HANDLE);
  EXPECT_EQ(pool.value().queue_family(), device_->graphics_family());
}

TEST_F(CommandTest, AllocatePrimaryProducesValidBuffer) {
  vg::CommandPool pool = make_pool(*device_);
  auto cmd = pool.allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();
  EXPECT_TRUE(cmd.value().valid());
  EXPECT_NE(cmd.value().handle(), VK_NULL_HANDLE);
}

TEST_F(CommandTest, BeginEndRoundTrips) {
  vg::CommandPool pool = make_pool(*device_);
  auto cmd = pool.allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();
  EXPECT_TRUE(cmd.value().begin().ok());
  EXPECT_TRUE(cmd.value().end().ok());
}

// End-to-end: record vkCmdFillBuffer into a host-visible buffer, submit on the
// graphics queue, fence-wait, then read the result back through the mapping.
// Proves CommandPool + CommandBuffer + Fence + Allocator compose into a real
// CPU -> GPU -> CPU round trip.
TEST_F(CommandTest, RecordFillSubmitReadback) {
  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  desc.memory = vg::MemoryUsage::HostVisible;
  desc.mapped = true;
  auto buffer = allocator.value().create_buffer(desc);
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();

  vg::CommandPool pool = make_pool(*device_);
  auto cmd = pool.allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();

  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  vkCmdFillBuffer(cmd.value().handle(), buffer.value().handle(), 0,
                  VK_WHOLE_SIZE, 0xABABABABu);
  ASSERT_TRUE(cmd.value().end().ok());

  auto fence = vg::Fence::create(device());
  ASSERT_TRUE(fence.ok()) << fence.status().message();

  VkCommandBuffer raw = cmd.value().handle();
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &raw;
  ASSERT_EQ(vkQueueSubmit(device_->graphics_queue(), 1, &submit,
                          fence.value().handle()),
            VK_SUCCESS);
  ASSERT_TRUE(fence.value().wait().ok());

  const auto* bytes =
      static_cast<const unsigned char*>(buffer.value().mapped());
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(bytes[i], 0xABu) << "byte " << i;
  }
}

TEST_F(CommandTest, PoolMoveLeavesSourceEmpty) {
  vg::CommandPool source = make_pool(*device_);
  ASSERT_TRUE(source.valid());

  vg::CommandPool moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.queue_family(), 0u);  // metadata zeroed, not just the handle
}

TEST_F(CommandTest, PoolMoveAssignOverLiveLeavesSourceEmpty) {
  vg::CommandPool dst = make_pool(*device_);
  vg::CommandPool src = make_pool(*device_);
  dst = std::move(src);  // frees dst's original pool, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(CommandTest, PoolSelfMoveAssignIsSafe) {
  vg::CommandPool pool = make_pool(*device_);
  vg::CommandPool* alias = &pool;  // launder past -Wself-move under -Werror
  pool = std::move(*alias);
  EXPECT_TRUE(pool.valid());
}

TEST_F(CommandTest, BufferMoveLeavesSourceEmpty) {
  vg::CommandPool pool = make_pool(*device_);
  auto made = pool.allocate_primary();
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::CommandBuffer source = std::move(made).value();
  ASSERT_TRUE(source.valid());

  vg::CommandBuffer moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.handle(), VK_NULL_HANDLE);
}

TEST_F(CommandTest, BufferMoveAssignOverLiveLeavesSourceEmpty) {
  vg::CommandPool pool = make_pool(*device_);
  auto a = pool.allocate_primary();
  auto b = pool.allocate_primary();
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::CommandBuffer dst = std::move(a).value();
  vg::CommandBuffer src = std::move(b).value();

  dst = std::move(src);  // frees dst's original buffer, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(CommandTest, BufferSelfMoveAssignIsSafe) {
  vg::CommandPool pool = make_pool(*device_);
  auto made = pool.allocate_primary();
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::CommandBuffer cmd = std::move(made).value();

  vg::CommandBuffer* alias = &cmd;  // launder past -Wself-move under -Werror
  cmd = std::move(*alias);
  EXPECT_TRUE(cmd.valid());
}

// No device needed: a default-constructed CommandBuffer owns nothing.
TEST(CommandBufferTest, DefaultConstructedIsEmpty) {
  vg::CommandBuffer cmd;
  EXPECT_FALSE(cmd.valid());
  EXPECT_EQ(cmd.handle(), VK_NULL_HANDLE);
}

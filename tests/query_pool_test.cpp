// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/query_pool.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

using QueryPoolTest = VulkanDeviceTest;

// A two-query timestamp pool on the test device. value() aborts only if the
// device is invalid, which the fixture has already asserted against.
vg::QueryPool make_pool(VkDevice device, uint32_t count = 2) {
  return std::move(vg::QueryPool::create(device, count)).value();
}

// How many low bits of a timestamp the graphics queue family reports as valid;
// zero means timestamps carry no signal on this device, so the round-trip test
// must not assert on the tick values.
uint32_t graphics_timestamp_valid_bits(VkPhysicalDevice physical,
                                       uint32_t family) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
  if (family >= count) return 0;
  return families[family].timestampValidBits;
}

}  // namespace

// --- Validation (no device needed) ------------------------------------------

TEST(QueryPoolValidationTest, CreateRejectsNullDevice) {
  auto pool = vg::QueryPool::create(VK_NULL_HANDLE, /*query_count=*/2);
  EXPECT_FALSE(pool.ok());
  EXPECT_EQ(pool.status().domain(), vg::Status::Code::InvalidArgument);
}

// --- Creation + accessors (device) ------------------------------------------

TEST_F(QueryPoolTest, CreateProducesValidPoolWithRequestedCount) {
  auto pool = vg::QueryPool::create(device(), /*query_count=*/4);
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  EXPECT_TRUE(pool.value().valid());
  EXPECT_NE(pool.value().handle(), VK_NULL_HANDLE);
  EXPECT_EQ(pool.value().query_count(), 4u);
}

TEST_F(QueryPoolTest, CreateRejectsZeroQueryCountOnRealDevice) {
  // The zero-count guard fires before vkCreateQueryPool even on a live device.
  auto pool = vg::QueryPool::create(device(), /*query_count=*/0);
  EXPECT_FALSE(pool.ok());
  EXPECT_EQ(pool.status().domain(), vg::Status::Code::InvalidArgument);
}

// --- Move-only lifecycle ----------------------------------------------------

TEST_F(QueryPoolTest, MoveLeavesSourceEmpty) {
  vg::QueryPool source = make_pool(device());
  ASSERT_TRUE(source.valid());

  vg::QueryPool moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.query_count(), 2u);
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.query_count(), 0u);  // metadata zeroed, not just the handle
}

TEST_F(QueryPoolTest, MoveAssignOverLiveLeavesSourceEmpty) {
  vg::QueryPool dst = make_pool(device(), /*count=*/2);
  vg::QueryPool src = make_pool(device(), /*count=*/3);

  dst = std::move(src);  // frees dst's original pool, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_EQ(dst.query_count(), 3u);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(src.query_count(), 0u);
}

TEST_F(QueryPoolTest, SelfMoveAssignIsSafe) {
  vg::QueryPool pool = make_pool(device());
  vg::QueryPool* alias = &pool;  // launder past -Wself-move under -Werror
  pool = std::move(*alias);
  EXPECT_TRUE(pool.valid());
  EXPECT_EQ(pool.query_count(), 2u);
}

// A moved-from pool holds a null VkQueryPool AND a null VkDevice (the default
// constructor is private, so that is the only way a consumer reaches this
// state), so every entry point must refuse rather than hand those to the
// loader.
TEST_F(QueryPoolTest, MovedFromPoolOperationsFailCleanly) {
  vg::QueryPool source = make_pool(device(), /*count=*/2);
  vg::QueryPool sink(std::move(source));
  ASSERT_TRUE(sink.valid());

  vg::QueryPool& pool = source;  // NOLINT(bugprone-use-after-move)
  ASSERT_FALSE(pool.valid());
  ASSERT_EQ(pool.query_count(), 0u);

  // The recording calls return void; "does not crash" is the whole assertion.
  pool.cmd_reset(VK_NULL_HANDLE, 0, 1);
  pool.cmd_write_timestamp(VK_NULL_HANDLE, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           0);

  uint64_t ticks[2] = {0, 0};
  const vg::Status read = pool.read_results(0, 2, ticks);
  ASSERT_FALSE(read.ok());
  EXPECT_EQ(read.domain(), vg::Status::Code::InvalidArgument);
}

// Ranges outside the pool are rejected rather than passed through to Vulkan --
// including count > query_count(), where a naive `first > query_count_ - count`
// bound would underflow and let everything through.
TEST_F(QueryPoolTest, RejectsOutOfRangeRequests) {
  vg::QueryPool pool = make_pool(device(), /*count=*/2);
  uint64_t ticks[4] = {};

  auto rejected = [&](uint32_t first, uint32_t count) {
    const vg::Status s = pool.read_results(first, count, ticks);
    EXPECT_FALSE(s.ok()) << "first=" << first << " count=" << count;
    EXPECT_EQ(s.domain(), vg::Status::Code::InvalidArgument);
  };
  rejected(0, 3);  // count past the end
  rejected(2, 1);  // first past the end
  rejected(1, 2);  // straddles the end
  rejected(0, 0);  // empty request

  const vg::Status null_out = pool.read_results(0, 2, nullptr);
  EXPECT_FALSE(null_out.ok());
  EXPECT_EQ(null_out.domain(), vg::Status::Code::InvalidArgument);
}

// --- End-to-end: reset + two timestamps, submit, read back ------------------

TEST_F(QueryPoolTest, ResetWriteSubmitReadBack) {
  // Recording a timestamp on a queue family with timestampValidBits == 0 is
  // invalid (VUID-vkCmdWriteTimestamp-timestampValidBits-00829) and the values
  // carry no signal, so skip before recording anything; some devices (e.g.
  // certain MoltenVK configs) report zero.
  const uint32_t valid_bits = graphics_timestamp_valid_bits(
      device_->physical_device(), device_->graphics_family());
  if (valid_bits == 0) {
    GTEST_SKIP() << "graphics queue reports timestampValidBits == 0; ticks "
                    "carry no signal on this device";
  }

  vg::QueryPool pool = make_pool(device(), /*count=*/2);

  vg::CommandPool cmd_pool =
      std::move(vg::CommandPool::create(device(), device_->graphics_family()))
          .value();
  auto cmd = cmd_pool.allocate_primary();
  ASSERT_TRUE(cmd.ok()) << cmd.status().message();

  ASSERT_TRUE(
      cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
  // A timestamp query must be reset before it is written.
  pool.cmd_reset(cmd.value().handle(), /*first=*/0, /*count=*/2);
  pool.cmd_write_timestamp(cmd.value().handle(),
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, /*index=*/0);
  pool.cmd_write_timestamp(cmd.value().handle(),
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, /*index=*/1);
  ASSERT_TRUE(cmd.value().end().ok());

  ASSERT_NO_FATAL_FAILURE(submit_and_wait(cmd.value().handle()));

  uint64_t ticks[2] = {0, 0};
  vg::Status read = pool.read_results(/*first=*/0, /*count=*/2, ticks);
  ASSERT_TRUE(read.ok()) << read.message();

  // Mask to the meaningful bits before comparing, since the high bits beyond
  // validBits are undefined.
  const uint64_t mask =
      valid_bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << valid_bits) - 1);
  const uint64_t begin = ticks[0] & mask;
  const uint64_t end = ticks[1] & mask;
  // BOTTOM_OF_PIPE drains no earlier than TOP_OF_PIPE, so end >= begin
  // (allowing equality on a coarse-granularity clock).
  EXPECT_GE(end, begin);
}

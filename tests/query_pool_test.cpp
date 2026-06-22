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

TEST(QueryPoolValidationTest, CreateRejectsZeroQueryCount) {
  // A non-null device is still rejected up front when the count is zero — the
  // count check is independent of any Vulkan call.
  auto pool = vg::QueryPool::create(VK_NULL_HANDLE, /*query_count=*/0);
  EXPECT_FALSE(pool.ok());
  EXPECT_EQ(pool.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST(QueryPoolValidationTest, DefaultConstructedIsEmpty) {
  vg::QueryPool pool;
  EXPECT_FALSE(pool.valid());
  EXPECT_EQ(pool.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(pool.query_count(), 0u);
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

// --- End-to-end: reset + two timestamps, submit, read back ------------------

TEST_F(QueryPoolTest, ResetWriteSubmitReadBack) {
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

  submit_and_wait(cmd.value().handle());

  uint64_t ticks[2] = {0, 0};
  vg::Status read = pool.read_results(/*first=*/0, /*count=*/2, ticks);
  ASSERT_TRUE(read.ok()) << read.message();

  // The tick values are only meaningful where the graphics queue family reports
  // a non-zero timestampValidBits; elsewhere (validBits == 0) the driver still
  // returns availability but the values carry no signal, so asserting on them
  // would flake. Gate the value checks on that capability.
  const uint32_t valid_bits = graphics_timestamp_valid_bits(
      device_->physical_device(), device_->graphics_family());
  if (valid_bits == 0) {
    GTEST_SKIP() << "graphics queue reports timestampValidBits == 0; ticks "
                    "carry no signal on this device";
  }

  // Mask to the meaningful bits before comparing, since the high bits beyond
  // validBits are undefined.
  const uint64_t mask =
      valid_bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << valid_bits) - 1);
  const uint64_t begin = ticks[0] & mask;
  const uint64_t end = ticks[1] & mask;
  EXPECT_NE(begin, 0u);
  EXPECT_NE(end, 0u);
  // BOTTOM_OF_PIPE drains no earlier than TOP_OF_PIPE, so end >= begin
  // (allowing equality on a coarse-granularity clock).
  EXPECT_GE(end, begin);
}

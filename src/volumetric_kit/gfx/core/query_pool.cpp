// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/query_pool.hpp"

#include <utility>

namespace volumetric_kit::gfx {

Result<QueryPool> QueryPool::create(VkDevice device, uint32_t query_count,
                                    VkQueryType type) {
  // Reject the deterministic misuses up front, before touching Vulkan: a null
  // device has nothing to create against, and a zero-query pool violates
  // VUID-VkQueryPoolCreateInfo-queryCount-02763 (a crash with validation off).
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument("QueryPool::create: device is null");
  }
  if (query_count == 0) {
    return Status::invalid_argument("QueryPool::create: query_count is zero");
  }

  VkQueryPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  info.queryType = type;
  info.queryCount = query_count;

  VkQueryPool handle = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateQueryPool(device, &info, nullptr, &handle));

  QueryPool pool;
  pool.handle_ = UniqueHandle<VkQueryPool, vkDestroyQueryPool>(device, handle);
  pool.query_count_ = query_count;
  return pool;
}

QueryPool::QueryPool(QueryPool&& other) noexcept
    : handle_(std::move(other.handle_)), query_count_(other.query_count_) {
  other.query_count_ = 0;
}

QueryPool& QueryPool::operator=(QueryPool&& other) noexcept {
  if (this != &other) {
    handle_ = std::move(other.handle_);  // frees this pool's current handle
    query_count_ = other.query_count_;
    other.query_count_ = 0;
  }
  return *this;
}

void QueryPool::cmd_reset(VkCommandBuffer cmd, uint32_t first,
                          uint32_t count) const noexcept {
  vkCmdResetQueryPool(cmd, handle_.get(), first, count);
}

void QueryPool::cmd_write_timestamp(VkCommandBuffer cmd,
                                    VkPipelineStageFlagBits stage,
                                    uint32_t index) const noexcept {
  vkCmdWriteTimestamp(cmd, stage, handle_.get(), index);
}

Status QueryPool::read_results(uint32_t first, uint32_t count,
                               uint64_t* out) const {
  // No VK_QUERY_RESULT_WAIT_BIT: read after the submission has retired so an
  // unavailable result is a caller error, surfaced as VK_NOT_READY rather than
  // a blocking wait. VK_QUERY_RESULT_64_BIT matches the uint64_t destination.
  VkResult result =
      vkGetQueryPoolResults(handle_.device(), handle_.get(), first, count,
                            static_cast<size_t>(count) * sizeof(uint64_t), out,
                            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
  if (result != VK_SUCCESS) {
    // VK_NOT_READY surfaces here as a non-OK Status, distinguishable via
    // code(), not treated as a hard failure.
    return vk_error(result, "vkGetQueryPoolResults");
  }
  return Status{};
}

}  // namespace volumetric_kit::gfx

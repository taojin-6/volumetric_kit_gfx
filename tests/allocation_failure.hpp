// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

#include <cstddef>

namespace vg_test {

// Records the allocation which returned a known object, then targets exactly
// that size and caller on a retry, leaving driver allocations untouched.
struct AllocationSite {
  std::size_t size;
  const void* caller;
};
// Collect upload-owned allocation sites, excluding driver/layer frames. The
// test's staging wrapper excludes the allocator call explicitly: hidden static
// library frames do not reliably carry names in dladdr on every platform.
void record_upload_allocations() noexcept;
std::size_t finish_upload_allocations() noexcept;
AllocationSite upload_allocation_site(std::size_t index) noexcept;
void fail_upload_allocation_at(AllocationSite site) noexcept;

void record_allocations(std::size_t size) noexcept;
const void* allocation_site(const void* object) noexcept;
const void* sole_allocation_site() noexcept;
void fail_allocation_at(std::size_t size, const void* caller) noexcept;
bool stop_targeted_failure() noexcept;

// Temporarily exclude dependency work from tracing and failure injection,
// preserving the pending failure for the caller's next allocation. Nestable.
class AllocationExclusion {
 public:
  AllocationExclusion() noexcept;
  ~AllocationExclusion();
  AllocationExclusion(const AllocationExclusion&) = delete;
  AllocationExclusion& operator=(const AllocationExclusion&) = delete;
  AllocationExclusion(AllocationExclusion&&) = delete;
  AllocationExclusion& operator=(AllocationExclusion&&) = delete;
};

// Fail this thread's allocations after `allowed` successful ones. Arm only
// around host-only API work, never a driver call or a GoogleTest assertion.
class AllocationFailure {
 public:
  explicit AllocationFailure(std::size_t allowed = 0,
                             bool persistent = false) noexcept;
  ~AllocationFailure();
  AllocationFailure(const AllocationFailure&) = delete;
  AllocationFailure& operator=(const AllocationFailure&) = delete;
  AllocationFailure(AllocationFailure&&) = delete;
  AllocationFailure& operator=(AllocationFailure&&) = delete;

  // Disarm before assertions, and report whether an allocation failed.
  bool stop() noexcept;
  // Allocation attempts observed before stop(), including a failed one.
  std::size_t allocations() const noexcept { return allocations_; }

 private:
  bool stopped_ = false;
  bool triggered_ = false;
  std::size_t allocations_ = 0;
};

}  // namespace vg_test

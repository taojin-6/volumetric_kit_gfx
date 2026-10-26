// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Complete replacements live apart from the tests so optimizers and the
// sanitizers see matching new/delete families, including sized and aligned
// forms. Only the dedicated allocation-failure test binary links this file.

#include "allocation_failure.hpp"

#include <dlfcn.h>
#include <execinfo.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace {

struct RecordedAllocation {
  const void* object;
  const void* caller;
};
thread_local RecordedAllocation records[512];
thread_local std::size_t recorded = 0;
thread_local std::size_t target_size = 0;
thread_local const void* target_caller = nullptr;
thread_local bool recording = false;
thread_local bool record_upload = false;
thread_local bool target_upload = false;
thread_local vg_test::AllocationSite upload_sites[512];
thread_local std::size_t upload_count = 0;

bool upload_owned_allocation() noexcept {
  Dl_info executable{};
  if (!dladdr(
          reinterpret_cast<const void*>(&vg_test::record_upload_allocations),
          &executable))
    return false;
  void* frames[48];
  const int count = backtrace(frames, 48);
  for (int i = 0; i < count; ++i) {
    Dl_info info{};
    if (!dladdr(frames[i], &info)) continue;
    // These APIs/frames own driver and allocator implementation work, not
    // the upload's host-side staging owner and retirement bookkeeping.
    if (!info.dli_fname) return false;
    const bool own_module =
        std::strcmp(info.dli_fname, executable.dli_fname) == 0 ||
        std::strstr(info.dli_fname, "libgfx_") != nullptr;
    const bool standard_library =
        std::strstr(info.dli_fname, "libc++") != nullptr ||
        std::strstr(info.dli_fname, "libstdc++") != nullptr;
    if (!own_module && !standard_library) return false;
    if (!info.dli_sname) continue;
    if (std::strstr(info.dli_sname, "create_buffer")) return false;
    if (std::strstr(info.dli_sname, "record_image_upload")) return true;
  }
  return false;
}
thread_local bool target_triggered = false;
thread_local bool armed = false;
thread_local bool persistent_failure = false;
thread_local bool triggered = false;
thread_local std::size_t remaining = 0;
thread_local std::size_t attempts = 0;

void fail_if_armed() {
  if (!armed) return;
  ++attempts;
  if (remaining != 0) {
    --remaining;
    return;
  }
  triggered = true;
  armed = persistent_failure;
  throw std::bad_alloc();
}

void* allocate(std::size_t size) {
  fail_if_armed();
  if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc();
}

void* allocate_aligned(std::size_t size, std::align_val_t alignment) {
  fail_if_armed();
  const auto align = static_cast<std::size_t>(alignment);
  // Round up for aligned_alloc's multiple-of-alignment size contract.
  const std::size_t bytes = size == 0 ? align : size;
  if (bytes > std::numeric_limits<std::size_t>::max() - (align - 1)) {
    throw std::bad_alloc();
  }
  const std::size_t padded = ((bytes + align - 1) / align) * align;
  if (void* p = std::aligned_alloc(align, padded)) return p;
  throw std::bad_alloc();
}

}  // namespace

namespace vg_test {

void record_upload_allocations() noexcept {
  // Warm stack unwinding before collection.
  (void)upload_owned_allocation();
  upload_count = 0;
  record_upload = true;
}

std::size_t finish_upload_allocations() noexcept {
  record_upload = false;
  return upload_count;
}

AllocationSite upload_allocation_site(std::size_t index) noexcept {
  return upload_sites[index];
}

void fail_upload_allocation_at(AllocationSite site) noexcept {
  fail_allocation_at(site.size, site.caller);
  target_upload = true;
}

void record_allocations(std::size_t size) noexcept {
  target_size = size;
  recorded = 0;
  recording = true;
  target_caller = nullptr;
}

const void* allocation_site(const void* object) noexcept {
  recording = false;
  for (std::size_t i = 0; i < recorded; ++i) {
    if (records[i].object == object) return records[i].caller;
  }
  return nullptr;
}

const void* sole_allocation_site() noexcept {
  recording = false;
  return recorded == 1 ? records[0].caller : nullptr;
}

void fail_allocation_at(std::size_t size, const void* caller) noexcept {
  target_size = size;
  target_caller = caller;
  target_triggered = false;
  target_upload = false;
}

bool stop_targeted_failure() noexcept {
  target_caller = nullptr;
  return target_triggered;
}

AllocationFailure::AllocationFailure(std::size_t allowed,
                                     bool persistent) noexcept {
  remaining = allowed;
  persistent_failure = persistent;
  triggered = false;
  attempts = 0;
  armed = true;
}

AllocationFailure::~AllocationFailure() { stop(); }

bool AllocationFailure::stop() noexcept {
  if (!stopped_) {
    armed = false;
    triggered_ = triggered;
    allocations_ = attempts;
    stopped_ = true;
  }
  return triggered_;
}

}  // namespace vg_test

// The standard library declares these replaceable functions with reserved
// parameter names; application definitions use ordinary portable names.
// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
void* operator new(std::size_t size) {
  const void* caller = __builtin_return_address(0);
  if (size == target_size && target_caller != nullptr &&
      target_caller == caller &&
      (!target_upload || upload_owned_allocation())) {
    target_caller = nullptr;
    target_triggered = true;
    throw std::bad_alloc();
  }
  if (record_upload && upload_owned_allocation()) {
    bool seen = false;
    for (std::size_t i = 0; i < upload_count; ++i) {
      seen = seen ||
             (upload_sites[i].size == size && upload_sites[i].caller == caller);
    }
    if (!seen && upload_count < 512)
      upload_sites[upload_count++] = {size, caller};
  }
  void* object = allocate(size);
  if (recording && size == target_size && recorded < 512) {
    records[recorded++] = {object, caller};
  }
  return object;
}
void* operator new[](std::size_t size) { return allocate(size); }
void* operator new(std::size_t size, std::align_val_t alignment) {
  return allocate_aligned(size, alignment);
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return allocate_aligned(size, alignment);
}
void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t& /*tag*/) noexcept {
  try {
    return allocate_aligned(size, alignment);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t& /*tag*/) noexcept {
  try {
    return allocate_aligned(size, alignment);
  } catch (...) {
    return nullptr;
  }
}
// NOLINTEND(readability-inconsistent-declaration-parameter-name)
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t /*size*/) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t /*size*/) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t /*size*/,
                     std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t /*size*/,
                       std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
void operator delete(void* p, const std::nothrow_t& /*tag*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, const std::nothrow_t& /*tag*/) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t /*alignment*/,
                     const std::nothrow_t& /*tag*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::align_val_t /*alignment*/,
                       const std::nothrow_t& /*tag*/) noexcept {
  std::free(p);
}

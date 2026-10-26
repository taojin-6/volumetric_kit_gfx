// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "../src/volumetric_kit/gfx/core/upload_steps.hpp"
#include "allocation_failure.hpp"

namespace volumetric_kit::gfx::detail {

core::Result<core::Buffer> make_staging_without_injection(
    core::Allocator& allocator, const void* src, VkDeviceSize size) {
  const vg_test::AllocationExclusion exclusion;
  return make_staging(allocator, src, size);
}

}  // namespace volumetric_kit::gfx::detail

// Compile the production upload with just its dependency call excluded from
// allocation injection. Resolving a create_buffer stack-frame name is not a
// reliable boundary when the static core library hides its symbols. Including
// upload_steps above preserves the real helper; this changes only its call in
// the implementation below, leaving gfx's staging owner and retirement work
// subject to injection. Normal test suites use the unmodified library object.
#define make_staging make_staging_without_injection
#include "../src/volumetric_kit/gfx/core/image_update.cpp"
#undef make_staging

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// gfx's memory types are volumetric_kit_core's (DECISIONS.md, "Memory comes
// from volumetric_kit_core"), so the allocator -- its placements, budgets and
// validation -- is tested where it lives, in the core. What gfx adds is the
// names, which must be the core's types so a buffer or image passes to a
// sibling library unchanged, and the defaults its call sites lean on. gfx's
// own placements are tested with their users: UploadBatch in
// texture_upload_test, the per-frame uniforms in pbr_scene_test.

#include <gtest/gtest.h>

#include <type_traits>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"

namespace vg = volumetric_kit::gfx;
namespace vkc = volumetric_kit::core;

static_assert(std::is_same_v<vg::Allocator, vkc::Allocator>);
static_assert(std::is_same_v<vg::Buffer, vkc::Buffer>);
static_assert(std::is_same_v<vg::Image, vkc::Image>);
static_assert(std::is_same_v<vg::BufferDesc, vkc::BufferDesc>);
static_assert(std::is_same_v<vg::ImageDesc, vkc::ImageDesc>);
static_assert(std::is_same_v<vg::ImageInfo, vkc::ImageInfo>);
static_assert(std::is_same_v<vg::MemoryInfo, vkc::MemoryInfo>);
static_assert(std::is_same_v<vg::MemoryUsage, vkc::MemoryUsage>);
static_assert(std::is_same_v<vg::HostAccess, vkc::HostAccess>);
static_assert(std::is_same_v<vg::HeapStats, vkc::HeapStats>);
static_assert(std::is_same_v<vg::MemoryStats, vkc::MemoryStats>);

// A buffer is device-only unless it says otherwise -- vertex, index and
// uniform data gfx uploads take the default -- and a mapped one is written,
// not read, unless it asks for cached memory.
TEST(GfxMemory, BuffersDefaultToDeviceOnlyMemory) {
  const vg::BufferDesc buffer;
  EXPECT_EQ(buffer.memory, vg::MemoryUsage::DeviceOnly);
  EXPECT_EQ(buffer.host_access, vg::HostAccess::SequentialWrite);
  EXPECT_EQ(buffer.queue_families, nullptr);
  const vg::ImageDesc image;
  EXPECT_TRUE(image.with_view);
  EXPECT_EQ(image.type, VK_IMAGE_TYPE_2D);
}

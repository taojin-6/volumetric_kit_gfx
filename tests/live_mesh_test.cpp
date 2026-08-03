// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// LiveMesh: the value-type contract for the reconstruction live handoff --
// default-empty, and valid() gating on all three borrowed buffers being bound.
// The end-to-end proof that an indirect live draw matches the equivalent direct
// draw lives in hybrid_mesh_pipeline_test.cpp, where the render harness is.

#include <gtest/gtest.h>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/pipelines/live_mesh.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

// --- No device needed --------------------------------------------------------

TEST(LiveMeshTest, DefaultConstructedIsEmpty) {
  pipelines::LiveMesh live;
  EXPECT_FALSE(live.valid());
  EXPECT_EQ(live.vertices, VK_NULL_HANDLE);
  EXPECT_EQ(live.indices, VK_NULL_HANDLE);
  EXPECT_EQ(live.indirect, VK_NULL_HANDLE);
  EXPECT_EQ(live.vertex_offset, 0u);
  EXPECT_EQ(live.index_offset, 0u);
  EXPECT_EQ(live.indirect_offset, 0u);
}

// --- valid() gating on real buffers ------------------------------------------

using LiveMeshDeviceTest = VulkanDeviceTest;

TEST_F(LiveMeshDeviceTest, ValidRequiresAllThreeBuffers) {
  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto make = [&](VkBufferUsageFlags usage) {
    vg::BufferDesc desc;
    desc.size = 256;
    desc.usage = usage;
    desc.memory = vg::MemoryUsage::DeviceLocal;
    return allocator.value().create_buffer(desc);
  };
  auto vtx = make(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  auto idx = make(VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  auto ind = make(VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  ASSERT_TRUE(vtx.ok()) << vtx.status().message();
  ASSERT_TRUE(idx.ok()) << idx.status().message();
  ASSERT_TRUE(ind.ok()) << ind.status().message();

  pipelines::LiveMesh full;
  full.vertices = vtx.value().handle();
  full.indices = idx.value().handle();
  full.indirect = ind.value().handle();
  EXPECT_TRUE(full.valid());

  // Dropping any one borrowed buffer makes the mesh non-drawable -- submit()
  // then skips it rather than record an indirect draw against an unbound
  // buffer.
  pipelines::LiveMesh missing = full;
  missing.vertices = VK_NULL_HANDLE;
  EXPECT_FALSE(missing.valid());
  missing = full;
  missing.indices = VK_NULL_HANDLE;
  EXPECT_FALSE(missing.valid());
  missing = full;
  missing.indirect = VK_NULL_HANDLE;
  EXPECT_FALSE(missing.valid());
}

}  // namespace

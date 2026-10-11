// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// LiveMesh: the value-type contract for the reconstruction live handoff --
// default-empty, and valid() gating on all three borrowed buffers being bound.
// The end-to-end proof that an indirect live draw matches the equivalent direct
// draw lives in hybrid_mesh_pipeline_test.cpp, where the render harness is.

#include <gtest/gtest.h>

#include "volumetric_kit/gfx/pipelines/live_mesh.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;

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

TEST(LiveMeshTest, ValidRequiresAllThreeBuffers) {
  // valid() only compares borrowed handles with null; these sentinels never
  // reach Vulkan. VkBuffer can be a pointer or an integer, depending on the
  // ABI.
#if VK_USE_64_BIT_PTR_DEFINES
  const VkBuffer bound = reinterpret_cast<VkBuffer>(1);
#else
  const VkBuffer bound = 1;
#endif
  pipelines::LiveMesh full;
  full.vertices = bound;
  full.indices = bound;
  full.indirect = bound;
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

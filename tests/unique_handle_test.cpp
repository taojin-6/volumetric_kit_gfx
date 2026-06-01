// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Device-free coverage of UniqueHandle — the shared move/destroy core of the
// sync primitives and CommandPool. A counting destroy function stands in for a
// vkDestroy* entry point, so "the handle is freed exactly once across every
// ownership transfer" becomes a deterministic assertion (not just something the
// sanitizers job might catch).

#include <gtest/gtest.h>

#include <utility>

#include "volumetric_kit/gfx/core/unique_handle.hpp"

namespace {

namespace vg = volumetric_kit::gfx;

int g_destroyed = 0;

VKAPI_ATTR void VKAPI_CALL count_destroy(VkDevice, void* /*handle*/,
                                         const VkAllocationCallbacks*) {
  ++g_destroyed;
}

using Stub = vg::UniqueHandle<void*, count_destroy>;

// A non-null stub handle; its value is never used, only its non-nullness.
void* fake_handle() { return &g_destroyed; }

}  // namespace

TEST(UniqueHandle, DestroysOwnedHandleOnce) {
  g_destroyed = 0;
  {
    Stub h(VK_NULL_HANDLE, fake_handle());
    EXPECT_TRUE(h.valid());
    EXPECT_EQ(h.get(), fake_handle());
  }
  EXPECT_EQ(g_destroyed, 1);
}

TEST(UniqueHandle, DefaultConstructedDestroysNothing) {
  g_destroyed = 0;
  {
    Stub h;
    EXPECT_FALSE(h.valid());
  }
  EXPECT_EQ(g_destroyed, 0);
}

TEST(UniqueHandle, MoveConstructTransfersOwnershipAndDestroysOnce) {
  g_destroyed = 0;
  {
    Stub source(VK_NULL_HANDLE, fake_handle());
    Stub moved(std::move(source));
    EXPECT_TRUE(moved.valid());
    EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.get(), VK_NULL_HANDLE);
    EXPECT_EQ(source.device(), VK_NULL_HANDLE);  // device zeroed too
  }
  EXPECT_EQ(g_destroyed, 1);  // freed once, by `moved`
}

TEST(UniqueHandle, MoveAssignOverLiveDestroysOverwrittenOnce) {
  g_destroyed = 0;
  {
    Stub dst(VK_NULL_HANDLE, fake_handle());
    Stub src(VK_NULL_HANDLE, fake_handle());
    dst = std::move(src);
    EXPECT_EQ(g_destroyed, 1);  // dst's original handle freed during the assign
    EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  }
  // + the adopted handle on scope exit = 2; a double-free (un-nulled src) would
  // make this 3.
  EXPECT_EQ(g_destroyed, 2);
}

TEST(UniqueHandle, SelfMoveAssignKeepsHandle) {
  g_destroyed = 0;
  {
    Stub h(VK_NULL_HANDLE, fake_handle());
    Stub* alias = &h;  // launder past -Wself-move under -Werror
    h = std::move(*alias);
    EXPECT_TRUE(h.valid());
    EXPECT_EQ(g_destroyed, 0);  // self-move must not free
  }
  EXPECT_EQ(g_destroyed, 1);
}

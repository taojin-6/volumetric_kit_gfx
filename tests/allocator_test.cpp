// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

// Allocation needs a real device + VMA allocator; skip when the runner has
// none. instance_ -> device_ -> allocator_ declaration order means the
// allocator is destroyed before the device, and buffers (local to each test)
// before either.
class AllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto instance = vg::Instance::create(vg::InstanceConfig{});
    if (!instance.ok()) {
      GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
    }
    instance_.emplace(std::move(instance).value());

    auto physical = instance_->select_physical_device();
    if (!physical.ok()) {
      GTEST_SKIP() << "no Vulkan device: " << physical.status().message();
    }

    auto device = vg::Device::create(instance_->handle(), physical.value(),
                                     vg::DeviceConfig{});
    ASSERT_TRUE(device.ok()) << device.status().message();
    device_.emplace(std::move(device).value());

    auto allocator = vg::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

  std::optional<vg::Instance> instance_;
  std::optional<vg::Device> device_;
  std::optional<vg::Allocator> allocator_;
};

// A small host-visible, mapped buffer — the common fixture for the move tests.
vg::BufferDesc host_visible_mapped_desc() {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vg::MemoryUsage::HostVisible;
  desc.mapped = true;
  return desc;
}

}  // namespace

TEST_F(AllocatorTest, HostVisibleMappedBufferRoundTrips) {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vg::MemoryUsage::HostVisible;
  desc.mapped = true;

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();
  EXPECT_EQ(buffer.value().size(), 64u);
  ASSERT_NE(buffer.value().mapped(), nullptr);

  // Write a pattern through the persistent mapping and read it back.
  auto* bytes = static_cast<unsigned char*>(buffer.value().mapped());
  for (int i = 0; i < 64; ++i) {
    bytes[i] = static_cast<unsigned char>(i);
  }
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(bytes[i], static_cast<unsigned char>(i));
  }
}

TEST_F(AllocatorTest, DeviceLocalBufferIsValidAndUnmapped) {
  vg::BufferDesc desc;
  desc.size = 256;
  desc.usage =
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  desc.memory = vg::MemoryUsage::DeviceLocal;

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();
  EXPECT_TRUE(buffer.value().valid());
  EXPECT_NE(buffer.value().handle(), VK_NULL_HANDLE);
  EXPECT_EQ(buffer.value().mapped(), nullptr);
}

TEST_F(AllocatorTest, ExportableBufferReturnsNotSupported) {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.exportable = true;

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().code(), VK_ERROR_FEATURE_NOT_PRESENT);
}

TEST_F(AllocatorTest, BufferMoveLeavesSourceEmpty) {
  auto made = allocator_->create_buffer(host_visible_mapped_desc());
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::Buffer source = std::move(made).value();
  ASSERT_TRUE(source.valid());

  vg::Buffer moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(source.size(), 0u);
}

TEST_F(AllocatorTest, BufferMoveAssignOverLiveLeavesSourceEmpty) {
  auto a = allocator_->create_buffer(host_visible_mapped_desc());
  auto b = allocator_->create_buffer(host_visible_mapped_desc());
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Buffer dst = std::move(a).value();
  vg::Buffer src = std::move(b).value();

  dst = std::move(src);  // runs dst's deleter once, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(AllocatorTest, BufferSelfMoveAssignIsSafe) {
  auto made = allocator_->create_buffer(host_visible_mapped_desc());
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::Buffer buffer = std::move(made).value();

  // Pointer-laundered self-move (dodges -Wself-move under -Werror); the
  // this != &other guard must keep the buffer intact and not run its deleter.
  vg::Buffer* alias = &buffer;
  buffer = std::move(*alias);
  EXPECT_TRUE(buffer.valid());
}

TEST_F(AllocatorTest, AllocatorSelfMoveAssignIsSafe) {
  vg::Allocator* alias = &*allocator_;
  *allocator_ = std::move(*alias);
  // Still usable after a self-move (no double-free of the VmaAllocator).
  auto buffer = allocator_->create_buffer(host_visible_mapped_desc());
  EXPECT_TRUE(buffer.ok()) << buffer.status().message();
}

TEST_F(AllocatorTest, AutoMemoryBufferIsValid) {
  vg::BufferDesc desc;
  desc.size = 128;
  desc.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  desc.memory = vg::MemoryUsage::Auto;

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();
  EXPECT_TRUE(buffer.value().valid());
}

TEST_F(AllocatorTest, MoveAssignOverLiveAllocatorStaysUsable) {
  // Exercises Allocator::operator=(Allocator&&) onto an already-live allocator:
  // the overwritten allocator must be released (no double-free / crash) and the
  // target must remain usable. A leak here would surface under validation
  // layers.
  auto other = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(other.ok()) << other.status().message();
  *allocator_ = std::move(other).value();

  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vg::MemoryUsage::HostVisible;
  desc.mapped = true;

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();
  EXPECT_TRUE(buffer.value().valid());
}

TEST_F(AllocatorTest, ZeroSizeBufferIsRejected) {
  vg::BufferDesc desc;
  desc.size = 0;  // invalid per the Vulkan spec — reject before touching VMA
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().code(), VK_ERROR_INITIALIZATION_FAILED);
}

TEST_F(AllocatorTest, ZeroUsageBufferIsRejected) {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = 0;  // a buffer with no usage flags is invalid — reject it

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().code(), VK_ERROR_INITIALIZATION_FAILED);
}

TEST_F(AllocatorTest, DeviceLocalMappedIsRejected) {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vg::MemoryUsage::DeviceLocal;
  desc.mapped =
      true;  // contradicts DeviceLocal — must be rejected, not demoted

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().code(), VK_ERROR_INITIALIZATION_FAILED);
}

// No device needed: a default-constructed Buffer owns nothing.
TEST(BufferTest, DefaultConstructedIsEmpty) {
  vg::Buffer buffer;
  EXPECT_FALSE(buffer.valid());
  EXPECT_EQ(buffer.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(buffer.size(), 0u);
  EXPECT_EQ(buffer.mapped(), nullptr);
}

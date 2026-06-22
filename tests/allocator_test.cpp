// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// Adds a VMA allocator on top of the shared device fixture. The derived
// allocator_ is destroyed before the base's device_/instance_ (derived members
// first), and buffers (local to each test) before any of them.
class AllocatorTest : public VulkanDeviceTest {
 protected:
  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (IsSkipped()) {
      return;  // no Vulkan device; the base already skipped
    }
    auto allocator = vg::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

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

TEST_F(AllocatorTest, ExternalMemoryBufferUnsupported) {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.external = vg::ExternalHandleType::OpaqueFd;

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().domain(), vg::Status::Code::Unsupported);
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
  // A non-null moved-from mapped() would dangle, aliasing `moved`'s host
  // memory.
  EXPECT_EQ(source.mapped(), nullptr);
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
  EXPECT_EQ(src.mapped(), nullptr);
  EXPECT_NE(dst.mapped(), nullptr);  // dst adopted src's live mapping
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

TEST_F(AllocatorTest, MoveConstructStaysUsable) {
  // Move-construct from the fixture's allocator; the moved-to instance owns the
  // VmaAllocator (transferred, not double-freed) and still produces buffers.
  vg::Allocator moved(std::move(*allocator_));
  auto buffer = moved.create_buffer(host_visible_mapped_desc());
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
  EXPECT_EQ(buffer.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, ZeroUsageBufferIsRejected) {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = 0;  // a buffer with no usage flags is invalid — reject it

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().domain(), vg::Status::Code::InvalidArgument);
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
  EXPECT_EQ(buffer.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, HostVisibleUnmappedBufferIsRejected) {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vg::MemoryUsage::HostVisible;
  desc.mapped = false;  // unreachable from the host without a mapping — reject

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_FALSE(buffer.ok());
  EXPECT_EQ(buffer.status().domain(), vg::Status::Code::InvalidArgument);
}

// No device needed: a default-constructed Buffer owns nothing.
TEST(BufferTest, DefaultConstructedIsEmpty) {
  vg::Buffer buffer;
  EXPECT_FALSE(buffer.valid());
  EXPECT_EQ(buffer.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(buffer.size(), 0u);
  EXPECT_EQ(buffer.mapped(), nullptr);
}

// No device needed: counting deleters via the public adopt-ctor make the
// move-assign double-free/leak path a deterministic assertion (not just
// something the sanitizers job catches).
TEST(BufferTest, MoveAssignRunsOverwrittenDeleterExactlyOnce) {
  int dst_runs = 0;
  int src_runs = 0;
  {
    vg::Buffer dst(VK_NULL_HANDLE, 0, nullptr, [&dst_runs]() { ++dst_runs; });
    vg::Buffer src(VK_NULL_HANDLE, 0, nullptr, [&src_runs]() { ++src_runs; });
    dst = std::move(src);
    EXPECT_EQ(dst_runs,
              1);  // dst's original deleter ran once, during the assign
    EXPECT_EQ(src_runs, 0);  // src's deleter was adopted, not run
  }
  // dst (now holding src's deleter) is destroyed -> src_runs == 1; the
  // moved-from src runs nothing. A failure to null the moved-from deleter would
  // make this 2.
  EXPECT_EQ(dst_runs, 1);
  EXPECT_EQ(src_runs, 1);
}

TEST_F(AllocatorTest, ColorImageHasImageAndView) {
  vg::TextureDesc desc;
  desc.extent = {64, 64};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  auto texture = allocator_->create_image(desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_TRUE(texture.value().valid());
  EXPECT_NE(texture.value().image(), VK_NULL_HANDLE);
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);
  EXPECT_EQ(texture.value().extent().width, 64u);
  EXPECT_EQ(texture.value().extent().height, 64u);
  EXPECT_EQ(texture.value().format(), VK_FORMAT_R8G8B8A8_UNORM);
}

TEST_F(AllocatorTest, DepthImageGetsDepthAspectView) {
  vg::TextureDesc desc;
  desc.extent = {32, 32};
  desc.format = VK_FORMAT_D32_SFLOAT;
  desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  // A depth format must yield a view; the wrong aspect would fail view
  // creation.
  auto texture = allocator_->create_image(desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);
}

TEST_F(AllocatorTest, ExternalMemoryImageUnsupported) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  desc.external = vg::ExternalHandleType::OpaqueFd;

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::Unsupported);
}

TEST_F(AllocatorTest, ZeroExtentImageIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {0, 0};  // invalid — reject before touching VMA
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, ZeroUsageImageIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = 0;  // no usage flags — reject before touching VMA

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, UndefinedFormatImageIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_UNDEFINED;  // invalid — reject before touching VMA
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, HostVisibleImageIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  desc.memory = vg::MemoryUsage::HostVisible;  // images have no host accessor

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, CombinedDepthStencilImageGetsDepthOnlyView) {
  vg::TextureDesc desc;
  desc.extent = {32, 32};
  desc.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
  desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  auto texture = allocator_->create_image(desc);
  if (!texture.ok()) {
    GTEST_SKIP() << "combined depth/stencil format unsupported: "
                 << texture.status().message();
  }
  // A single VkImageView may not mix depth and stencil aspects; the DEPTH-only
  // default makes a valid, immediately-bindable view.
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);
}

TEST_F(AllocatorTest, Volume3DImageGets3DView) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.depth = 8;
  desc.type = VK_IMAGE_TYPE_3D;
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  auto texture = allocator_->create_image(desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_NE(texture.value().image(), VK_NULL_HANDLE);
  EXPECT_NE(texture.value().view(), VK_NULL_HANDLE);  // 3D view
  EXPECT_EQ(texture.value().extent().width, 16u);
  EXPECT_EQ(texture.value().depth(), 8u);  // slice count recoverable
}

TEST_F(AllocatorTest, MippedArrayImageIsValid) {
  vg::TextureDesc desc;
  desc.extent = {64, 64};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  desc.mip_levels = 4;
  desc.array_layers = 3;

  auto texture = allocator_->create_image(desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_NE(texture.value().view(),
            VK_NULL_HANDLE);  // 2D-array view over 4 mips
  EXPECT_EQ(texture.value().mip_levels(),
            4u);  // threaded through from the desc
}

TEST_F(AllocatorTest, DepthWithout3DTypeIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.depth = 4;  // depth > 1 but type is the default 2D
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, Arrayed3DImageIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.depth = 4;
  desc.type = VK_IMAGE_TYPE_3D;
  desc.array_layers = 2;  // 3D images cannot be arrayed
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, ZeroMipLevelsImageIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  desc.mip_levels = 0;

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, MultisampledMippedImageIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  desc.samples = VK_SAMPLE_COUNT_4_BIT;
  desc.mip_levels = 2;  // MSAA images must be single-mip — contradiction

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, ViewlessTransferImageHasNoView) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  desc.with_view = false;  // transfer-only image: no view-compatible usage

  auto texture = allocator_->create_image(desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();
  EXPECT_NE(texture.value().image(), VK_NULL_HANDLE);
  EXPECT_EQ(texture.value().view(), VK_NULL_HANDLE);  // viewless
}

TEST_F(AllocatorTest, ViewIncompatibleUsageWithViewIsRejected) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // names no view-compatible bit
  // desc.with_view defaults to true → contradiction.

  auto texture = allocator_->create_image(desc);
  ASSERT_FALSE(texture.ok());
  EXPECT_EQ(texture.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(AllocatorTest, SequentialWriteMappedBufferRoundTrips) {
  vg::BufferDesc desc;
  desc.size = 64;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vg::MemoryUsage::HostVisible;
  desc.mapped = true;
  desc.host_access = vg::HostAccess::SequentialWrite;

  auto buffer = allocator_->create_buffer(desc);
  ASSERT_TRUE(buffer.ok()) << buffer.status().message();
  ASSERT_NE(buffer.value().mapped(), nullptr);
  // Write-combined memory is write-only from the CPU; exercise sequential
  // writes (reading it back would be undefined for SequentialWrite).
  auto* bytes = static_cast<unsigned char*>(buffer.value().mapped());
  for (int i = 0; i < 64; ++i) {
    bytes[i] = static_cast<unsigned char>(i);
  }
}

TEST_F(AllocatorTest, TextureMoveLeavesSourceEmpty) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  auto made = allocator_->create_image(desc);
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::Texture source = std::move(made).value();
  ASSERT_TRUE(source.valid());

  vg::Texture moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.image(), VK_NULL_HANDLE);
  // A stale moved-from view() would name a VkImageView the deleter already
  // freed.
  EXPECT_EQ(source.view(), VK_NULL_HANDLE);
  EXPECT_EQ(source.extent().width, 0u);
  EXPECT_EQ(source.depth(), 1u);
  EXPECT_EQ(source.mip_levels(), 1u);
  EXPECT_EQ(source.format(), VK_FORMAT_UNDEFINED);
}

TEST_F(AllocatorTest, TextureMoveAssignOverLiveLeavesSourceEmpty) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  auto a = allocator_->create_image(desc);
  auto b = allocator_->create_image(desc);
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  vg::Texture dst = std::move(a).value();
  vg::Texture src = std::move(b).value();

  dst = std::move(src);  // runs dst's deleter once, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(src.view(), VK_NULL_HANDLE);
}

TEST_F(AllocatorTest, TextureSelfMoveAssignIsSafe) {
  vg::TextureDesc desc;
  desc.extent = {16, 16};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

  auto made = allocator_->create_image(desc);
  ASSERT_TRUE(made.ok()) << made.status().message();
  vg::Texture texture = std::move(made).value();

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must keep the texture intact and not run its deleter.
  vg::Texture* alias = &texture;
  texture = std::move(*alias);
  EXPECT_TRUE(texture.valid());
}

// No device needed: a default-constructed Texture owns nothing.
TEST(TextureTest, DefaultConstructedIsEmpty) {
  vg::Texture texture;
  EXPECT_FALSE(texture.valid());
  EXPECT_EQ(texture.image(), VK_NULL_HANDLE);
  EXPECT_EQ(texture.view(), VK_NULL_HANDLE);
  EXPECT_EQ(texture.extent().width, 0u);
  EXPECT_EQ(texture.depth(), 1u);
  EXPECT_EQ(texture.mip_levels(), 1u);
  EXPECT_EQ(texture.format(), VK_FORMAT_UNDEFINED);
}

// No device needed: counting deleters via the public adopt-ctor makes Texture's
// move-assign double-free/leak path a deterministic assertion (mirrors the
// BufferTest counting test), not just something the sanitizers job catches.
TEST(TextureTest, MoveAssignRunsOverwrittenDeleterExactlyOnce) {
  int dst_runs = 0;
  int src_runs = 0;
  {
    vg::Texture dst(VK_NULL_HANDLE, VK_NULL_HANDLE, {}, 1, 1,
                    VK_FORMAT_UNDEFINED, [&dst_runs]() { ++dst_runs; });
    vg::Texture src(VK_NULL_HANDLE, VK_NULL_HANDLE, {}, 1, 1,
                    VK_FORMAT_UNDEFINED, [&src_runs]() { ++src_runs; });
    dst = std::move(src);
    EXPECT_EQ(dst_runs,
              1);  // dst's original deleter ran once, during the assign
    EXPECT_EQ(src_runs, 0);  // src's deleter was adopted, not run
  }
  // A failure to null the moved-from deleter would make dst_runs/src_runs == 2.
  EXPECT_EQ(dst_runs, 1);
  EXPECT_EQ(src_runs, 1);
}

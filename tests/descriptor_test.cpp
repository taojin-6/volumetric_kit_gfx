// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// DescriptorSetLayout / DescriptorPool: argument validation and the move-only
// lifecycle (each owns a Vulkan handle). End-to-end allocation, a uniform
// write, and bind are covered by the MVP draw in graphics_pipeline_test.

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "vulkan_test_fixture.hpp"

namespace vg = volumetric_kit::gfx;

namespace {

using DescriptorDeviceTest = VulkanDeviceTest;

VkDescriptorSetLayoutBinding uniform_binding(uint32_t binding) {
  VkDescriptorSetLayoutBinding b{};
  b.binding = binding;
  b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  b.descriptorCount = 1;
  b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  return b;
}

VkDescriptorSetLayoutBinding sampler_binding(uint32_t binding) {
  VkDescriptorSetLayoutBinding b{};
  b.binding = binding;
  b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  b.descriptorCount = 1;
  b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  return b;
}

// Adds a VMA allocator (for a sampled image to bind) on top of the device
// fixture; skips with the base when no Vulkan device is present.
class DescriptorImageTest : public VulkanDeviceTest {
 protected:
  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (IsSkipped()) {
      return;
    }
    auto allocator = vg::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

  std::optional<vg::Allocator> allocator_;
};

}  // namespace

// --- Empty defaults + validation (checked before the Vulkan call) ------------

TEST(DescriptorTest, DefaultsAreEmpty) {
  EXPECT_FALSE(vg::DescriptorSetLayout().valid());
  EXPECT_EQ(vg::DescriptorSetLayout().handle(), VK_NULL_HANDLE);
  EXPECT_FALSE(vg::DescriptorPool().valid());
  EXPECT_EQ(vg::DescriptorPool().handle(), VK_NULL_HANDLE);
  EXPECT_FALSE(vg::DescriptorSet().valid());
  EXPECT_EQ(vg::DescriptorSet().handle(), VK_NULL_HANDLE);
}

TEST(DescriptorTest, LayoutNullBindingsWithCountRejected) {
  auto layout = vg::DescriptorSetLayout::create(VK_NULL_HANDLE, nullptr, 1);
  ASSERT_FALSE(layout.ok());
  EXPECT_EQ(layout.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST(DescriptorTest, PoolNullSizesRejected) {
  auto pool = vg::DescriptorPool::create(VK_NULL_HANDLE, nullptr, 0, 1);
  ASSERT_FALSE(pool.ok());
  EXPECT_EQ(pool.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST(DescriptorTest, PoolZeroMaxSetsRejected) {
  const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
  auto pool = vg::DescriptorPool::create(VK_NULL_HANDLE, &size, 1, 0);
  ASSERT_FALSE(pool.ok());
  EXPECT_EQ(pool.status().domain(), vg::Status::Code::InvalidArgument);
}

// --- Real creation + move-only lifecycle: needs a device ---------------------

TEST_F(DescriptorDeviceTest, LayoutCreateAndMove) {
  const VkDescriptorSetLayoutBinding b = uniform_binding(0);
  auto created = vg::DescriptorSetLayout::create(device(), &b, 1);
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::DescriptorSetLayout layout = std::move(created).value();
  ASSERT_TRUE(layout.valid());

  vg::DescriptorSetLayout moved(std::move(layout));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(layout.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(layout.handle(), VK_NULL_HANDLE);

  // Move-assign over a live layout: frees the destination, then adopts.
  auto other = vg::DescriptorSetLayout::create(device(), &b, 1);
  ASSERT_TRUE(other.ok());
  vg::DescriptorSetLayout dst = std::move(other).value();
  dst = std::move(moved);
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(moved.valid());  // NOLINT(bugprone-use-after-move)

  // Self-move (pointer-laundered to dodge -Wself-move) keeps it intact.
  vg::DescriptorSetLayout* alias = &dst;
  dst = std::move(*alias);
  EXPECT_TRUE(dst.valid());
}

TEST_F(DescriptorDeviceTest, EmptyLayoutForResourcelessSet) {
  // A set with no bindings is valid -- it is how the pipeline fills a gap
  // between used set indices.
  auto layout = vg::DescriptorSetLayout::create(device(), nullptr, 0);
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_TRUE(layout.value().valid());
}

TEST_F(DescriptorDeviceTest, PoolCreateAllocateAndMove) {
  const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2};
  auto created = vg::DescriptorPool::create(device(), &size, 1, 2);
  ASSERT_TRUE(created.ok()) << created.status().message();
  vg::DescriptorPool pool = std::move(created).value();
  ASSERT_TRUE(pool.valid());

  const VkDescriptorSetLayoutBinding b = uniform_binding(0);
  auto layout = vg::DescriptorSetLayout::create(device(), &b, 1);
  ASSERT_TRUE(layout.ok());
  auto set = pool.allocate(layout.value().handle());
  ASSERT_TRUE(set.ok()) << set.status().message();
  EXPECT_TRUE(set.value().valid());

  vg::DescriptorPool moved(std::move(pool));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(pool.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(pool.handle(), VK_NULL_HANDLE);

  auto other = vg::DescriptorPool::create(device(), &size, 1, 2);
  ASSERT_TRUE(other.ok());
  vg::DescriptorPool dst = std::move(other).value();
  dst = std::move(moved);  // free dst's pool, then adopt moved's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(moved.valid());  // NOLINT(bugprone-use-after-move)

  vg::DescriptorPool* alias = &dst;
  dst = std::move(*alias);  // self-move
  EXPECT_TRUE(dst.valid());
}

TEST_F(DescriptorImageTest, CombinedImageSamplerWriteSucceeds) {
  // Bind a sampled image + sampler into a set: exercises the
  // write_combined_image_sampler path end to end. A malformed write is caught
  // by the validation layers under the sanitizer job; here we assert the set
  // stays valid through the write.
  vg::TextureDesc tex_desc;
  tex_desc.extent = {4, 4};
  tex_desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  tex_desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  auto texture = allocator_->create_image(tex_desc);
  ASSERT_TRUE(texture.ok()) << texture.status().message();

  auto sampler = vg::Sampler::create(device());
  ASSERT_TRUE(sampler.ok()) << sampler.status().message();

  const VkDescriptorSetLayoutBinding b = sampler_binding(0);
  auto layout = vg::DescriptorSetLayout::create(device(), &b, 1);
  ASSERT_TRUE(layout.ok()) << layout.status().message();

  const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  auto pool = vg::DescriptorPool::create(device(), &size, 1, 1);
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto set = pool.value().allocate(layout.value().handle());
  ASSERT_TRUE(set.ok()) << set.status().message();

  set.value().write_combined_image_sampler(
      0, texture.value().view(), sampler.value().handle(),
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  EXPECT_TRUE(set.value().valid());
}

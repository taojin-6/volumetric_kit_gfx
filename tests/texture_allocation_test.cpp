// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "../src/volumetric_kit/gfx/pipelines/ycbcr_check.hpp"
#include "allocation_failure.hpp"
#include "gfx_test_support.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/gfx/core/image_update.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_set.hpp"

namespace pipelines = volumetric_kit::gfx::pipelines;

TEST(YcbcrCodingAllocationTest, ValidCodingNeedsNoHostAllocation) {
  for (auto siting :
       {pipelines::ChromaSiting::Center, pipelines::ChromaSiting::Left}) {
    pipelines::YcbcrCoding coding;
    coding.chroma_siting = siting;
    // The successful NV12 conversion check must not allocate its call name.
    // This host-only check keeps injection out of the driver and layer.
    vg_test::AllocationFailure failure(0, true);
    const vkc::Status checked = pipelines::detail::check_ycbcr_coding(
        "ImagePipeline::record_convert", coding);
    const bool failed = failure.stop();
    EXPECT_FALSE(failed);
    EXPECT_TRUE(checked.ok()) << checked.message();
  }
}

class TextureAllocationTest : public vg_test::RendererDeviceTest {
 protected:
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::On;
  }
};

TEST_F(TextureAllocationTest,
       InvalidArgumentsContainPersistentDiagnosticFailure) {
  const pipelines::HybridMeshPipeline pipeline;
  const pipelines::YcbcrPicture no_picture{};
  vkc::Allocator& alloc = allocator();
  {
    vg_test::AllocationFailure failure(0, true);
    const auto result = pipelines::HybridMeshTextureSet::create(
        device(), alloc, pipeline, VK_NULL_HANDLE);
    const bool failed = failure.stop();
    EXPECT_TRUE(failed);
    EXPECT_EQ(result.status().domain(), vkc::Status::Code::OutOfMemory);
  }
  {
    vg_test::AllocationFailure failure(0, true);
    const auto result = pipelines::HybridMeshTextureSet::create(
        device(), alloc, pipeline, no_picture);
    const bool failed = failure.stop();
    EXPECT_TRUE(failed);
    EXPECT_EQ(result.status().domain(), vkc::Status::Code::OutOfMemory);
  }
  {
    vg_test::AllocationFailure failure(0, true);
    const auto result = vg::Sampler::create(VK_NULL_HANDLE, {});
    const bool failed = failure.stop();
    EXPECT_TRUE(failed);
    EXPECT_EQ(result.status().domain(), vkc::Status::Code::OutOfMemory);
  }
  EXPECT_EQ(pipelines::HybridMeshTextureSet::create(device(), alloc, pipeline,
                                                    VK_NULL_HANDLE)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
  EXPECT_EQ(pipelines::HybridMeshTextureSet::create(device(), alloc, pipeline,
                                                    no_picture)
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);
  EXPECT_EQ(vg::Sampler::create(VK_NULL_HANDLE, {}).status().domain(),
            vkc::Status::Code::InvalidArgument);
}

TEST_F(TextureAllocationTest, WrapperAllocationFailureCleansUpAndAllowsRetry) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
  layout.color_count = 1;
  layout.depth_format = VK_FORMAT_D32_SFLOAT;
  auto pipeline =
      pipelines::HybridMeshPipeline::create(device(), allocator(), layout);
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
  vkc::ImageDesc image_desc;
  image_desc.extent = {1, 1};
  image_desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  image_desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  auto image = allocator().create_image(image_desc);
  ASSERT_TRUE(image.ok()) << image.status().message();

  // Discover the wrapper's allocation by its resulting pointer. Its (size,
  // caller) identifies this library allocation even when validation layers
  // and the driver allocate objects of the same size.
  vg_test::record_allocations(sizeof(pipelines::HybridMeshTextureSet));
  auto original = pipelines::HybridMeshTextureSet::create(
      device(), allocator(), pipeline.value(), image.value().view());
  ASSERT_TRUE(original.ok()) << original.status().message();
  const void* site = vg_test::allocation_site(original.value().get());
  ASSERT_NE(site, nullptr);
  original.value().reset();

  vg_test::fail_allocation_at(sizeof(pipelines::HybridMeshTextureSet), site);
  const auto refused = pipelines::HybridMeshTextureSet::create(
      device(), allocator(), pipeline.value(), image.value().view());
  const bool failed = vg_test::stop_targeted_failure();
  ASSERT_TRUE(failed);
  EXPECT_EQ(refused.status().domain(), vkc::Status::Code::OutOfMemory);

  auto retried = pipelines::HybridMeshTextureSet::create(
      device(), allocator(), pipeline.value(), image.value().view());
  ASSERT_TRUE(retried.ok()) << retried.status().message();
  EXPECT_NE(retried.value()->set(), VK_NULL_HANDLE);
  retried.value()->rebind(image.value().view());

  // A picture's refusal, too, reports a failed diagnostic as OutOfMemory.
  const pipelines::YcbcrPicture no_picture{};
  vkc::Status rebound;
  {
    vg_test::AllocationFailure failure(0, true);
    rebound = retried.value()->rebind(no_picture);
    const bool rebind_failed = failure.stop();
    EXPECT_TRUE(rebind_failed);
  }
  EXPECT_EQ(rebound.domain(), vkc::Status::Code::OutOfMemory);
  EXPECT_EQ(retried.value()->rebind(no_picture).domain(),
            vkc::Status::Code::InvalidArgument);
}

TEST_F(TextureAllocationTest,
       UploadEnrollmentFailureRecordsNothingAndAllowsRetry) {
  auto timeline = vkc::TimelineSemaphore::create(device());
  ASSERT_TRUE(timeline.ok());
  vg::RetireQueue counted(timeline.value());
  vg_test::record_allocations(sizeof(vg::RetireList<std::uint64_t>));
  counted.push(1, [] {});
  const void* site = vg_test::sole_allocation_site();
  ASSERT_NE(site, nullptr);
  counted.reclaim();  // the empty closure guards no GPU resource

  vkc::ImageDesc desc;
  desc.extent = {1, 1};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  auto image = allocator().create_image(desc);
  ASSERT_TRUE(image.ok());
  vg::RetireQueue retire(timeline.value());
  const std::uint32_t pixel = 0xff123456;
  vkc::Status recorded;
  bool failed = false;
  auto pool =
      vkc::CommandPool::create(device().handle(), device().queue_family());
  ASSERT_TRUE(pool.ok());
  auto cmd = pool.value().allocate_primary();
  ASSERT_TRUE(cmd.ok());
  ASSERT_TRUE(cmd.value().begin().ok());
  // Only the queue's owner allocation fails; validation and driver host
  // allocations remain unaffected. Check the unchanged state before submitting,
  // so the regression never submits dangling staging on a broken
  // implementation.
  vg_test::fail_allocation_at(sizeof(vg::RetireList<std::uint64_t>), site);
  recorded = vg::record_image_upload(cmd.value().handle(), allocator(), retire,
                                     1, image.value(), &pixel, sizeof(pixel));
  failed = vg_test::stop_targeted_failure();
  ASSERT_TRUE(failed);
  EXPECT_EQ(recorded.domain(), vkc::Status::Code::OutOfMemory);
  ASSERT_EQ(image.value().layout(), VK_IMAGE_LAYOUT_UNDEFINED);
  EXPECT_EQ(retire.pending(), 0u);
  ASSERT_TRUE(cmd.value().end().ok());
  EXPECT_TRUE(device().submit_and_wait(cmd.value().handle()).ok());

  const vkc::Status retried =
      device().submit_single_time([&](VkCommandBuffer cmd) {
        recorded = vg::record_image_upload(
            cmd, allocator(), retire, 1, image.value(), &pixel, sizeof(pixel));
      });
  ASSERT_TRUE(retried.ok()) << retried.message();
  ASSERT_TRUE(recorded.ok()) << recorded.message();
  EXPECT_EQ(image.value().layout(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  EXPECT_EQ(retire.pending(), 1u);
  ASSERT_TRUE(timeline.value().signal(1).ok());
  EXPECT_EQ(retire.poll(), 1u);
}

TEST_F(TextureAllocationTest, ImageUpdateDiagnosticsContainAllocationFailure) {
  vkc::Buffer source;
  vkc::Image image;
  VkBufferImageCopy region{};
  vg_test::AllocationFailure failure(0, true);
  const vkc::Status refused =
      vg::record_image_update(VK_NULL_HANDLE, source, image, &region, 1);
  const bool failed = failure.stop();
  EXPECT_TRUE(failed);
  EXPECT_EQ(refused.domain(), vkc::Status::Code::OutOfMemory);
  EXPECT_EQ(image.layout(), VK_IMAGE_LAYOUT_UNDEFINED);
  EXPECT_EQ(vg::record_image_update(VK_NULL_HANDLE, source, image, &region, 1)
                .domain(),
            vkc::Status::Code::InvalidArgument);
}

TEST_F(TextureAllocationTest, EveryUploadHostAllocationIsTransactional) {
  auto timeline = vkc::TimelineSemaphore::create(device());
  ASSERT_TRUE(timeline.ok());
  vkc::ImageDesc desc;
  desc.extent = {1, 1};
  desc.format = VK_FORMAT_R8G8B8A8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  const std::uint32_t pixel = 0xff654321;
  auto image = allocator().create_image(desc);
  ASSERT_TRUE(image.ok());
  vg::RetireQueue baseline(timeline.value());
  vkc::Status recorded;
  std::size_t sites = 0;
  ASSERT_TRUE(device()
                  .submit_single_time([&](VkCommandBuffer cmd) {
                    vg_test::record_upload_allocations();
                    recorded = vg::record_image_upload(
                        cmd, allocator(), baseline, 1, image.value(), &pixel,
                        sizeof(pixel));
                    sites = vg_test::finish_upload_allocations();
                  })
                  .ok());
  ASSERT_TRUE(recorded.ok());
  // At least the staging shared owner, queue owner and queue entry storage.
  ASSERT_GE(sites, 3u);
  ASSERT_LT(sites, 512u);
  RecordProperty("upload_host_allocation_sites", static_cast<int>(sites));
  ASSERT_TRUE(timeline.value().signal(1).ok());
  ASSERT_EQ(baseline.poll(), 1u);

  for (std::size_t i = 0; i < sites; ++i) {
    SCOPED_TRACE(i);
    auto destination = allocator().create_image(desc);
    ASSERT_TRUE(destination.ok());
    auto pool =
        vkc::CommandPool::create(device().handle(), device().queue_family());
    ASSERT_TRUE(pool.ok());
    auto cmd = pool.value().allocate_primary();
    ASSERT_TRUE(cmd.ok());
    ASSERT_TRUE(cmd.value().begin().ok());
    vg::RetireQueue retire(timeline.value());
    const std::uint64_t retire_at = i + 2;
    vg_test::fail_upload_allocation_at(vg_test::upload_allocation_site(i));
    recorded = vg::record_image_upload(cmd.value().handle(), allocator(),
                                       retire, retire_at, destination.value(),
                                       &pixel, sizeof(pixel));
    const bool failed = vg_test::stop_targeted_failure();
    const auto pending = retire.pending();
    // On a regression, do not let an unsubmitted value hang test teardown.
    // No command has been submitted, so reclaiming these owners is safe.
    if (pending != 0) retire.reclaim();
    ASSERT_TRUE(failed);
    EXPECT_EQ(recorded.domain(), vkc::Status::Code::OutOfMemory);
    ASSERT_EQ(destination.value().layout(), VK_IMAGE_LAYOUT_UNDEFINED);
    ASSERT_EQ(pending, 0u);

    // Retry in the same recording: no abandoned value may remain queued and
    // no partially recorded copy may have changed the image.
    recorded = vg::record_image_upload(cmd.value().handle(), allocator(),
                                       retire, retire_at, destination.value(),
                                       &pixel, sizeof(pixel));
    ASSERT_TRUE(recorded.ok()) << recorded.message();
    ASSERT_EQ(retire.pending(), 1u);
    ASSERT_TRUE(cmd.value().end().ok());
    ASSERT_TRUE(device().submit_and_wait(cmd.value().handle()).ok());
    ASSERT_TRUE(timeline.value().signal(retire_at).ok());
    EXPECT_EQ(retire.poll(), 1u);
  }
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace assets = volumetric_kit::gfx::assets;
namespace pipelines = volumetric_kit::gfx::pipelines;

// Adds a VMA allocator on top of the shared device fixture; skips with the base
// when no Vulkan device is present.
class GpuMeshTest : public VulkanDeviceTest {
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

  // A minimal two-triangle quad (4 default vertices, 6 indices).
  static assets::Mesh quad() {
    assets::Mesh mesh;
    mesh.vertices.resize(4);
    mesh.indices = {0, 1, 2, 0, 2, 3};
    return mesh;
  }

  std::optional<vg::Allocator> allocator_;
};

}  // namespace

TEST_F(GpuMeshTest, UploadsVerticesAndIndices) {
  auto mesh = pipelines::upload_mesh(*device_, *allocator_, quad());
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();
  EXPECT_TRUE(mesh.value().valid());
  EXPECT_EQ(mesh.value().index_count(), 6u);
}

TEST_F(GpuMeshTest, EmptyMeshIsRejected) {
  const assets::Mesh empty;  // no vertices / indices
  auto mesh = pipelines::upload_mesh(*device_, *allocator_, empty);
  ASSERT_FALSE(mesh.ok());
  EXPECT_EQ(mesh.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GpuMeshTest, BatchUploadsManyMeshesInOneSubmit) {
  auto batch = vg::UploadBatch::begin(*device_, *allocator_);
  ASSERT_TRUE(batch.ok()) << batch.status().message();

  // An invalid mesh is rejected before anything records: the batch stays open
  // and usable.
  const assets::Mesh empty;
  EXPECT_EQ(pipelines::upload_mesh(batch.value(), empty).status().domain(),
            vg::Status::Code::InvalidArgument);

  auto a = pipelines::upload_mesh(batch.value(), quad());
  auto b = pipelines::upload_mesh(batch.value(), quad());
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();

  // One submit for both meshes; both come back draw-ready.
  const vg::Status finished = batch.value().finish();
  ASSERT_TRUE(finished.ok()) << finished.message();
  EXPECT_TRUE(a.value().valid());
  EXPECT_TRUE(b.value().valid());
  EXPECT_EQ(a.value().index_count(), 6u);
  EXPECT_EQ(b.value().index_count(), 6u);
}

TEST_F(GpuMeshTest, BatchFormRejectsEmptyBatch) {
  vg::UploadBatch batch;  // default-constructed: owns nothing
  auto mesh = pipelines::upload_mesh(batch, quad());
  ASSERT_FALSE(mesh.ok());
  EXPECT_EQ(mesh.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(GpuMeshTest, MoveLeavesSourceEmpty) {
  auto made = pipelines::upload_mesh(*device_, *allocator_, quad());
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::GpuMesh source = std::move(made).value();
  ASSERT_TRUE(source.valid());

  pipelines::GpuMesh moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.index_count(), 6u);
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.index_count(), 0u);
}

TEST_F(GpuMeshTest, MoveAssignOverLiveLeavesSourceEmpty) {
  auto a = pipelines::upload_mesh(*device_, *allocator_, quad());
  auto b = pipelines::upload_mesh(*device_, *allocator_, quad());
  ASSERT_TRUE(a.ok()) << a.status().message();
  ASSERT_TRUE(b.ok()) << b.status().message();
  pipelines::GpuMesh dst = std::move(a).value();
  pipelines::GpuMesh src = std::move(b).value();

  dst = std::move(src);  // frees dst's buffers, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(src.index_count(), 0u);
}

TEST_F(GpuMeshTest, SelfMoveAssignIsSafe) {
  auto made = pipelines::upload_mesh(*device_, *allocator_, quad());
  ASSERT_TRUE(made.ok()) << made.status().message();
  pipelines::GpuMesh mesh = std::move(made).value();

  // Pointer-laundered self-move (dodges -Wself-move); the this != &other guard
  // must keep the mesh intact.
  pipelines::GpuMesh* alias = &mesh;
  mesh = std::move(*alias);
  EXPECT_TRUE(mesh.valid());
}

// No device needed: a default-constructed GpuMesh owns nothing.
TEST(GpuMeshDefaultTest, DefaultConstructedIsEmpty) {
  const pipelines::GpuMesh mesh;
  EXPECT_FALSE(mesh.valid());
  EXPECT_EQ(mesh.index_count(), 0u);
}

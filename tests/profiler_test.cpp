// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/profiler.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/frame_metrics.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

// Field assignment, not designated initializers (those are C++20; this project
// is C++17).
vg::ProfilerConfig make_config(uint32_t frames, uint32_t max_gpu_sections) {
  vg::ProfilerConfig config;
  config.frames_in_flight = frames;
  config.max_gpu_sections_per_frame = max_gpu_sections;
  return config;
}

class ProfilerTest : public VulkanDeviceTest {
 protected:
  // Build the instance/device with debug-utils enabled (and validation as the
  // label-balance backstop), so gpu_scope's label emit/close path is exercised;
  // fall back to debug-utils-only, then plain, where a layer or the extension
  // is unavailable. Overrides VulkanDeviceTest::SetUp, whose default configs
  // leave debug-utils off.
  void SetUp() override {
    vg::InstanceConfig icfg;
    icfg.enable_validation = true;
    icfg.enable_debug_utils = true;
    auto instance = vg::Instance::create(icfg);
    if (!instance.ok()) {
      icfg.enable_validation = false;
      instance = vg::Instance::create(icfg);
    }
    if (!instance.ok()) {
      GTEST_SKIP() << "no Vulkan instance: " << instance.status().message();
    }
    instance_.emplace(std::move(instance).value());

    auto physical = instance_->select_physical_device();
    if (!physical.ok()) {
      GTEST_SKIP() << "no Vulkan device: " << physical.status().message();
    }
    physical_ = physical.value();

    vg::DeviceConfig dcfg;
    dcfg.enable_debug_utils = instance_->debug_utils_enabled();
    auto device = vg::Device::create(instance_->handle(), physical_, dcfg);
    ASSERT_TRUE(device.ok()) << device.status().message();
    device_.emplace(std::move(device).value());
  }

  vg::Profiler make_profiler(uint32_t frames = 1, uint32_t max_gpu = 8) {
    auto result = vg::Profiler::create(*device_, make_config(frames, max_gpu));
    EXPECT_TRUE(result.ok()) << result.status().message();
    return std::move(result).value();
  }

  // Runs one whole frame on `slot` inside a one-time submit that retires before
  // returning, so the slot's GPU timestamps are readable on its next
  // begin_frame. `body` records work between begin_frame and end_frame.
  template <class Body>
  void run_frame(vg::Profiler& profiler, uint32_t slot, Body&& body) {
    vg::Status status = device_->submit_single_time([&](VkCommandBuffer cmd) {
      profiler.begin_frame(slot, cmd);
      body(cmd);
      profiler.end_frame();
    });
    ASSERT_TRUE(status.ok()) << status.message();
  }
};

}  // namespace

// gpu_timing() mirrors the device's timestamp-validity capability exactly: it
// is the single wiring invariant the rest of the GPU path keys off.
TEST_F(ProfilerTest, GpuTimingMatchesDeviceCapability) {
  vg::Profiler profiler = make_profiler();
  EXPECT_EQ(profiler.gpu_timing(),
            device_->graphics_timestamp_valid_bits() != 0);
}

// The published snapshot lags the in-flight depth: with one slot, the frame
// just recorded is not visible until the slot recurs on the next begin_frame.
TEST_F(ProfilerTest, MetricsEmptyUntilSlotRecurs) {
  vg::Profiler profiler = make_profiler(/*frames=*/1, /*max_gpu=*/8);
  EXPECT_TRUE(profiler.metrics().sections.empty());
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    auto s = profiler.gpu_scope(cmd, "work");
  });
  // Buffered against slot 0 but not resolved (slot 0 has not begun again).
  EXPECT_TRUE(profiler.metrics().sections.empty());
}

// End-to-end GPU path: a gpu_scope in frame 1 is resolved and published when
// slot 0 recurs in frame 2. Where the device has timestamp timing (lavapipe
// CI), has_gpu is set and gpu_ms is a real, non-negative measurement; where it
// does not (MoltenVK may report zero valid bits), the stage degrades to
// CPU-only.
TEST_F(ProfilerTest, PublishesStageAfterSlotRecurs) {
  vg::Profiler profiler = make_profiler(/*frames=*/1, /*max_gpu=*/8);
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    auto pass = profiler.gpu_scope(cmd, "pass");
  });
  run_frame(profiler, 0,
            [&](VkCommandBuffer) {});  // resolves + publishes frame 1

  const vg::FrameMetrics& m = profiler.metrics();
  ASSERT_EQ(m.sections.size(), 1u);
  EXPECT_STREQ(m.sections[0].name, "pass");
  EXPECT_GE(m.sections[0].cpu_ms, 0.0);
  EXPECT_EQ(m.sections[0].has_gpu, profiler.gpu_timing());
  if (profiler.gpu_timing()) {
    EXPECT_GE(m.sections[0].gpu_ms, 0.0);
  }
}

// A cpu_scope is always CPU-only: has_gpu stays false even on a device with GPU
// timing, since the flag reports per-stage GPU availability, not the device's.
TEST_F(ProfilerTest, CpuScopeNeverReportsGpu) {
  vg::Profiler profiler = make_profiler();
  run_frame(profiler, 0,
            [&](VkCommandBuffer) { auto s = profiler.cpu_scope("update"); });
  run_frame(profiler, 0, [&](VkCommandBuffer) {});

  const vg::FrameMetrics& m = profiler.metrics();
  ASSERT_EQ(m.sections.size(), 1u);
  EXPECT_STREQ(m.sections[0].name, "update");
  EXPECT_FALSE(m.sections[0].has_gpu);
  EXPECT_GE(m.sections[0].cpu_ms, 0.0);
}

// Stages appear in record order, mixing GPU and CPU scopes.
TEST_F(ProfilerTest, RecordsStagesInRecordOrder) {
  vg::Profiler profiler = make_profiler();
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    auto outer = profiler.gpu_scope(cmd, "render");
    auto inner = profiler.cpu_scope("cull");
  });
  run_frame(profiler, 0, [&](VkCommandBuffer) {});

  const vg::FrameMetrics& m = profiler.metrics();
  ASSERT_EQ(m.sections.size(), 2u);
  EXPECT_STREQ(m.sections[0].name, "render");
  EXPECT_STREQ(m.sections[1].name, "cull");
}

// With a memory source set, end_frame samples the allocator into the snapshot's
// aggregate figures. Any real device exposes at least one heap with a budget.
TEST_F(ProfilerTest, MemorySourcePopulatesAggregateMemory) {
  auto alloc = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(alloc.ok()) << alloc.status().message();
  vg::Allocator allocator = std::move(alloc).value();

  vg::Profiler profiler = make_profiler();
  profiler.set_memory_source(&allocator);
  run_frame(profiler, 0, [&](VkCommandBuffer) {});
  run_frame(profiler, 0, [&](VkCommandBuffer) {});

  EXPECT_GT(profiler.metrics().memory_budget_bytes, 0u);
}

// --- GPU resolve: multi-slot + multi-section indexing ----------------------

// Exercises the per-slot query base (slot*max*2) and the intra-frame stride (a
// second section's queries at local offset 2) — both always 0 in the other
// tests (frames=1, one section). A wrong per-slot or per-section offset reads
// an unwritten / other-slot query, so read_results returns NOT_READY and
// has_gpu flips off; the names also let a swapped pairing show.
TEST_F(ProfilerTest, MultiSlotMultiSectionResolves) {
  vg::Profiler profiler = make_profiler(/*frames=*/2, /*max_gpu=*/8);

  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    auto a = profiler.gpu_scope(cmd, "a");
    auto b = profiler.gpu_scope(cmd, "b");
  });
  run_frame(profiler, 1, [&](VkCommandBuffer cmd) {
    auto c = profiler.gpu_scope(cmd, "c");
  });
  run_frame(profiler, 0, [&](VkCommandBuffer) {
  });  // slot 0 recurs: publishes the first frame

  const vg::FrameMetrics& m = profiler.metrics();
  ASSERT_EQ(m.sections.size(), 2u);
  EXPECT_STREQ(m.sections[0].name, "a");
  EXPECT_STREQ(m.sections[1].name, "b");
  EXPECT_EQ(m.sections[0].has_gpu, profiler.gpu_timing());
  EXPECT_EQ(m.sections[1].has_gpu, profiler.gpu_timing());
  if (profiler.gpu_timing()) {
    EXPECT_TRUE(std::isfinite(m.sections[0].gpu_ms));
    EXPECT_TRUE(std::isfinite(m.sections[1].gpu_ms));
  }
}

// A gpu_scope whose cmd does not match begin_frame's cmd must not write a
// timestamp into the unreset query range — it degrades to CPU-only. The frame
// is begun CPU-only (VK_NULL_HANDLE, no reset) but the scope is handed the real
// recording buffer; the stage must resolve has_gpu == false.
TEST_F(ProfilerTest, GpuScopeWithMismatchedCmdIsCpuOnly) {
  vg::Profiler profiler = make_profiler();
  vg::Status status = device_->submit_single_time([&](VkCommandBuffer cmd) {
    profiler.begin_frame(0, VK_NULL_HANDLE);  // CPU-only frame: no reset
    {
      auto s = profiler.gpu_scope(cmd, "stage");
    }
    profiler.end_frame();
  });
  ASSERT_TRUE(status.ok()) << status.message();
  run_frame(profiler, 0, [&](VkCommandBuffer) {});

  const vg::FrameMetrics& m = profiler.metrics();
  ASSERT_EQ(m.sections.size(), 1u);
  EXPECT_FALSE(m.sections[0].has_gpu);
}

// A Scope still open at end_frame is finalized by end_frame (label closed, end
// timestamp written), so the submitted buffer carries no unbalanced debug-utils
// region and the stage still resolves; the Scope's later destruction is a safe
// no-op. (The @warning still asks callers to close scopes within the frame.)
TEST_F(ProfilerTest, ScopeOpenAtEndFrameStillResolves) {
  vg::Profiler profiler = make_profiler();
  vg::Status status = device_->submit_single_time([&](VkCommandBuffer cmd) {
    profiler.begin_frame(0, cmd);
    vg::Profiler::Scope open = profiler.gpu_scope(cmd, "open");
    profiler.end_frame();  // finalizes `open` before the buffer is submitted
    // `open` is destroyed here, after end_frame — finalize must no-op safely.
  });
  ASSERT_TRUE(status.ok()) << status.message();
  run_frame(profiler, 0, [&](VkCommandBuffer) {});

  const vg::FrameMetrics& m = profiler.metrics();
  ASSERT_EQ(m.sections.size(), 1u);
  EXPECT_STREQ(m.sections[0].name, "open");
  EXPECT_EQ(m.sections[0].has_gpu, profiler.gpu_timing());
}

// A null stage name must not reach vkCmdBeginDebugUtilsLabelEXT (pLabelName
// must be non-null); the label is skipped and the submit still succeeds even
// with debug-utils active.
TEST_F(ProfilerTest, GpuScopeNullNameEmitsNoLabel) {
  vg::Profiler profiler = make_profiler();
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    auto s = profiler.gpu_scope(cmd, nullptr);
  });
  run_frame(profiler, 0, [&](VkCommandBuffer) {});

  ASSERT_EQ(profiler.metrics().sections.size(), 1u);
}

// --- move-only contract: Profiler ------------------------------------------

TEST_F(ProfilerTest, MoveConstructLeavesSourceInvalid) {
  vg::Profiler source = make_profiler();
  ASSERT_TRUE(source.valid());

  vg::Profiler moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(ProfilerTest, MoveAssignOverLiveProfiler) {
  vg::Profiler source = make_profiler();
  vg::Profiler dest = make_profiler();  // a live profiler we assign over
  ASSERT_TRUE(source.valid());

  dest = std::move(source);
  EXPECT_TRUE(dest.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(ProfilerTest, SelfMoveIsSafe) {
  vg::Profiler profiler = make_profiler();
  vg::Profiler* alias = &profiler;
  profiler = std::move(*alias);  // pointer-laundered to dodge -Wself-move
  EXPECT_TRUE(profiler.valid());
}

// --- move-only contract: Scope ---------------------------------------------
// Driven CPU-only (begin_frame with a null command buffer) so the move
// bookkeeping is exercised without a submit; the scopes close before end_frame,
// as in real per-frame use.

TEST_F(ProfilerTest, ScopeMoveConstructLeavesSourceInert) {
  vg::Profiler profiler = make_profiler();
  profiler.begin_frame(0, VK_NULL_HANDLE);
  {
    vg::Profiler::Scope source = profiler.cpu_scope("stage");
    ASSERT_TRUE(source.active());

    vg::Profiler::Scope moved(std::move(source));
    EXPECT_TRUE(moved.active());
    EXPECT_FALSE(source.active());  // NOLINT(bugprone-use-after-move)
  }
  profiler.end_frame();
}

TEST_F(ProfilerTest, ScopeMoveAssignOverLiveScope) {
  vg::Profiler profiler = make_profiler();
  profiler.begin_frame(0, VK_NULL_HANDLE);
  {
    vg::Profiler::Scope a = profiler.cpu_scope("a");
    vg::Profiler::Scope b = profiler.cpu_scope("b");

    a = std::move(b);  // a finalizes its own stage, then adopts b's
    EXPECT_TRUE(a.active());
    EXPECT_FALSE(b.active());  // NOLINT(bugprone-use-after-move)
  }
  profiler.end_frame();
}

TEST_F(ProfilerTest, ScopeSelfMoveIsSafe) {
  vg::Profiler profiler = make_profiler();
  profiler.begin_frame(0, VK_NULL_HANDLE);
  {
    vg::Profiler::Scope scope = profiler.cpu_scope("stage");
    vg::Profiler::Scope* alias = &scope;
    scope = std::move(*alias);  // pointer-laundered to dodge -Wself-move
    EXPECT_TRUE(scope.active());
  }
  profiler.end_frame();
}

// A default-constructed scope is inert and finalizes nothing; moving it stays
// inert. Holds with no device, so it documents the contract directly.
TEST(ProfilerInertTest, DefaultScopeIsInert) {
  vg::Profiler::Scope scope;
  EXPECT_FALSE(scope.active());

  vg::Profiler::Scope moved(std::move(scope));
  EXPECT_FALSE(moved.active());
  EXPECT_FALSE(scope.active());  // NOLINT(bugprone-use-after-move)
}

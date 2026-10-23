// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/profiler.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <utility>
#include <vector>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/frame_metrics.hpp"

namespace {

// Field assignment, not designated initializers (those are C++20; this project
// is C++17).
vg::ProfilerConfig make_config(uint32_t frames, uint32_t max_gpu_sections) {
  vg::ProfilerConfig config;
  config.frames_in_flight = frames;
  config.max_gpu_sections_per_frame = max_gpu_sections;
  return config;
}

class ProfilerTest : public vg_test::RendererDeviceTest {
 protected:
  // Validation as the label-balance backstop, on the instance's debug-utils
  // (which every instance requests), so gpu_scope's label emit/close path is
  // exercised; the core's instance continues debug-utils-only where the layer
  // is unavailable.
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::On;
  }

  vg::Profiler make_profiler(uint32_t frames = 1, uint32_t max_gpu = 8) {
    auto result = vg::Profiler::create(device(), make_config(frames, max_gpu));
    EXPECT_TRUE(result.ok()) << result.status().message();
    return std::move(result).value();
  }

  // Runs one whole frame on `slot` inside a one-time submit that retires before
  // returning, so the slot's GPU timestamps are readable on its next
  // begin_frame. `body` records work between begin_frame and end_frame.
  template <class Body>
  void run_frame(vg::Profiler& profiler, uint32_t slot, Body&& body) {
    vkc::Status status = device().submit_single_time([&](VkCommandBuffer cmd) {
      profiler.begin_frame(slot, cmd);
      body(cmd);
      profiler.end_frame();
    });
    ASSERT_TRUE(status.ok()) << status.message();
  }

  // A 16 MiB device-only buffer: a fill of it is work the GPU takes a
  // measurable time over.
  vkc::Buffer fill_target() {
    vkc::BufferDesc desc;
    desc.size = 16u << 20;
    desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto buffer = allocator().create_buffer(desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
  }
};

}  // namespace

// gpu_timing() mirrors the device's timestamp-validity capability exactly: it
// is the single wiring invariant the rest of the GPU path keys off.
TEST_F(ProfilerTest, GpuTimingMatchesDeviceCapability) {
  vg::Profiler profiler = make_profiler();
  EXPECT_EQ(profiler.gpu_timing(), device().timestamp_valid_bits() != 0);
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
  vg::Profiler profiler = make_profiler();
  profiler.set_memory_source(&allocator());
  run_frame(profiler, 0, [&](VkCommandBuffer) {});
  run_frame(profiler, 0, [&](VkCommandBuffer) {});

  EXPECT_GT(profiler.metrics().memory_budget_bytes, 0u);
}

// --- GPU resolve: multi-slot + multi-section indexing ----------------------

// Exercises two frames in flight, each holding its own range of the timer, and
// a second section's tiles in one frame — neither of which the one-slot,
// one-section tests reach. A frame resolved from another's range, or a section
// summing the wrong tiles, reads a mark not written for it, so the row reads
// NotTimed and has_gpu flips off; the names also let a swapped pairing show.
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
  vkc::Status status = device().submit_single_time([&](VkCommandBuffer cmd) {
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
  vkc::Status status = device().submit_single_time([&](VkCommandBuffer cmd) {
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

// A published row's name is never null (core::StageRow), so a stage opened
// without one is recorded as "(unnamed)" -- which also keeps a null off
// vkCmdBeginDebugUtilsLabelEXT, whose pLabelName must be non-null, so the
// submit succeeds with debug-utils and validation active. The rows then fold
// into the core's StageMetrics, which aborts on a null name.
TEST_F(ProfilerTest, NullNameStageIsRecordedUnnamed) {
  vg::Profiler profiler = make_profiler();
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    auto gpu = profiler.gpu_scope(cmd, nullptr);
  });
  run_frame(profiler, 0,
            [&](VkCommandBuffer) { auto cpu = profiler.cpu_scope(nullptr); });
  ASSERT_EQ(profiler.metrics().sections.size(), 1u);
  EXPECT_STREQ(profiler.metrics().sections[0].name, "(unnamed)");

  run_frame(profiler, 0, [&](VkCommandBuffer) {});
  ASSERT_EQ(profiler.metrics().sections.size(), 1u);
  EXPECT_STREQ(profiler.metrics().sections[0].name, "(unnamed)");

  vkc::StageMetrics stages;
  for (const vkc::StageRow& row : profiler.metrics().sections) {
    stages.add_cpu(row.name, row.cpu_ms);
  }
  EXPECT_EQ(stages.rows().size(), 1u);
}

// A bound past what the core's timer holds for the frames in flight is
// lowered, not refused: two frames take 2048 pairs of queries at most, so the
// bound becomes 1023 and each frame's 1024 pairs fit beside the other's. A
// frame that took a pair more would leave the next no range, so both frames
// in flight are timed.
TEST_F(ProfilerTest, ABoundPastWhatTheTimerHoldsIsLowered) {
  vg::Profiler profiler = make_profiler(/*frames=*/2, /*max_gpu=*/0x7FFFFFFFu);
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    auto a = profiler.gpu_scope(cmd, "a");
  });
  run_frame(profiler, 1, [&](VkCommandBuffer cmd) {
    auto b = profiler.gpu_scope(cmd, "b");
  });
  const vkc::DeviceTime timed = profiler.gpu_timing()
                                    ? vkc::DeviceTime::Measured
                                    : vkc::DeviceTime::Unavailable;
  run_frame(profiler, 0, [](VkCommandBuffer) {});
  ASSERT_EQ(profiler.metrics().sections.size(), 1u);
  EXPECT_STREQ(profiler.metrics().sections[0].name, "a");
  EXPECT_EQ(profiler.metrics().sections[0].device, timed);
  run_frame(profiler, 1, [](VkCommandBuffer) {});
  ASSERT_EQ(profiler.metrics().sections.size(), 1u);
  EXPECT_STREQ(profiler.metrics().sections[0].name, "b");
  EXPECT_EQ(profiler.metrics().sections[0].device, timed);
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

// --- device halves, on the core's tiles -------------------------------------

// Each row says what its device half holds: a GPU scope within the bound is
// measured, one past it and one on another command buffer are not timed, a
// CPU scope has none; where the queue cannot time, every GPU scope reads
// unavailable.
TEST_F(ProfilerTest, EachRowSaysWhatItsDeviceHalfHolds) {
  vg::Profiler profiler = make_profiler(/*frames=*/1, /*max_gpu=*/1);
  // Each scope closes at the end of its statement.
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    (void)profiler.gpu_scope(cmd, "timed");
    (void)profiler.gpu_scope(cmd, "past the bound");
    (void)profiler.cpu_scope("host");
  });
  vkc::Status status = device().submit_single_time([&](VkCommandBuffer cmd) {
    profiler.begin_frame(0, VK_NULL_HANDLE);  // publishes the first frame
    (void)profiler.gpu_scope(cmd, "other buffer");
    profiler.end_frame();
  });
  ASSERT_TRUE(status.ok()) << status.message();
  const vkc::DeviceTime measured = profiler.gpu_timing()
                                       ? vkc::DeviceTime::Measured
                                       : vkc::DeviceTime::Unavailable;
  const vkc::DeviceTime untimed = profiler.gpu_timing()
                                      ? vkc::DeviceTime::NotTimed
                                      : vkc::DeviceTime::Unavailable;

  const std::vector<vkc::StageRow> first = profiler.metrics().sections;
  ASSERT_EQ(first.size(), 3u);
  EXPECT_EQ(first[0].device, measured);
  EXPECT_EQ(first[0].has_gpu, profiler.gpu_timing());
  EXPECT_EQ(first[1].device, untimed);
  EXPECT_FALSE(first[1].has_gpu);
  EXPECT_EQ(first[2].device, vkc::DeviceTime::None);

  run_frame(profiler, 0, [](VkCommandBuffer) {});
  const vg::FrameMetrics& second = profiler.metrics();
  ASSERT_EQ(second.sections.size(), 1u);
  EXPECT_EQ(second.sections[0].device, untimed);
}

// A GPU scope's time runs from its open to its close, so it holds the scopes
// nested in it, and a scope's time ends at its close: each nested fill counts
// to its own scope and to the one around both.
TEST_F(ProfilerTest, AGpuScopeHoldsTheScopesNestedInIt) {
  vg::Profiler profiler = make_profiler();
  vkc::Buffer first = fill_target();
  vkc::Buffer second = fill_target();
  ASSERT_TRUE(first.valid() && second.valid());
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    auto outer = profiler.gpu_scope(cmd, "outer");
    {
      auto a = profiler.gpu_scope(cmd, "a");
      vkCmdFillBuffer(cmd, first.handle(), 0, VK_WHOLE_SIZE, 1u);
    }
    {
      auto b = profiler.gpu_scope(cmd, "b");
      vkCmdFillBuffer(cmd, second.handle(), 0, VK_WHOLE_SIZE, 2u);
    }
  });
  run_frame(profiler, 0, [](VkCommandBuffer) {});

  const vg::FrameMetrics& m = profiler.metrics();
  ASSERT_EQ(m.sections.size(), 3u);
  const vkc::StageRow& outer = m.sections[0];
  const vkc::StageRow& a = m.sections[1];
  const vkc::StageRow& b = m.sections[2];
  if (!profiler.gpu_timing()) {
    EXPECT_EQ(outer.device, vkc::DeviceTime::Unavailable);
    return;
  }
  ASSERT_EQ(outer.device, vkc::DeviceTime::Measured);
  ASSERT_EQ(a.device, vkc::DeviceTime::Measured);
  ASSERT_EQ(b.device, vkc::DeviceTime::Measured);
  EXPECT_GE(outer.gpu_ms, a.gpu_ms + b.gpu_ms);
  EXPECT_GE(a.gpu_ms, 0.0);
  EXPECT_GE(b.gpu_ms, 0.0);
  // Each fill counts to its own scope: what outer holds beyond a and b is the
  // gaps between the marks, small beside either fill. A scope credited the
  // tile after its own would hold a gap, and outer both fills over it.
  EXPECT_LE(outer.gpu_ms - a.gpu_ms - b.gpu_ms,
            0.5 * std::min(a.gpu_ms, b.gpu_ms) + 1e-9)
      << "outer " << outer.gpu_ms << " ms, a " << a.gpu_ms << ", b "
      << b.gpu_ms;
}

// A scope's host time splits into busy and blocked as a core scope's does: a
// core wait inside it -- here a timeline semaphore another thread signals --
// is its wait, and the wait of the scope around it, and no other scope's.
TEST_F(ProfilerTest, ACoreWaitInsideAScopeIsItsWait) {
  vg::Profiler profiler = make_profiler();
  auto made = vkc::TimelineSemaphore::create(device());
  ASSERT_TRUE(made.ok()) << made.status().message();
  vkc::TimelineSemaphore timeline = std::move(made).value();
  vkc::Status waited;
  vkc::Status signalled;
  profiler.begin_frame(0, VK_NULL_HANDLE);
  {
    auto frame = profiler.cpu_scope("frame");
    {
      auto waiting = profiler.cpu_scope("waiting");
      std::thread signaller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        signalled = timeline.signal(1);
      });
      waited = timeline.wait(1);
      signaller.join();
    }
    auto after = profiler.cpu_scope("after");
  }
  profiler.end_frame();
  ASSERT_TRUE(signalled.ok()) << signalled.message();
  ASSERT_TRUE(waited.ok()) << waited.message();
  profiler.begin_frame(0, VK_NULL_HANDLE);  // publishes the frame
  profiler.end_frame();

  const std::vector<vkc::StageRow>& rows = profiler.metrics().sections;
  ASSERT_EQ(rows.size(), 3u);
  const vkc::StageRow& frame = rows[0];
  const vkc::StageRow& waiting = rows[1];
  const vkc::StageRow& after = rows[2];
  EXPECT_GT(waiting.wait_ms, 0.0);
  EXPECT_LE(waiting.wait_ms, waiting.cpu_ms);
  EXPECT_EQ(frame.wait_ms, waiting.wait_ms);
  EXPECT_LE(frame.wait_ms, frame.cpu_ms);
  EXPECT_EQ(after.wait_ms, 0.0);
}

// A render loop's rows come after the frame's own stages, whether added while
// the frame records or once it has ended, and rows of one name accumulate;
// outside a frame, nothing is added.
TEST_F(ProfilerTest, AddedRowsFollowTheFramesOwnStages) {
  vg::Profiler profiler = make_profiler();
  vkc::StageMetrics waited;
  waited.add_cpu("frame wait", 2.0);
  waited.add_wait("frame wait", 2.0);
  vkc::StageMetrics presented;
  presented.add_cpu("present", 0.5);
  presented.add_wait("present", 0.5);

  profiler.add_rows(presented);  // no frame yet
  run_frame(profiler, 0, [&](VkCommandBuffer cmd) {
    profiler.add_rows(waited);
    auto pass = profiler.gpu_scope(cmd, "pass");
  });
  profiler.add_rows(presented);
  profiler.add_rows(presented);
  run_frame(profiler, 0, [](VkCommandBuffer) {});

  const vg::FrameMetrics& m = profiler.metrics();
  ASSERT_EQ(m.sections.size(), 3u);
  EXPECT_STREQ(m.sections[0].name, "pass");
  EXPECT_STREQ(m.sections[1].name, "frame wait");
  EXPECT_DOUBLE_EQ(m.sections[1].wait_ms, 2.0);
  EXPECT_STREQ(m.sections[2].name, "present");
  EXPECT_DOUBLE_EQ(m.sections[2].cpu_ms, 1.0);
  EXPECT_DOUBLE_EQ(m.sections[2].wait_ms, 1.0);
  EXPECT_EQ(m.sections[2].device, vkc::DeviceTime::None);
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

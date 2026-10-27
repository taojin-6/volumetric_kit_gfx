// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The frame loop over offscreen targets: frames rendered and read back per
// slot, the numbering, the pace, the profiler's rows, timeline waits and
// signals with their refusals, and teardown. It needs no surface, so it runs
// on the shared renderer device on every GPU leg, NVIDIA's included, where
// the swapchain loop's tests skip (windowing_test.cpp). Synchronization
// validation checks each slot's target as it is prepared, drawn, read back
// and reused across submits.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gfx_test_support.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/profiler.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/retire_queue.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"

namespace win = volumetric_kit::gfx::windowing;

namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

constexpr VkFormat kColorFormat = VK_FORMAT_B8G8R8A8_SRGB;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkExtent2D kExtent{8, 4};

// A texel as a B8G8R8A8 readback holds it.
struct Bgra {
  uint8_t b, g, r, a;
};

// An opaque clear whose channels are 0 or 1, which every color format
// stores exactly, and the bytes it reads back as.
struct Clear {
  float r, g, b;
  Bgra bytes() const {
    const auto byte = [](float c) { return static_cast<uint8_t>(c * 255.0f); };
    return {byte(b), byte(g), byte(r), 255};
  }
};

constexpr std::array<Clear, 6> kClears = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {0, 1, 1}, {1, 0, 1}}};

double ms_between(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

// A published frame's rows by name, in order.
std::vector<std::string> row_names(const vg::FrameMetrics& metrics) {
  std::vector<std::string> names;
  for (const vkc::StageRow& row : metrics.sections) {
    names.emplace_back(row.name);
  }
  return names;
}

class OffscreenFrameLoopTest : public vg_test::RendererDeviceTest {
 protected:
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::Sync;
  }

  static win::OffscreenLoopDesc desc(std::chrono::nanoseconds interval = {}) {
    win::OffscreenLoopDesc d;
    d.target.extent = kExtent;
    d.target.color_format = kColorFormat;
    d.target.depth_format = kDepthFormat;
    d.frame_interval = interval;
    return d;
  }

  // A loop on the shared device, or null, failing the test, when create
  // fails.
  std::unique_ptr<win::FrameLoop> make_loop(
      uint32_t frames_in_flight = 2, std::chrono::nanoseconds interval = {}) {
    auto loop = win::FrameLoop::create(device(), allocator(), desc(interval),
                                       frames_in_flight);
    EXPECT_TRUE(loop.ok()) << loop.status().message();
    return loop.ok() ? std::move(loop).value() : nullptr;
  }

  // Record a clear of the frame's color and depth, as every frame here draws.
  static void record_clear(const win::Frame& frame, Clear clear = {0, 0, 0}) {
    vg::RenderTargetBeginInfo begin;
    begin.clear_color.float32[0] = clear.r;
    begin.clear_color.float32[1] = clear.g;
    begin.clear_color.float32[2] = clear.b;
    begin.clear_color.float32[3] = 1.0f;
    frame.target->begin(frame.cmd, begin);
    frame.target->end(frame.cmd);
  }

  // Drive `count` clear-only frames through the loop.
  static vkc::Status run_frames(win::FrameLoop& loop, int count) {
    for (int i = 0; i < count; ++i) {
      auto frame = loop.begin_frame();
      if (!frame.ok()) {
        return frame.status();
      }
      record_clear(frame.value());
      vkc::Status end = loop.end_frame(frame.value());
      if (!end.ok()) {
        return end;
      }
    }
    return {};
  }

  // A timeline semaphore on the shared device, at 0.
  vkc::TimelineSemaphore make_timeline() {
    auto timeline = vkc::TimelineSemaphore::create(device());
    EXPECT_TRUE(timeline.ok()) << timeline.status().message();
    return timeline.ok() ? std::move(timeline).value()
                         : vkc::TimelineSemaphore{};
  }

  // A 4-byte buffer: device-only with transfer usage, or a host-readable
  // readback.
  vkc::Buffer make_word_buffer(bool readback) {
    vkc::BufferDesc d;
    d.size = sizeof(uint32_t);
    d.usage = readback ? VK_BUFFER_USAGE_TRANSFER_DST_BIT
                       : VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (readback) {
      d.memory = vkc::MemoryUsage::Staging;
      d.host_access = vkc::HostAccess::Random;
    }
    auto buffer = allocator().create_buffer(d);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
  }
};

// Every texel `target` read back is `expected`.
void expect_every_texel(const vg::OffscreenTarget& target, Bgra expected) {
  const auto* px = static_cast<const uint8_t*>(target.pixels());
  ASSERT_NE(px, nullptr);
  const size_t texels = size_t{kExtent.width} * kExtent.height;
  for (size_t i = 0; i < texels; ++i) {
    const uint8_t* t = px + 4 * i;
    if (t[0] != expected.b || t[1] != expected.g || t[2] != expected.r ||
        t[3] != expected.a) {
      ADD_FAILURE() << "texel " << i << " is (" << int{t[0]} << ", "
                    << int{t[1]} << ", " << int{t[2]} << ", " << int{t[3]}
                    << "), expected (" << int{expected.b} << ", "
                    << int{expected.g} << ", " << int{expected.r} << ", "
                    << int{expected.a} << ")";
      return;
    }
  }
}

// Each slot owns a target of the description's extent and formats, prepared
// for the frame; a frame drawn into it and read back is in its pixels once
// the slot recurs, and after the device idles for the last frames.
TEST_F(OffscreenFrameLoopTest, EachSlotReadsBackItsFrameWhenTheSlotRecurs) {
  constexpr uint32_t kSlots = 2;
  std::unique_ptr<win::FrameLoop> loop = make_loop(kSlots);
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(loop->frames_in_flight(), kSlots);
  const vg::RenderTargetLayout layout = loop->layout();
  EXPECT_EQ(layout.color_count, 1u);
  EXPECT_EQ(layout.color_formats[0], kColorFormat);
  EXPECT_EQ(layout.depth_format, kDepthFormat);
  EXPECT_EQ(layout.samples, VK_SAMPLE_COUNT_1_BIT);

  std::array<const vg::OffscreenTarget*, kSlots> targets{};
  std::array<int, kSlots> drawn{-1, -1};  // the frame each slot holds
  for (int n = 0; n < static_cast<int>(kClears.size()); ++n) {
    SCOPED_TRACE(n);
    auto frame = loop->begin_frame();
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    const win::Frame& f = frame.value();
    EXPECT_EQ(f.number, static_cast<uint64_t>(n) + 1);
    ASSERT_EQ(f.slot, static_cast<uint32_t>(n) % kSlots);
    EXPECT_EQ(f.image_index, f.slot);
    ASSERT_NE(f.offscreen, nullptr);
    ASSERT_NE(f.target, nullptr);
    EXPECT_EQ(f.target->extent().width, kExtent.width);
    EXPECT_EQ(f.target->extent().height, kExtent.height);
    EXPECT_TRUE(f.target->layout().compatible_with(layout));
    EXPECT_EQ(f.offscreen->extent().width, kExtent.width);
    EXPECT_NE(f.offscreen->depth_image(), VK_NULL_HANDLE);
    if (targets[f.slot] == nullptr) {
      targets[f.slot] = f.offscreen;
    } else {
      EXPECT_EQ(f.offscreen, targets[f.slot]);  // the slot's own target
    }
    if (drawn[f.slot] >= 0) {
      expect_every_texel(*f.offscreen, kClears[drawn[f.slot]].bytes());
    }
    record_clear(f, kClears[n]);
    f.offscreen->record_readback(f.cmd);
    drawn[f.slot] = n;
    ASSERT_TRUE(loop->end_frame(f).ok());
  }
  EXPECT_NE(targets[0], targets[1]);

  ASSERT_TRUE(device().wait_idle().ok());
  for (uint32_t slot = 0; slot < kSlots; ++slot) {
    SCOPED_TRACE(slot);
    expect_every_texel(*targets[slot], kClears[drawn[slot]].bytes());
  }
}

// Frames are numbered 1, 2, ... on the loop's timeline, which reaches a number
// once that frame completes, within the in-flight depth of the frame begun. A
// RetireQueue on the timeline frees what a frame used once it completes.
TEST_F(OffscreenFrameLoopTest, FramesCompleteInOrderAndRetireOnTheTimeline) {
  constexpr uint32_t kSlots = 2;
  std::unique_ptr<win::FrameLoop> loop = make_loop(kSlots);
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(loop->submitted(), 0u);
  ASSERT_TRUE(loop->completed().ok());
  EXPECT_EQ(loop->completed().value(), 0u);
  vg::RetireQueue retire(loop->timeline());
  std::vector<uint64_t> released;

  uint64_t last_completed = 0;
  for (uint64_t n = 1; n <= 8; ++n) {
    auto frame = loop->begin_frame();
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    EXPECT_EQ(frame.value().number, n);
    EXPECT_EQ(frame.value().slot, (n - 1) % kSlots);

    auto completed = loop->completed();
    ASSERT_TRUE(completed.ok()) << completed.status().message();
    EXPECT_GE(completed.value() + kSlots, n);  // at most kSlots frames ahead
    EXPECT_GE(completed.value(), last_completed);
    EXPECT_LE(completed.value(), loop->submitted());
    last_completed = completed.value();

    retire.push(n, [&released, n]() { released.push_back(n); });
    retire.poll();
    // Never before its frame completed: read after the poll, the timeline is
    // at least where the poll saw it.
    const auto after_poll = loop->completed();
    ASSERT_TRUE(after_poll.ok()) << after_poll.status().message();
    for (const uint64_t r : released) {
      EXPECT_LE(r, after_poll.value());
    }
    record_clear(frame.value());
    ASSERT_TRUE(loop->end_frame(frame.value()).ok());
    EXPECT_EQ(loop->submitted(), n);
  }
  ASSERT_TRUE(loop->timeline().wait(8).ok());
  EXPECT_EQ(loop->completed().value(), 8u);
  retire.poll();
  EXPECT_EQ(released, (std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8}));
}

// The pace holds frames at least an interval apart, as a display would: N
// frames take (N - 1) intervals or more. An interval of 0, the default, holds
// none back.
TEST_F(OffscreenFrameLoopTest, PaceHoldsFramesAnIntervalApart) {
  EXPECT_EQ(win::OffscreenLoopDesc{}.frame_interval.count(), 0);

  constexpr milliseconds kInterval(20);
  constexpr int kFrames = 5;
  std::unique_ptr<win::FrameLoop> paced = make_loop(2, kInterval);
  ASSERT_NE(paced, nullptr);
  const Clock::time_point start = Clock::now();
  ASSERT_TRUE(run_frames(*paced, kFrames).ok());
  EXPECT_GE(Clock::now() - start, (kFrames - 1) * kInterval);

  // Three frames held a long interval apart would take 1 s or more.
  std::unique_ptr<win::FrameLoop> unpaced = make_loop(2);
  ASSERT_NE(unpaced, nullptr);
  const Clock::time_point unpaced_start = Clock::now();
  ASSERT_TRUE(run_frames(*unpaced, 3).ok());
  EXPECT_LT(Clock::now() - unpaced_start, milliseconds(1000));
}

// Without a profiler the loop adds no rows, but the pace still reaches the
// caller's scope as blocked time, as an acquire does on a swapchain loop.
TEST_F(OffscreenFrameLoopTest, PaceReachesTheCallersScopeAsBlockedTime) {
  constexpr milliseconds kInterval(40);
  std::unique_ptr<win::FrameLoop> loop = make_loop(2, kInterval);
  ASSERT_NE(loop, nullptr);
  const Clock::time_point first = Clock::now();
  ASSERT_TRUE(run_frames(*loop, 1).ok());

  vkc::StageMetrics ticks;
  Clock::time_point before;
  {
    const vkc::StageScope tick(ticks, "tick");
    before = Clock::now();
    ASSERT_TRUE(run_frames(*loop, 1).ok());
  }
  ASSERT_EQ(ticks.rows().size(), 1u);
  const vkc::StageRow& row = ticks.rows()[0];
  EXPECT_GT(row.wait_ms, 0.0);
  EXPECT_LT(row.wait_ms, row.cpu_ms);
  // The second frame may not begin before an interval after the first did,
  // which began after `first`: the scope waited at least that long.
  const double held_ms = ms_between(before, first + kInterval);
  EXPECT_GE(row.wait_ms, held_ms - 1.0);
}

// An attached profiler gets the caller's stages, then the loop's frame wait
// and submit; an offscreen loop presents nothing, so has no present row. The
// frame's GPU scope is timed as on a swapchain loop.
TEST_F(OffscreenFrameLoopTest, ProfilerGetsFrameWaitAndSubmitButNoPresent) {
  vg::ProfilerConfig pcfg;
  pcfg.frames_in_flight = 2;
  auto profiler = vg::Profiler::create(device(), pcfg);
  ASSERT_TRUE(profiler.ok()) << profiler.status().message();
  // A short pace, so every frame's wait has a span to measure.
  std::unique_ptr<win::FrameLoop> loop = make_loop(2, milliseconds(2));
  ASSERT_NE(loop, nullptr);
  loop->set_profiler(&profiler.value());

  for (int i = 0; i < 4; ++i) {
    auto frame = loop->begin_frame();
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    {
      vg::Profiler::Scope pass =
          profiler.value().gpu_scope(frame.value().cmd, "frame");
      record_clear(frame.value());
    }
    ASSERT_TRUE(loop->end_frame(frame.value()).ok());
  }
  ASSERT_TRUE(device().wait_idle().ok());

  // The second frame, which the fourth's begin_frame published.
  const vg::FrameMetrics& m = profiler.value().metrics();
  ASSERT_EQ(row_names(m),
            (std::vector<std::string>{"frame", "frame wait", "submit"}));
  EXPECT_EQ(m.sections[0].has_gpu, profiler.value().gpu_timing());
  EXPECT_GT(m.cpu_frame_ms, 0.0);
  const vkc::StageRow& waited = m.sections[1];
  const vkc::StageRow& submitted = m.sections[2];
  EXPECT_GT(waited.wait_ms, 0.0);
  EXPECT_DOUBLE_EQ(waited.cpu_ms, waited.wait_ms);
  EXPECT_GT(submitted.cpu_ms, 0.0);
  EXPECT_DOUBLE_EQ(submitted.wait_ms, 0.0);
  for (const vkc::StageRow* row : {&waited, &submitted}) {
    EXPECT_EQ(row->device, vkc::DeviceTime::None) << row->name;
  }
  loop->set_profiler(nullptr);
}

// A frame's wait for a value another submit sets is honoured: the producer
// fills a buffer and sets `produced`; once that value is reached, the frame
// copies the buffer out at the TRANSFER stage. Synchronization validation
// would report a frame that ignored its wait: nothing else orders its copy
// after the fill. What the frame signals is set once it completes, and
// joins the core's record of submitted values.
TEST_F(OffscreenFrameLoopTest, FrameWaitsForAndSignalsTimelineValues) {
  std::unique_ptr<win::FrameLoop> loop = make_loop();
  ASSERT_NE(loop, nullptr);
  vkc::TimelineSemaphore produced = make_timeline();
  vkc::TimelineSemaphore done = make_timeline();
  vkc::Buffer source = make_word_buffer(/*readback=*/false);
  vkc::Buffer readback = make_word_buffer(/*readback=*/true);
  ASSERT_TRUE(source.valid() && readback.valid());

  constexpr uint32_t kProduced = 0xabcdabcdu;
  const VkBuffer source_handle = source.handle();
  auto producer = device().submit_pending(
      [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, source_handle, 0, VK_WHOLE_SIZE, kProduced);
      },
      {}, {{&produced, 1}});
  ASSERT_TRUE(producer.ok()) << producer.status().message();

  auto frame = loop->begin_frame();
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  win::Frame& f = frame.value();
  VkBufferCopy region{};
  region.size = sizeof(uint32_t);
  vkCmdCopyBuffer(f.cmd, source_handle, readback.handle(), 1, &region);
  VkMemoryBarrier to_host{};
  to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(f.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, nullptr,
                       0, nullptr);
  record_clear(f);
  f.waits.push_back({{&produced, 1}, VK_PIPELINE_STAGE_TRANSFER_BIT});
  f.signals.push_back({&done, 10});
  ASSERT_TRUE(produced.wait(1).ok());
  EXPECT_EQ(done.value().value(), 0u);
  ASSERT_TRUE(loop->end_frame(f).ok());
  // A submit after the frame may not set a value at or below the frame's.
  EXPECT_EQ(device()
                .submit_pending([](VkCommandBuffer) {}, {}, {{&done, 5}})
                .status()
                .domain(),
            vkc::Status::Code::InvalidArgument);

  ASSERT_TRUE(loop->timeline().wait(f.number).ok());
  ASSERT_TRUE(done.wait(10).ok());
  ASSERT_TRUE(producer.value().wait().ok());
  uint32_t copied = 0;
  std::memcpy(&copied, readback.mapped(), sizeof(copied));
  EXPECT_EQ(copied, kProduced);

  // A later frame may wait for what an earlier one set.
  auto next = loop->begin_frame();
  ASSERT_TRUE(next.ok()) << next.status().message();
  record_clear(next.value());
  next.value().waits.push_back({{&done, 10}});
  ASSERT_TRUE(loop->end_frame(next.value()).ok());
  ASSERT_TRUE(device().wait_idle().ok());
}

// end_frame refuses a wait or signal it cannot submit, as on a swapchain
// loop. A refused frame never runs, but an empty submit stands in for it: its
// number is still set, so what waits for it does not hang, its profiler frame
// is never read, and the loop runs on with no rebuild. A wait for a completed
// frame is accepted.
TEST_F(OffscreenFrameLoopTest, EndFrameRefusesPointsItCannotHonour) {
  vg::ProfilerConfig pcfg;
  pcfg.frames_in_flight = 2;
  auto profiler = vg::Profiler::create(device(), pcfg);
  ASSERT_TRUE(profiler.ok()) << profiler.status().message();
  std::unique_ptr<win::FrameLoop> loop = make_loop();
  ASSERT_NE(loop, nullptr);
  loop->set_profiler(&profiler.value());
  ASSERT_TRUE(run_frames(*loop, 1).ok());
  vkc::TimelineSemaphore other = make_timeline();
  ASSERT_TRUE(other.signal(2).ok());  // nothing is submitted to set more
  vkc::TimelineSemaphore empty;
  const vkc::TimelineSemaphore& own = loop->timeline();

  struct Case {
    const char* what;
    std::vector<win::FrameWait> waits;
    std::vector<vkc::TimelinePoint> signals;
  };
  const std::vector<Case> cases = {
      {"null wait", {{{nullptr, 1}}}, {}},
      {"empty wait", {{{&empty, 1}}}, {}},
      {"no stages", {{{&other, 1}, 0}}, {}},
      {"unsubmitted value", {{{&other, 3}}}, {}},
      {"unsubmitted frame", {{{&own, 0}}}, {}},  // set to the frame's own
      {"null signal", {}, {{nullptr, 1}}},
      {"own timeline signal", {}, {{&own, 100}}},
      {"signal set twice", {}, {{&other, 5}, {&other, 6}}},
      {"signal not above the counter", {}, {{&other, 2}}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.what);
    auto frame = loop->begin_frame();
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    win::Frame& f = frame.value();
    {
      vg::Profiler::Scope pass = profiler.value().gpu_scope(f.cmd, "refused");
      record_clear(f);
    }
    f.waits = c.waits;
    f.signals = c.signals;
    for (win::FrameWait& wait : f.waits) {
      if (wait.point.semaphore == &own) wait.point.value = f.number;
    }
    const vkc::Status ended = loop->end_frame(f);
    EXPECT_EQ(ended.domain(), vkc::Status::Code::InvalidArgument);
    EXPECT_EQ(loop->submitted(), f.number);
    EXPECT_TRUE(own.wait(f.number).ok());
  }

  auto frame = loop->begin_frame();
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  win::Frame& f = frame.value();
  {
    vg::Profiler::Scope pass = profiler.value().gpu_scope(f.cmd, "accepted");
    record_clear(f);
  }
  f.waits.push_back({{&own, f.number - 1}});
  EXPECT_TRUE(loop->end_frame(f).ok());
  // The next frame on f's slot publishes f's times.
  EXPECT_TRUE(run_frames(*loop, 2).ok());
  ASSERT_TRUE(device().wait_idle().ok());
  const vg::FrameMetrics& m = profiler.value().metrics();
  ASSERT_EQ(row_names(m),
            (std::vector<std::string>{"accepted", "frame wait", "submit"}));
  EXPECT_EQ(m.sections[0].has_gpu, profiler.value().gpu_timing());
  loop->set_profiler(nullptr);
}

// end_frame takes only the frame begin_frame handed out: a hand-built, wrong,
// repeated or another loop's frame is refused, consuming nothing, and the
// real frame still ends.
TEST_F(OffscreenFrameLoopTest, EndFrameRefusesAFrameItDidNotHandOut) {
  std::unique_ptr<win::FrameLoop> loop = make_loop(/*frames_in_flight=*/1);
  std::unique_ptr<win::FrameLoop> other = make_loop(/*frames_in_flight=*/1);
  ASSERT_NE(loop, nullptr);
  ASSERT_NE(other, nullptr);
  EXPECT_EQ(loop->end_frame(win::Frame{}).domain(),
            vkc::Status::Code::InvalidArgument);  // none handed out yet

  auto begun = loop->begin_frame();
  auto other_begun = other->begin_frame();
  ASSERT_TRUE(begun.ok()) << begun.status().message();
  ASSERT_TRUE(other_begun.ok()) << other_begun.status().message();
  const win::Frame frame = begun.value();
  std::vector<win::Frame> wrong(6, frame);
  wrong[0] = win::Frame{};
  wrong[1].slot = 1;  // past the one slot
  wrong[2].image_index = 1;
  wrong[3].number = frame.number + 1;
  wrong[4].cmd = VK_NULL_HANDLE;
  wrong[5] = other_begun.value();  // same number, slot and index
  for (size_t i = 0; i < wrong.size(); ++i) {
    EXPECT_EQ(loop->end_frame(wrong[i]).domain(),
              vkc::Status::Code::InvalidArgument)
        << "wrong[" << i << "]";
    EXPECT_EQ(loop->submitted(), 0u);
  }

  record_clear(frame);
  record_clear(other_begun.value());
  ASSERT_TRUE(loop->end_frame(frame).ok());
  ASSERT_TRUE(other->end_frame(other_begun.value()).ok());
  EXPECT_EQ(loop->end_frame(frame).domain(),
            vkc::Status::Code::InvalidArgument);  // already ended
  EXPECT_EQ(loop->submitted(), 1u);
  ASSERT_TRUE(device().wait_idle().ok());
}

// create refuses, before it allocates, what cannot make a loop: no slots, a
// negative pace, and the targets OffscreenTarget::create refuses.
TEST_F(OffscreenFrameLoopTest, CreateRefusesWhatCannotMakeALoop) {
  const auto refused = [&](const win::OffscreenLoopDesc& d, uint32_t n) {
    auto loop = win::FrameLoop::create(device(), allocator(), d, n);
    EXPECT_FALSE(loop.ok());
    return loop.ok() ? vkc::Status::Code::Ok : loop.status().domain();
  };
  EXPECT_EQ(refused(desc(), 0), vkc::Status::Code::InvalidArgument);
  EXPECT_EQ(refused(desc(std::chrono::nanoseconds(-1)), 2),
            vkc::Status::Code::InvalidArgument);
  win::OffscreenLoopDesc zero = desc();
  zero.target.extent = {0, 4};
  EXPECT_EQ(refused(zero, 2), vkc::Status::Code::InvalidArgument);
  win::OffscreenLoopDesc undefined = desc();
  undefined.target.color_format = VK_FORMAT_UNDEFINED;
  EXPECT_EQ(refused(undefined, 2), vkc::Status::Code::InvalidArgument);
}

// An offscreen loop follows no window: the extent-taking begin_frame ignores
// the extent -- a minimized window's 0 x 0 or a resize -- never skips, and
// never runs the recreate callback.
TEST_F(OffscreenFrameLoopTest, ManagedBeginFrameNeverSkips) {
  std::unique_ptr<win::FrameLoop> loop = make_loop();
  ASSERT_NE(loop, nullptr);
  int recreated = 0;
  loop->set_recreate_callback([&recreated](VkExtent2D) {
    ++recreated;
    return vkc::Status{};
  });
  for (const VkExtent2D extent : {VkExtent2D{0, 0}, VkExtent2D{320, 240},
                                  VkExtent2D{kExtent.width, kExtent.height}}) {
    auto frame = loop->begin_frame(extent);
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    ASSERT_TRUE(frame.value().has_value());
    EXPECT_EQ(frame.value()->target->extent().width, kExtent.width);
    EXPECT_EQ(frame.value()->target->extent().height, kExtent.height);
    record_clear(*frame.value());
    ASSERT_TRUE(loop->end_frame(*frame.value()).ok());
  }
  EXPECT_EQ(recreated, 0);
  EXPECT_EQ(loop->submitted(), 3u);
  ASSERT_TRUE(device().wait_idle().ok());
}

// Destruction drains in-flight frames: no device wait before the loop goes
// out of scope. Validation fails the test if the destructor freed command
// buffers or targets the GPU still used; the sanitizer job checks the rest.
TEST_F(OffscreenFrameLoopTest, DestructionDrainsInFlightFrames) {
  std::unique_ptr<win::FrameLoop> loop = make_loop();
  ASSERT_NE(loop, nullptr);
  for (int i = 0; i < 3; ++i) {
    auto frame = loop->begin_frame();
    ASSERT_TRUE(frame.ok()) << frame.status().message();
    record_clear(frame.value());
    frame.value().offscreen->record_readback(frame.value().cmd);
    ASSERT_TRUE(loop->end_frame(frame.value()).ok());
  }
  loop.reset();
}

}  // namespace

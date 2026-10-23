// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/profiler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/core/log.hpp"

namespace volumetric_kit::gfx {
namespace {

using Clock = std::chrono::steady_clock;

// A published row's name is never null (core::StageRow::name), so a stage
// opened without one is recorded -- and labelled -- under this.
constexpr const char* kUnnamedStage = "(unnamed)";

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

}  // namespace

// pImpl: keeps the timer and <chrono> out of the public header. One
// owns-everything struct; the owned Vulkan resources are the core's timer and
// the frames' tiles (move-only), so Profiler's move/dtor are defaulted around
// the unique_ptr.
//
// A frame's device time is one core::GpuTiles: a mark at begin_frame, at each
// GPU scope's open and close, and at end_frame. Each mark starts a tile named
// by its index, so a scope's time is the sum of the tiles from its open to its
// close, the scopes nested in it included, as a timestamp pair at its ends
// measured it.
struct Profiler::Impl {
  // A stage's core scope while it is open: on its thread's stack of open
  // scopes, so the core's waits inside it -- a fence, a timeline, a blocking
  // submit -- reach it as blocked time, as they reach a caller's
  // core::StageScope. Heap-held, as the stack holds it by address while
  // `current` reallocates; it times the stage's host span too.
  struct Host {
    core::StageMetrics rows;
    std::optional<core::StageScope> scope;
  };

  // A stage recorded in the current frame; its host time and wait are filled
  // in on finalize.
  struct Section {
    const char* name = nullptr;
    std::unique_ptr<Host> host;  // while open
    double cpu_ms = 0.0;
    double wait_ms = 0.0;
    bool finished = false;
    // Its open and close mark tiles: the stage's device time is tiles
    // [first_tile, end_tile).
    bool tiled = false;
    uint32_t first_tile = 0;
    uint32_t end_tile = 0;
    // The device half when no tile times it: None for a CPU stage.
    core::DeviceTime untimed = core::DeviceTime::None;
    // begin_debug_label was called for this stage, so finalize makes the
    // matching end_debug_label call (both no-ops without debug labels).
    bool label = false;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
  };

  // The frame buffered against one in-flight slot, resolved + published when
  // the slot recurs (its GPU submission has retired by then).
  struct SlotFrame {
    std::vector<Section> sections;
    core::StageMetrics loop;  // add_rows', shown after the sections
    core::GpuTiles tiles;
    double cpu_frame_ms = 0.0;
    double fps = 0.0;
    uint64_t mem_used = 0;
    uint64_t mem_budget = 0;
    bool pending = false;
  };

  // Capabilities + config, captured once at create().
  uint32_t frames_in_flight = 1;
  uint32_t max_gpu_sections = 0;  // the bound, lowered to what the timer holds
  // Borrowed: records the labels, and outlives the profiler.
  const core::Device* device = nullptr;
  // The core's timer: every frame in flight holds a range of its queries
  // until its slot recurs. Available exactly where the device can time.
  core::GpuTimer timer;
  // Tile t's name, its index; a frame has at most 2 * max_gpu_sections + 1.
  std::vector<std::string> tile_names;
  const core::Allocator* allocator = nullptr;  // borrowed; sampled at end_frame

  std::vector<SlotFrame> slots;  // one per in-flight slot

  // The frame currently being recorded (between begin_frame and end_frame).
  std::vector<Section> current;
  core::StageMetrics current_loop;
  core::GpuTiles current_tiles;
  uint32_t tile = 0;  // the open tile's index
  uint32_t current_slot = 0;
  VkCommandBuffer current_cmd = VK_NULL_HANDLE;
  uint32_t gpu_count = 0;
  Clock::time_point frame_start{};
  bool in_frame = false;
  bool warned_overflow = false;
  // The slot of the frame end_frame last ended, until the next begin_frame:
  // where add_rows goes between frames.
  std::optional<uint32_t> ended_slot;

  // fps smoothed (EMA) over the begin-to-begin cadence.
  Clock::time_point last_begin{};
  bool have_last_begin = false;
  double fps = 0.0;

  FrameMetrics published;  // latest resolved snapshot

  // Why a GPU stage no tile times reads untimed.
  core::DeviceTime untimed_state() const noexcept {
    return timer.available() ? core::DeviceTime::NotTimed
                             : core::DeviceTime::Unavailable;
  }

  // Start the next tile: a mark in the frame's command buffer.
  uint32_t boundary(VkCommandBuffer cmd) {
    ++tile;
    current_tiles.mark(cmd, tile_names[tile].c_str());
    return tile;
  }

  // Open a stage's core scope, which starts its host clock.
  static void open(Section& s) {
    s.host = std::make_unique<Host>();
    s.host->scope.emplace(s.host->rows, s.name);
  }

  // Stop a stage's host clock and, for a GPU stage, mark its close and close
  // its label. Idempotent so a finalize-then-destroy Scope is safe.
  void finalize(uint32_t index) {
    if (index >= current.size()) {
      return;
    }
    Section& s = current[index];
    if (s.finished) {
      return;
    }
    s.host->scope.reset();
    const core::StageRow& host = s.host->rows.rows().front();
    s.cpu_ms = host.cpu_ms;
    s.wait_ms = host.wait_ms;
    s.host.reset();
    if (s.tiled) {
      s.end_tile = boundary(s.cmd);
    }
    if (s.label) {
      device->end_debug_label(s.cmd, s.name);
    }
    s.finished = true;
  }

  // Resolve a retired slot's tiles and pair them with its buffered CPU halves
  // into the published snapshot.
  void publish(SlotFrame& f) {
    FrameMetrics m;
    m.cpu_frame_ms = f.cpu_frame_ms;
    m.fps = f.fps;
    m.memory_used_bytes = f.mem_used;
    m.memory_budget_bytes = f.mem_budget;

    // Never waits: the caller waited the slot's fence, and a mark not yet
    // readable leaves its tiles NotTimed.
    f.tiles.resolve();
    std::vector<double> tile_ms(tile_names.size(), 0.0);
    std::vector<core::DeviceTime> tile_state(tile_names.size(),
                                             core::DeviceTime::NotTimed);
    for (const core::StageRow& row : f.tiles.rows()) {
      const unsigned long t = std::strtoul(row.name, nullptr, 10);
      if (t < tile_names.size()) {
        tile_ms[t] = row.gpu_ms;
        tile_state[t] = row.device;
      }
    }

    m.sections.reserve(f.sections.size() + f.loop.rows().size());
    for (const Section& s : f.sections) {
      core::StageRow out;
      out.name = s.name;
      out.cpu_ms = s.cpu_ms;
      out.wait_ms = s.wait_ms;
      out.device = s.untimed;
      if (s.tiled) {
        out.device = core::DeviceTime::Measured;
        for (uint32_t t = s.first_tile; t < s.end_tile; ++t) {
          out.gpu_ms += tile_ms[t];
          out.device = std::max(out.device, tile_state[t]);
        }
        if (out.device != core::DeviceTime::Measured) out.gpu_ms = 0.0;
      }
      out.has_gpu = out.device == core::DeviceTime::Measured;
      m.sections.push_back(out);
    }
    m.sections.insert(m.sections.end(), f.loop.rows().begin(),
                      f.loop.rows().end());
    published = std::move(m);
  }
};

core::Result<Profiler> Profiler::create(const core::Device& device,
                                        const ProfilerConfig& config) {
  if (config.frames_in_flight == 0) {
    return core::Status::invalid_argument(
        "Profiler::create: frames_in_flight is zero");
  }
  if (config.max_gpu_sections_per_frame == 0) {
    return core::Status::invalid_argument(
        "Profiler::create: max_gpu_sections_per_frame is zero");
  }
  // The profiler times frames the renderer records: a device made for another
  // library may lack the renderer's floor, a graphics queue included.
  VKC_TRY(device.check_enabled(device_requirements())
              .with_context("Profiler::create"));

  // A frame of n GPU scopes takes 2n + 2 marks, n + 1 of the timer's pairs,
  // and every frame in flight holds its own: lower the bound until they fit.
  constexpr uint32_t kSpans = core::GpuTimer::kMaxSpans;
  const uint32_t per_frame = kSpans / config.frames_in_flight;
  const uint32_t sections =
      per_frame == 0
          ? 0
          : std::min(config.max_gpu_sections_per_frame, per_frame - 1);
  if (sections < config.max_gpu_sections_per_frame) {
    log_message(core::LogLevel::Warning,
                "Profiler::create: max_gpu_sections_per_frame lowered to " +
                    std::to_string(sections) + ", the most the core's timer " +
                    "holds for " + std::to_string(config.frames_in_flight) +
                    " frames in flight");
  }

  auto impl = std::make_unique<Impl>();
  impl->frames_in_flight = config.frames_in_flight;
  impl->max_gpu_sections = sections;
  impl->device = &device;
  impl->slots.resize(config.frames_in_flight);
  impl->tile_names.reserve(2 * static_cast<size_t>(sections) + 1);
  for (uint32_t t = 0; t <= 2 * sections; ++t) {
    impl->tile_names.push_back(std::to_string(t));
  }
  // Unavailable rather than failing where the queue has no timestamps or the
  // pool will not allocate (logged): every GPU scope is then a CPU-timed
  // (and possibly labelled) no-op on the GPU clock.
  const uint32_t spans =
      std::max(1U, std::min(kSpans, config.frames_in_flight * (sections + 1)));
  VKC_ASSIGN(impl->timer, core::GpuTimer::create(device, spans));

  Profiler profiler;
  profiler.impl_ = std::move(impl);
  return profiler;
}

Profiler::Profiler() noexcept = default;
Profiler::~Profiler() = default;
Profiler::Profiler(Profiler&&) noexcept = default;
Profiler& Profiler::operator=(Profiler&&) noexcept = default;

Profiler::Scope::~Scope() {
  if (impl_ != nullptr) {
    impl_->finalize(index_);
  }
}

Profiler::Scope::Scope(Scope&& other) noexcept
    : impl_(other.impl_), index_(other.index_) {
  other.impl_ = nullptr;
  other.index_ = 0;
}

Profiler::Scope& Profiler::Scope::operator=(Scope&& other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      impl_->finalize(index_);  // close our own stage before adopting other's
    }
    impl_ = other.impl_;
    index_ = other.index_;
    other.impl_ = nullptr;
    other.index_ = 0;
  }
  return *this;
}

void Profiler::set_memory_source(const core::Allocator* allocator) noexcept {
  if (impl_) {
    impl_->allocator = allocator;
  }
}

void Profiler::begin_frame(uint32_t slot, VkCommandBuffer cmd) noexcept {
  if (!impl_) {
    return;
  }
  Impl& d = *impl_;
  d.ended_slot.reset();
  if (slot >= d.frames_in_flight) {
    // Out-of-range slot is a misuse; stay out of a frame so scopes are inert
    // rather than indexing past the slot's query range.
    d.in_frame = false;
    return;
  }

  // The frame this slot last carried has retired (the caller waited its fence);
  // resolve its GPU ticks and publish, then clear the slot for reuse.
  Impl::SlotFrame& f = d.slots[slot];
  if (f.pending) {
    d.publish(f);
    f = Impl::SlotFrame{};
  }

  const Clock::time_point now = Clock::now();
  if (d.have_last_begin) {
    const double delta_ms =
        std::chrono::duration<double, std::milli>(now - d.last_begin).count();
    if (delta_ms > 0.0) {
      const double instant = 1000.0 / delta_ms;
      d.fps = d.fps > 0.0 ? (0.9 * d.fps + 0.1 * instant) : instant;
    }
  }
  d.last_begin = now;
  d.have_last_begin = true;

  d.current.clear();
  d.current_loop.clear();
  d.current_slot = slot;
  d.current_cmd = cmd;
  d.gpu_count = 0;
  d.tile = 0;
  d.frame_start = now;
  d.in_frame = true;

  // A frame begun but never ended -- its command buffer never ran -- returns
  // its range first, so this frame finds room. Then take the frame's range up
  // front, with its reset and first mark, outside any render pass. Tile 0
  // holds what the frame records before its first GPU scope.
  d.current_tiles = core::GpuTiles{};
  if (cmd != VK_NULL_HANDLE) {
    d.current_tiles =
        core::GpuTiles::begin(d.timer, cmd, 2 * d.max_gpu_sections + 1);
    d.current_tiles.mark(cmd, d.tile_names[0].c_str());
  }
}

void Profiler::end_frame() noexcept {
  if (!impl_ || !impl_->in_frame) {
    return;
  }
  Impl& d = *impl_;

  uint64_t used = 0;
  uint64_t budget = 0;
  if (d.allocator != nullptr) {
    const core::MemoryStats stats = d.allocator->memory_stats();
    for (uint32_t i = 0; i < stats.heap_count; ++i) {
      used += stats.heaps[i].usage_bytes;
      budget += stats.heaps[i].budget_bytes;
    }
  }

  // Finalize any stage whose Scope is still open (close its label, mark its
  // close) so the submitted command buffer never carries an unbalanced
  // debug-utils region and every stage's device time ends. Idempotent: a stage
  // its Scope already closed is skipped via Section::finished.
  for (uint32_t i = 0; i < static_cast<uint32_t>(d.current.size()); ++i) {
    d.finalize(i);
  }
  d.current_tiles.end(d.current_cmd);

  Impl::SlotFrame& f = d.slots[d.current_slot];
  f.sections = std::move(d.current);
  f.loop = std::move(d.current_loop);
  f.tiles = std::move(d.current_tiles);
  f.cpu_frame_ms = ms_since(d.frame_start);
  f.fps = d.fps;
  f.mem_used = used;
  f.mem_budget = budget;
  f.pending = true;

  d.current.clear();  // re-empty the moved-from accumulators
  d.current_loop.clear();
  d.in_frame = false;
  d.ended_slot = d.current_slot;
}

void Profiler::add_rows(const core::StageMetrics& rows) {
  if (!impl_) {
    return;
  }
  Impl& d = *impl_;
  if (d.in_frame) {
    d.current_loop.merge(rows);
  } else if (d.ended_slot && d.slots[*d.ended_slot].pending) {
    d.slots[*d.ended_slot].loop.merge(rows);
  }
}

Profiler::Scope Profiler::cpu_scope(const char* name) {
  if (!impl_ || !impl_->in_frame) {
    return Scope{};
  }
  Impl& d = *impl_;
  Impl::Section s;
  s.name = name != nullptr ? name : kUnnamedStage;
  Impl::open(s);
  const uint32_t index = static_cast<uint32_t>(d.current.size());
  d.current.push_back(std::move(s));
  return Scope(impl_.get(), index);
}

Profiler::Scope Profiler::gpu_scope(VkCommandBuffer cmd, const char* name) {
  if (!impl_ || !impl_->in_frame || cmd == VK_NULL_HANDLE) {
    return Scope{};
  }
  Impl& d = *impl_;
  Impl::Section s;
  s.name = name != nullptr ? name : kUnnamedStage;
  Impl::open(s);
  s.cmd = cmd;
  s.untimed = d.untimed_state();

  // GPU work (label + marks) records into the frame's command buffer — the
  // one begin_frame took the tiles on. A cmd that does not match it (a
  // CPU-only frame begun with VK_NULL_HANDLE, or simply a different buffer)
  // leaves the stage CPU-timed only, so no mark lands in another buffer.
  const bool records_gpu = (cmd == d.current_cmd);

  // A label is independent of timestamp timing. The device emits it where
  // VK_EXT_debug_utils is enabled, and skips the begin/end pair alike where it
  // is not. s.name is never null, as a label's name must not be.
  if (records_gpu) {
    d.device->begin_debug_label(cmd, s.name);
    s.label = true;
  }

  // Mark the stage's open while the per-frame bound holds; past it the stage
  // is CPU-only. On a timer that cannot time the marks write nothing, and the
  // tiles read Unavailable.
  if (records_gpu) {
    if (d.gpu_count < d.max_gpu_sections) {
      s.tiled = true;
      s.first_tile = d.boundary(cmd);
      ++d.gpu_count;
    } else if (d.timer.available() && !d.warned_overflow) {
      d.warned_overflow = true;
      log_message(
          core::LogLevel::Warning,
          "Profiler: gpu_scope calls exceeded "
          "max_gpu_sections_per_frame; extra stages are CPU-timed only");
    }
  }

  const uint32_t index = static_cast<uint32_t>(d.current.size());
  d.current.push_back(std::move(s));
  return Scope(impl_.get(), index);
}

const FrameMetrics& Profiler::metrics() const noexcept {
  static const FrameMetrics kEmpty;
  return impl_ ? impl_->published : kEmpty;
}

bool Profiler::gpu_timing() const noexcept {
  return impl_ && impl_->timer.available();
}

}  // namespace volumetric_kit::gfx

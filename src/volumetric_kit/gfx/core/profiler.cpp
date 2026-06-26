// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/profiler.hpp"

#include <chrono>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/impl/debug_utils_table.hpp"
#include "volumetric_kit/gfx/core/log.hpp"
#include "volumetric_kit/gfx/core/query_pool.hpp"

namespace volumetric_kit::gfx {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

}  // namespace

// pImpl: keeps the timestamp pool, the marker table, and <chrono> out of the
// public header. One owns-everything struct; the only owned Vulkan resource is
// the QueryPool (move-only), so Profiler's move/dtor are defaulted around the
// unique_ptr.
struct Profiler::Impl {
  // A stage recorded in the current frame; CPU time is filled in on finalize.
  struct Section {
    const char* name = nullptr;
    Clock::time_point cpu_start{};
    double cpu_ms = 0.0;
    bool finished = false;
    bool has_gpu = false;  // a timestamp pair was written for this stage
    bool label = false;    // a debug-utils region was opened for this stage
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    uint32_t begin_query = 0;  // absolute query index; the end query is +1
  };

  // The frame buffered against one in-flight slot, resolved + published when
  // the slot recurs (its GPU submission has retired by then).
  struct SlotFrame {
    std::vector<Section> sections;
    double cpu_frame_ms = 0.0;
    double fps = 0.0;
    uint64_t mem_used = 0;
    uint64_t mem_budget = 0;
    uint32_t query_base = 0;
    uint32_t gpu_count = 0;
    bool pending = false;
  };

  // Capabilities + config, captured once at create().
  uint32_t frames_in_flight = 1;
  uint32_t max_gpu_sections = 0;
  uint32_t valid_bits = 0;
  bool gpu_timing = false;
  float ts_period_ns = 0.0f;
  DebugUtilsTable table;
  // QueryPool is create-only (no public default ctor), so hold it optionally;
  // engaged only where gpu_timing is supported.
  std::optional<QueryPool> query_pool;
  const Allocator* allocator = nullptr;  // borrowed; sampled at end_frame

  std::vector<SlotFrame> slots;  // one per in-flight slot

  // The frame currently being recorded (between begin_frame and end_frame).
  std::vector<Section> current;
  uint32_t current_slot = 0;
  VkCommandBuffer current_cmd = VK_NULL_HANDLE;
  uint32_t gpu_count = 0;
  Clock::time_point frame_start{};
  bool in_frame = false;
  bool warned_overflow = false;

  // fps smoothed (EMA) over the begin-to-begin cadence.
  Clock::time_point last_begin{};
  bool have_last_begin = false;
  double fps = 0.0;

  FrameMetrics published;  // latest resolved snapshot

  // Base query index for a slot's range: two queries (begin + end) per section,
  // max_gpu_sections sections per slot. The single source of the pool layout.
  uint32_t slot_query_base(uint32_t slot) const {
    return slot * max_gpu_sections * 2;
  }

  // Stop a stage's CPU clock and, for a GPU stage, record its end timestamp and
  // close its label. Idempotent so a finalize-then-destroy Scope is safe.
  void finalize(uint32_t index) {
    if (index >= current.size()) {
      return;
    }
    Section& s = current[index];
    if (s.finished) {
      return;
    }
    s.cpu_ms = ms_since(s.cpu_start);
    if (s.has_gpu && query_pool) {
      query_pool->cmd_write_timestamp(
          s.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.begin_query + 1);
    }
    if (s.label) {
      table.cmd_end(s.cmd);
    }
    s.finished = true;
  }

  // Resolve a retired slot's GPU ticks and pair them with its buffered CPU
  // halves into the published snapshot.
  void publish(const SlotFrame& f) {
    FrameMetrics m;
    m.cpu_frame_ms = f.cpu_frame_ms;
    m.fps = f.fps;
    m.memory_used_bytes = f.mem_used;
    m.memory_budget_bytes = f.mem_budget;

    std::vector<uint64_t> ticks;
    bool gpu_ok = false;
    if (f.gpu_count > 0 && query_pool) {
      ticks.resize(static_cast<size_t>(f.gpu_count) * 2);
      gpu_ok =
          query_pool->read_results(f.query_base, f.gpu_count * 2, ticks.data())
              .ok();
    }
    // Only the low `valid_bits` of each tick are meaningful; mask before
    // subtracting so high-bit noise never corrupts the delta.
    const uint64_t mask = valid_bits >= 64
                              ? ~uint64_t{0}
                              : (uint64_t{1} << valid_bits) - uint64_t{1};

    m.sections.reserve(f.sections.size());
    for (const Section& s : f.sections) {
      FrameMetrics::Section out;
      out.name = s.name;
      out.cpu_ms = s.cpu_ms;
      if (s.has_gpu && gpu_ok) {
        const uint32_t local = s.begin_query - f.query_base;
        const uint64_t begin = ticks[local] & mask;
        const uint64_t end = ticks[local + 1] & mask;
        out.gpu_ms = ticks_to_ms(end - begin, ts_period_ns);
        out.has_gpu = true;
      }
      m.sections.push_back(out);
    }
    published = std::move(m);
  }
};

Result<Profiler> Profiler::create(const Device& device,
                                  const ProfilerConfig& config) {
  if (config.frames_in_flight == 0) {
    return Status::invalid_argument(
        "Profiler::create: frames_in_flight is zero");
  }
  if (config.max_gpu_sections_per_frame == 0) {
    return Status::invalid_argument(
        "Profiler::create: max_gpu_sections_per_frame is zero");
  }

  auto impl = std::make_unique<Impl>();
  impl->frames_in_flight = config.frames_in_flight;
  impl->max_gpu_sections = config.max_gpu_sections_per_frame;
  impl->valid_bits = device.graphics_timestamp_valid_bits();
  impl->gpu_timing = impl->valid_bits != 0;
  impl->ts_period_ns = device.caps().limits().timestampPeriod;
  impl->table = device.debug_utils();
  impl->slots.resize(config.frames_in_flight);

  // The timestamp pool exists only where timing is supported; without it every
  // GPU scope is a CPU-timed (and possibly labelled) no-op on the GPU clock.
  if (impl->gpu_timing) {
    const uint32_t query_count =
        config.frames_in_flight * config.max_gpu_sections_per_frame * 2;
    Result<QueryPool> pool = QueryPool::create(device.handle(), query_count);
    if (!pool) {
      return pool.status();
    }
    impl->query_pool.emplace(std::move(pool).value());
  }

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

void Profiler::set_memory_source(const Allocator* allocator) noexcept {
  if (impl_) {
    impl_->allocator = allocator;
  }
}

void Profiler::begin_frame(uint32_t slot, VkCommandBuffer cmd) noexcept {
  if (!impl_) {
    return;
  }
  Impl& d = *impl_;
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
  d.current_slot = slot;
  d.current_cmd = cmd;
  d.gpu_count = 0;
  d.frame_start = now;
  d.in_frame = true;

  // Reset the slot's whole timestamp range up front (outside any render pass)
  // so each gpu_scope can write into it.
  if (d.gpu_timing && cmd != VK_NULL_HANDLE && d.query_pool) {
    const uint32_t base = d.slot_query_base(slot);
    d.query_pool->cmd_reset(cmd, base, d.max_gpu_sections * 2);
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
    const MemoryStats stats = d.allocator->memory_stats();
    for (uint32_t i = 0; i < stats.heap_count; ++i) {
      used += stats.heaps[i].usage_bytes;
      budget += stats.heaps[i].budget_bytes;
    }
  }

  // Finalize any stage whose Scope is still open (close its label, write its
  // end timestamp) so the submitted command buffer never carries an unbalanced
  // debug-utils region and no GPU end timestamp is left unwritten. Idempotent:
  // a stage its Scope already closed is skipped via Section::finished.
  for (uint32_t i = 0; i < static_cast<uint32_t>(d.current.size()); ++i) {
    d.finalize(i);
  }

  Impl::SlotFrame& f = d.slots[d.current_slot];
  f.sections = std::move(d.current);
  f.cpu_frame_ms = ms_since(d.frame_start);
  f.fps = d.fps;
  f.mem_used = used;
  f.mem_budget = budget;
  f.query_base = d.slot_query_base(d.current_slot);
  f.gpu_count = d.gpu_count;
  f.pending = true;

  d.current.clear();  // re-empty the moved-from accumulator
  d.in_frame = false;
}

Profiler::Scope Profiler::cpu_scope(const char* name) {
  if (!impl_ || !impl_->in_frame) {
    return Scope{};
  }
  Impl& d = *impl_;
  Impl::Section s;
  s.name = name;
  s.cpu_start = Clock::now();
  const uint32_t index = static_cast<uint32_t>(d.current.size());
  d.current.push_back(s);
  return Scope(impl_.get(), index);
}

Profiler::Scope Profiler::gpu_scope(VkCommandBuffer cmd, const char* name) {
  if (!impl_ || !impl_->in_frame || cmd == VK_NULL_HANDLE) {
    return Scope{};
  }
  Impl& d = *impl_;
  Impl::Section s;
  s.name = name;
  s.cpu_start = Clock::now();
  s.cmd = cmd;

  // GPU work (label + timestamps) records into the frame's command buffer — the
  // one begin_frame reset the query range on. A cmd that does not match it (a
  // CPU-only frame begun with VK_NULL_HANDLE, or simply a different buffer)
  // leaves the stage CPU-timed only, so a timestamp is never written into an
  // unreset query.
  const bool records_gpu = (cmd == d.current_cmd);

  // A label is independent of timestamp timing: open it whenever debug-utils is
  // active, even if no GPU timing is available. A null name is skipped —
  // VkDebugUtilsLabelEXT::pLabelName must be non-null.
  if (records_gpu && d.table.active() && name != nullptr) {
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    d.table.cmd_begin(cmd, &label);
    s.label = true;
  }

  // Write the begin timestamp when timing is available and the per-frame GPU
  // budget is not yet exhausted; otherwise this stage is CPU-only.
  if (records_gpu && d.gpu_timing && d.query_pool) {
    if (d.gpu_count < d.max_gpu_sections) {
      s.begin_query = d.slot_query_base(d.current_slot) + d.gpu_count * 2;
      s.has_gpu = true;
      d.query_pool->cmd_write_timestamp(
          cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.begin_query);
      ++d.gpu_count;
    } else if (!d.warned_overflow) {
      d.warned_overflow = true;
      log_message(
          LogLevel::Warning,
          "Profiler: gpu_scope calls exceeded "
          "max_gpu_sections_per_frame; extra stages are CPU-timed only");
    }
  }

  const uint32_t index = static_cast<uint32_t>(d.current.size());
  d.current.push_back(s);
  return Scope(impl_.get(), index);
}

const FrameMetrics& Profiler::metrics() const noexcept {
  static const FrameMetrics kEmpty;
  return impl_ ? impl_->published : kEmpty;
}

bool Profiler::gpu_timing() const noexcept {
  return impl_ && impl_->gpu_timing;
}

}  // namespace volumetric_kit::gfx

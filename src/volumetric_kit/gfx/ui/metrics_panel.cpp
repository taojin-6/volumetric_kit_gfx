// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/metrics_panel.hpp"

#include <algorithm>
#include <string>

#include "imgui.h"

namespace volumetric_kit::gfx::ui {
namespace {

// What a row's device half holds. A row that sets only has_gpu, built before
// `device` existed, is measured, as the core's StageMetrics reads it.
core::DeviceTime device_of(const core::StageRow& row) {
  if (row.device == core::DeviceTime::None && row.has_gpu) {
    return core::DeviceTime::Measured;
  }
  return row.device;
}

// The host span less the part of it spent blocked.
double busy_ms(const core::StageRow& row) {
  return std::max(0.0, row.cpu_ms - row.wait_ms);
}

// A row that only waited, as a core::WaitScope's: its busy time would print
// 0.000, which a mean's rounding may leave a few ulps above 0.
bool wait_only(const core::StageRow& row) {
  return row.wait_ms > 0.0 && busy_ms(row) < 0.0005;
}

void text_cell(bool dim, const char* text) {
  if (dim) {
    ImGui::TextDisabled("%s", text);
  } else {
    ImGui::TextUnformatted(text);
  }
}

void ms_cell(bool dim, double ms) {
  if (dim) {
    ImGui::TextDisabled("%.3f", ms);
  } else {
    ImGui::Text("%.3f", ms);
  }
}

// One profiler snapshot is one frame's; it states a window only where a stage
// has a device half to say the frame's timing of.
StageWindow frame_window(const std::vector<core::StageRow>& rows) {
  StageWindow window;
  for (const core::StageRow& row : rows) {
    const core::DeviceTime device = device_of(row);
    if (device == core::DeviceTime::None) continue;
    window.frames = 1;
    if (device == core::DeviceTime::Measured) window.timed = 1;
  }
  return window;
}

}  // namespace

void draw_metrics_panel(const FrameMetrics& metrics, const char* title,
                        bool* open) {
  // ImGui::Begin asserts on a null window name; fall back to the default title
  // so an explicit nullptr can't crash the panel.
  if (title == nullptr) title = "Performance";
  // Begin returns false when the window is collapsed/clipped, but End must be
  // called unconditionally; guard only the widget building.
  if (ImGui::Begin(title, open)) {
    // The period from the rate; the recording time is a part of it.
    const double period_ms = metrics.fps > 0.0 ? 1000.0 / metrics.fps : 0.0;
    ImGui::Text("%.1f fps · %.2f ms a frame · %.2f ms recording", metrics.fps,
                period_ms, metrics.cpu_frame_ms);

    if (metrics.memory_budget_bytes > 0) {
      constexpr double kToMiB = 1.0 / (1024.0 * 1024.0);
      ImGui::Text("Memory: %.1f / %.1f MiB",
                  static_cast<double>(metrics.memory_used_bytes) * kToMiB,
                  static_cast<double>(metrics.memory_budget_bytes) * kToMiB);
    }

    ImGui::Separator();
    draw_stage_table("stages", metrics.sections,
                     frame_window(metrics.sections));
  }
  ImGui::End();
}

void draw_stage_table(const char* id, const std::vector<core::StageRow>& rows,
                      const StageWindow& window) {
  std::string line;
  if (window.frames > 0) {
    line = std::string("per ") +
           (window.unit != nullptr ? window.unit : "frame") + " · last " +
           std::to_string(window.frames) + " · GPU timed in " +
           std::to_string(window.timed);
  }
  if (window.note != nullptr) {
    line += (line.empty() ? "" : " · ") + std::string(window.note);
  }
  if (!line.empty()) {
    ImGui::TextDisabled("%s", line.c_str());
  }
  if (rows.empty()) {
    ImGui::TextDisabled("(no timed stages)");
    return;
  }
  const bool waited =
      std::any_of(rows.begin(), rows.end(),
                  [](const core::StageRow& row) { return row.wait_ms > 0.0; });
  // SizingFixedFit (columns size to content), not SizingStretchProp: the
  // latter divides the available width among columns, which is NaN in an
  // auto-sizing window before its width is known (its first, content-measuring
  // frame) — a float-to-int cast that traps under UBSan.
  if (!ImGui::BeginTable(id, waited ? 4 : 3,
                         ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_SizingFixedFit)) {
    return;
  }
  ImGui::TableSetupColumn("Stage");
  ImGui::TableSetupColumn("host ms");
  if (waited) ImGui::TableSetupColumn("wait ms");
  ImGui::TableSetupColumn("GPU ms");
  ImGui::TableHeadersRow();
  for (const core::StageRow& row : rows) {
    const bool dim = wait_only(row);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    text_cell(dim, row.name != nullptr ? row.name : "");
    ImGui::TableNextColumn();
    if (!dim) ms_cell(false, busy_ms(row));
    if (waited) {
      ImGui::TableNextColumn();
      if (row.wait_ms > 0.0) ms_cell(dim, row.wait_ms);
    }
    ImGui::TableNextColumn();
    // Each state its own cell, so an unmeasured one says why rather than
    // showing a meaningless 0.000; a host-only row's stays blank.
    switch (device_of(row)) {
      case core::DeviceTime::Measured:
        ms_cell(dim, row.gpu_ms);
        break;
      case core::DeviceTime::NotTimed:
        ImGui::TextDisabled("n/t");
        break;
      case core::DeviceTime::Unavailable:
        ImGui::TextDisabled("n/a");
        break;
      case core::DeviceTime::None:
        break;
    }
  }
  ImGui::EndTable();
}

}  // namespace volumetric_kit::gfx::ui

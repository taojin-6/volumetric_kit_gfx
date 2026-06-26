// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/metrics_panel.hpp"

#include "imgui.h"

namespace volumetric_kit::gfx::ui {

void draw_metrics_panel(const FrameMetrics& metrics, const char* title,
                        bool* open) {
  // Begin returns false when the window is collapsed/clipped, but End must be
  // called unconditionally; guard only the widget building.
  if (ImGui::Begin(title, open)) {
    ImGui::Text("%.1f FPS  (%.2f ms/frame)", metrics.fps, metrics.cpu_frame_ms);

    if (metrics.memory_budget_bytes > 0) {
      constexpr double kToMiB = 1.0 / (1024.0 * 1024.0);
      ImGui::Text("Memory: %.1f / %.1f MiB",
                  static_cast<double>(metrics.memory_used_bytes) * kToMiB,
                  static_cast<double>(metrics.memory_budget_bytes) * kToMiB);
    }

    ImGui::Separator();

    if (metrics.sections.empty()) {
      ImGui::TextDisabled("(no timed stages)");
      // SizingFixedFit (columns size to content), not SizingStretchProp: the
      // latter divides the available width among columns, which is NaN in an
      // auto-sizing window before its width is known (its first, content-
      // measuring frame) — a float-to-int cast that traps under UBSan.
    } else if (ImGui::BeginTable("stages", 3,
                                 ImGuiTableFlags_Borders |
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_SizingFixedFit)) {
      ImGui::TableSetupColumn("Stage");
      ImGui::TableSetupColumn("CPU ms");
      ImGui::TableSetupColumn("GPU ms");
      ImGui::TableHeadersRow();
      for (const FrameMetrics::Section& section : metrics.sections) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(section.name != nullptr ? section.name : "");
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", section.cpu_ms);
        ImGui::TableNextColumn();
        // has_gpu is the capability report: blank the GPU column for a CPU-only
        // stage rather than printing a meaningless 0.000.
        if (section.has_gpu) {
          ImGui::Text("%.3f", section.gpu_ms);
        } else {
          ImGui::TextDisabled("-");
        }
      }
      ImGui::EndTable();
    }
  }
  ImGui::End();
}

}  // namespace volumetric_kit::gfx::ui

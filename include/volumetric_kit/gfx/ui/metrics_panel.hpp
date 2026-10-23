// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file metrics_panel.hpp
/// @brief A Dear ImGui debug panel for a @ref FrameMetrics snapshot, and its
///        table of stages.

#include <cstdint>
#include <vector>

#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/gfx/core/frame_metrics.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief What a stage table's figures cover, stated on the line above it:
///        the frames they are taken over, how many of those had their device
///        work timed, and what the application adds -- when it times, how its
///        clocks are calibrated -- which only it knows.
///
/// @code
/// ui::StageWindow window;
/// window.frames = 150;  // the rows are means over the last 150 frames
/// window.timed = 150;   // each of them device-timed
/// window.unit = "stream frame";
/// window.note = "GPU timing while this panel is open";
/// ui::draw_stage_table("worker", rows, window);
/// // draws "per stream frame · last 150 · GPU timed in 150 · GPU timing while
/// // this panel is open" above the table
/// @endcode
struct StageWindow {
  /// The frames the figures are taken over: 1 for one frame's, more for
  /// their mean. 0, the default, states no counts.
  std::uint32_t frames = 0;
  /// Of @ref frames, how many had their device work timed.
  std::uint32_t timed = 0;
  /// What one frame is, e.g. `"display frame"`; null reads `"frame"`.
  const char* unit = nullptr;
  /// Text drawn after the counts, e.g. the timing mode and the clocks'
  /// calibration; read only while the table is drawn. Null, the default,
  /// adds none; with no counts and no note, no line is drawn.
  const char* note = nullptr;
};

/// @brief Build a debug panel for a profiler snapshot with `ImGui::` calls.
///
/// Emits one ImGui window: the frame rate, the frame period it gives and the
/// frame's recording time (`FrameMetrics::cpu_frame_ms`); the aggregate
/// device memory (only when a budget is reported); and its stages' @ref
/// draw_stage_table, stated as one frame's where a stage has a device half.
/// It only *builds* widgets into the ImGui frame the caller is already
/// driving (it calls neither `NewFrame` nor `Render`), so the caller frames it
/// — e.g. between @ref ImGuiOverlay::new_frame and @ref ImGuiOverlay::render.
///
/// @param metrics  The snapshot to display (e.g. from @ref Profiler::metrics).
/// @param title    The ImGui window title and id.
/// @param open     Optional `bool` the window's close button toggles; `nullptr`
///                 gives a panel with no close button.
/// @pre An ImGui frame is active this frame (`new_frame` was called) with its
///      context current.
///
/// @code
/// overlay.new_frame();
/// ui::draw_metrics_panel(profiler.metrics());
/// target->begin(cmd, {});
/// overlay.render(cmd);
/// target->end(cmd);
/// @endcode
VG_UI_API void draw_metrics_panel(const FrameMetrics& metrics,
                                  const char* title = "Performance",
                                  bool* open = nullptr);

/// @brief Draw @p rows as a table in the caller's window, opening none of its
///        own, under a line stating @p window.
///
/// Its columns are Stage, host ms, wait ms and GPU ms:
/// - host is the row's busy time, `cpu_ms - wait_ms`, so it never holds a
///   wait; a row that only waited leaves it blank and is dimmed;
/// - wait is the time the row blocked, blank where it did not; the column is
///   drawn only when a row waited;
/// - GPU says what the row's `core::DeviceTime` holds: the time when
///   measured, `n/t` when its device work ran untimed, `n/a` when the queue
///   cannot time, and blank for a host-only row. A row that sets only
///   `has_gpu` reads as measured.
///
/// No rows draw "(no timed stages)".
///
/// @code
/// ImGui::Begin("##stats", nullptr, ImGuiWindowFlags_NoDecoration);
/// ui::draw_stage_table("stages", rows);
/// ImGui::End();
/// @endcode
/// @param id      The table's ImGui id, unique in its window.
/// @param rows    The rows, in order.
/// @param window  What the figures cover; the default states nothing.
/// @pre An ImGui frame is active, with its context current, inside a window.
VG_UI_API void draw_stage_table(const char* id,
                                const std::vector<core::StageRow>& rows,
                                const StageWindow& window = {});

}  // namespace volumetric_kit::gfx::ui

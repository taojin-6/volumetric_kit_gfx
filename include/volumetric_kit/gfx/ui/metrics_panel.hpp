// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file metrics_panel.hpp
/// @brief A Dear ImGui debug panel for a @ref FrameMetrics snapshot, and its
///        table of stages.

#include <vector>

#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/gfx/core/frame_metrics.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief Build a debug panel for a profiler snapshot with `ImGui::` calls.
///
/// Emits one ImGui window: the frame's fps and CPU frame time, the aggregate
/// device memory (only when a budget is reported), and its stages' @ref
/// draw_stage_table. It only *builds* widgets into the ImGui frame the
/// caller is already driving (it calls neither `NewFrame` nor `Render`), so the
/// caller frames it — e.g. between @ref ImGuiOverlay::new_frame and @ref
/// ImGuiOverlay::render.
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

/// @brief Draw a Stage / CPU ms / GPU ms table of @p rows in the caller's
///        window, opening none of its own: `-` in the GPU column for a row
///        with no device span (`has_gpu` false), "(no timed stages)" for no
///        rows.
///
/// @code
/// ImGui::Begin("##stats", nullptr, ImGuiWindowFlags_NoDecoration);
/// ui::draw_stage_table("stages", rows);
/// ImGui::End();
/// @endcode
/// @param id    The table's ImGui id, unique in its window.
/// @param rows  The rows, in order.
/// @pre An ImGui frame is active, with its context current, inside a window.
VG_UI_API void draw_stage_table(const char* id,
                                const std::vector<core::StageRow>& rows);

}  // namespace volumetric_kit::gfx::ui

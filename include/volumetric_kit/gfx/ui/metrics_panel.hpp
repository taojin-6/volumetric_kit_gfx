// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file metrics_panel.hpp
/// @brief A Dear ImGui debug panel for a @ref FrameMetrics snapshot.

#include "volumetric_kit/gfx/core/frame_metrics.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief Build a debug panel for a profiler snapshot with `ImGui::` calls.
///
/// Emits one ImGui window: the frame's fps and CPU frame time, the aggregate
/// device memory (only when a budget is reported), and a per-stage table of CPU
/// and GPU milliseconds; the GPU column stays blank for a CPU-only stage (where
/// `has_gpu` is false). It only *builds* widgets into the ImGui frame the
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

}  // namespace volumetric_kit::gfx::ui

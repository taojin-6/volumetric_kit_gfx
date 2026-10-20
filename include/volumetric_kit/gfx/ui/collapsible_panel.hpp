// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file collapsible_panel.hpp
/// @brief A Dear ImGui panel that a tap or click on its header opens and
///        closes, and whose body leaves the pointer to the scene.

#include "imgui.h"
#include "volumetric_kit/gfx/ui/export.hpp"

namespace volumetric_kit::gfx::ui {

/// @brief Begin a panel that a tap or click on its header opens and closes.
///
/// The header, a disclosure arrow (right when closed, down when open) and
/// @p title, is one button at least the frame height tall (see @ref
/// ImGuiOverlayConfig::min_frame_height) and the only part that takes input.
/// The body is a window of its own directly below the header, from the first
/// frame, that takes none, so a finger or the mouse on it still reaches the
/// scene. Both windows have a 60 % background, are placed and sized to fit,
/// never move on a drag and are not saved.
///
/// @code
/// const ImGuiViewport* v = ImGui::GetMainViewport();
/// if (ui::begin_collapsible_panel("##stats", "Stats", v->WorkPos, &open)) {
///   ImGui::TextUnformatted("bitrate 11.8 Mbit/s");
///   ui::draw_stage_table("stages", rows);
///   ui::end_collapsible_panel();
/// }
/// @endcode
/// @param id     The header window's ImGui name, unique among windows; the
///               body's is derived from it.
/// @param title  The header's text.
/// @param pos    The header's top-left, in ImGui display coordinates.
/// @param open   The caller's state; a tap or click on the header flips it.
/// @return true when the body is open and ImGui draws it this frame: build it,
///         then call @ref end_collapsible_panel. false when it is closed or
///         ImGui skips it (its `Begin` returned false): every window is
///         already ended, so call nothing.
/// @pre An ImGui frame is active, with its context current; @p id, @p title
///      and @p open are not null.
VG_UI_API bool begin_collapsible_panel(const char* id, const char* title,
                                       ImVec2 pos, bool* open);

/// @brief End the body of a @ref begin_collapsible_panel that returned true.
VG_UI_API void end_collapsible_panel();

}  // namespace volumetric_kit::gfx::ui

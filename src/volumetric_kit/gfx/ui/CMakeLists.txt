# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_ui: Dear ImGui debug-overlay tier ----------------------------------
# Wraps ImGui's Vulkan renderer backend (imgui_impl_vulkan) so a debug UI draws
# into a RenderTarget through dynamic rendering, and draws marks over an image
# view. The GLFW platform backend (imgui_impl_glfw) is gfx_glfw's
# (glfw/CMakeLists.txt), so this tier stays GLFW-free like windowing; its own
# platform half, ImGuiTouch, reads no window system. Depends on gfx_core,
# gfx_camera and gfx_windowing.
include(GenerateExportHeader)

# The ui tier builds generated copies of two ImGui sources, patched below. The
# fetched sources stay untouched, and a patch whose upstream text is gone fails
# the configure, so an ImGui update reviews each patch.
function(_vg_patch_imgui code old new message)
  string(FIND "${${code}}" "${old}" _vg_match)
  if(_vg_match EQUAL -1)
    message(FATAL_ERROR "${message}")
  endif()
  string(REPLACE "${old}" "${new}" _vg_patched "${${code}}")
  set(${code}
      "${_vg_patched}"
      PARENT_SCOPE)
endfunction()

# Convert both scissor endpoints to integers before subtracting. Truncating
# their floating-point difference drops the last pixel when the framebuffer or a
# parent clip cuts off ImageOverlay's upper-edge nudge.
set(_vg_imgui_backend "${VG_IMGUI_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp")
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${_vg_imgui_backend}")
file(READ "${_vg_imgui_backend}" _vg_imgui_backend_code)
foreach(_vg_axis x y)
  _vg_patch_imgui(
    _vg_imgui_backend_code
    "(uint32_t)(clip_max.${_vg_axis} - clip_min.${_vg_axis})"
    "(uint32_t)(clip_max.${_vg_axis}) - (uint32_t)scissor.offset.${_vg_axis}"
    "ImGui scissor conversion changed; review the edge-pixel fix")
endforeach()
set(_vg_imgui_backend "${CMAKE_CURRENT_BINARY_DIR}/imgui_impl_vulkan.cpp")
file(
  GENERATE
  OUTPUT "${_vg_imgui_backend}"
  CONTENT "${_vg_imgui_backend_code}")

# Upstream's fragment shader, decoding ImGui's sRGB-encoded vertex colours with
# gfx's curve before it writes them, for a target that takes linear light
# (ImGuiOverlay::create; DECISIONS.md, "ImGui's colours as authored").
set(_vg_imgui_frag "${VG_IMGUI_SOURCE_DIR}/backends/vulkan/glsl_shader.frag")
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${_vg_imgui_frag}")
file(READ "${_vg_imgui_frag}" _vg_imgui_frag_code)
set(_vg_frag_message "ImGui's fragment shader changed; review its sRGB variant")
_vg_patch_imgui(
  _vg_imgui_frag_code
  "#version 450 core"
  "#version 450 core\n#extension GL_GOOGLE_include_directive : require\n#include \"common/srgb.glsl\""
  "${_vg_frag_message}")
_vg_patch_imgui(
  _vg_imgui_frag_code "fColor = In.Color * "
  "fColor = vec4(srgb_to_linear(In.Color.rgb), In.Color.a) * "
  "${_vg_frag_message}")
set(_vg_imgui_frag "${CMAKE_CURRENT_BINARY_DIR}/imgui_srgb.frag")
file(
  GENERATE
  OUTPUT "${_vg_imgui_frag}"
  CONTENT "${_vg_imgui_frag_code}")

# ImGui's own translation units carry their own warning posture and would trip
# this project's -Werror, so they compile in a separate vendored static lib that
# does NOT get vg_target_warnings -- gfx_ui below builds only first-party code
# under the strict flags. PIC so it links into a shared gfx_ui. This is an
# internal build target (no volumetric_kit:: alias, not exported).
#
# imgui_demo.cpp (ShowDemoWindow) is deliberately excluded: the tier never calls
# it, and a PRIVATE-linked demo symbol would not propagate out of a shared
# gfx_ui. The example + test that actually call ShowDemoWindow compile that one
# TU themselves (see examples/ and tests/ CMakeLists).
add_library(
  vg_imgui STATIC
  "${VG_IMGUI_SOURCE_DIR}/imgui.cpp" "${VG_IMGUI_SOURCE_DIR}/imgui_draw.cpp"
  "${VG_IMGUI_SOURCE_DIR}/imgui_tables.cpp"
  "${VG_IMGUI_SOURCE_DIR}/imgui_widgets.cpp" "${_vg_imgui_backend}")
# imgui.h lives at the root, imgui_impl_vulkan.h under backends/. SYSTEM so the
# strict -Werror first-party flags never apply to ImGui's headers where they are
# pulled into gfx_ui's own TU (e.g. GCC -Wpedantic differs from AppleClang).
target_include_directories(
  vg_imgui SYSTEM PUBLIC "${VG_IMGUI_SOURCE_DIR}"
                         "${VG_IMGUI_SOURCE_DIR}/backends")
# A single stroke can exceed 64K vertices; vertex offsets only split between
# allocations. Use 32-bit indices throughout ImGui and publish the same ABI to
# gfx_ui consumers below, including their platform backend and demo sources.
set(_vg_imgui_definitions "ImDrawIdx=unsigned int")
target_compile_definitions(vg_imgui PUBLIC ${_vg_imgui_definitions})
# imgui_impl_vulkan.cpp uses the linked Vulkan prototypes (no volk): Headers to
# compile against; the loader symbols resolve when this links into gfx_ui.
target_link_libraries(vg_imgui PUBLIC Vulkan::Headers)
set_target_properties(vg_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_ui)
add_library(volumetric_kit::gfx_ui ALIAS gfx_ui)

target_sources(
  gfx_ui PRIVATE collapsible_panel.cpp image_overlay.cpp imgui_overlay.cpp
                 imgui_touch.cpp metrics_panel.cpp)

generate_export_header(
  gfx_ui
  BASE_NAME
  VG_UI
  EXPORT_MACRO_NAME
  VG_UI_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/ui/export.hpp")

set_target_properties(
  gfx_ui
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vg_type gfx_ui TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_ui PUBLIC VG_UI_STATIC_DEFINE)
endif()

target_compile_features(gfx_ui PUBLIC cxx_std_17)
target_compile_definitions(gfx_ui PUBLIC ${_vg_imgui_definitions})

target_include_directories(
  gfx_ui PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                "$<BUILD_INTERFACE:${_vg_generated_inc}>")

# Consumers write raw ImGui:: calls, so imgui.h must reach them -- SYSTEM so
# their -Werror does not fire on it. vg_imgui is PRIVATE below, so its include
# dirs do not propagate to consumers; this is what exposes imgui.h, minus the
# backends/ dir.
target_include_directories(gfx_ui SYSTEM PUBLIC "${VG_IMGUI_SOURCE_DIR}")

# gfx_core, gfx_camera and gfx_windowing are PUBLIC (RenderTargetLayout,
# ImageView2D and overlay_config's Swapchain are in this tier's API); the ImGui
# backend and the Vulkan loader are PRIVATE -- neither imgui_impl_vulkan nor a
# loader symbol appears in this tier's public surface. vg_imgui's backends/
# include dir stays PRIVATE this way, so consumers see imgui.h (added PUBLIC
# above) but not the renderer backend header.
target_link_libraries(
  gfx_ui
  PUBLIC volumetric_kit::gfx_core volumetric_kit::gfx_camera
         volumetric_kit::gfx_windowing Vulkan::Headers
  PRIVATE vg_imgui Vulkan::Vulkan)

# The sRGB variant's SPIR-V, embedded as imgui_srgb_frag.spv.hpp, as the
# pipelines tier embeds its shaders.
vkc_embed_shaders(
  gfx_ui
  SYMBOL_PREFIX
  vg_
  TARGET_ENV
  vulkan1.3
  INCLUDE_DIRS
  "${PROJECT_SOURCE_DIR}/shaders"
  SHADERS
  "${_vg_imgui_frag}")

vg_target_warnings(gfx_ui)

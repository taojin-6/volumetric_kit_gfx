# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_ui: Dear ImGui debug-overlay tier ----------------------------------
# Wraps ImGui's Vulkan renderer backend (imgui_impl_vulkan) so a debug UI draws
# into a RenderTarget through dynamic rendering. The platform backend
# (imgui_impl_glfw) is deliberately NOT built here -- it stays in the example,
# so this tier inherits the windowing tier's GLFW-free rule. Depends only on
# gfx_core.
include(GenerateExportHeader)
include(GNUInstallDirs)

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
  "${VG_IMGUI_SOURCE_DIR}/imgui.cpp"
  "${VG_IMGUI_SOURCE_DIR}/imgui_draw.cpp"
  "${VG_IMGUI_SOURCE_DIR}/imgui_tables.cpp"
  "${VG_IMGUI_SOURCE_DIR}/imgui_widgets.cpp"
  "${VG_IMGUI_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp")
# imgui.h lives at the root, imgui_impl_vulkan.h under backends/. SYSTEM so the
# strict -Werror first-party flags never apply to ImGui's headers where they are
# pulled into gfx_ui's own TU (e.g. GCC -Wpedantic differs from AppleClang).
target_include_directories(
  vg_imgui SYSTEM PUBLIC "${VG_IMGUI_SOURCE_DIR}"
                         "${VG_IMGUI_SOURCE_DIR}/backends")
# imgui_impl_vulkan.cpp uses the linked Vulkan prototypes (no volk): Headers to
# compile against; the loader symbols resolve when this links into gfx_ui.
target_link_libraries(vg_imgui PUBLIC Vulkan::Headers)
set_target_properties(vg_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_ui)
add_library(volumetric_kit::gfx_ui ALIAS gfx_ui)

target_sources(gfx_ui PRIVATE imgui_overlay.cpp metrics_panel.cpp)

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

target_include_directories(
  gfx_ui
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Consumers write raw ImGui:: calls, so imgui.h must reach them -- SYSTEM so
# their -Werror does not fire on it, and BUILD_INTERFACE-only for now (vg_imgui
# is PRIVATE below, so its include dirs do not propagate to consumers; this is
# what exposes imgui.h, minus the backends/ dir). Co-installing these headers is
# the deferred packaging step (see the install TODO below).
target_include_directories(gfx_ui SYSTEM
                           PUBLIC "$<BUILD_INTERFACE:${VG_IMGUI_SOURCE_DIR}>")

# gfx_core is PUBLIC (RenderTargetLayout is in this tier's API); the ImGui
# backend and the Vulkan loader are PRIVATE -- neither imgui_impl_vulkan nor a
# loader symbol appears in this tier's public surface. vg_imgui's backends/
# include dir stays PRIVATE this way, so consumers see imgui.h (added PUBLIC
# above) but not the renderer backend header.
target_link_libraries(
  gfx_ui
  PUBLIC volumetric_kit::gfx_core Vulkan::Headers
  PRIVATE vg_imgui Vulkan::Vulkan)

vg_target_warnings(gfx_ui)

# TODO: install/export gfx_ui once the ImGui headers it exposes (PUBLIC, build
# tree only) and the PRIVATE vg_imgui static lib are co-installed -- the
# packaging step deferred from this tier's first PR. Until then gfx_ui is
# build-tree-only: examples + tests link it directly, and the umbrella pulls it
# in through BUILD_INTERFACE (see src/CMakeLists.txt), so the installed package
# is unaffected.

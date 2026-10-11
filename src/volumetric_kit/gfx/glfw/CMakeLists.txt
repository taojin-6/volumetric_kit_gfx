# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_glfw: GLFW window glue ---------------------------------------------
# What every GLFW application of the renderer repeats: the instance extensions,
# the surface factory, the framebuffer extent, the cursor in framebuffer pixels,
# a device shared with a compute library, and ImGui's GLFW platform backend.
# Depends on gfx_app and gfx_ui; built only with VG_WITH_GLFW and an installed
# GLFW (src/CMakeLists.txt).
include(GenerateExportHeader)
include(GNUInstallDirs)

# imgui_impl_glfw.cpp, compiled once here in its own static library without the
# first-party warning flags, as vg_imgui is (ui/CMakeLists.txt). It calls the
# ImGui core gfx_ui links, so it links gfx_ui rather than a second copy of
# vg_imgui, whose globals a shared build would then hold twice. Internal: no
# alias, not exported.
add_library(vg_imgui_glfw STATIC
            "${VG_IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.cpp")
target_include_directories(vg_imgui_glfw SYSTEM
                           PUBLIC "${VG_IMGUI_SOURCE_DIR}/backends")
target_link_libraries(vg_imgui_glfw PUBLIC volumetric_kit::gfx_ui glfw)
set_target_properties(vg_imgui_glfw PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_glfw)
add_library(volumetric_kit::gfx_glfw ALIAS gfx_glfw)

target_sources(gfx_glfw PRIVATE window.cpp shared_device.cpp imgui_glfw.cpp)

generate_export_header(
  gfx_glfw
  BASE_NAME
  VG_GLFW
  EXPORT_MACRO_NAME
  VG_GLFW_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/glfw/export.hpp")

set_target_properties(
  gfx_glfw
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vg_type gfx_glfw TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_glfw PUBLIC VG_GLFW_STATIC_DEFINE)
endif()

target_compile_features(gfx_glfw PUBLIC cxx_std_17)

target_include_directories(
  gfx_glfw
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# gfx_app (the surface factory's type) and gfx_ui (the ImGui context the backend
# binds, and imgui.h) are PUBLIC, as are GLFW, whose window the API takes, and
# glm (cursor_in_framebuffer). The backend is PRIVATE: no ImGui_ImplGlfw_*
# symbol is part of this tier's API.
target_link_libraries(
  gfx_glfw
  PUBLIC volumetric_kit::gfx_app volumetric_kit::gfx_ui glfw glm::glm
  PRIVATE vg_imgui_glfw)
vg_glm_conventions(gfx_glfw PUBLIC)

vg_target_warnings(gfx_glfw)

# TODO: install/export gfx_glfw with gfx_ui, once the ImGui headers are
# co-installed (ui/CMakeLists.txt). Until then it is build-tree-only: the
# umbrella takes it through BUILD_INTERFACE and the header install excludes
# glfw/ (src/CMakeLists.txt).

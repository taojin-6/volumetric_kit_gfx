# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_app: app tier -------------------------------------------------------
# Bring-up facades: WindowedApp collapses the instance -> surface -> device ->
# allocator -> swapchain -> frame-loop chain into one create() call (with the
# surface threaded consistently through selection, needs_present, and device
# creation); HeadlessApp is the surface-free instance -> device -> allocator
# chain. Depends on gfx_core + gfx_windowing; GLFW-free like windowing -- the
# consumer supplies the required instance extensions and a surface-creation
# callback.
#
# TODO: app/streamed_app (dynamic content; double-buffer + frame queue) joins
# this tier once the interop tier that feeds it lands.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_app)
add_library(volumetric_kit::gfx_app ALIAS gfx_app)

target_sources(gfx_app PRIVATE windowed_app.cpp headless_app.cpp)

generate_export_header(
  gfx_app
  BASE_NAME
  VG_APP
  EXPORT_MACRO_NAME
  VG_APP_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/app/export.hpp")

set_target_properties(
  gfx_app
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vg_type gfx_app TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_app PUBLIC VG_APP_STATIC_DEFINE)
endif()

target_compile_features(gfx_app PUBLIC cxx_std_17)

target_include_directories(
  gfx_app
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# gfx_core + gfx_windowing are PUBLIC (their types appear in this tier's API);
# the Vulkan loader is PRIVATE (wait_idle calls vkDeviceWaitIdle, but no loader
# symbol surfaces publicly).
target_link_libraries(
  gfx_app
  PUBLIC volumetric_kit::gfx_core volumetric_kit::gfx_windowing
  PRIVATE Vulkan::Vulkan)

vg_target_warnings(gfx_app)

if(VG_INSTALL)
  install(
    TARGETS gfx_app
    EXPORT volumetric_kit_gfxTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${_vg_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()

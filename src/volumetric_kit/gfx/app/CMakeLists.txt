# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_app: app tier -------------------------------------------------------
# Bring-up facades: WindowedApp collapses the instance -> surface -> device ->
# allocator -> swapchain -> frame-loop chain into one create() call (with the
# surface threaded consistently through selection, needs_present, and device
# creation); HeadlessApp is the surface-free instance -> device -> allocator
# chain; make_shared_device builds the device a windowed app shares with a
# compute library. Depends on gfx_core + gfx_windowing; GLFW-free like
# windowing, so the consumer supplies a windowing::SurfaceSource, or the
# instance extensions and a surface-creation callback.
#
# TODO: app/streamed_app (dynamic content; double-buffer + frame queue) joins
# this tier once the interop tier that feeds it lands.
include(GenerateExportHeader)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_app)
add_library(volumetric_kit::gfx_app ALIAS gfx_app)

target_sources(gfx_app PRIVATE windowed_app.cpp headless_app.cpp
                               shared_device_config.cpp)

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
  gfx_app PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                 "$<BUILD_INTERFACE:${_vg_generated_inc}>")

# gfx_core + gfx_windowing are PUBLIC (their types appear in this tier's API);
# the Vulkan loader is PRIVATE (wait_idle calls vkDeviceWaitIdle, but no loader
# symbol surfaces publicly).
target_link_libraries(
  gfx_app
  PUBLIC volumetric_kit::gfx_core volumetric_kit::gfx_windowing
  PRIVATE Vulkan::Vulkan)

vg_target_warnings(gfx_app)

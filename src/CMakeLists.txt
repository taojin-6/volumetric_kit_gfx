# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Orchestrator for the component tiers. Each subdirectory owns its own target
# and build settings; the strict dependency rule (a tier may depend only on
# tiers to its left) is expressed by what each links.

add_subdirectory(volumetric_kit/gfx/core)
add_subdirectory(volumetric_kit/gfx/windowing)
add_subdirectory(volumetric_kit/gfx/camera)
add_subdirectory(volumetric_kit/gfx/assets)
add_subdirectory(volumetric_kit/gfx/io)
add_subdirectory(volumetric_kit/gfx/ui)
add_subdirectory(volumetric_kit/gfx/pipelines)
add_subdirectory(volumetric_kit/gfx/app)
# The GLFW window glue, when VG_WITH_GLFW found GLFW (CMakeLists.txt).
if(VG_WITH_GLFW AND TARGET glfw)
  add_subdirectory(volumetric_kit/gfx/glfw)
elseif(VG_WITH_GLFW)
  message(STATUS "gfx_glfw: skipped (GLFW 3.3 or newer not found)")
endif()
# TODO: add_subdirectory() for passes as that tier lands.

# --- gfx: umbrella alias ----------------------------------------------------
# Consumers link `volumetric_kit::gfx` and get whatever tiers are built.
add_library(gfx INTERFACE)
add_library(volumetric_kit::gfx ALIAS gfx)
target_link_libraries(
  gfx
  INTERFACE gfx_core
            gfx_windowing
            gfx_camera
            gfx_assets
            gfx_io
            gfx_pipelines
            gfx_app
            gfx_ui)
if(TARGET gfx_glfw)
  target_link_libraries(gfx INTERFACE gfx_glfw)
endif()

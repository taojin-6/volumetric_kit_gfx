# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_windowing: windowing tier ------------------------------------------
# Surface + swapchain + the frames-in-flight FrameLoop: turns a window-system
# VkSurfaceKHR into RenderTargets and drives the present loop. Depends only on
# gfx_core; deliberately GLFW-free (it takes a VkSurfaceKHR), so the swapchain
# is unit-testable via VK_EXT_headless_surface and a consumer picks its own
# window system. GLFW lives in the examples, not here.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_windowing)
add_library(volumetric_kit::gfx_windowing ALIAS gfx_windowing)

target_sources(gfx_windowing PRIVATE surface.cpp swapchain.cpp frame_loop.cpp)

generate_export_header(
  gfx_windowing
  BASE_NAME
  VG_WINDOWING
  EXPORT_MACRO_NAME
  VG_WINDOWING_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/windowing/export.hpp")

set_target_properties(
  gfx_windowing
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vg_type gfx_windowing TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_windowing PUBLIC VG_WINDOWING_STATIC_DEFINE)
endif()

target_compile_features(gfx_windowing PUBLIC cxx_std_17)

target_include_directories(
  gfx_windowing
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# gfx_core is PUBLIC (its types appear in this tier's API); the Vulkan loader is
# PRIVATE (this tier calls the WSI entry points -- vkCreateSwapchainKHR,
# vkAcquireNextImageKHR, ... -- but never surfaces a loader symbol publicly).
target_link_libraries(
  gfx_windowing
  PUBLIC volumetric_kit::gfx_core Vulkan::Headers
  PRIVATE Vulkan::Vulkan)

# swapchain.cpp classifies the depth format through core/impl/vk_format.hpp
# (vkuFormat*), so this tier's own TUs need the same pinned Vulkan-Headers +
# Vulkan-Utility-Libraries repointing as gfx_core: BEFORE so they win over the
# Vulkan::Headers include above, SYSTEM to keep vendored warnings out of
# -Werror, and PRIVATE so nothing leaks to consumers.
target_include_directories(
  gfx_windowing SYSTEM BEFORE PRIVATE "${VG_VULKAN_HEADERS_INCLUDE}"
                                      "${VG_VULKAN_UTILITY_INCLUDE}")

vg_target_warnings(gfx_windowing)

if(VG_INSTALL)
  install(
    TARGETS gfx_windowing
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

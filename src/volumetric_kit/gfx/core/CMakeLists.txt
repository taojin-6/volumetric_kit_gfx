# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_core: Tier 1 -------------------------------------------------------
# Foundation tier: version API, gfx's names for volumetric_kit_core's
# Status/Result error handling, log sink and contract checks, instance, device
# and allocator (buffers and images), plus sync primitives and uploads. Every
# other tier depends on this; it depends only on volumetric_kit_core's base and
# vulkan tiers + Vulkan + spirv-cross (a private impl detail backing SPIR-V
# reflection in shader.cpp).
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_core)
add_library(volumetric_kit::gfx_core ALIAS gfx_core)

target_sources(
  gfx_core
  PRIVATE version.cpp
          debug_label.cpp
          sync.cpp
          query_pool.cpp
          profiler.cpp
          retire_queue.cpp
          sampler.cpp
          shader.cpp
          descriptor.cpp
          graphics_pipeline.cpp
          render_target.cpp
          image_barrier.cpp
          offscreen_target.cpp
          command_pool.cpp
          command_buffer.cpp
          texture_upload.cpp)

# Generated headers: the VG_CORE_API export macro + the configured version
# header. The export header is per-tier (core/export.hpp + VG_CORE_API +
# VG_CORE_STATIC_DEFINE) so each future tier (passes/pipelines/app) generates
# its own isolated macro/guard and they never collide on one path or
# mis-attribute symbols across shared libraries.
generate_export_header(
  gfx_core
  BASE_NAME
  VG_CORE
  EXPORT_MACRO_NAME
  VG_CORE_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/core/export.hpp")
configure_file("${PROJECT_SOURCE_DIR}/include/volumetric_kit/gfx/version.hpp.in"
               "${_vg_generated_inc}/volumetric_kit/gfx/version.hpp" @ONLY)

set_target_properties(
  gfx_core
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

# With a static archive, VG_CORE_API must expand to nothing for consumers.
get_target_property(_vg_type gfx_core TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_core PUBLIC VG_CORE_STATIC_DEFINE)
endif()

target_compile_features(gfx_core PUBLIC cxx_std_17)

target_include_directories(
  gfx_core
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Public deps are independently re-findable, so the installed package's
# find_dependency() calls stay simple: volumetric_kit_core -- its base tier
# (Status / Result / the log sink in every tier's signatures) and its vulkan
# tier (the Instance, PhysicalDeviceInfo, Device and Allocator gfx's API takes,
# the umbrella header core/vulkan.hpp forwards to, the VkResult bridge
# result.hpp names, and the format metadata) -- and the system's Vulkan headers,
# which every gfx TU compiles against. gfx links the Vulkan loader PRIVATE --
# its own implementation calls it, and no public gfx signature needs it -- but
# the core's vulkan tier links it PUBLIC, so consumers get it through that. VMA
# is the core's, compiled into its allocator; gfx no longer builds a copy. glm
# is intentionally absent: no public gfx_core header names a glm type (see the
# rationale at the top-level find_package site in CMakeLists.txt).
target_link_libraries(
  gfx_core
  PUBLIC volumetric_kit::core_base volumetric_kit::core_vulkan Vulkan::Headers
  PRIVATE Vulkan::Vulkan)

# spirv-cross: SPIR-V reflection for descriptor-set layouts, used privately in
# shader.cpp -- its headers/types never reach gfx_core's public API. SYSTEM-mark
# its include dir so its third-party-header warnings stay out of -Werror.
# spirv-cross runs its own install/export under VG_INSTALL (see
# third_party/CMakeLists.txt), so the installed package's
# find_dependency(spirv_cross_core) re-supplies this private static link.
target_include_directories(
  gfx_core SYSTEM
  PRIVATE $<TARGET_PROPERTY:spirv-cross-core,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(gfx_core PRIVATE spirv-cross-core)

vg_target_warnings(gfx_core)

# Install the target into the shared export set + the headers this tier
# generates. The umbrella target, public-header tree, and package config are
# finalized by the parent (src/CMakeLists.txt).
if(VG_INSTALL)
  install(
    TARGETS gfx_core
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

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_core: Tier 1 -------------------------------------------------------
# Foundation tier: version API, logging, contract checks, Vulkan instance/device
# setup, sync primitives, and GPU memory allocation (buffers/images via VMA).
# Every other tier depends on this; it depends only on Vulkan + VMA.
#
# TODO: add the spirv-cross reflection surface.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_core)
add_library(volumetric_kit::gfx_core ALIAS gfx_core)

target_sources(
  gfx_core
  PRIVATE version.cpp
          vma_impl.cpp
          log.cpp
          check.cpp
          instance.cpp
          device.cpp
          sync.cpp
          retire_queue.cpp
          allocator.cpp
          buffer.cpp
          texture.cpp
          shader.cpp)

# Generated headers: the VG_API export macro + the configured version header.
generate_export_header(
  gfx_core
  BASE_NAME
  VG
  EXPORT_MACRO_NAME
  VG_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/export.hpp")
configure_file("${PROJECT_SOURCE_DIR}/include/volumetric_kit/gfx/version.hpp.in"
               "${_vg_generated_inc}/volumetric_kit/gfx/version.hpp" @ONLY)

set_target_properties(
  gfx_core
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

# With a static archive, VG_API must expand to nothing for consumers.
get_target_property(_vg_type gfx_core TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_core PUBLIC VG_STATIC_DEFINE)
endif()

target_compile_features(gfx_core PUBLIC cxx_std_17)

target_include_directories(
  gfx_core
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# VMA's header is a vendored private dep (implemented in vma_impl.cpp). Mark it
# SYSTEM so its warnings -- nullability-extension/-completeness under
# -Wall/-Wpedantic -- never reach our build, keeping first-party code clean
# enough for -Werror.
target_include_directories(gfx_core SYSTEM PRIVATE "${VG_VMA_INCLUDE_DIR}")

# Public deps (header packages) are independently re-findable, so the installed
# package's find_dependency() calls stay simple. The Vulkan loader is PRIVATE:
# our implementation (VMA) links it, but it never surfaces in the public API.
target_link_libraries(
  gfx_core
  PUBLIC Vulkan::Headers glm::glm
  PRIVATE Vulkan::Vulkan)

if(VG_WITH_VALIDATION)
  target_compile_definitions(gfx_core PRIVATE VG_WITH_VALIDATION)
endif()

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
  install(DIRECTORY "${_vg_generated_inc}/"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

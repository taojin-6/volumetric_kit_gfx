# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

# --- gfx_core: Tier 1 -------------------------------------------------------
# A deliberately minimal core (just the version API) so the build, dependency,
# export, and CI pipeline are proven end-to-end. The device / memory /
# reflection surface lands later.
add_library(gfx_core)
add_library(volumetric_kit::gfx_core ALIAS gfx_core)

target_sources(gfx_core PRIVATE volumetric_kit/gfx/version.cpp
                                volumetric_kit/gfx/core/vma_impl.cpp)

# Generated headers: the VG_API export macro + the configured version header.
generate_export_header(
  gfx_core
  BASE_NAME
  VG
  EXPORT_MACRO_NAME
  VG_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/export.hpp")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../include/volumetric_kit/gfx/version.hpp.in"
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
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  # Absorbed private dep: VMA's header, implemented in vma_impl.cpp.
  PRIVATE "${VG_VMA_INCLUDE_DIR}")

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

# --- gfx: umbrella alias ----------------------------------------------------
# Aggregates the component targets; consumers link `volumetric_kit::gfx` and get
# whatever tiers are built. Grows as passes/pipelines/app land.
add_library(gfx INTERFACE)
add_library(volumetric_kit::gfx ALIAS gfx)
target_link_libraries(gfx INTERFACE gfx_core)

# --- Install / export -------------------------------------------------------
if(VG_INSTALL)
  install(
    TARGETS gfx_core gfx
    EXPORT volumetric_kit_gfxTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(DIRECTORY "${_vg_generated_inc}/"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  install(
    EXPORT volumetric_kit_gfxTargets
    NAMESPACE volumetric_kit::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/volumetric_kit_gfx")

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/volumetric_kit_gfxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/volumetric_kit_gfxConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/volumetric_kit_gfx")
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/volumetric_kit_gfxConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/volumetric_kit_gfxConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/volumetric_kit_gfxConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/volumetric_kit_gfx")
endif()

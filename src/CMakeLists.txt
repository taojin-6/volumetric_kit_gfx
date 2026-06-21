# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Orchestrator for the component tiers. Each subdirectory owns its own target
# and build settings and installs itself into the shared
# volumetric_kit_gfxTargets export set; the strict dependency rule (a tier may
# depend only on tiers to its left) is expressed by what each links.
include(GNUInstallDirs)

add_subdirectory(volumetric_kit/gfx/core)
add_subdirectory(volumetric_kit/gfx/windowing)
add_subdirectory(volumetric_kit/gfx/camera)
# TODO: add_subdirectory() for passes / pipelines / app as those tiers land.

# --- gfx: umbrella alias ----------------------------------------------------
# Consumers link `volumetric_kit::gfx` and get whatever tiers are built.
add_library(gfx INTERFACE)
add_library(volumetric_kit::gfx ALIAS gfx)
target_link_libraries(gfx INTERFACE gfx_core gfx_windowing gfx_camera)

# --- Install / export -------------------------------------------------------
# Tiers install their own targets into volumetric_kit_gfxTargets (above); here
# we add the umbrella, ship the public header tree, and finalize the export set
# and package config.
if(VG_INSTALL)
  install(TARGETS gfx EXPORT volumetric_kit_gfxTargets)

  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp"
    # impl/vk_format.hpp is internal (no public header includes it) and pulls
    # the vendored <vulkan/utility/...> header that the package neither ships
    # nor re-exports, so an installed copy would not compile -- keep it out.
    PATTERN "vk_format.hpp" EXCLUDE)

  install(
    EXPORT volumetric_kit_gfxTargets
    NAMESPACE volumetric_kit::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/volumetric_kit_gfx")

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/volumetric_kit_gfxConfig.cmake.in"
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

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
add_subdirectory(volumetric_kit/gfx/assets)
add_subdirectory(volumetric_kit/gfx/io)
add_subdirectory(volumetric_kit/gfx/ui)
add_subdirectory(volumetric_kit/gfx/pipelines)
add_subdirectory(volumetric_kit/gfx/app)
# TODO: add_subdirectory() for passes as that tier lands.

# --- gfx: umbrella alias ----------------------------------------------------
# Consumers link `volumetric_kit::gfx` and get whatever tiers are built. gfx_ui
# is build-tree only (BUILD_INTERFACE) until its ImGui-header packaging is
# sorted, so it joins the umbrella for in-tree consumers without entering the
# installed export set (see ui/CMakeLists.txt).
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
            $<BUILD_INTERFACE:gfx_ui>)

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
    # ui/ needs the generated ui/export.hpp and the ImGui headers, and neither
    # is installed while gfx_ui stays build-tree-only (see the install TODO in
    # volumetric_kit/gfx/ui/CMakeLists.txt), so an installed copy would not
    # compile -- keep the tier out until gfx_ui is exported.
    REGEX "volumetric_kit/gfx/ui(/|$)" EXCLUDE
    # impl/vk_format.hpp is internal (no public header includes it) and pulls
    # the vendored <vulkan/utility/...> header that the package neither ships
    # nor re-exports, so an installed copy would not compile -- keep it out.
    PATTERN "vk_format.hpp" EXCLUDE
    # impl/vk_query.hpp is internal too ("Not a public header."); no installed
    # header includes it, so keep it out. The other impl/ headers ship because
    # public headers include them (impl/result.hpp, impl/retire_list.hpp,
    # impl/debug_utils_table.hpp, ...).
    REGEX "volumetric_kit/gfx/core/impl/vk_query\\.hpp$" EXCLUDE)

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

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_camera: camera tier -------------------------------------------------
# Pure-CPU view + projection math: Camera (Vulkan-clip-space view/projection),
# OrbitCamera (turntable controls), and CameraRig (6-DoF verb-driven pose). Its
# only dependency is glm; it links neither gfx_core nor Vulkan, so it is the one
# tier that builds and tests with no Vulkan device or loader present. The strict
# tier rule (depend only on tiers to the left) is satisfied trivially: it
# depends on no other tier.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_camera)
add_library(volumetric_kit::gfx_camera ALIAS gfx_camera)

target_sources(gfx_camera PRIVATE camera.cpp orbit_camera.cpp camera_rig.cpp)

generate_export_header(
  gfx_camera
  BASE_NAME
  VG_CAMERA
  EXPORT_MACRO_NAME
  VG_CAMERA_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/camera/export.hpp")

set_target_properties(
  gfx_camera
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vg_type gfx_camera TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_camera PUBLIC VG_CAMERA_STATIC_DEFINE)
endif()

target_compile_features(gfx_camera PUBLIC cxx_std_17)

target_include_directories(
  gfx_camera
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# glm is PUBLIC: it appears in this tier's public API (glm::mat4 / glm::vec3
# return and parameter types). It is the only link dependency -- no gfx_core, no
# Vulkan -- which keeps the camera tier CPU-only and the tier rule enforced.
target_link_libraries(gfx_camera PUBLIC glm::glm)

# Pin glm's clip-space depth convention as a PUBLIC compile definition, not just
# in a header. GLM_FORCE_DEPTH_ZERO_TO_ONE selects Vulkan's [0, 1] depth (vs
# GL's [-1, 1]); because glm is header-only it changes the *definition* of
# glm::perspective/ortho, so it must be identical in every TU that instantiates
# them -- the library's own and every consumer's. PUBLIC puts it on the command
# line of anything linking gfx_camera (build tree and, via the install
# interface, installed consumers), so the guarantee no longer depends on a
# consumer including camera.hpp before any <glm/...>. impl/glm_config.hpp
# mirrors the define at the include site; this is the authoritative,
# order-independent half.
target_compile_definitions(gfx_camera PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

vg_target_warnings(gfx_camera)

# Install the target into the shared export set + the headers this tier
# generates. The umbrella target, public-header tree, and package config are
# finalized by the parent (src/CMakeLists.txt).
if(VG_INSTALL)
  install(
    TARGETS gfx_camera
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

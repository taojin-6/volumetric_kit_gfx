# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_io: asset file loaders ---------------------------------------------
# Format loaders that parse files into the gfx_assets model (glTF now; OBJ / PLY
# / assimp later). Pure CPU: depends on gfx_assets (the model) + tinygltf only,
# never gfx_core or Vulkan, so it stays usable from a tool or a device-less
# test. The strict tier rule (depend only on tiers to the left) holds: io ->
# assets.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_io)
add_library(volumetric_kit::gfx_io ALIAS gfx_io)

target_sources(gfx_io PRIVATE tinygltf_impl.cpp gltf_loader.cpp)

generate_export_header(
  gfx_io
  BASE_NAME
  VG_IO
  EXPORT_MACRO_NAME
  VG_IO_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/io/export.hpp")

set_target_properties(
  gfx_io
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vg_type gfx_io TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_io PUBLIC VG_IO_STATIC_DEFINE)
endif()

target_compile_features(gfx_io PUBLIC cxx_std_17)

target_include_directories(
  gfx_io
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# tinygltf (with its bundled json + stb) is a vendored PRIVATE dep, implemented
# in tinygltf_impl.cpp. Mark it SYSTEM so its warnings never reach our -Werror
# build, and PRIVATE so it stays out of the public API -- no installed header
# pulls <tiny_gltf.h>.
target_include_directories(gfx_io SYSTEM PRIVATE "${VG_TINYGLTF_INCLUDE_DIR}")

# Read-only loaders, so drop tinygltf's image *writer* (stb_image_write and the
# default WriteImageData callback). PRIVATE and target-wide -- not a per-file
# define -- because TINYGLTF_NO_STB_IMAGE_WRITE changes the TinyGLTF class
# declaration: every io TU that includes <tiny_gltf.h> must agree, or the
# implementation TU and the loader TU disagree on the type (ODR), surfacing as
# an undefined WriteImageData symbol at link time.
target_compile_definitions(gfx_io PRIVATE TINYGLTF_NO_STB_IMAGE_WRITE)

# gfx_assets (PUBLIC) supplies the model types named in the loader's public
# header and brings glm transitively. No Vulkan and no gfx_core -- io depends
# only on the assets tier to its left.
target_link_libraries(gfx_io PUBLIC gfx_assets)

vg_target_warnings(gfx_io)

# Install the target + the export header this tier generates. The umbrella
# target, public-header tree, and package config are finalized by the parent
# (src/CMakeLists.txt).
if(VG_INSTALL)
  install(
    TARGETS gfx_io
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

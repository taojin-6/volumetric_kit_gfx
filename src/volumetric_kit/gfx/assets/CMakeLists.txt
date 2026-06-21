# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_assets: assets tier ------------------------------------------------
# Format-neutral, CPU-side asset model (Mesh / Material / Image / PointCloud /
# Scene / Model) plus the glTF loader. Pure CPU: depends on glm + tinygltf only,
# never gfx_core or Vulkan, so it has no GPU dependency and stays usable from a
# tool or a test with no device. The tier rule (a tier may depend only on tiers
# to its left) is enforced by what this links -- there is no gfx_core/Vulkan
# edge below, by design.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_assets)
add_library(volumetric_kit::gfx_assets ALIAS gfx_assets)

target_sources(gfx_assets PRIVATE tinygltf_impl.cpp gltf_loader.cpp)

generate_export_header(
  gfx_assets
  BASE_NAME
  VG_ASSETS
  EXPORT_MACRO_NAME
  VG_ASSETS_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/assets/export.hpp")

set_target_properties(
  gfx_assets
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vg_type gfx_assets TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_assets PUBLIC VG_ASSETS_STATIC_DEFINE)
endif()

target_compile_features(gfx_assets PUBLIC cxx_std_17)

target_include_directories(
  gfx_assets
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# tinygltf (with its bundled json + stb) is a vendored PRIVATE dep, implemented
# in tinygltf_impl.cpp. Mark it SYSTEM so its warnings never reach our -Werror
# build, and PRIVATE so it stays out of the public API -- no installed header
# pulls <tiny_gltf.h>.
target_include_directories(gfx_assets SYSTEM
                           PRIVATE "${VG_TINYGLTF_INCLUDE_DIR}")

# This tier only reads assets, so drop tinygltf's image *writer*
# (stb_image_write and the default WriteImageData callback). PRIVATE and
# target-wide -- not a per-file define -- because TINYGLTF_NO_STB_IMAGE_WRITE
# changes the TinyGLTF class declaration: every assets TU that includes
# <tiny_gltf.h> must agree, or the implementation TU and the loader TU disagree
# on the type (ODR), surfacing as an undefined WriteImageData symbol at link
# time.
target_compile_definitions(gfx_assets PRIVATE TINYGLTF_NO_STB_IMAGE_WRITE)

# glm is PUBLIC: glm::vec3/vec4/mat4 appear throughout this tier's model
# headers, so a consumer compiling against gfx_assets needs glm's include path
# too. No Vulkan and no gfx_core link -- the tier rule keeps this tier GPU-free.
target_link_libraries(gfx_assets PUBLIC glm::glm)

vg_target_warnings(gfx_assets)

# Install the target + the export header this tier generates. The umbrella
# target, public-header tree, and package config are finalized by the parent
# (src/CMakeLists.txt).
if(VG_INSTALL)
  install(
    TARGETS gfx_assets
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

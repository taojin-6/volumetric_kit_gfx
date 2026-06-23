# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_pipelines: pipelines tier ------------------------------------------
# Ready-to-use rendering building blocks composed on core (GPU resources) and
# the assets data model. The first module is GPU mesh upload (assets::Mesh ->
# vertex/index buffers + an indexed draw). Per the tier rule it depends only on
# tiers to its left -- core + assets -- plus the Vulkan loader (privately, for
# the vkCmd* calls in record_draw).
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vg_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(gfx_pipelines)
add_library(volumetric_kit::gfx_pipelines ALIAS gfx_pipelines)

target_sources(gfx_pipelines PRIVATE gpu_mesh.cpp pbr_pipeline.cpp)

generate_export_header(
  gfx_pipelines
  BASE_NAME
  VG_PIPELINES
  EXPORT_MACRO_NAME
  VG_PIPELINES_API
  EXPORT_FILE_NAME
  "${_vg_generated_inc}/volumetric_kit/gfx/pipelines/export.hpp")

set_target_properties(
  gfx_pipelines
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vg_type gfx_pipelines TYPE)
if(_vg_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(gfx_pipelines PUBLIC VG_PIPELINES_STATIC_DEFINE)
endif()

target_compile_features(gfx_pipelines PUBLIC cxx_std_17)

target_include_directories(
  gfx_pipelines
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vg_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# core (Allocator/Buffer/Result) and assets (the Mesh data model) appear in this
# tier's public API, so both are PUBLIC. The Vulkan loader is PRIVATE:
# record_draw records vkCmd*, but no loader type crosses the public API.
target_link_libraries(
  gfx_pipelines
  PUBLIC volumetric_kit::gfx_core volumetric_kit::gfx_assets
  PRIVATE Vulkan::Vulkan)

# --- Embedded PBR shaders ---------------------------------------------------
# PbrPipeline ships its own technique: compile the GLSL to SPIR-V, then embed it
# as constexpr byte arrays compiled into the library, so a consumer needs no
# shader files. The generated headers sit on a PRIVATE include path -- internal,
# not installed.
set(_vg_spv_dir "${CMAKE_CURRENT_BINARY_DIR}/spv")
set(_vg_embed_inc "${CMAKE_CURRENT_BINARY_DIR}/embedded")
vg_compile_shaders(
  gfx_pipelines OUTPUT_DIR "${_vg_spv_dir}" SHADERS
  "${PROJECT_SOURCE_DIR}/shaders/model.vert"
  "${PROJECT_SOURCE_DIR}/shaders/model.frag")
foreach(_stage IN ITEMS vert frag)
  add_custom_command(
    OUTPUT "${_vg_embed_inc}/model_${_stage}.spv.hpp"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_vg_embed_inc}"
    COMMAND
      ${CMAKE_COMMAND} "-DSPV=${_vg_spv_dir}/model.${_stage}.spv"
      "-DSYMBOL=vg_model_${_stage}_spv"
      "-DOUT=${_vg_embed_inc}/model_${_stage}.spv.hpp" -P
      "${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake"
    DEPENDS "${_vg_spv_dir}/model.${_stage}.spv"
            "${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake"
    COMMENT "Embedding model.${_stage}.spv"
    VERBATIM)
endforeach()
add_custom_target(
  gfx_pipelines_embedded_shaders DEPENDS "${_vg_embed_inc}/model_vert.spv.hpp"
                                         "${_vg_embed_inc}/model_frag.spv.hpp")
add_dependencies(gfx_pipelines gfx_pipelines_embedded_shaders)
target_include_directories(gfx_pipelines PRIVATE "${_vg_embed_inc}")

vg_target_warnings(gfx_pipelines)

# Install the target into the shared export set + the headers this tier
# generates. The umbrella target, public-header tree, and package config are
# finalized by the parent (src/CMakeLists.txt).
if(VG_INSTALL)
  install(
    TARGETS gfx_pipelines
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

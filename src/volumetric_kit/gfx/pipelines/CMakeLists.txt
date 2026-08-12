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

target_sources(
  gfx_pipelines
  PRIVATE gpu_mesh.cpp
          hybrid_mesh_pipeline.cpp
          ibl.cpp
          live_mesh.cpp
          owned_descriptor_set.cpp
          patch_mesh_pipeline.cpp
          pbr_material.cpp
          pbr_model.cpp
          pbr_pipeline.cpp
          pbr_scene.cpp)

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
# record_draw records vkCmd*, but no loader type crosses the public API. Threads
# is PRIVATE too: the IBL bake convolves faces via std::async, but no thread
# type crosses the public API.
find_package(Threads REQUIRED)
target_link_libraries(
  gfx_pipelines
  PUBLIC volumetric_kit::gfx_core volumetric_kit::gfx_assets
  PRIVATE Vulkan::Vulkan Threads::Threads)

# --- Embedded PBR shaders ---------------------------------------------------
# PbrPipeline ships its own technique: compile shaders/model.{vert,frag} to
# SPIR-V and embed each as a constexpr byte array compiled into the library
# (pbr_pipeline.cpp #includes model_{vert,frag}.spv.hpp), so a consumer needs no
# shader files. vg_embed_shaders puts the generated headers on a PRIVATE include
# path -- internal, not installed.
vg_embed_shaders(
  gfx_pipelines SHADERS "${PROJECT_SOURCE_DIR}/shaders/model.vert"
  "${PROJECT_SOURCE_DIR}/shaders/model.frag")

# --- Embedded reconstruction hybrid-mesh shaders ----------------------------
# HybridMeshPipeline (the volumetric_kit_recon handoff) likewise embeds its
# technique: hybrid_mesh.{vert,frag} -> SPIR-V compiled into the library
# (hybrid_mesh_pipeline.cpp #includes hybrid_mesh_{vert,frag}.spv.hpp).
# PatchMeshPipeline's technique rides the same list: it draws the same handoff
# shaded from a per-triangle patch atlas instead of a per-vertex uv.
vg_embed_shaders(
  gfx_pipelines
  SHADERS
  "${PROJECT_SOURCE_DIR}/shaders/hybrid_mesh.vert"
  "${PROJECT_SOURCE_DIR}/shaders/hybrid_mesh.frag"
  "${PROJECT_SOURCE_DIR}/shaders/patch_mesh.vert"
  "${PROJECT_SOURCE_DIR}/shaders/patch_mesh.frag")

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

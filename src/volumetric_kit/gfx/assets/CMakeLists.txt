# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- gfx_assets: asset data model (header-only) -----------------------------
# Format-neutral, CPU-side asset value types (Vertex / Mesh / Material / Image /
# PointCloud / Scene / Model). Header-only and GPU-API-free: it depends on glm
# only -- never gfx_core, Vulkan, or any file-format parser. This is the stable
# interchange representation the io loaders produce and downstream tiers
# consume, so it deliberately drags no device or parser dependency. The file
# loaders live in the io tier, which links this.
include(GNUInstallDirs)

add_library(gfx_assets INTERFACE)
add_library(volumetric_kit::gfx_assets ALIAS gfx_assets)

target_compile_features(gfx_assets INTERFACE cxx_std_17)

target_include_directories(
  gfx_assets INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                       "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# glm is PUBLIC (INTERFACE): glm::vec3/vec4/mat4 appear throughout the model
# headers, so every consumer needs glm's include path. No Vulkan, no gfx_core --
# the tier rule keeps the model GPU-free.
target_link_libraries(gfx_assets INTERFACE glm::glm)

# Header-only: nothing is compiled here, so there is no export header, no
# visibility preset, and no warning pass. The public header tree is installed by
# the parent (src/CMakeLists.txt); here we only add the INTERFACE target to the
# shared export set.
if(VG_INSTALL)
  install(TARGETS gfx_assets EXPORT volumetric_kit_gfxTargets)
endif()

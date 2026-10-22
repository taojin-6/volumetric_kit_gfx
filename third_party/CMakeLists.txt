# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Dependencies. System via find_package: the Vulkan headers and loader, and
# glslc/glslangValidator (shader compilation, through the core's
# vkc_compile_shaders). The library compiles against the system's Vulkan
# headers, as the family's core does (its DECISIONS.md, "Vulkan headers come
# from the system"), and takes its VkFormat metadata from the core's format.hpp.
# glm (camera tier) is preferred from the system too (resolved at the top
# level), falling back to a pinned vendored copy below when none is installed.
# Vendored via pinned FetchContent: spirv-cross (SPIR-V reflection; a static
# archive linked privately into gfx_core), tinygltf, ImGui and googletest. VMA
# is the core's: its allocator compiles it, and gfx no longer builds a copy
# (DECISIONS.md, "Memory comes from volumetric_kit_core").

include(FetchContent)

# Public deps (Vulkan, glm's preference, volumetric_kit_core) are resolved at
# the top level so their imported targets are visible across sibling
# subdirectories. This file owns only the vendored deps.

# --- glm: header-only vector/matrix math (camera + assets tiers) --------------
# Only vendored when the top level did not find a system glm. glm's CMake
# defines the same glm::glm target FetchContent exposes, so the tiers link one
# name either way. Pinned to a tagged release; its tests and install are off.
if(NOT VG_HAVE_SYSTEM_GLM)
  set(GLM_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(GLM_BUILD_INSTALL
      OFF
      CACHE BOOL "" FORCE)
  FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.3
    GIT_SHALLOW TRUE)
endif()

# --- tinygltf: glTF 2.0 loader for the io tier (header-only) -----------------
# tinygltf bundles its deps (nlohmann/json + stb_image) in the repo root and is
# implemented in one gfx_io TU (tinygltf_impl.cpp). SOURCE_SUBDIR names a
# nonexistent dir so MakeAvailable populates the sources without running
# tinygltf's own CMakeLists (which builds its loader_example and installs
# targets we do not want). The io tier compiles against this single include dir
# and links no Vulkan -- it is pure CPU (gfx_assets + tinygltf only).
FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG v2.9.7
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR _headers_only_)

# --- Dear ImGui: immediate-mode debug UI (ui tier; Vulkan backend only) -------
# ImGui ships no root CMakeLists -- you compile its sources into your own
# target. SOURCE_SUBDIR names a nonexistent dir so MakeAvailable only populates
# the sources (no add_subdirectory), exactly as tinygltf above; the ui tier
# compiles imgui core + backends/imgui_impl_vulkan into a vendored static lib
# (see src/.../ui/CMakeLists.txt), and gfx_glfw compiles the GLFW platform
# backend (imgui_impl_glfw) from the same sources. Pinned to a tag whose Vulkan
# backend supports dynamic rendering (the kit's only rendering path).
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.92.9
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR _headers_only_)

# --- spirv-cross: SPIR-V reflection (reflection-driven descriptor layouts) ----
# Build spirv-cross-core as a static archive, linked privately into gfx_core,
# without its CLI, tests or install rules.
set(SPIRV_CROSS_STATIC
    ON
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_CLI
    OFF
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_GLSL
    ON
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL
    OFF
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL
    OFF
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_CPP
    OFF
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_REFLECT
    ON
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_UTIL
    OFF
    CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL
    ON
    CACHE BOOL "" FORCE)
FetchContent_Declare(
  spirv-cross
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
  GIT_TAG vulkan-sdk-1.4.357.0
  GIT_SHALLOW TRUE)

# tinygltf + spirv-cross + ImGui are all needed by the library: spirv-cross
# backs gfx_core's SPIR-V reflection (shader.cpp), and ImGui backs the ui tier's
# overlay (imgui_overlay.cpp).
FetchContent_MakeAvailable(tinygltf spirv-cross imgui)

# glm only when no system copy was found (see the declaration guard above).
if(NOT VG_HAVE_SYSTEM_GLM)
  FetchContent_MakeAvailable(glm)
endif()

# tinygltf + its bundled json/stb headers live in the repo root; the io tier
# compiles against this dir and implements the library in one TU.
set(VG_TINYGLTF_INCLUDE_DIR
    "${tinygltf_SOURCE_DIR}"
    PARENT_SCOPE)

# ImGui sources live in the repo root (imgui.cpp, ...) with the renderer backend
# under backends/; the ui tier compiles a curated subset from this dir.
set(VG_IMGUI_SOURCE_DIR
    "${imgui_SOURCE_DIR}"
    PARENT_SCOPE)

if(VG_BUILD_TESTS)
  # googletest is only needed when building the test suite.
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

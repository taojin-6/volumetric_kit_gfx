# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Dependencies. System via find_package: the Vulkan loader (linked at runtime)
# and glslc/glslangValidator (shader compilation) -- re-findable, keeping the
# installed export clean (consumers find_dependency(Vulkan)). glm (camera tier)
# is preferred from the system too (resolved at the top level), falling back to
# a pinned vendored copy below when none is installed. Vendored via pinned
# FetchContent: Vulkan-Headers + Vulkan-Utility-Libraries (the library compiles
# against these for authoritative VkFormat metadata), VMA (header-only, built in
# one gfx_core TU), spirv-cross (SPIR-V reflection; a static archive linked
# privately into gfx_core), and googletest.
#
# The library compiles against the pinned Vulkan-Headers and links the system
# loader -- this also pre-stages the header vendoring the iOS/Android work
# needs.

include(FetchContent)

# Public deps (Vulkan) are resolved at the top level so their imported targets
# are visible across sibling subdirectories. This file owns only the vendored
# deps.

# --- VMA: Vulkan Memory Allocator (header-only; implemented in one gfx_core TU)
# ----
FetchContent_Declare(
  VulkanMemoryAllocator
  GIT_REPOSITORY
    https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.4.0
  GIT_SHALLOW TRUE)

# --- Vulkan-Headers + Vulkan-Utility-Libraries (authoritative VkFormat info) --
# The library compiles against these *pinned* headers (the system loader still
# links at runtime), so format metadata comes from Khronos' vkuFormat* helpers
# -- one source of truth, complete as of its pinned tag -- rather than a
# hand-maintained table. Fetched headers-only: SOURCE_SUBDIR names a nonexistent
# dir so MakeAvailable populates the sources without running their CMakeLists,
# which would otherwise redefine the Vulkan::Headers target that
# find_package(Vulkan) already created. Pinned to the same vulkan-sdk tag as
# spirv-cross. (The Ubuntu CI's libvulkan-dev carries neither the utility
# headers nor recent format enums -- hence vendoring, not the system headers.)
FetchContent_Declare(
  VulkanHeaders
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG vulkan-sdk-1.4.357.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR _headers_only_)
FetchContent_Declare(
  VulkanUtilityLibraries
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Utility-Libraries.git
  GIT_TAG vulkan-sdk-1.4.357.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR _headers_only_)

# --- glm: header-only vector/matrix math (camera + assets tiers) --------------
# Only vendored when the top level did not find a system glm. glm's CMake
# defines the same glm::glm target FetchContent exposes, so the tiers link one
# name either way. Pinned to a tagged release; its tests are off. Its install is
# tied to VG_INSTALL: gfx_camera/gfx_assets link glm PUBLIC, so when we
# install/export we need glm in an export set too -- letting glm run its own
# install (which exports the glm target and ships a glmConfig into our prefix)
# satisfies that and lets the package's find_dependency(glm) resolve the
# co-installed copy.
if(NOT VG_HAVE_SYSTEM_GLM)
  set(GLM_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(GLM_BUILD_INSTALL
      ${VG_INSTALL}
      CACHE BOOL "" FORCE)
  FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.3
    GIT_SHALLOW TRUE)
endif()

# --- tinygltf: glTF 2.0 loader for the io tier (header-only) -----------------
# tinygltf bundles its deps (nlohmann/json + stb_image) in the repo root and is
# implemented in one gfx_io TU (tinygltf_impl.cpp), exactly like VMA above.
# SOURCE_SUBDIR names a nonexistent dir so MakeAvailable populates the sources
# without running tinygltf's own CMakeLists (which builds its loader_example and
# installs targets we do not want). The io tier compiles against this single
# include dir and links no Vulkan -- it is pure CPU (gfx_assets + tinygltf
# only).
FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG v2.9.7
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR _headers_only_)

# --- Dear ImGui: immediate-mode debug UI (ui tier; Vulkan backend only) -------
# ImGui ships no root CMakeLists -- you compile its sources into your own
# target. SOURCE_SUBDIR names a nonexistent dir so MakeAvailable only populates
# the sources (no add_subdirectory), exactly as VulkanHeaders/tinygltf above;
# the ui tier compiles imgui core + backends/imgui_impl_vulkan into a vendored
# static lib (see src/.../ui/CMakeLists.txt). The GLFW platform backend
# (imgui_impl_glfw) is deliberately NOT built here -- it stays in examples, so
# this tier inherits the windowing tier's GLFW-free rule. Pinned to a tag whose
# Vulkan backend supports dynamic rendering (the kit's only rendering path).
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.92.8
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR _headers_only_)

# --- spirv-cross: SPIR-V reflection (reflection-driven descriptor layouts) ----
# Build spirv-cross-core as a static archive. SPIRV_CROSS_STATIC defaults ON
# upstream, but spirv-cross *gates the install/export of the core lib* on it
# (its CMakeLists guards the core-lib install block with
# `if(SPIRV_CROSS_STATIC)`) -- so pin it ON to keep our packaging self-contained
# under VG_INSTALL rather than reliant on that default holding.
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
# Tie spirv-cross's own install to VG_INSTALL: gfx_core links spirv-cross-core
# (privately, for SPIR-V reflection), so when we install/export,
# spirv-cross-core must live in an export set too. Letting spirv-cross run its
# own install (exporting its targets + shipping spirv_cross_*Config into our
# prefix) satisfies that and lets the package's
# find_dependency(spirv_cross_core) resolve it -- the same arrangement glm uses
# above.
if(VG_INSTALL)
  set(SPIRV_CROSS_SKIP_INSTALL
      OFF
      CACHE BOOL "" FORCE)
else()
  set(SPIRV_CROSS_SKIP_INSTALL
      ON
      CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(
  spirv-cross
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
  GIT_TAG vulkan-sdk-1.4.357.0
  GIT_SHALLOW TRUE)

# VMA + the vendored Vulkan headers + tinygltf + spirv-cross + ImGui are all
# needed by the library: spirv-cross backs gfx_core's SPIR-V reflection
# (shader.cpp), and ImGui backs the ui tier's overlay (imgui_overlay.cpp).
FetchContent_MakeAvailable(VulkanMemoryAllocator VulkanHeaders
                           VulkanUtilityLibraries tinygltf spirv-cross imgui)

# glm only when no system copy was found (see the declaration guard above).
if(NOT VG_HAVE_SYSTEM_GLM)
  FetchContent_MakeAvailable(glm)
endif()

# Expose the vendored header locations to the library target. VMA is implemented
# in one TU; the Vulkan-Headers + Utility-Libraries dirs are what gfx_core
# compiles against (see the BUILD_INTERFACE include in core/CMakeLists.txt).
set(VG_VMA_INCLUDE_DIR
    "${vulkanmemoryallocator_SOURCE_DIR}/include"
    PARENT_SCOPE)
set(VG_VULKAN_HEADERS_INCLUDE
    "${vulkanheaders_SOURCE_DIR}/include"
    PARENT_SCOPE)
set(VG_VULKAN_UTILITY_INCLUDE
    "${vulkanutilitylibraries_SOURCE_DIR}/include"
    PARENT_SCOPE)

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
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

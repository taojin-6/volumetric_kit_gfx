# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Dependencies. System via find_package: Vulkan (headers + loader) -- widely
# packaged and re-findable, keeping the installed export clean. (glm joins this
# list when the camera/assets tiers land; no tier names a glm type yet.)
# Vendored via pinned FetchContent: VMA (header-only, built in one gfx_core TU),
# spirv-cross (reflection; built by tests now to prove it compiles), and
# googletest.
#
# Desktop foundation: Vulkan comes from the system; deferred iOS/Android work
# will vendor a pinned Vulkan-Headers for those targets.

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
  GIT_TAG v3.3.0
  GIT_SHALLOW TRUE)

# --- spirv-cross: SPIR-V reflection (reflection-driven descriptor layouts) ----
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
  GIT_TAG vulkan-sdk-1.4.350.0
  GIT_SHALLOW TRUE)

# VMA is always needed by the library; only fetch the SPIR-V sources here.
FetchContent_MakeAvailable(VulkanMemoryAllocator)

# Expose VMA's header location to the library target (implemented in one TU
# there).
set(VG_VMA_INCLUDE_DIR
    "${vulkanmemoryallocator_SOURCE_DIR}/include"
    PARENT_SCOPE)

if(VG_BUILD_TESTS)
  # spirv-cross + googletest are only needed when building the test suite.
  # Pulling spirv-cross now (even before the core tier uses it) proves the
  # reflection toolchain compiles on every CI platform ahead of its first use.
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(spirv-cross googletest)
endif()

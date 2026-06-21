# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

cmake_minimum_required(VERSION 3.21)

project(
  volumetric_kit_gfx
  VERSION 0.0.1
  DESCRIPTION
    "Standalone, reusable Vulkan renderer library (MoltenVK on Apple) for volumetric workloads"
  HOMEPAGE_URL "https://github.com/taojin-6/volumetric_kit_gfx"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Are we the top-level project, or pulled in via add_subdirectory/FetchContent?
# Defaults for "build the extras" and "generate install rules" follow from this.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(VG_IS_TOP_LEVEL ON)
else()
  set(VG_IS_TOP_LEVEL OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Options -----------------------------------------------------------------
option(VG_BUILD_EXAMPLES "Build example programs" ${VG_IS_TOP_LEVEL})
option(VG_BUILD_TESTS "Build tests" ${VG_IS_TOP_LEVEL})
option(VG_INSTALL "Generate install/export rules" ${VG_IS_TOP_LEVEL})
# TODO: VG_WITH_GLFW / VG_WITH_CUDA are reserved for the windowing / interop
# tiers and are not consumed yet (no target reads them, so toggling them is a
# no-op). Wire them up — or drop them and the matching CI GLFW packages — when
# those tiers land.
option(VG_WITH_GLFW "Enable GLFW desktop windowing (reserved; not yet wired)"
       ON)
option(
  VG_WITH_CUDA
  "Enable CUDA <-> Vulkan external-memory interop (reserved; not yet wired)"
  OFF)
option(VG_WARNINGS_AS_ERRORS "Treat first-party compiler warnings as errors" ON)

# Validation is a runtime decision (InstanceConfig::enable_validation), gated at
# instance creation by layer availability — not a compile-time switch. The debug
# messenger/layer wiring is negligible code size, so there is no build option to
# strip it.

# Semicolon list of sanitizers (e.g. "address;undefined"); empty = off. Applied
# globally — see cmake/vg_sanitizers.cmake for why.
set(VG_SANITIZE
    ""
    CACHE STRING "Sanitizers to enable, e.g. address;undefined (empty = off)")

include(vg_warnings)
include(vg_sanitizers)
include(vg_shaders)

# --- Public dependencies -----------------------------------------------------
# Resolved at the top level so their imported targets are visible to every
# subdirectory, not just the one find_package() ran in. Provides Vulkan::Headers
# (the public-API types) and Vulkan::Vulkan (the loader our implementation links
# privately).
find_package(Vulkan REQUIRED)

# glm: the camera tier's only dependency (its public matrix/vector types).
# Prefer a system glm (its config defines glm::glm); fall back to the pinned
# vendored copy from third_party/ when none is installed -- so the build is
# self-contained on CI runners that don't ship glm, matching how the other deps
# are vendored. VG_HAVE_SYSTEM_GLM records which path won, so third_party/ only
# vendors glm (and lets it run its own install/export) when no system copy
# exists. glm is a PUBLIC dependency either way, so the installed package config
# re-finds it unconditionally (find_dependency(glm)).
find_package(glm CONFIG QUIET)
if(glm_FOUND)
  set(VG_HAVE_SYSTEM_GLM ON)
else()
  set(VG_HAVE_SYSTEM_GLM OFF)
endif()

# --- Dependencies + library --------------------------------------------------
add_subdirectory(third_party)
add_subdirectory(src)

if(VG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Guard: examples/ may not exist yet, so don't error when it is absent.
#
# TODO: drop the guard once examples/ is added.
if(VG_BUILD_EXAMPLES AND EXISTS
                         "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()

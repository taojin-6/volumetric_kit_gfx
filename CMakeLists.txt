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
option(VG_WITH_GLFW "Enable GLFW desktop windowing" ON)
option(VG_WITH_CUDA "Enable CUDA <-> Vulkan external-memory interop" OFF)
# Off: gfx_core compiles the absorbed VMA implementation TU, and -Werror trips
# on third-party warnings from it.
#
# TODO: isolate that TU from the first-party warning flags, then default this
# ON.
option(VG_WARNINGS_AS_ERRORS "Treat first-party compiler warnings as errors"
       OFF)

# Validation layers are useful in unoptimized builds and a needless cost
# otherwise.
if(CMAKE_BUILD_TYPE STREQUAL "Debug"
   OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"
   OR NOT CMAKE_BUILD_TYPE)
  set(_vg_validation_default ON)
else()
  set(_vg_validation_default OFF)
endif()
option(VG_WITH_VALIDATION "Enable Vulkan validation layers + debug messenger"
       ${_vg_validation_default})

include(vg_warnings)

# --- Public dependencies -----------------------------------------------------
# Resolved at the top level so their imported targets (Vulkan::Headers,
# glm::glm) are visible to every subdirectory, not just the one find_package()
# ran in. Provides Vulkan::Headers (the public-API types) and Vulkan::Vulkan
# (the loader our implementation links privately).
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

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

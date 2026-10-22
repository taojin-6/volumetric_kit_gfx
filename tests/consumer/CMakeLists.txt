# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# A stand-in for an application that fetches gfx (the player, ios), configured
# by tests/CMakeLists.txt with -DVG_SOURCE_DIR=<gfx> -DVKC_SOURCE_DIR=<core>. It
# makes the core available first, with the vulkan tier on, so gfx's pin yields
# to it, and gfx is not the top-level project, so its examples, tests and
# -Werror must stay off.
cmake_minimum_required(VERSION 3.21)
project(vg_consumer LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(volumetric_kit_core SOURCE_DIR "${VKC_SOURCE_DIR}")
set(VKC_WITH_VULKAN ON)
FetchContent_MakeAvailable(volumetric_kit_core)
# Cleared first, so a reconfigure checks gfx's defaults, not a cached value.
set(_options VG_BUILD_EXAMPLES VG_BUILD_TESTS VG_WARNINGS_AS_ERRORS)
foreach(_option IN LISTS _options)
  unset(${_option} CACHE)
endforeach()
add_subdirectory("${VG_SOURCE_DIR}" volumetric_kit_gfx)
foreach(_option IN LISTS _options)
  if(${_option})
    message(FATAL_ERROR "${_option} must default OFF for a gfx subproject")
  endif()
endforeach()

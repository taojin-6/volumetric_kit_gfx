# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# A stand-in for a project that consumes gfx, configured by tests/CMakeLists.txt
# the two ways the family does:
#
# -DVG_CONSUME=subdirectory -DVG_SOURCE_DIR=<gfx> -DVKC_SOURCE_DIR=<core>
# -DVG_CONSUME=find_package -DCMAKE_PREFIX_PATH=<install prefixes>
#
# The first is an application that fetches gfx (ios): it makes the core
# available first, with VKC_WITH_VULKAN as given, so gfx's pin yields to it, and
# gfx is not the top-level project, so its examples, tests, install rules and
# -Werror must stay off. -DVG_CONSUMER_CORE_VERSION=<version> stamps that
# version on the core, standing in for a core older than it is.
cmake_minimum_required(VERSION 3.21)
project(vg_consumer LANGUAGES CXX)

set(VG_CONSUME
    "find_package"
    CACHE STRING "find_package or subdirectory")
set(VG_CONSUMER_CORE_VERSION
    ""
    CACHE STRING "A version to stamp on the core (subdirectory only)")

if(VG_CONSUME STREQUAL "find_package")
  find_package(volumetric_kit_gfx REQUIRED CONFIG)
elseif(VG_CONSUME STREQUAL "subdirectory")
  include(FetchContent)
  FetchContent_Declare(volumetric_kit_core SOURCE_DIR "${VKC_SOURCE_DIR}")
  FetchContent_MakeAvailable(volumetric_kit_core)
  if(VG_CONSUMER_CORE_VERSION)
    get_target_property(_core volumetric_kit::core_base ALIASED_TARGET)
    set_target_properties(${_core} PROPERTIES VKC_CORE_VERSION
                                              ${VG_CONSUMER_CORE_VERSION})
  endif()
  # Cleared first, so a reconfigure checks gfx's defaults, not a cached value.
  set(_options VG_BUILD_EXAMPLES VG_BUILD_TESTS VG_INSTALL
               VG_WARNINGS_AS_ERRORS)
  foreach(_option IN LISTS _options)
    unset(${_option} CACHE)
  endforeach()
  add_subdirectory("${VG_SOURCE_DIR}" volumetric_kit_gfx)
  foreach(_option IN LISTS _options)
    if(${_option})
      message(FATAL_ERROR "${_option} must default OFF for a gfx subproject")
    endif()
  endforeach()
else()
  message(FATAL_ERROR "VG_CONSUME must be find_package or subdirectory")
endif()

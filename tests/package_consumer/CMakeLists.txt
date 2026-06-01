# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Standalone downstream consumer used by the `package` CI job — NOT part of the
# main build (tests/CMakeLists.txt does not add it). Configured two ways: -
# find_package: cmake -S . -DCMAKE_PREFIX_PATH=<install prefix> -
# add_subdirectory (FetchContent-style): cmake -S . -DVG_CONSUMER_FETCH=ON \
# -DVG_SOURCE_DIR=<repo root>
cmake_minimum_required(VERSION 3.21)
project(vg_package_consumer LANGUAGES CXX)

option(VG_CONSUMER_FETCH
       "Consume via add_subdirectory(VG_SOURCE_DIR) instead of find_package"
       OFF)

if(VG_CONSUMER_FETCH)
  if(NOT VG_SOURCE_DIR)
    message(
      FATAL_ERROR "VG_CONSUMER_FETCH requires -DVG_SOURCE_DIR=<repo root>")
  endif()
  # Mirrors a FetchContent consumer: the library is added as a subproject with
  # VG_IS_TOP_LEVEL OFF, so its tests/examples/install default off.
  add_subdirectory("${VG_SOURCE_DIR}" volumetric_kit_gfx)
else()
  find_package(volumetric_kit_gfx CONFIG REQUIRED)
endif()

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE volumetric_kit::gfx_core)

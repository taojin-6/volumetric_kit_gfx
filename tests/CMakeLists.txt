# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

include(GoogleTest)

add_executable(vg_version_test version_test.cpp)
target_link_libraries(
  vg_version_test
  PRIVATE volumetric_kit::gfx_core
          GTest::gtest_main
          # spirv-cross is not used yet, but linking it forces CMake to build
          # the reflection library on every CI platform now -- an early
          # compile-check ahead of its first use.
          spirv-cross-core)
vg_target_warnings(vg_version_test)
gtest_discover_tests(vg_version_test)

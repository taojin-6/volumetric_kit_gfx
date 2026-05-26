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

# core tier: Result/Status (CPU-only) + Instance/Device (GPU; skips when no
# device).
add_executable(
  vg_core_test result_test.cpp device_test.cpp log_test.cpp sync_test.cpp
               retire_list_test.cpp retire_queue_test.cpp)
target_link_libraries(vg_core_test PRIVATE volumetric_kit::gfx_core
                                           GTest::gtest_main)
vg_target_warnings(vg_core_test)
gtest_discover_tests(vg_core_test)

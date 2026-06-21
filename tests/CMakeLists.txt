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
  vg_core_test
  result_test.cpp
  device_test.cpp
  log_test.cpp
  sync_test.cpp
  unique_handle_test.cpp
  retire_list_test.cpp
  retire_queue_test.cpp
  allocator_test.cpp
  shader_test.cpp
  graphics_pipeline_test.cpp
  render_target_test.cpp
  offscreen_target_test.cpp
  command_buffer_test.cpp)
target_link_libraries(vg_core_test PRIVATE volumetric_kit::gfx_core
                                           GTest::gtest_main)
vg_target_warnings(vg_core_test)

# shader_test loads a compiled SPIR-V blob; compile the triangle shaders next to
# the test binary and pass it the directory via VG_SHADER_DIR.
vg_compile_shaders(
  vg_core_test
  OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/shaders"
  SHADERS
  "${PROJECT_SOURCE_DIR}/shaders/triangle.vert"
  "${PROJECT_SOURCE_DIR}/shaders/triangle.frag"
  "${PROJECT_SOURCE_DIR}/shaders/mesh.vert"
  "${PROJECT_SOURCE_DIR}/shaders/mesh.frag")
target_compile_definitions(
  vg_core_test PRIVATE VG_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")

gtest_discover_tests(vg_core_test)

# windowing tier: surface / swapchain / FrameLoop driven through a headless
# surface (VK_EXT_headless_surface). Skips wholesale where that surface or a
# present-capable device is unavailable (e.g. MoltenVK).
add_executable(vg_windowing_test windowing_test.cpp)
target_link_libraries(vg_windowing_test PRIVATE volumetric_kit::gfx_windowing
                                                GTest::gtest_main)
vg_target_warnings(vg_windowing_test)
gtest_discover_tests(vg_windowing_test)

# camera tier: Camera / OrbitCamera math. Pure CPU (glm only, no Vulkan), so
# every case runs on every platform regardless of device availability.
add_executable(vg_camera_test camera_test.cpp)
target_link_libraries(vg_camera_test PRIVATE volumetric_kit::gfx_camera
                                             GTest::gtest_main)
vg_target_warnings(vg_camera_test)
gtest_discover_tests(vg_camera_test)

# assets tier: format-neutral model + glTF loader (CPU-only; runs anywhere, no
# device needed). The glTF test loads the Khronos Box.glb sample (CC0); fetch it
# at configure time so CI does not need it committed. The download is best-
# effort -- if the network is unavailable the file is simply absent and the
# glTF-loading tests skip (the PointCloud tests need no fixture). VG_ASSETS_DIR
# is defined only when the fixture is present.
set(_vg_assets_dir "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(_vg_box_glb "${_vg_assets_dir}/Box.glb")
if(NOT EXISTS "${_vg_box_glb}")
  file(MAKE_DIRECTORY "${_vg_assets_dir}")
  message(STATUS "assets test: fetching Box.glb fixture")
  file(
    DOWNLOAD
    "https://github.com/KhronosGroup/glTF-Sample-Assets/raw/main/Models/Box/glTF-Binary/Box.glb"
    "${_vg_box_glb}"
    STATUS _vg_box_status
    TIMEOUT 30)
  list(GET _vg_box_status 0 _vg_box_rc)
  if(NOT _vg_box_rc EQUAL 0)
    list(GET _vg_box_status 1 _vg_box_err)
    message(
      STATUS
        "assets test: Box.glb download failed (${_vg_box_err}); glTF tests will skip"
    )
    file(REMOVE "${_vg_box_glb}")
  endif()
endif()

add_executable(vg_assets_test assets_test.cpp)
target_link_libraries(vg_assets_test PRIVATE volumetric_kit::gfx_assets
                                             GTest::gtest_main)
vg_target_warnings(vg_assets_test)
if(EXISTS "${_vg_box_glb}")
  target_compile_definitions(vg_assets_test
                             PRIVATE VG_ASSETS_DIR="${_vg_assets_dir}")
endif()
# Prefix the ctest names ("assets.GltfLoader.*", "assets.PointCloud.*") so this
# CPU-only tier is selectable as a group with `ctest -R assets`.
gtest_discover_tests(vg_assets_test TEST_PREFIX "assets.")

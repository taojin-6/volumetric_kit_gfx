# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

include(GoogleTest)

# ctest cannot see a GTEST_SKIP() on its own: the binary still exits 0, so a
# skipped case reports "Passed" unless a SKIP_REGULAR_EXPRESSION matches gtest's
# skip banner. That matters because most of this suite skips wholesale when no
# Vulkan device is present -- so a GPU leg that quietly lost its device (a
# reconfigured container runtime, a driver upgrade, a changed
# NVIDIA_DRIVER_CAPABILITIES) would validate nothing while `ci / required`
# stayed green, indefinitely and invisibly. sanitizers.yml already takes the
# stricter line for its own leg (hard-fail when no device comes up); this is the
# reporting half of the same policy.
#
# CMake's GoogleTest module sets that property itself on every test it
# discovers, in every version since before the project's 3.21 floor (3.20's
# GoogleTestAddTests.cmake already does), so pass nothing of our own. A second
# SKIP_REGULAR_EXPRESSION is worse than redundant: before 3.26 the module hands
# the properties through a macro, which re-parses them, so "\\[  SKIPPED \\]"
# arrives as the character class "[  SKIPPED ]". That matches nearly any output,
# and as the last value set it overrode the module's correct one: on Ubuntu
# 22.04's CMake 3.22 every test, passing or failing, was reported "Skipped".
# From 3.26 a second one is a malformed duplicate.
#
# VG_TEST_SHARDS > 0 registers each test binary as that many CTest entries
# instead of one per test: shard i runs the binary with GTEST_TOTAL_SHARDS and
# GTEST_SHARD_INDEX set, so its share of the tests runs in one process and
# shares the core test fixtures' Vulkan devices. CI uses it, as on NVIDIA's
# Linux driver every device costs ~70-90 ms of work the driver serializes
# machine-wide. 0, the default, keeps one entry per test, so `ctest -R <test>`
# still runs a single test.
#
# A shard gets no SKIP_REGULAR_EXPRESSION: one skipped test among its many would
# mark the whole shard skipped, failures included. It passes or fails on its
# exit code alone. Skips stay visible where it matters through
# VKC_REQUIRE_VULKAN_DEVICE, which CI sets so that a fixture test without a
# Vulkan device fails rather than skips.
set(VG_TEST_SHARDS
    0
    CACHE STRING
          "Run each test binary as this many CTest shards (0: one per test)")

# The GPU tests derive from the core's test fixtures
# (tests/gfx_test_support.hpp; the core's README, "Use it in your project"),
# which gfx's top-level CMakeLists.txt asks the core to build.
if(NOT TARGET volumetric_kit::core_test_support)
  message(
    FATAL_ERROR
      "gfx's tests need volumetric_kit::core_test_support: set "
      "VKC_BUILD_TEST_SUPPORT ON before the first "
      "FetchContent_MakeAvailable(volumetric_kit_core); an installed core has "
      "none.")
endif()

function(vg_discover_tests target)
  if(NOT VG_TEST_SHARDS GREATER 0)
    gtest_discover_tests(${target} ${ARGN})
    return()
  endif()
  cmake_parse_arguments(arg "" "TEST_PREFIX" "" ${ARGN})
  math(EXPR last "${VG_TEST_SHARDS} - 1")
  foreach(index RANGE ${last})
    set(name "${arg_TEST_PREFIX}${target}.shard${index}")
    add_test(NAME "${name}" COMMAND ${target})
    set_tests_properties(
      "${name}"
      PROPERTIES
        ENVIRONMENT
        "GTEST_TOTAL_SHARDS=${VG_TEST_SHARDS};GTEST_SHARD_INDEX=${index}")
  endforeach()
endfunction()

add_executable(vg_version_test version_test.cpp)
target_link_libraries(vg_version_test PRIVATE volumetric_kit::gfx_core
                                              GTest::gtest_main)
vg_target_warnings(vg_version_test)
vg_discover_tests(vg_version_test)

# core tier: FrameMetrics (CPU-only) + Instance/Device (GPU; skips when no
# device).
add_executable(
  vg_core_test
  frame_metrics_test.cpp
  device_test.cpp
  gfx_test_support_test.cpp
  debug_label_test.cpp
  log_test.cpp
  profiler_test.cpp
  retire_list_test.cpp
  retire_queue_test.cpp
  shader_test.cpp
  graphics_pipeline_test.cpp
  render_target_test.cpp
  image_barrier_test.cpp
  buffer_barrier_test.cpp
  offscreen_target_test.cpp
  sampler_test.cpp
  texture_upload_test.cpp)
target_link_libraries(
  vg_core_test PRIVATE volumetric_kit::gfx_core
                       volumetric_kit::core_test_support GTest::gtest_main)
vg_target_warnings(vg_core_test)

# shader_test loads a compiled SPIR-V blob; compile the triangle shaders next to
# the test binary and pass it the directory via VG_SHADER_DIR.
vkc_compile_shaders(
  vg_core_test
  OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/shaders"
  TARGET_ENV
  vulkan1.3
  SHADERS
  "${PROJECT_SOURCE_DIR}/shaders/triangle.vert"
  "${PROJECT_SOURCE_DIR}/shaders/triangle.frag"
  "${PROJECT_SOURCE_DIR}/shaders/mesh.vert"
  "${PROJECT_SOURCE_DIR}/shaders/mesh.frag"
  "${PROJECT_SOURCE_DIR}/shaders/mesh_mvp.vert"
  "${PROJECT_SOURCE_DIR}/shaders/reflect_probe.frag"
  "${PROJECT_SOURCE_DIR}/shaders/reflect_storage.comp"
  "${PROJECT_SOURCE_DIR}/shaders/spec_probe.frag")
target_compile_definitions(
  vg_core_test PRIVATE VG_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")

vg_discover_tests(vg_core_test)

# windowing tier: surface / swapchain / FrameLoop driven through a headless
# surface (VK_EXT_headless_surface). Skips wholesale where that surface or a
# present-capable device is unavailable.
add_executable(vg_windowing_test windowing_test.cpp)
target_link_libraries(
  vg_windowing_test PRIVATE volumetric_kit::gfx_windowing
                            volumetric_kit::core_test_support GTest::gtest_main)
vg_target_warnings(vg_windowing_test)
vg_discover_tests(vg_windowing_test)

# app tier: the WindowedApp/HeadlessApp bring-up facades. WindowedApp is driven
# through a VK_EXT_headless_surface factory (the windowing-test vehicle) and
# skips where that surface or a present-capable device is unavailable;
# HeadlessApp runs wherever a Vulkan device exists.
add_executable(vg_app_test app_test.cpp)
target_link_libraries(
  vg_app_test PRIVATE volumetric_kit::gfx_app volumetric_kit::core_test_support
                      GTest::gtest_main)
vg_target_warnings(vg_app_test)
vg_discover_tests(vg_app_test TEST_PREFIX "app.")

# camera tier: Camera / OrbitCamera / CameraRig math. Pure CPU (glm only, no
# Vulkan), so every case runs on every platform regardless of device
# availability.
add_executable(vg_camera_test camera_test.cpp camera_rig_test.cpp
                              image_view_2d_test.cpp)
target_link_libraries(vg_camera_test PRIVATE volumetric_kit::gfx_camera
                                             GTest::gtest_main)
vg_target_warnings(vg_camera_test)
vg_discover_tests(vg_camera_test)

# pipelines tier: GPU mesh upload + the PBR building blocks, driven through the
# shared device fixture (skips when no device). Links gfx_pipelines, which
# brings core + assets, plus gfx_io so pbr_model_test can drive the real
# load_gltf -> PbrModel::create path against the Box.glb fixture (defined below,
# next to the fixture download).
add_executable(
  vg_pipelines_test
  gpu_mesh_test.cpp
  hybrid_mesh_pipeline_test.cpp
  ibl_test.cpp
  image_pipeline_test.cpp
  live_mesh_test.cpp
  pbr_material_test.cpp
  pbr_model_test.cpp
  pbr_pipeline_test.cpp
  pbr_scene_test.cpp)
target_link_libraries(
  vg_pipelines_test
  PRIVATE volumetric_kit::gfx_pipelines volumetric_kit::gfx_io
          volumetric_kit::core_test_support GTest::gtest_main)
vg_target_warnings(vg_pipelines_test)
# image_pipeline_test's producer: a compute pass writing a source plane, so the
# update's barriers are checked against a writer in a stage of its own.
vkc_compile_shaders(
  vg_pipelines_test
  OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/shaders"
  TARGET_ENV
  vulkan1.3
  SHADERS
  "${PROJECT_SOURCE_DIR}/shaders/write_words.comp")
target_compile_definitions(
  vg_pipelines_test PRIVATE VG_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
vg_discover_tests(vg_pipelines_test TEST_PREFIX "pipelines.")

# assets tier: the format-neutral model is header-only, so its test (PointCloud)
# links gfx_assets alone -- no loader, no parser, no fixture -- which proves the
# model stands on its own.
add_executable(vg_assets_test assets_test.cpp)
target_link_libraries(vg_assets_test PRIVATE volumetric_kit::gfx_assets
                                             GTest::gtest_main)
vg_target_warnings(vg_assets_test)
# Prefix the ctest names ("assets.PointCloud.*") so this CPU-only tier is
# selectable as a group with `ctest -R assets`.
vg_discover_tests(vg_assets_test TEST_PREFIX "assets.")

# io tier: the glTF loader test loads the Khronos Box.glb sample (CC0). Fetch it
# at configure time so CI does not need it committed; the download is
# best-effort -- if the network is unavailable the file is simply absent and the
# glTF tests skip. VG_ASSETS_DIR is defined only when the fixture is present.
set(_vg_assets_dir "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(_vg_box_glb "${_vg_assets_dir}/Box.glb")
# Pinned to a specific glTF-Sample-Assets commit (not the moving `main`) plus a
# content hash, so the fixture is reproducible and a tampered/truncated payload
# is rejected rather than silently tested against.
set(_vg_box_url
    "https://github.com/KhronosGroup/glTF-Sample-Assets/raw/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/Box/glTF-Binary/Box.glb"
)
set(_vg_box_sha256
    "ed52f7192b8311d700ac0ce80644e3852cd01537e4d62241b9acba023da3d54e")
# Drop a stale/corrupt prior download (e.g. a truncated file from an interrupted
# run) so the verified fetch below replaces it instead of testing against junk.
if(EXISTS "${_vg_box_glb}")
  file(SHA256 "${_vg_box_glb}" _vg_box_have)
  if(NOT _vg_box_have STREQUAL _vg_box_sha256)
    file(REMOVE "${_vg_box_glb}")
  endif()
endif()
if(NOT EXISTS "${_vg_box_glb}")
  file(MAKE_DIRECTORY "${_vg_assets_dir}")
  message(STATUS "io test: fetching Box.glb fixture")
  file(
    DOWNLOAD "${_vg_box_url}" "${_vg_box_glb}"
    EXPECTED_HASH SHA256=${_vg_box_sha256}
    STATUS _vg_box_status
    TIMEOUT 30)
  list(GET _vg_box_status 0 _vg_box_rc)
  if(NOT _vg_box_rc EQUAL 0)
    list(GET _vg_box_status 1 _vg_box_err)
    message(
      STATUS "io test: Box.glb unavailable (${_vg_box_err}); glTF tests skip")
    # A failed EXPECTED_HASH download can leave a partial file; remove it so a
    # later configure refetches and the tests skip cleanly in the meantime.
    file(REMOVE "${_vg_box_glb}")
  endif()
endif()

add_executable(vg_io_test io_test.cpp)
target_link_libraries(vg_io_test PRIVATE volumetric_kit::gfx_io
                                         GTest::gtest_main)
vg_target_warnings(vg_io_test)
if(EXISTS "${_vg_box_glb}")
  target_compile_definitions(vg_io_test
                             PRIVATE VG_ASSETS_DIR="${_vg_assets_dir}")
  # pbr_model_test (in vg_pipelines_test, above) drives the same fixture through
  # load_gltf -> PbrModel::create; gate its define the same way.
  target_compile_definitions(vg_pipelines_test
                             PRIVATE VG_ASSETS_DIR="${_vg_assets_dir}")
endif()
# Prefix the ctest names ("io.GltfLoader.*") so this CPU-only tier is selectable
# as a group with `ctest -R io`.
vg_discover_tests(vg_io_test TEST_PREFIX "io.")

# ui tier: ImGuiOverlay driven headlessly -- create + new_frame + render into an
# offscreen target (dynamic rendering), the move-only lifecycle, and argument
# validation. The GPU cases skip where no Vulkan device is available; the test
# links gfx_ui (which exposes imgui.h for the UI-building calls). It compiles
# imgui_demo.cpp itself for ShowDemoWindow (the tier keeps the demo out of the
# library), and links Vulkan::Vulkan directly because it records vkCmd*/
# vkQueueSubmit (the loader does not propagate through a shared gfx_ui for the
# test's own calls).
add_executable(vg_ui_test ui_test.cpp "${VG_IMGUI_SOURCE_DIR}/imgui_demo.cpp")
target_link_libraries(
  vg_ui_test PRIVATE volumetric_kit::gfx_ui volumetric_kit::core_test_support
                     GTest::gtest_main Vulkan::Vulkan)
vg_target_warnings(vg_ui_test)
set_source_files_properties("${VG_IMGUI_SOURCE_DIR}/imgui_demo.cpp"
                            PROPERTIES COMPILE_OPTIONS "-w")
vg_discover_tests(vg_ui_test TEST_PREFIX "ui.")

# gfx as its consumers take it (consumer/), configure only, with the dependency
# sources this build fetched: as an application's subproject, whose defaults
# hold, and as an installed package.
set(_vg_consumer
    "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_SOURCE_DIR}/consumer" -G
    "${CMAKE_GENERATOR}" "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
if(CMAKE_PREFIX_PATH)
  string(REPLACE ";" "$<SEMICOLON>" _vg_prefixes "${CMAKE_PREFIX_PATH}")
  list(APPEND _vg_consumer "-DCMAKE_PREFIX_PATH=${_vg_prefixes}")
endif()
set(_vg_consumer_dir "${CMAKE_CURRENT_BINARY_DIR}/consumer")

FetchContent_GetProperties(volumetric_kit_core SOURCE_DIR _vg_core_source)
set(_vg_subdirectory
    -DVG_CONSUME=subdirectory "-DVG_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DVKC_SOURCE_DIR=${_vg_core_source}")
foreach(_vg_dep VulkanMemoryAllocator tinygltf spirv-cross imgui glm)
  unset(_vg_dep_source)
  FetchContent_GetProperties(${_vg_dep} SOURCE_DIR _vg_dep_source)
  if(_vg_dep_source)
    string(TOUPPER "${_vg_dep}" _vg_dep_upper)
    list(APPEND _vg_subdirectory
         "-DFETCHCONTENT_SOURCE_DIR_${_vg_dep_upper}=${_vg_dep_source}")
  endif()
endforeach()

add_test(NAME vg_consumer_subdirectory
         COMMAND ${_vg_consumer} -B "${_vg_consumer_dir}/subdirectory"
                 ${_vg_subdirectory} -DVKC_WITH_VULKAN=ON)

if(VG_INSTALL)
  set(_vg_stage "${_vg_consumer_dir}/stage")
  add_test(NAME vg_consumer_install
           COMMAND "${CMAKE_COMMAND}" --install "${PROJECT_BINARY_DIR}"
                   --config "$<CONFIG>" --prefix "${_vg_stage}")
  set_tests_properties(vg_consumer_install PROPERTIES FIXTURES_SETUP
                                                      vg_consumer_stage)

  add_test(NAME vg_consumer_find_package
           COMMAND ${_vg_consumer} -B "${_vg_consumer_dir}/find_package"
                   -DVG_CONSUME=find_package "-DCMAKE_PREFIX_PATH=${_vg_stage}")
  set_tests_properties(vg_consumer_find_package PROPERTIES FIXTURES_REQUIRED
                                                           vg_consumer_stage)
endif()

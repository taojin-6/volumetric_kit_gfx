# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

include(GoogleTest)

add_executable(vg_version_test version_test.cpp)
target_link_libraries(vg_version_test PRIVATE volumetric_kit::gfx_core
                                              GTest::gtest_main)
vg_target_warnings(vg_version_test)
gtest_discover_tests(vg_version_test)

# core tier: Result/Status + FrameMetrics (CPU-only) + Instance/Device (GPU;
# skips when no device).
add_executable(
  vg_core_test
  result_test.cpp
  frame_metrics_test.cpp
  device_test.cpp
  debug_label_test.cpp
  log_test.cpp
  sync_test.cpp
  query_pool_test.cpp
  profiler_test.cpp
  unique_handle_test.cpp
  retire_list_test.cpp
  retire_queue_test.cpp
  allocator_test.cpp
  shader_test.cpp
  descriptor_test.cpp
  graphics_pipeline_test.cpp
  render_target_test.cpp
  offscreen_target_test.cpp
  command_buffer_test.cpp
  sampler_test.cpp
  texture_upload_test.cpp)
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
  "${PROJECT_SOURCE_DIR}/shaders/mesh.frag"
  "${PROJECT_SOURCE_DIR}/shaders/mesh_mvp.vert"
  "${PROJECT_SOURCE_DIR}/shaders/reflect_probe.frag"
  "${PROJECT_SOURCE_DIR}/shaders/reflect_storage.comp")
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

# pipelines tier: GPU mesh upload, driven through the shared device fixture
# (skips when no device). Links gfx_pipelines, which brings core + assets.
add_executable(vg_pipelines_test gpu_mesh_test.cpp pbr_material_test.cpp
                                 pbr_pipeline_test.cpp pbr_scene_test.cpp)
target_link_libraries(vg_pipelines_test PRIVATE volumetric_kit::gfx_pipelines
                                                GTest::gtest_main)
vg_target_warnings(vg_pipelines_test)
gtest_discover_tests(vg_pipelines_test TEST_PREFIX "pipelines.")

# assets tier: the format-neutral model is header-only, so its test (PointCloud)
# links gfx_assets alone -- no loader, no parser, no fixture -- which proves the
# model stands on its own.
add_executable(vg_assets_test assets_test.cpp)
target_link_libraries(vg_assets_test PRIVATE volumetric_kit::gfx_assets
                                             GTest::gtest_main)
vg_target_warnings(vg_assets_test)
# Prefix the ctest names ("assets.PointCloud.*") so this CPU-only tier is
# selectable as a group with `ctest -R assets`.
gtest_discover_tests(vg_assets_test TEST_PREFIX "assets.")

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
endif()
# Prefix the ctest names ("io.GltfLoader.*") so this CPU-only tier is selectable
# as a group with `ctest -R io`.
gtest_discover_tests(vg_io_test TEST_PREFIX "io.")

# ui tier: ImGuiOverlay driven headlessly -- create + new_frame + render into an
# offscreen target (dynamic rendering), the move-only lifecycle, and argument
# validation. The GPU cases skip where no Vulkan device is available; the test
# links gfx_ui (which exposes imgui.h for the UI-building calls). It compiles
# imgui_demo.cpp itself for ShowDemoWindow (the tier keeps the demo out of the
# library), and links Vulkan::Vulkan directly because it records vkCmd*/
# vkQueueSubmit (the loader does not propagate through a shared gfx_ui for the
# test's own calls).
add_executable(vg_ui_test ui_test.cpp "${VG_IMGUI_SOURCE_DIR}/imgui_demo.cpp")
target_link_libraries(vg_ui_test PRIVATE volumetric_kit::gfx_ui
                                         GTest::gtest_main Vulkan::Vulkan)
vg_target_warnings(vg_ui_test)
set_source_files_properties("${VG_IMGUI_SOURCE_DIR}/imgui_demo.cpp"
                            PROPERTIES COMPILE_OPTIONS "-w")
gtest_discover_tests(vg_ui_test TEST_PREFIX "ui.")

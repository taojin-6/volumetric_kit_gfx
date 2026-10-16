# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The examples open a GLFW window through gfx_glfw. Skip cleanly -- rather than
# erroring -- when it is not built (VG_WITH_GLFW is OFF or GLFW is not
# installed), so a library-only / headless build still configures.
if(NOT TARGET volumetric_kit::gfx_glfw)
  message(STATUS "examples: skipped (no gfx_glfw: VG_WITH_GLFW=${VG_WITH_GLFW},"
                 " or GLFW not found)")
  return()
endif()

configure_file(config/model-screenshot.json config/model-screenshot.json
               COPYONLY)

if(NOT TARGET volumetric_kit::core_config)
  message(FATAL_ERROR "gfx examples need volumetric_kit::core_config: enable "
                      "VKC_WITH_CONFIG before making core available")
endif()

# 01_triangle: the hello-triangle in a window; the bring-up chain comes from the
# app tier (WindowedApp), the window glue from gfx_glfw. Links the Vulkan loader
# directly because it records vkCmd* itself.
add_executable(example_01_triangle 01_triangle/main.cpp)
target_link_libraries(example_01_triangle PRIVATE volumetric_kit::gfx_glfw
                                                  Vulkan::Vulkan)
vg_target_warnings(example_01_triangle)

vkc_compile_shaders(
  example_01_triangle
  OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/shaders"
  TARGET_ENV
  vulkan1.3
  SHADERS
  "${PROJECT_SOURCE_DIR}/shaders/triangle.vert"
  "${PROJECT_SOURCE_DIR}/shaders/triangle.frag")
target_compile_definitions(
  example_01_triangle
  PRIVATE VG_EXAMPLE_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")

# 02_imgui: a Dear ImGui debug overlay over the swapchain: the ui tier's
# renderer backend and gfx_glfw's platform backend. It compiles the demo window
# (imgui_demo.cpp), which the library tier deliberately does not build (see
# ui/CMakeLists.txt); imgui.h comes transitively from gfx_ui (SYSTEM). The
# Vulkan loader is linked directly for the vkDeviceWaitIdle.
add_executable(example_02_imgui 02_imgui/main.cpp
                                "${VG_IMGUI_SOURCE_DIR}/imgui_demo.cpp")
target_link_libraries(example_02_imgui PRIVATE volumetric_kit::gfx_glfw
                                               Vulkan::Vulkan)
# Strict first-party warnings apply to the example's own main.cpp; the bundled
# third-party demo TU opts out -- the source-level -w follows the target flags
# on the command line, so it wins over -Werror.
vg_target_warnings(example_02_imgui)
set_source_files_properties("${VG_IMGUI_SOURCE_DIR}/imgui_demo.cpp"
                            PROPERTIES COMPILE_OPTIONS "-w")

# 03_model: load a glTF model (io tier) and view it depth-tested, with an
# interactive CameraRig (camera tier) and a reflected per-draw MVP push
# constant; the windowed and screenshot bring-up chains come from the app tier
# (WindowedApp / HeadlessApp). Links the io and pipelines tiers on top of app
# (gfx_pipelines brings the camera tier), and gfx_glfw for the window and the
# Dear ImGui profiler-metrics overlay. The Vulkan loader is linked directly
# (records vkCmd*).
add_executable(example_03_model 03_model/main.cpp)
target_link_libraries(
  example_03_model PRIVATE volumetric_kit::gfx_glfw volumetric_kit::gfx_io
                           volumetric_kit::gfx_pipelines Vulkan::Vulkan)
vg_target_warnings(example_03_model)

# The model technique now ships embedded in gfx_pipelines (PbrPipeline); the
# example only compiles the skybox shaders it still loads from disk.
vkc_compile_shaders(
  example_03_model
  OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/shaders"
  TARGET_ENV
  vulkan1.3
  INCLUDE_DIRS
  "${PROJECT_SOURCE_DIR}/shaders"
  SHADERS
  "${PROJECT_SOURCE_DIR}/shaders/skybox.vert"
  "${PROJECT_SOURCE_DIR}/shaders/skybox.frag")
target_compile_definitions(
  example_03_model
  PRIVATE VG_EXAMPLE_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")

# 04_image: a 2D image (a zone plate) drawn through pipelines::ImagePipeline and
# panned / zoomed with camera::ImageView2D; the picture is re-converted from a
# device buffer every frame, as a live stream would be. Its shaders are embedded
# in gfx_pipelines, which brings gfx_camera; it records no vkCmd* of its own.
add_executable(example_04_image 04_image/main.cpp)
target_link_libraries(example_04_image PRIVATE volumetric_kit::gfx_glfw
                                               volumetric_kit::gfx_pipelines)
vg_target_warnings(example_04_image)

foreach(example IN ITEMS 01_triangle 02_imgui 03_model 04_image)
  target_link_libraries(example_${example} PRIVATE volumetric_kit::core_config)
  if(VG_BUILD_TESTS)
    add_test(
      NAME example_config.${example}
      COMMAND
        "${CMAKE_COMMAND}" "-DEXAMPLE_EXE=$<TARGET_FILE:example_${example}>"
        "-DINVALID_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/${example}-invalid.json"
        -P "${PROJECT_SOURCE_DIR}/tests/example_config.cmake")
  endif()
endforeach()

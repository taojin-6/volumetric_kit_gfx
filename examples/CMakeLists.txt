# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Examples need a window system (GLFW). Skip cleanly — rather than erroring —
# when GLFW is disabled or not installed, so a library-only / headless build
# still configures. CI provides GLFW via libglfw3-dev (Linux) / brew glfw
# (macOS).
find_package(glfw3 QUIET)
if(NOT VG_WITH_GLFW OR NOT glfw3_FOUND)
  message(STATUS "examples: skipped (VG_WITH_GLFW=${VG_WITH_GLFW}, "
                 "glfw3_FOUND=${glfw3_FOUND})")
  return()
endif()

# 01_triangle: the hello-triangle in a window, via Surface + Swapchain +
# FrameLoop. Links the Vulkan loader directly because it records vkCmd* itself.
add_executable(example_01_triangle 01_triangle/main.cpp)
target_link_libraries(example_01_triangle PRIVATE volumetric_kit::gfx_windowing
                                                  glfw Vulkan::Vulkan)
vg_target_warnings(example_01_triangle)

vg_compile_shaders(
  example_01_triangle OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders" SHADERS
  "${PROJECT_SOURCE_DIR}/shaders/triangle.vert"
  "${PROJECT_SOURCE_DIR}/shaders/triangle.frag")
target_compile_definitions(
  example_01_triangle
  PRIVATE VG_EXAMPLE_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")

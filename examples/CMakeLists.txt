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

# 02_imgui: a Dear ImGui debug overlay over the swapchain, via the ui tier. The
# example owns the platform backend (imgui_impl_glfw) — the GLFW-free ui tier
# wraps only imgui_impl_vulkan — and the demo window (imgui_demo.cpp), which the
# library tier deliberately does not build (see ui/CMakeLists.txt), so it
# compiles both ImGui TUs here. imgui.h comes transitively from gfx_ui (SYSTEM);
# the backends/ dir is added for imgui_impl_glfw.h. Vulkan loader is linked
# directly for the vkDeviceWaitIdle.
add_executable(
  example_02_imgui
  02_imgui/main.cpp "${VG_IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
  "${VG_IMGUI_SOURCE_DIR}/imgui_demo.cpp")
target_include_directories(example_02_imgui SYSTEM
                           PRIVATE "${VG_IMGUI_SOURCE_DIR}/backends")
target_link_libraries(
  example_02_imgui PRIVATE volumetric_kit::gfx_ui volumetric_kit::gfx_windowing
                           glfw Vulkan::Vulkan)
# Strict first-party warnings apply to the example's own main.cpp; the bundled
# third-party ImGui TUs opt out (the source-level -w follows the target flags on
# the command line, so it wins over -Werror).
vg_target_warnings(example_02_imgui)
set_source_files_properties(
  "${VG_IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
  "${VG_IMGUI_SOURCE_DIR}/imgui_demo.cpp" PROPERTIES COMPILE_OPTIONS "-w")

# 03_orbit_cube: an interactive orbit-camera RGB cube -- ties the camera tier to
# the core descriptor/MVP path over the swapchain, with a wireframe toggle. It
# reuses the mesh_mvp shaders and links gfx_camera (OrbitCamera) alongside
# gfx_windowing (which brings gfx_core); the Vulkan loader is linked directly
# for the vkCmd* it records.
add_executable(example_03_orbit_cube 03_orbit_cube/main.cpp)
target_link_libraries(
  example_03_orbit_cube PRIVATE volumetric_kit::gfx_windowing
                                volumetric_kit::gfx_camera glfw Vulkan::Vulkan)
vg_target_warnings(example_03_orbit_cube)
vg_compile_shaders(
  example_03_orbit_cube OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders"
  SHADERS "${PROJECT_SOURCE_DIR}/shaders/mesh_mvp.vert"
  "${PROJECT_SOURCE_DIR}/shaders/mesh.frag")
target_compile_definitions(
  example_03_orbit_cube
  PRIVATE VG_EXAMPLE_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")

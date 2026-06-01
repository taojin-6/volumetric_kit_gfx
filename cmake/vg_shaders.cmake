# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# vg_compile_shaders(<target> [OUTPUT_DIR <dir>] SHADERS <file>...)
#
# Compiles each GLSL shader to <dir>/<name>.spv (e.g. triangle.vert ->
# triangle.vert.spv) and makes <target> depend on the results, so building
# <target> (re)compiles any stale shader. Uses the SPIR-V compiler that
# find_package(Vulkan) located -- glslc (shaderc) preferred, glslangValidator as
# fallback -- and only errors when invoked, so a build compiling no shaders
# needs no compiler installed. Safe to call more than once per target.
function(vg_compile_shaders target)
  cmake_parse_arguments(ARG "" "OUTPUT_DIR" "SHADERS" ${ARGN})
  if(NOT ARG_SHADERS)
    message(FATAL_ERROR "vg_compile_shaders(${target}): no SHADERS given")
  endif()
  if(NOT ARG_OUTPUT_DIR)
    set(ARG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  endif()

  if(Vulkan_GLSLC_EXECUTABLE)
    set(_compiler "${Vulkan_GLSLC_EXECUTABLE}")
    set(_mode glslc)
  elseif(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    set(_compiler "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
    set(_mode glslang)
  else()
    message(
      FATAL_ERROR
        "vg_compile_shaders(${target}): no GLSL->SPIR-V compiler found. Install "
        "shaderc (glslc) or glslang -- both ship with the Vulkan SDK (macOS: "
        "`brew install shaderc`).")
  endif()
  # Surface the chosen compiler so the per-runner choice (glslc on macOS,
  # glslangValidator on the Linux runners) is greppable in CI logs.
  message(STATUS "vg_compile_shaders(${target}): ${_mode} -> ${_compiler}")

  set(_spv_outputs)
  set(_seen_names)
  foreach(_src IN LISTS ARG_SHADERS)
    get_filename_component(_name "${_src}" NAME)
    # Outputs are keyed by basename, so two sources that share one (from
    # different directories) would clobber each other -- reject that up front.
    if(_name IN_LIST _seen_names)
      message(
        FATAL_ERROR
          "vg_compile_shaders(${target}): duplicate shader name '${_name}'")
    endif()
    list(APPEND _seen_names "${_name}")

    set(_out "${ARG_OUTPUT_DIR}/${_name}.spv")
    if(_mode STREQUAL glslc)
      set(_cmd "${_compiler}" --target-env=vulkan1.3 -o "${_out}" "${_src}")
    else()
      set(_cmd
          "${_compiler}"
          -V
          --target-env
          vulkan1.3
          -o
          "${_out}"
          "${_src}")
    endif()
    # make_directory runs at build time (not just configure), so the compile
    # still works if the output dir was cleaned without re-running CMake.
    add_custom_command(
      OUTPUT "${_out}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_OUTPUT_DIR}"
      COMMAND ${_cmd}
      DEPENDS "${_src}"
      COMMENT "Compiling shader ${_name}"
      VERBATIM)
    list(APPEND _spv_outputs "${_out}")
  endforeach()

  # A custom target carrying the .spv outputs; <target> depends on it so the
  # shaders are (re)built ahead of the consumer. The name is uniquified via a
  # build-global counter so repeated calls don't collide on one target name.
  get_property(_seq GLOBAL PROPERTY _vg_shader_set_seq)
  if(NOT _seq)
    set(_seq 0)
  endif()
  math(EXPR _seq "${_seq} + 1")
  set_property(GLOBAL PROPERTY _vg_shader_set_seq "${_seq}")

  add_custom_target(${target}_shaders_${_seq} DEPENDS ${_spv_outputs})
  add_dependencies(${target} ${target}_shaders_${_seq})
endfunction()

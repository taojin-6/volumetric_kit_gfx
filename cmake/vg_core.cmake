# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# vg_require_core_vulkan(<pin>)
#
# Fails the configure unless the volumetric_kit_core gfx got has its vulkan tier
# at <pin> or newer, the commit gfx pins. Run after the core is made available:
# gfx's pin and its VKC_WITH_VULKAN yield to a project that made the core
# available first, and FetchContent may have found an installed core instead, so
# this checks what arrived rather than trusting the request. The core's version
# does not advance between commits, so <pin> is recognized by the newest
# declaration gfx uses from it, core::TimelinePoint in
# volumetric_kit/core/vulkan/sync.hpp.
function(vg_require_core_vulkan pin)
  get_target_property(_imported volumetric_kit::core_base IMPORTED)
  if(_imported)
    set(_origin "the core found installed at ${volumetric_kit_core_DIR}")
  elseif(FETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE)
    set(_origin
        "the core checkout at ${FETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE}")
  else()
    set(_origin "the core another project made available first")
  endif()

  if(NOT TARGET volumetric_kit::core_vulkan)
    if(_imported)
      string(
        CONCAT _fix "install one built with VKC_WITH_VULKAN ON, or let "
               "gfx build its own (FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)")
    else()
      string(CONCAT _fix "set VKC_WITH_VULKAN ON before its first "
                    "FetchContent_MakeAvailable(volumetric_kit_core)")
    endif()
    message(
      FATAL_ERROR
        "gfx needs volumetric_kit_core's vulkan tier, which ${_origin} was "
        "built without: ${_fix}.")
  endif()

  # A target built here lists its include roots inside $<BUILD_INTERFACE:...>;
  # an installed one lists plain paths.
  set(_dirs)
  foreach(_target IN ITEMS volumetric_kit::core_vulkan
                           volumetric_kit::core_base)
    get_target_property(_aliased ${_target} ALIASED_TARGET)
    if(_aliased)
      set(_target ${_aliased})
    endif()
    get_target_property(_target_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_target_dirs)
      list(APPEND _dirs ${_target_dirs})
    endif()
  endforeach()
  string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>;]*)>" "\\1" _dirs "${_dirs}")
  list(FILTER _dirs EXCLUDE REGEX "^\\$<")
  foreach(_dir IN LISTS _dirs)
    set(_sync "${_dir}/volumetric_kit/core/vulkan/sync.hpp")
    if(EXISTS "${_sync}")
      file(STRINGS "${_sync}" _declares REGEX "^struct TimelinePoint")
      if(_declares)
        return()
      endif()
    endif()
  endforeach()
  message(
    FATAL_ERROR
      "gfx needs volumetric_kit_core at ${pin} or newer, and ${_origin} is "
      "older (its volumetric_kit/core/vulkan/sync.hpp declares no "
      "TimelinePoint): use a core at or after that commit.")
endfunction()

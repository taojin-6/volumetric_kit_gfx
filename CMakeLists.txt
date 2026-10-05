# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

cmake_minimum_required(VERSION 3.21)

project(
  volumetric_kit_gfx
  VERSION 0.0.1
  DESCRIPTION
    "Standalone, reusable Vulkan renderer library (MoltenVK on Apple) for volumetric workloads"
  HOMEPAGE_URL "https://github.com/taojin-6/volumetric_kit_gfx"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Are we the top-level project, or pulled in via add_subdirectory/FetchContent?
# Defaults for "build the extras" and "generate install rules" follow from this.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(VG_IS_TOP_LEVEL ON)
else()
  set(VG_IS_TOP_LEVEL OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Options -----------------------------------------------------------------
option(VG_BUILD_EXAMPLES "Build example programs" ${VG_IS_TOP_LEVEL})
option(VG_BUILD_TESTS "Build tests" ${VG_IS_TOP_LEVEL})
option(VG_INSTALL "Generate install/export rules" ${VG_IS_TOP_LEVEL})
# VG_WITH_GLFW gates the examples: examples/CMakeLists.txt skips the whole
# directory when it is OFF (or when GLFW is not installed). The library tiers
# never read it — windowing is GLFW-free by design.
#
# TODO: wire VG_WITH_CUDA into the interop tier when it lands; until then no
# target reads it, so toggling it is a no-op.
option(VG_WITH_GLFW "Enable GLFW desktop windowing (gates examples/)" ON)
option(
  VG_WITH_CUDA
  "Enable CUDA <-> Vulkan external-memory interop (reserved; not yet wired)"
  OFF)
option(VG_WARNINGS_AS_ERRORS "Treat first-party compiler warnings as errors" ON)

# Validation is a runtime decision (InstanceConfig::enable_validation), gated at
# instance creation by layer availability — not a compile-time switch. The debug
# messenger/layer wiring is negligible code size, so there is no build option to
# strip it.

# Semicolon list of sanitizers (e.g. "address;undefined"); empty = off. Applied
# globally — see cmake/vg_sanitizers.cmake for why.
set(VG_SANITIZE
    ""
    CACHE STRING "Sanitizers to enable, e.g. address;undefined (empty = off)")

include(vg_warnings)
include(vg_glm)
include(vg_sanitizers)
include(vg_shaders)
include(vg_embed)

# --- Public dependencies -----------------------------------------------------
# Resolved at the top level so their imported targets are visible to every
# subdirectory, not just the one find_package() ran in. Provides Vulkan::Headers
# (the public-API types) and Vulkan::Vulkan (the loader our implementation links
# privately).
find_package(Vulkan REQUIRED)

# glm: the matrix/vector types in the camera and assets tiers' public headers.
# Prefer a system glm (its config defines glm::glm); fall back to the pinned
# vendored copy from third_party/ when none is installed -- so the build is
# self-contained on CI runners that don't ship glm, matching how the other deps
# are vendored. VG_HAVE_SYSTEM_GLM records which path won, so third_party/ only
# vendors glm (and lets it run its own install/export) when no system copy
# exists. glm is a PUBLIC dependency either way, so the installed package config
# re-finds it unconditionally (find_dependency(glm)).
find_package(glm CONFIG QUIET)
if(glm_FOUND)
  set(VG_HAVE_SYSTEM_GLM ON)
else()
  set(VG_HAVE_SYSTEM_GLM OFF)
endif()

# volumetric_kit_core: gfx's Status / Result / VG_CHECK / log sink come from its
# base tier (DECISIONS.md, 2026-10-04), so they are the same types recon and
# calib use, and its vulkan tier supplies the format metadata gfx's image code
# reads (format.hpp). A compiled library and a PUBLIC dependency of gfx_core,
# resolved here rather than in third_party/: when FetchContent satisfies it with
# find_package (FETCHCONTENT_TRY_FIND_PACKAGE_MODE), the imported
# volumetric_kit::core_base and volumetric_kit_core_VERSION are visible only in
# this directory and below. Pinned by commit, never main; an application that
# also fetches recon declares the core first, and FetchContent keeps the first
# declaration of a name, so this pin then yields to the application's. Work on
# both checkouts at once with
# -DFETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE=<core checkout>. The core's
# tests and -Werror stay off in a subproject.
#
# The pin is main after core PR #13, which settled that the family builds on the
# system's Vulkan headers (the core's DECISIONS.md, "Vulkan headers come from
# the system").
#
# TODO: pin a release tag once the core tags one (v0.1.0).
include(FetchContent)
FetchContent_Declare(
  volumetric_kit_core
  GIT_REPOSITORY https://github.com/taojin-6/volumetric_kit_core.git
  GIT_TAG a60398b7aeb2cad722a3f79cb71696bda10fc054)
# gfx's install must carry the core its exported targets link, so VG_INSTALL
# turns the core's install rules on. It never turns them off: they default ON,
# and a sibling that installs (recon) needs them whichever project populates the
# core first.
if(VG_INSTALL)
  set(VKC_INSTALL ON)
endif()
# gfx builds on the vulkan tier, so it asks for it; both compile against the
# Vulkan headers find_package(Vulkan) found above.
set(VKC_WITH_VULKAN ON)
FetchContent_MakeAvailable(volumetric_kit_core)
# The set() above is lost when calib or the application made the core available
# first, without its vulkan tier.
if(NOT TARGET volumetric_kit::core_vulkan)
  message(
    FATAL_ERROR
      "gfx needs volumetric_kit_core's vulkan tier: set VKC_WITH_VULKAN ON "
      "before the first FetchContent_MakeAvailable of the core")
endif()

# --- Dependencies + library --------------------------------------------------
add_subdirectory(third_party)
add_subdirectory(src)

if(VG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VG_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

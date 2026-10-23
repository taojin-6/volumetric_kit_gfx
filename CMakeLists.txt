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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Options -----------------------------------------------------------------
# Examples, tests and -Werror default ON only when gfx is the top-level project,
# so an application that fetches gfx builds the libraries alone, with no
# warnings-as-errors from its own compiler.
option(VG_BUILD_EXAMPLES "Build example programs" ${PROJECT_IS_TOP_LEVEL})
option(VG_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(VG_WARNINGS_AS_ERRORS "Treat first-party compiler warnings as errors"
       ${PROJECT_IS_TOP_LEVEL})
# VG_WITH_GLFW builds gfx_glfw, the GLFW window glue, and the examples on it;
# both are skipped when it is OFF or GLFW is not installed. No other tier reads
# GLFW.
#
# TODO: wire VG_WITH_CUDA into the interop tier when it lands; until then no
# target reads it, so toggling it is a no-op.
option(VG_WITH_GLFW "Build gfx_glfw (GLFW window glue) and the examples" ON)
option(
  VG_WITH_CUDA
  "Enable CUDA <-> Vulkan external-memory interop (reserved; not yet wired)"
  OFF)

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

# --- Public dependencies -----------------------------------------------------
# Resolved at the top level so their imported targets are visible to every
# subdirectory, not just the one find_package() ran in: GLFW here, then the
# core, then Vulkan and glm after it.

# GLFW, for gfx_glfw and the examples: optional, so a headless build configures
# without it. A `glfw` target the application already made is used as it is. CI
# provides it through libglfw3-dev (Linux) and brew glfw (macOS).
if(VG_WITH_GLFW AND NOT TARGET glfw)
  # 3.3: gfx_glfw reads GLFW's errors with glfwGetError.
  find_package(glfw3 3.3 QUIET)
endif()

# volumetric_kit_core: gfx takes Status / Result / VKC_CHECK / log sink from its
# base tier (DECISIONS.md, 2026-10-04), so they are the same types recon and
# calib use. Its vulkan tier supplies the Vulkan umbrella header, the VkResult
# bridge, the format metadata gfx's image code reads (format.hpp) and the shader
# build functions (vkc_compile_shaders / vkc_embed_shaders). Both tiers are
# compiled libraries and PUBLIC dependencies of gfx_core, resolved here rather
# than in third_party/: when FetchContent satisfies them with find_package
# (FETCHCONTENT_TRY_FIND_PACKAGE_MODE), the imported targets are visible only in
# this directory and below. Pinned by commit, never main; an application that
# also fetches recon declares the core first, and FetchContent keeps the first
# declaration of a name, so this pin then yields to the application's. Work on
# both checkouts at once with
# -DFETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE=<core checkout>. The core's
# tests and -Werror stay off in a subproject.
#
# The pin (core 0.19.0, #49's merge) has what gfx relies on: the timeline-value
# checks, the optional JSON configuration component, core_test_support with its
# two-family fixture, and the iOS module (cmake/vkc_ios.cmake) that answers
# gfx's finds of Vulkan and glm in an iOS build. _build.yml's homebrew-key
# action runs at the same SHA.
include(FetchContent)
FetchContent_Declare(
  volumetric_kit_core
  GIT_REPOSITORY https://github.com/taojin-6/volumetric_kit_core.git
  GIT_TAG ff44a981055d505590e295ff6a943bb1b6e96109)
# gfx builds on the vulkan tier, so it asks for it; both compile against the
# Vulkan headers the tier's find_package(Vulkan) caches, which gfx's below
# reads.
set(VKC_WITH_VULKAN ON)
# Only the example executables read run configuration. Renderer libraries and
# headless consumers do not acquire the optional JSON configuration dependency.
if(VG_BUILD_EXAMPLES
   AND VG_WITH_GLFW
   AND TARGET glfw)
  set(VKC_WITH_CONFIG ON)
endif()
# gfx's tests build on the core's test fixtures (tests/CMakeLists.txt).
if(VG_BUILD_TESTS)
  set(VKC_BUILD_TEST_SUPPORT ON)
endif()
FetchContent_MakeAvailable(volumetric_kit_core)
# The oldest core gfx builds with, and its vulkan tier, checked in the core that
# arrived: the pin and the set() above are lost when calib or the application
# made the core available first, and FetchContent may have found an installed
# core.
set(VG_VKC_MIN_VERSION 0.1.0)
vkc_require_core(${VG_VKC_MIN_VERSION} VULKAN)

# Vulkan and glm are found after the core is added: in an iOS build the core's
# cmake/vkc_ios.cmake, which runs as the core is added, answers both, with
# MoltenVK's static library and the host's glm headers.
#
# Vulkan::Headers (the public-API types) and Vulkan::Vulkan (the loader, or
# MoltenVK's static library on iOS, which our implementation links privately).
find_package(Vulkan REQUIRED)

# glm: the matrix/vector types in the camera and assets tiers' public headers.
# Prefer a system glm (its config defines glm::glm); fall back to the pinned
# vendored copy from third_party/ when none is installed -- so the build is
# self-contained on CI runners that don't ship glm, matching how the other deps
# are vendored. VG_HAVE_SYSTEM_GLM records which path won, so third_party/ only
# vendors glm when no system copy exists. An iOS build takes the core's answer
# or stops with its message; it never fetches glm.
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  find_package(glm CONFIG REQUIRED)
else()
  find_package(glm CONFIG QUIET)
endif()
if(glm_FOUND)
  set(VG_HAVE_SYSTEM_GLM ON)
else()
  set(VG_HAVE_SYSTEM_GLM OFF)
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

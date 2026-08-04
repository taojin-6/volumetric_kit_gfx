# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# vg_glm_conventions(<target> <PUBLIC|INTERFACE>)
#
# Pin glm's clip-space depth convention on a target that exposes glm to its
# consumers. GLM_FORCE_DEPTH_ZERO_TO_ONE selects Vulkan's [0, 1] depth range
# over GL's [-1, 1]; because glm is header-only it changes the *definition* of
# glm::perspective / glm::ortho, so it must be identical in every translation
# unit that instantiates them -- the library's own and every consumer's.
#
# Call this from EVERY tier that puts glm on a consumer's include path, not just
# the tier that calls glm::perspective itself. gfx_pipelines, for instance,
# exposes a caller-built `glm::mat4 view_proj` on HybridMeshFrame / PbrFrame: a
# consumer that links gfx_pipelines WITHOUT gfx_camera builds that matrix in its
# own TUs, and without the define it produces a [-1, 1] projection for a [0, 1]
# depth attachment. Vulkan clips at 0 <= z <= w, so the near half of the frustum
# silently disappears -- no compile error, no VUID, nothing to grep for.
#
# The definition is attached to *our* targets rather than to glm::glm so it
# survives install/export: a consumer resolving glm through find_dependency(glm)
# gets an unmodified glm, and only the usage requirements we export can carry
# the convention across the package boundary.
function(vg_glm_conventions target scope)
  if(NOT scope STREQUAL "PUBLIC" AND NOT scope STREQUAL "INTERFACE")
    message(
      FATAL_ERROR "vg_glm_conventions(${target}): scope must be PUBLIC or "
                  "INTERFACE, got '${scope}'")
  endif()
  target_compile_definitions(${target} ${scope} GLM_FORCE_DEPTH_ZERO_TO_ONE)
endfunction()

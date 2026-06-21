// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file glm_config.hpp
/// @brief glm configuration for the camera tier -- defined once, before glm.
///
/// Every camera header and TU includes this *before* any `<glm/...>` header so
/// glm's clip-space convention is fixed in one place:
///
/// - `GLM_FORCE_DEPTH_ZERO_TO_ONE` makes glm's `perspective`/`ortho` emit a
///   clip-space `z` in `[0, 1]` -- Vulkan's depth range -- instead of OpenGL's
///   `[-1, 1]`. This is a compile-time switch on a header-only library, so it
///   *must* be set consistently across every TU that instantiates glm's
///   projections. The authoritative, order-independent guarantee is a PUBLIC
///   compile definition on the `gfx_camera` target (see its CMakeLists), which
///   reaches the library's own TUs and -- via the install interface -- every
///   consumer regardless of include order. This header mirrors the same define
///   at the include site so it stays local and reviewable, and so a TU that
///   includes `camera.hpp` without linking the target still gets it.
///
/// The complementary framebuffer-Y flip is *not* a glm macro -- glm has no
/// "Y down" mode -- so it lives in the projection math (negating `proj[1][1]`),
/// documented on @ref volumetric_kit::gfx::camera::Camera.
///
/// This is an internal header (`impl/`); consumers include `camera.hpp`, which
/// pulls this in first.

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

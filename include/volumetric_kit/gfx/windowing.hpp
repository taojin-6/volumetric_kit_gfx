// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file windowing.hpp
/// @brief Convenience umbrella for the windowing tier — pulls in @ref
///        volumetric_kit::gfx::windowing::Surface, @ref
///        volumetric_kit::gfx::windowing::Swapchain, and @ref
///        volumetric_kit::gfx::windowing::FrameLoop in one include.
///
/// The three windowing types are always used together to drive a present loop
/// (surface → swapchain → frames-in-flight), so this header lets a consumer
/// write one `#include` instead of three. It is purely additive: the individual
/// headers remain public for code that wants finer-grained dependencies.
///
/// @note Like the headers it aggregates, this transitively includes
///       `core/vulkan.hpp`. If you rely on GLFW's Vulkan
///       WSI helpers (e.g. `glfwCreateWindowSurface`), include a Vulkan header
///       before `<GLFW/glfw3.h>` so `glfw3.h` sees Vulkan and declares them —
///       the examples do this by including `core/vulkan.hpp` first.
///
/// @code
/// #include "volumetric_kit/gfx/windowing.hpp"
/// // -> windowing::Surface, windowing::Swapchain, windowing::FrameLoop
/// @endcode

#include "volumetric_kit/gfx/windowing/frame_loop.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

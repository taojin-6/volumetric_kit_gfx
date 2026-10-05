// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file vulkan.hpp
/// @brief The single umbrella header through which first-party code includes
///        Vulkan.
///
/// The single point where first-party code pulls in Vulkan. Always include this
/// header — never `<vulkan/vulkan.h>` or a loader header directly — so the
/// loader / dispatch choice (currently the link-time loader `Vulkan::Vulkan`)
/// stays a detail of this one file: adopting volk for iOS/Android would be a
/// change here plus the link line, with no churn at call sites.
///
/// It forwards to volumetric_kit_core's umbrella, so gfx compiles against the
/// same system Vulkan headers as the core, held to the oldest the core supports
/// (1.3.204; 1.3.208 on Apple) by that header's check.
///
/// @code
/// #include "volumetric_kit/gfx/core/vulkan.hpp"
/// VkFence fence = VK_NULL_HANDLE;
/// @endcode

#include "volumetric_kit/core/vulkan/vulkan.hpp"  // IWYU pragma: export

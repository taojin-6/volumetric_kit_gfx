// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file physical_device_info.hpp
/// @brief gfx's name for volumetric_kit_core's `PhysicalDeviceInfo`: a
///        physical device's capabilities, captured once.
///
/// The core's type (`volumetric_kit/core/vulkan/physical_device_info.hpp`):
/// properties and limits, memory heaps and types, queue families, extensions,
/// features and format queries, and the Vulkan version usable on the device
/// through its instance. @ref Instance::select_physical_device returns one for
/// the best device meeting a set of requirements.
///
/// @code
/// PhysicalDeviceInfo caps =
///     PhysicalDeviceInfo::query(physical, instance.api_version());
/// if (!caps.supports_dynamic_rendering()) return Status::unsupported("...");
/// @endcode

#include "volumetric_kit/core/vulkan/physical_device_info.hpp"

namespace volumetric_kit::gfx {

using core::PhysicalDeviceInfo;

}  // namespace volumetric_kit::gfx

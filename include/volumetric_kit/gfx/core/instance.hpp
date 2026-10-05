// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file instance.hpp
/// @brief gfx's names for volumetric_kit_core's `Instance` and
///        `InstanceConfig`.
///
/// The instance is the family's shared one (the core's
/// `volumetric_kit/core/vulkan/instance.hpp`, DECISIONS.md, 2026-10-04, "The
/// device comes from volumetric_kit_core"), so an instance gfx makes is the
/// type recon adopts, and the reverse. It asks for Vulkan 1.3 or the loader's
/// lower version, routes validation to the family's log sink, requests
/// `VK_EXT_debug_utils` by default (`InstanceConfig::request_debug_utils`), and
/// enables portability enumeration where the loader offers it.
///
/// @code
/// InstanceConfig config;
/// config.app_name = "viewer";
/// config.enable_validation = true;
/// VG_ASSIGN(Instance instance, Instance::create(config));
/// DeviceRequirements reqs = device_requirements();
/// VG_ASSIGN(PhysicalDeviceInfo gpu, instance.select_physical_device(reqs));
/// @endcode

#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/gfx/core/physical_device_info.hpp"
#include "volumetric_kit/gfx/core/result.hpp"

namespace volumetric_kit::gfx {

using core::Instance;
using core::InstanceConfig;

}  // namespace volumetric_kit::gfx

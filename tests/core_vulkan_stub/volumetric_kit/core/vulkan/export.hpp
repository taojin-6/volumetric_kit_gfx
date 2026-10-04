// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Stands in for the export header the core generates when it builds its vulkan
// tier, which gfx's build does not (see core_vulkan_tier_test.cpp). The test
// calls none of the functions this marks.
#define VKC_VULKAN_API

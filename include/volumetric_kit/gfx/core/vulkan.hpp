// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file vulkan.hpp
/// The single point where first-party code pulls in Vulkan. Always include this
/// header — never `<vulkan/vulkan.h>` or a loader header directly — so the
/// loader / dispatch choice (currently the link-time loader `Vulkan::Vulkan`)
/// stays a detail of this one file: adopting volk for iOS/Android would be a
/// change here plus the link line, with no churn at call sites.

#include <vulkan/vulkan.h>

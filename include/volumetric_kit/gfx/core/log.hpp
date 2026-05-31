// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file log.hpp
/// A pluggable logging seam. The library imposes no logging framework on
/// consumers: it emits through a handler they can install, defaulting to stderr
/// for warnings and errors. The Vulkan debug messenger routes here too.

#include <functional>
#include <string_view>

#include "volumetric_kit/gfx/core/export.hpp"

namespace volumetric_kit::gfx {

enum class LogLevel { Debug, Info, Warning, Error };

using LogHandler = std::function<void(LogLevel, std::string_view)>;

/// Install the diagnostic sink. Pass a default-constructed (empty) handler to
/// restore the built-in default (warnings + errors to stderr). Thread-safe.
VG_CORE_API void set_log_handler(LogHandler handler);

/// Emit a diagnostic through the current handler (or the default sink).
/// Thread-safe.
VG_CORE_API void log_message(LogLevel level, std::string_view message);

}  // namespace volumetric_kit::gfx

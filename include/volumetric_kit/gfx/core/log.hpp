// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file log.hpp
/// @brief gfx's side of the family's one log sink.
///
/// The sink is volumetric_kit_core's: one process-wide handler for calib,
/// recon and gfx alike, defaulting to stderr for warnings and errors. The
/// library imposes no logging framework on consumers; an application installs
/// its own handler with @ref set_log_handler (the core's, named here) and
/// receives each message with its level and source. gfx's messages carry the
/// source @ref kLogSource, so the default sink keeps printing `[vg <level>]`.
/// Contract failures are the exception: a failed @ref VG_CHECK, or reading the
/// value of an error `Result`, is reported by the core with source `"core"`
/// (`[core error] contract check failed: ...`), so a handler that routes by
/// source sees gfx's contract failures under `"core"`.
///
/// @code
/// set_log_handler([](LogLevel level, std::string_view source,
///                    std::string_view message) {
///   if (level >= LogLevel::Warning) forward(source, message);
/// });
/// @endcode

#include <string_view>

#include "volumetric_kit/core/base/log.hpp"

namespace volumetric_kit::gfx {

// TODO: name the log sink as the core does (core::set_log_handler,
// vkc::set_log_handler outside gfx) and drop these aliases (DECISIONS.md,
// "Memory comes from volumetric_kit_core").
using core::LogHandler;
using core::LogLevel;
using core::set_log_handler;

/// @brief The source gfx's diagnostics carry, contract failures aside; the
///        default sink prints `[vg <level>]`.
inline constexpr std::string_view kLogSource = "vg";

/// @brief Emit a gfx diagnostic through the family's sink, with source
///        @ref kLogSource. Thread-safe.
/// @param level    The message's severity.
/// @param message  The message; it need not be NUL-terminated.
inline void log_message(LogLevel level, std::string_view message) {
  core::log_message(level, kLogSource, message);
}

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file check.hpp
/// Fail-fast contract checks for *programmer errors* (precondition violations),
/// as distinct from Vulkan runtime errors -- those flow through `Status` /
/// `Result`. On failure `VG_CHECK` logs at Error through the diagnostic sink,
/// then calls `std::abort()`. Active in every build (not just debug).
///
/// The library is built and consumed with `-fno-exceptions` on mobile, so abort
/// -- not `throw` -- is the portable way to terminate on a bug: it raises
/// SIGABRT, which crash reporters (Crashlytics, os_log, Android tombstones)
/// capture, and it never leaves the empty-`optional` / use-after-error UB that
/// a silently-skipped check would.

#include <string_view>

#include "volumetric_kit/gfx/core/export.hpp"

namespace volumetric_kit::gfx::detail {

/// Report a failed `VG_CHECK` (log + abort). Never returns.
[[noreturn]] VG_CORE_API void check_failed(const char* file, int line,
                                           const char* expr,
                                           std::string_view msg);

}  // namespace volumetric_kit::gfx::detail

/// Abort (after logging) unless `cond` holds. For programmer errors only --
/// recoverable runtime failures must use `Status` / `Result` instead.
#define VG_CHECK(cond, msg)                                                  \
  do {                                                                       \
    if (!(cond)) {                                                           \
      ::volumetric_kit::gfx::detail::check_failed(__FILE__, __LINE__, #cond, \
                                                  (msg));                    \
    }                                                                        \
  } while (0)

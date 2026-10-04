// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file check.hpp
/// @brief gfx's name for the core's fail-fast contract check.
///
/// `VG_CHECK` is volumetric_kit_core's `VKC_CHECK`: for *programmer errors*
/// (precondition violations), as distinct from recoverable runtime failures,
/// which flow through `Status` / `Result`. On failure it logs at Error through
/// the family's log sink (source `"core"`), then calls `std::abort()`, in every
/// build. Abort rather than `throw`, because mobile consumers build with
/// `-fno-exceptions` and crash reporters capture SIGABRT.
///
/// @code
/// VG_CHECK(index < size, "index past the end");
/// @endcode

#include "volumetric_kit/core/base/check.hpp"

// TODO: rename to VKC_CHECK with VG_TRY / VG_ASSIGN (result.hpp), then delete
// this alias.

/// @brief The core's `VKC_CHECK` under gfx's name, used as
///        `VG_CHECK(cond, msg)`: abort, after logging, unless the precondition
///        `cond` holds; `msg` describes the contract.
///
/// An object-like alias rather than a function-like forward, which would expand
/// macros in `cond` before `VKC_CHECK` stringizes it, so a failure would report
/// `n <= 16U` for `n <= VK_UUID_SIZE`.
#define VG_CHECK VKC_CHECK

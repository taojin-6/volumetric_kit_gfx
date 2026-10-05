// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file result.hpp
/// @brief gfx's error handling: volumetric_kit_core's `Status`, `Result` and
///        bridge from a failed `VkResult`, named in this namespace.
///
/// gfx defines no error types of its own. It uses the family's shared ones,
/// from volumetric_kit_core's base tier (DECISIONS.md, 2026-10-04, "Error
/// handling comes from volumetric_kit_core"), so a `Status` from gfx is the
/// same type as one from recon or calib and passes between them unchanged. The
/// `VkResult` bridge is the core's vulkan tier's. The using-declarations below
/// let gfx and its consumers keep writing `Status`, `Result<T>`, `vk_error`,
/// `vk_result` and `to_string` in this namespace.
///
/// No exceptions cross the API boundary: mobile consumers build with
/// `-fno-exceptions`. Fallible calls return `Status` or `Result<T>`, both
/// `[[nodiscard]]`. `Status` is backend-neutral: a failed Vulkan call is a
/// backend status, `Status::Code::Backend` with the `VkResult` as its
/// `detail()`, made by @ref vk_error or @ref VG_VK_TRY and read back by
/// @ref vk_result. Name a domain or a `VkResult` with `to_string`:
/// `to_string(status.domain())`. Reading the value of an error `Result` is a
/// programmer error and aborts. The full contract is in the core's
/// `volumetric_kit/core/base/result.hpp`.
///
/// @code
/// Result<Device> r = Device::create(instance, physical, reqs);
/// if (!r) return r.status();   // propagate failure to our caller
/// Device& device = r.value();  // safe: guarded by the !r check above
/// @endcode

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/core/check.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

// TODO: name Status, Result and the VkResult bridge below as the core does
// (core::Status, vkc::Status outside gfx) and drop these aliases, with the
// macros' rename below (DECISIONS.md, "Memory comes from
// volumetric_kit_core").
using core::Result;
using core::Status;
// The core's VkResult bridge (volumetric_kit/core/vulkan/vk_result.hpp), under
// gfx's names: vk_error(result, what) wraps a failed VkResult as a backend
// Status, vk_result(status) reads it back -- empty unless the domain is Backend
// and the detail fits VkResult's 32 bits -- and to_string names a VkResult or a
// Status::Code. vk_result cannot tell a Vulkan status from a CUDA one, which
// shares the Backend domain: ask it only of a status from a Vulkan call, as
// every backend status gfx returns is.
using core::to_string;
using core::vk_error;
using core::vk_result;

}  // namespace volumetric_kit::gfx

// TODO: rename VG_TRY / VG_ASSIGN (and VG_CHECK, check.hpp) to the core's
// VKC_TRY / VKC_ASSIGN / VKC_CHECK across gfx once its open branches have
// landed, then delete these aliases, as recon plans for its VR_* names. Until
// then the old names keep those branches merging cleanly, and new code uses
// them too.

/// @brief gfx's name for the core's `VKC_TRY`, used as `VG_TRY(expr)`:
///        evaluate an expression yielding a `Status` and early-return it if
///        not OK.
///
/// Usable only inside a function returning `Status` or `Result<T>`. It and
/// @ref VG_ASSIGN are object-like aliases, as @ref VG_CHECK is (check.hpp).
///
/// @code
/// Status init() {
///   VG_TRY(create_instance());   // returns the error if this fails
///   return {};                   // success
/// }
/// @endcode
#define VG_TRY VKC_TRY

/// @brief gfx's name for the core's `VKC_ASSIGN`, used as
///        `VG_ASSIGN(decl, expr)`: unwrap the `Result<T>` that `expr` yields
///        into the variable declaration `decl`, or early-return its `Status`.
///
/// It declares `decl` in the enclosing scope, so it is a statement sequence:
/// never the unbraced body of an `if`/`for`/`while`. A type with a top-level
/// comma needs an alias first.
///
/// @code
/// Result<Pipeline> build(VkDevice device) {
///   VG_ASSIGN(ShaderModule vert, ShaderModule::create(device, code, bytes));
///   return assemble(vert);
/// }
/// @endcode
#define VG_ASSIGN VKC_ASSIGN

/// @brief gfx's name for the core's `VKC_VK_TRY`, used as `VG_VK_TRY(expr)`:
///        evaluate an expression yielding a raw `VkResult` and early-return
///        a backend `Status` unless it is `VK_SUCCESS`. The expression's text
///        becomes the failure's message, so the failing call names itself.
///
/// @warning Valid only for calls whose sole success code is `VK_SUCCESS`: it
///          treats every other code -- including the positive success codes
///          `VK_SUBOPTIMAL_KHR`, `VK_INCOMPLETE`, `VK_NOT_READY`, and
///          `VK_TIMEOUT` -- as a failure to early-return. For a call that can
///          return more than one success code (e.g. `vkAcquireNextImageKHR`),
///          hand-roll the check as @ref Fence::wait does.
///
/// @code
/// VG_VK_TRY(vkCreateDevice(phys, &ci, nullptr, &dev_));
/// @endcode
#define VG_VK_TRY VKC_VK_TRY

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file result.hpp
/// @brief gfx's error handling: volumetric_kit_core's `Status` and `Result`,
///        named in this namespace, and the bridge from a failed `VkResult`.
///
/// gfx defines no error types of its own. It uses the family's shared ones,
/// from volumetric_kit_core's base tier (DECISIONS.md, 2026-10-04, "Error
/// handling comes from volumetric_kit_core"), so a `Status` from gfx is the
/// same type as one from recon or calib and passes between them unchanged. The
/// using-declarations below let gfx and its consumers keep writing `Status` and
/// `Result<T>` in this namespace.
///
/// No exceptions cross the API boundary: mobile consumers build with
/// `-fno-exceptions`. Fallible calls return `Status` or `Result<T>`, both
/// `[[nodiscard]]`. `Status` is backend-neutral: a failed Vulkan call is
/// `Status::Code::Backend` with the `VkResult` as its `detail()`, made by
/// @ref vk_error or @ref VG_VK_TRY and read back by @ref vk_result. Reading the
/// value of an error `Result` is a programmer error and aborts. The full
/// contract is in the core's `volumetric_kit/core/base/result.hpp`.
///
/// @code
/// Result<Device> r = Device::create(instance, physical, config);
/// if (!r) return r.status();   // propagate failure to our caller
/// Device& device = r.value();  // safe: guarded by the !r check above
/// @endcode

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/gfx/core/check.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

using core::Result;
using core::Status;
using core::to_string;

// TODO: take vk_error, vk_result, to_string(VkResult) and VG_VK_TRY from the
// core's vulkan tier (volumetric_kit/core/vulkan/vk_result.hpp, the same
// names and contracts) once gfx adopts that tier; until then gfx keeps them, as
// the core's base tier includes no GPU API.

/// @brief Wrap a failed `VkResult` as a backend @ref Status.
/// @param result  What the failed Vulkan call returned.
/// @param what    Context for the message, e.g. the failing call.
/// @pre @p result is not `VK_SUCCESS`: a success code is no failure, and
///      `Status::backend_error` aborts on one.
/// @return A non-OK `Status`, domain `Status::Code::Backend`, whose
///         `detail()` is @p result.
inline Status vk_error(VkResult result, std::string_view what) {
  return Status::backend_error(static_cast<std::int64_t>(result),
                               std::string(what));
}

/// @brief The `VkResult` a backend @ref Status carries.
/// @param status  Any status.
/// @return Its `detail()` as a `VkResult` when its domain is
///         `Status::Code::Backend`; empty for any other domain, success
///         included.
///
/// @code
/// const Status s = swapchain.present(queue);
/// if (vk_result(s) == VK_ERROR_OUT_OF_DATE_KHR) recreate();
/// @endcode
inline std::optional<VkResult> vk_result(const Status& status) noexcept {
  if (status.domain() != Status::Code::Backend) return std::nullopt;
  return static_cast<VkResult>(status.detail());
}

/// @brief Human-readable name for a `VkResult` (e.g. "VK_ERROR_DEVICE_LOST").
/// @param result  Any `VkResult`.
/// @return A static `string_view`; unrecognized codes yield
///         "VK_RESULT_UNKNOWN".
VG_CORE_API std::string_view to_string(VkResult result) noexcept;

}  // namespace volumetric_kit::gfx

// TODO: rename VG_TRY / VG_ASSIGN (and VG_CHECK, check.hpp) to the core's
// VKC_TRY / VKC_ASSIGN / VKC_CHECK across gfx once its open branches have
// landed, then delete these aliases, as recon plans for its VR_* names. Until
// then the old names keep those branches merging cleanly, and new code uses
// them too.

/// @brief gfx's name for the core's `VKC_TRY`: evaluate a `Status` expression
///        and early-return it if not OK.
/// @param expr  An expression yielding a `Status`.
///
/// Usable only inside a function returning `Status` or `Result<T>`.
///
/// @code
/// Status init() {
///   VG_TRY(create_instance());   // returns the error if this fails
///   return {};                   // success
/// }
/// @endcode
#define VG_TRY(expr) VKC_TRY(expr)

/// @brief gfx's name for the core's `VKC_ASSIGN`: unwrap a `Result<T>` into
///        @p decl, or early-return its `Status`.
/// @param decl  A variable declaration bound to the unwrapped value.
/// @param expr  An expression yielding a `Result<T>`.
///
/// It declares @p decl in the enclosing scope, so it is a statement sequence:
/// never the unbraced body of an `if`/`for`/`while`. A type with a top-level
/// comma needs an alias first.
///
/// @code
/// Result<Pipeline> build(VkDevice device) {
///   VG_ASSIGN(ShaderModule vert, ShaderModule::create(device, code, bytes));
///   return assemble(vert);
/// }
/// @endcode
#define VG_ASSIGN(decl, expr) VKC_ASSIGN(decl, expr)

/// @brief Evaluate a raw `VkResult` and early-return a backend `Status` on
///        failure.
/// @param expr  An expression yielding a `VkResult`. The expression text is
///              stringified (via `#expr`) as the error context, so the failing
///              call names itself -- no separate message argument.
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
#define VG_VK_TRY(expr)                                      \
  do {                                                       \
    const VkResult _vg_vk = (expr);                          \
    if (_vg_vk != VK_SUCCESS) {                              \
      return ::volumetric_kit::gfx::vk_error(_vg_vk, #expr); \
    }                                                        \
  } while (0)

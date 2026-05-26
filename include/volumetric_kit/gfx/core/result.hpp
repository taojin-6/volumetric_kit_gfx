// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file result.hpp
/// @brief Exception-free error handling for the public API.
///
/// No exceptions cross the library boundary: mobile (iOS/Android) consumers
/// frequently build with `-fno-exceptions`, where a throwing API is unusable.
/// Fallible calls therefore report failure by value:
///
/// - @ref Status    -- success, or a `VkResult` code plus a context message.
/// - @ref Result    -- a `T` on success, or a `Status` on failure.
///
/// Two macros remove the check-and-propagate boilerplate: @ref VG_TRY (for a
/// `Status` expression) and @ref VG_VK_TRY (for a raw `VkResult`). Both
/// early-return on failure, so they appear only inside functions that
/// themselves return `Status` or `Result<T>`.
///
/// Misuse -- reading the value of an error `Result` -- is a programmer error,
/// not a runtime one: it fails fast via `VG_CHECK` (see check.hpp) rather than
/// throwing.
///
/// @code
/// Result<Device> r = Device::create(instance, physical, config);
/// if (!r) return r.status();   // propagate failure to our caller
/// Device& device = r.value();  // safe: guarded by the !r check above
/// @endcode

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Success, or a Vulkan error code paired with a human-readable message.
///
/// A default-constructed `Status` is success; build a failure with @ref error.
/// Convertible to `bool` (true == success) for terse checks.
///
/// @code
/// Status s = upload();
/// if (!s) {
///   log_message(LogLevel::Error, s.message());
///   return s;
/// }
/// @endcode
class Status {
 public:
  /// Construct a success status.
  Status() = default;

  /// @brief Build a failure status.
  /// @param code     Vulkan result code; must not be `VK_SUCCESS`.
  /// @param message  Human-readable context (e.g. the failing call site).
  /// @return A non-OK `Status` carrying @p code and @p message.
  static Status error(VkResult code, std::string message) {
    return Status{code, std::move(message)};
  }

  /// @return `true` if this is a success status.
  bool ok() const noexcept { return code_ == VK_SUCCESS; }
  /// @return `true` on success (same as @ref ok).
  explicit operator bool() const noexcept { return ok(); }

  /// @return The Vulkan result code; `VK_SUCCESS` when @ref ok.
  VkResult code() const noexcept { return code_; }
  /// @return The failure context message; empty when @ref ok.
  const std::string& message() const noexcept { return message_; }

 private:
  Status(VkResult code, std::string message)
      : code_(code), message_(std::move(message)) {}

  VkResult code_ = VK_SUCCESS;
  std::string message_;
};

/// @brief Build an error @ref Status from a failed code and a `string_view`.
/// @param code  A failed `VkResult` (not `VK_SUCCESS`).
/// @param what  Short context string, copied into the Status message.
/// @return A non-OK `Status`.
inline Status vk_error(VkResult code, std::string_view what) {
  return Status::error(code, std::string(what));
}

/// @brief A value of type `T` on success, or a non-OK @ref Status on failure.
/// @tparam T  The success value type (must be movable).
///
/// Constructs implicitly from either a `T` (success) or a `Status` (failure),
/// so a function can `return value;` or `return some_error;` directly. Always
/// check @ref ok (or the `bool` conversion) before reading @ref value.
///
/// @code
/// Result<Buffer> make_buffer(std::size_t bytes) {
///   if (bytes == 0) return vk_error(VK_ERROR_INITIALIZATION_FAILED, "empty");
///   return Buffer{bytes};   // implicit success
/// }
/// @endcode
template <class T>
class Result {
 public:
  /// Construct a success Result holding @p value.
  Result(T value);  // NOLINT(google-explicit-constructor) — ergonomic success
                    // return
  /// Construct a failure Result; @p err must be non-OK (checked by VG_CHECK).
  Result(Status err);  // NOLINT(google-explicit-constructor) — ergonomic error
                       // return

  /// @return `true` if this holds a value rather than an error.
  bool ok() const noexcept { return status_.ok(); }
  /// @return `true` if this holds a value (same as @ref ok).
  explicit operator bool() const noexcept { return ok(); }
  /// @return The status; non-OK exactly when this is an error Result.
  const Status& status() const noexcept { return status_; }

  /// @brief Access the held value.
  /// @pre @ref ok is true. Calling this on an error Result is a programmer
  ///      error: it aborts via `VG_CHECK` (it never throws), so guard with
  ///      @ref ok first.
  /// @return Reference to the held value.
  T& value() &;
  const T& value() const&;
  T&& value() &&;

  /// @brief Pointer/reference access to the held value.
  /// @pre @ref ok is true; otherwise aborts, as in @ref value.
  T* operator->();
  const T* operator->() const;
  T& operator*() &;
  const T& operator*() const&;

 private:
  Status status_;
  std::optional<T> value_;
};

}  // namespace volumetric_kit::gfx

/// @brief Evaluate a `Status` expression and early-return it if not OK.
/// @param expr  An expression yielding a `Status`.
///
/// Usable only inside a function returning `Status` or `Result<T>` -- the early
/// `return` carries the failure outward.
///
/// @code
/// Status init() {
///   VG_TRY(create_instance());   // returns the error if this fails
///   return {};                   // success
/// }
/// @endcode
#define VG_TRY(expr)                                   \
  do {                                                 \
    ::volumetric_kit::gfx::Status _vg_status = (expr); \
    if (!_vg_status.ok()) return _vg_status;           \
  } while (0)

/// @brief Evaluate a raw `VkResult` and early-return a `Status` on failure.
/// @param expr  An expression yielding a `VkResult`. The expression text is
///              stringified (via `#expr`) as the error context, so the failing
///              call names itself -- no separate message argument.
///
/// @code
/// VG_VK_TRY(vkCreateDevice(phys, &ci, nullptr, &dev_));
/// @endcode
#define VG_VK_TRY(expr)                                      \
  do {                                                       \
    VkResult _vg_vk = (expr);                                \
    if (_vg_vk != VK_SUCCESS)                                \
      return ::volumetric_kit::gfx::vk_error(_vg_vk, #expr); \
  } while (0)

#include "volumetric_kit/gfx/core/impl/result.ipp"

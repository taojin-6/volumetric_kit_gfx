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
/// - @ref Status    -- success, or an error domain (@ref Status::Code) with an
///                     optional `VkResult` detail and a context message.
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

#include "volumetric_kit/gfx/core/check.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Success, or an error: a domain (@ref Code), an optional `VkResult`
///        detail, and a human-readable message.
///
/// A default-constructed `Status` is success. Build a failure with a domain
/// factory (@ref invalid_argument, @ref not_found, @ref unsupported,
/// @ref out_of_memory, @ref io_error) or, for a failed Vulkan call, @ref error
/// or @ref vk_error (which set @ref domain to @ref Code::Vulkan and carry the
/// `VkResult`). Convertible to `bool` (true == success) for terse checks.
///
/// @code
/// Status s = upload();
/// if (!s) {
///   std::string detail(to_string(s.domain()));
///   if (s.domain() == Status::Code::Vulkan) {
///     detail += '/';
///     detail += to_string(s.code());  // recover the specific VkResult
///   }
///   log_message(LogLevel::Error, detail + ": " + s.message());
///   return s;
/// }
/// @endcode
class Status {
 public:
  /// @brief The kind of failure a non-OK `Status` reports.
  ///
  /// This is the primary discriminator. @ref code carries a meaningful
  /// `VkResult` only when the domain is @ref Code::Vulkan; for every other
  /// domain it is `VK_SUCCESS`.
  enum class Code {
    Ok,               ///< Success.
    InvalidArgument,  ///< A malformed or contradictory argument value.
    NotFound,         ///< A named resource or file does not exist.
    Unsupported,      ///< A valid request the device or build cannot satisfy.
    OutOfMemory,      ///< A host or device allocation failed.
    IoError,          ///< A read/write/decode/encode operation failed.
    Vulkan,           ///< A Vulkan call failed; @ref code holds the `VkResult`.
  };

  /// Construct a success status.
  Status() = default;

  /// @brief Build a Vulkan failure status (domain @ref Code::Vulkan).
  /// @param code     Vulkan result code; must not be `VK_SUCCESS`.
  /// @param message  Human-readable context (e.g. the failing call site).
  /// @return A non-OK `Status` carrying @p code and @p message.
  static Status error(VkResult code, std::string message) {
    VG_CHECK(code != VK_SUCCESS, "Status::error needs a failed VkResult");
    return Status{Code::Vulkan, code, std::move(message)};
  }

  /// @brief Build a non-Vulkan failure status in the named domain.
  /// @param message  Human-readable context.
  /// @return A non-OK `Status` whose @ref domain is the factory's domain and
  ///         whose @ref code is `VK_SUCCESS` (no Vulkan call was involved).
  static Status invalid_argument(std::string message) {
    return Status{Code::InvalidArgument, std::move(message)};
  }
  /// @copydoc invalid_argument
  static Status not_found(std::string message) {
    // TODO: emitted by the assets tier (named resource / file lookup); no core
    // producer yet.
    return Status{Code::NotFound, std::move(message)};
  }
  /// @copydoc invalid_argument
  static Status unsupported(std::string message) {
    return Status{Code::Unsupported, std::move(message)};
  }
  /// @copydoc invalid_argument
  static Status out_of_memory(std::string message) {
    // TODO: map genuine VMA/Vulkan OOM (VK_ERROR_OUT_OF_*_MEMORY) into this
    // domain in the allocator/interop tier; today such failures stay
    // Code::Vulkan with the VkResult in code(), so this factory has no core
    // producer yet.
    return Status{Code::OutOfMemory, std::move(message)};
  }
  /// @copydoc invalid_argument
  static Status io_error(std::string message) {
    // TODO: emitted by the assets tier (read/write/decode/encode); no core
    // producer yet.
    return Status{Code::IoError, std::move(message)};
  }

  /// @return `true` if this is a success status.
  bool ok() const noexcept { return domain_ == Code::Ok; }
  /// @return `true` on success (same as @ref ok).
  explicit operator bool() const noexcept { return ok(); }

  /// @return The error domain; @ref Code::Ok exactly when @ref ok.
  Code domain() const noexcept { return domain_; }
  /// @return The Vulkan result code. Meaningful only when @ref domain is
  ///         @ref Code::Vulkan; `VK_SUCCESS` otherwise.
  VkResult code() const noexcept { return code_; }
  /// @return The failure context message; empty when @ref ok.
  const std::string& message() const noexcept { return message_; }

 private:
  // Non-Vulkan domains carry no VkResult; this overload fixes code_ to
  // VK_SUCCESS so a domain factory cannot pair a real VkResult with a
  // non-Vulkan domain. Only error() takes an explicit VkResult.
  Status(Code domain, std::string message)
      : domain_(domain), message_(std::move(message)) {}
  Status(Code domain, VkResult code, std::string message)
      : domain_(domain), code_(code), message_(std::move(message)) {}

  Code domain_ = Code::Ok;
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

/// @brief Human-readable name for a @ref Status::Code (e.g. "InvalidArgument").
/// @param code  A domain value.
/// @return A static, never-empty `string_view`.
VG_CORE_API std::string_view to_string(Status::Code code) noexcept;

/// @brief Human-readable name for a `VkResult` (e.g. "VK_ERROR_DEVICE_LOST").
/// @param result  Any `VkResult`.
/// @return A static `string_view`; unrecognized codes yield
/// "VK_RESULT_UNKNOWN".
VG_CORE_API std::string_view to_string(VkResult result) noexcept;

/// @brief A value of type `T` on success, or a non-OK @ref Status on failure.
/// @tparam T  The success value type (must be movable).
///
/// Constructs implicitly from either a `T` (success) or a `Status` (failure),
/// so a function can `return value;` or `return some_error;` directly. Always
/// check @ref ok (or the `bool` conversion) before reading @ref value.
///
/// @code
/// Result<Buffer> make_buffer(std::size_t bytes) {
///   if (bytes == 0) return Status::invalid_argument("empty");
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
  T&& operator*() &&;

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
    VkResult _vg_vk = (expr);                                \
    if (_vg_vk != VK_SUCCESS)                                \
      return ::volumetric_kit::gfx::vk_error(_vg_vk, #expr); \
  } while (0)

/// @brief Evaluate a `Result<T>` expression, early-return its `Status` on
///        failure, otherwise move the value into @p decl.
/// @param decl  A variable declaration (e.g. `Device device`) bound to the
///              unwrapped value on success.
/// @param expr  An expression yielding a `Result<T>`.
///
/// The `Result<T>` analogue of @ref VG_TRY: it removes the check-status-then-
/// move-value boilerplate that fallible-value call sites otherwise repeat.
/// Usable only inside a function returning `Status` or `Result<U>`. Because it
/// declares @p decl in the enclosing scope, it expands to a statement sequence
/// (not a `do { } while`), so it is not a single statement -- never use it as
/// the unbraced body of an `if`/`for`/`while`. The hidden temporary is keyed on
/// `__COUNTER__` (not `__LINE__`), so multiple `VG_ASSIGN`s in one scope never
/// collide -- even two on the same source line. @p decl is a single macro
/// argument, so a type written with a top-level comma needs an alias first
/// (e.g. `using Pair = std::pair<int, int>;` then `VG_ASSIGN(Pair p, expr)`).
///
/// @code
/// Result<Pipeline> build(VkDevice device) {
///   VG_ASSIGN(ShaderModule vert, ShaderModule::create(device, code, bytes));
///   // `vert` holds the value here; a failure already returned its Status.
///   return assemble(vert);
/// }
/// @endcode
#define VG_ASSIGN(decl, expr) VG_ASSIGN_(decl, expr, __COUNTER__)
#define VG_ASSIGN_(decl, expr, id) VG_ASSIGN_IMPL_(decl, expr, id)
#define VG_ASSIGN_IMPL_(decl, expr, id)                       \
  auto _vg_result_##id = (expr);                              \
  if (!_vg_result_##id.ok()) return _vg_result_##id.status(); \
  decl = std::move(_vg_result_##id).value()

#include "volumetric_kit/gfx/core/impl/result.hpp"

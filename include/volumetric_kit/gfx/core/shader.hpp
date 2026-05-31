// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file shader.hpp
/// @brief A `VkShaderModule` built from a SPIR-V blob.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Owns a `VkShaderModule` compiled from SPIR-V words.
///
/// A module is a thin wrapper around the SPIR-V handed to pipeline creation; a
/// pipeline copies what it needs at build time, so the module can be destroyed
/// once every pipeline using it has been created. A default-constructed
/// `ShaderModule` is empty (`valid()` is false) and safe to move-assign into.
///
/// @warning The @p device passed to @ref create must outlive the module: the
///          destructor frees through it, so destroying the device first is
///          undefined behavior.
///
/// @code
/// // `spirv` holds the compiled .spv as 32-bit words, loaded by the caller.
/// Result<ShaderModule> vert = ShaderModule::create(
///     device, spirv.data(), spirv.size() * sizeof(uint32_t));
/// if (!vert) return vert.status();
/// // ... reference vert.value().handle() in a VkPipelineShaderStageCreateInfo
/// @endcode
class VG_CORE_API ShaderModule {
 public:
  /// @brief Construct an empty module (owns nothing; `valid()` is false).
  ShaderModule() = default;

  /// @brief Create a shader module from a SPIR-V blob.
  /// @param device     The logical device that owns the module.
  /// @param code       Pointer to the SPIR-V words (4-byte aligned by type).
  /// @param size_bytes Size of @p code in **bytes** — a non-zero multiple of 4.
  /// @pre @p code is non-null and @p size_bytes is a non-zero multiple of 4;
  ///      these are validated before Vulkan is touched and otherwise yield a
  ///      non-OK @ref Status carrying `VK_ERROR_INITIALIZATION_FAILED`.
  /// @return The module on success, or a non-OK @ref Status.
  static Result<ShaderModule> create(VkDevice device, const uint32_t* code,
                                     size_t size_bytes);

  ~ShaderModule() = default;
  ShaderModule(ShaderModule&&) noexcept = default;
  ShaderModule& operator=(ShaderModule&&) noexcept = default;
  ShaderModule(const ShaderModule&) = delete;
  ShaderModule& operator=(const ShaderModule&) = delete;

  /// @return The underlying `VkShaderModule` (`VK_NULL_HANDLE` when empty).
  VkShaderModule handle() const noexcept { return module_.get(); }

  /// @return `true` if this owns a module.
  bool valid() const noexcept { return module_.valid(); }

 private:
  UniqueHandle<VkShaderModule, vkDestroyShaderModule> module_;
};

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file shader.hpp
/// @brief A `VkShaderModule` built from a SPIR-V blob, plus the descriptor
///        interface recovered from it by reflection.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief One descriptor-bound resource a shader declares, recovered from the
///        SPIR-V by reflection — the currency a pipeline builds its
///        descriptor-set layouts from.
///
/// @code
/// for (const ReflectedResource& r : vert.resources())
///   if (r.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) { ... }
/// @endcode
struct ReflectedResource {
  uint32_t set = 0;      ///< Descriptor set index (GLSL `layout(set = N)`).
  uint32_t binding = 0;  ///< Binding within the set (`layout(binding = M)`).
  /// The Vulkan descriptor type (uniform buffer, combined image sampler, …).
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
  /// Array length; 1 for a scalar binding, 0 for a runtime-sized array.
  uint32_t count = 1;
  VkShaderStageFlags stages = 0;  ///< Stage(s) that declare it.
};

/// @brief Owns a `VkShaderModule` compiled from SPIR-V words and the descriptor
///        interface reflected from it.
///
/// A module is a thin wrapper around the SPIR-V handed to pipeline creation; a
/// pipeline copies what it needs at build time, so the module can be destroyed
/// once every pipeline using it has been created. @ref create also reflects the
/// SPIR-V (via spirv-cross) into the @ref ReflectedResource list and
/// push-constant size a pipeline turns into descriptor-set + pipeline layouts.
/// A default-constructed `ShaderModule` is empty (`valid()` is false) and safe
/// to move-assign into.
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
  ///      non-OK @ref Status with domain @ref Status::Code::InvalidArgument.
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

  /// @return The stage reflected from the SPIR-V's execution model (e.g.
  ///         `VK_SHADER_STAGE_VERTEX_BIT`).
  VkShaderStageFlagBits stage() const noexcept { return stage_; }

  /// @return The descriptor-bound resources the shader declares, in no
  ///         particular order (empty when it binds none).
  const std::vector<ReflectedResource>& resources() const noexcept {
    return resources_;
  }

  /// @return The push-constant block size in bytes (0 when none is declared).
  uint32_t push_constant_size() const noexcept { return push_constant_size_; }

 private:
  UniqueHandle<VkShaderModule, vkDestroyShaderModule> module_;
  std::vector<ReflectedResource> resources_;
  uint32_t push_constant_size_ = 0;
  VkShaderStageFlagBits stage_ = VK_SHADER_STAGE_ALL;
};

}  // namespace volumetric_kit::gfx

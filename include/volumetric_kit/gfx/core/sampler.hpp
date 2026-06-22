// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sampler.hpp
/// @brief A `VkSampler`: the filtering and addressing state a shader reads a
///        texture through.

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Filtering, mip, and addressing state for a @ref Sampler.
///
/// The defaults describe the common glTF-texture sampler: trilinear filtering
/// (linear min/mag with linear blending between mip levels) over the whole mip
/// chain, with `REPEAT` wrapping on every axis. Override per field for, e.g.,
/// `NEAREST` filtering or `CLAMP_TO_EDGE` wrapping.
struct SamplerDesc {
  VkFilter mag_filter = VK_FILTER_LINEAR;  ///< Filter when magnifying.
  VkFilter min_filter = VK_FILTER_LINEAR;  ///< Filter when minifying.
  VkSamplerMipmapMode mipmap_mode =
      VK_SAMPLER_MIPMAP_MODE_LINEAR;  ///< Blend between mip levels.
  VkSamplerAddressMode address_mode_u =
      VK_SAMPLER_ADDRESS_MODE_REPEAT;  ///< Wrap on U (S).
  VkSamplerAddressMode address_mode_v =
      VK_SAMPLER_ADDRESS_MODE_REPEAT;  ///< Wrap on V (T).
  VkSamplerAddressMode address_mode_w =
      VK_SAMPLER_ADDRESS_MODE_REPEAT;  ///< Wrap on W (R; 3D textures).
  float min_lod = 0.0f;  ///< Lowest mip LOD the sampler clamps to.
  /// Highest mip LOD the sampler clamps to; the `VK_LOD_CLAMP_NONE` default
  /// leaves the entire mip chain reachable. Must be >= @ref min_lod.
  float max_lod = VK_LOD_CLAMP_NONE;
  /// Texel returned for coordinates outside `[0,1]` on a
  /// `VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER` axis; ignored by every other
  /// address mode.
  VkBorderColor border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
};
// TODO: anisotropic filtering (maxAnisotropy) -- needs the device
// samplerAnisotropy feature enabled and a clamp to the physical device's
// maxSamplerAnisotropy limit, so create() would take the Device (for its caps),
// not a raw VkDevice.

/// @brief Owns a `VkSampler` and frees it via @ref UniqueHandle.
///
/// A sampler is a pure device object (no backing memory), independent of any
/// one texture: bind it alongside a sampled @ref Texture in a descriptor set.
///
/// @warning The @p device passed to @ref create must outlive the sampler: the
///          destructor frees through it, so destroying the device first is
///          undefined behavior.
///
/// @code
/// Result<Sampler> sampler = Sampler::create(device.handle());
/// if (!sampler) return sampler.status();
/// // ... bind sampler.value().handle() with a texture's view in a descriptor
/// // set ...
/// @endcode
class VG_CORE_API Sampler {
 public:
  /// @brief Create a sampler.
  /// @param device  The logical device that owns the sampler.
  /// @param desc    Filtering, mip, and addressing state (defaults to trilinear
  ///                + REPEAT; see @ref SamplerDesc).
  /// @return The sampler on success, or a non-OK @ref Status: @ref
  ///         Status::Code::InvalidArgument for a null @p device or a
  ///         `desc.max_lod < desc.min_lod`; otherwise a Vulkan-domain Status
  ///         carrying the `VkResult` if `vkCreateSampler` fails.
  static Result<Sampler> create(VkDevice device, const SamplerDesc& desc = {});

  ~Sampler() = default;
  Sampler(Sampler&&) noexcept = default;
  Sampler& operator=(Sampler&&) noexcept = default;
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  /// @return The underlying `VkSampler` handle (`VK_NULL_HANDLE` when empty).
  VkSampler handle() const noexcept { return handle_.get(); }

  /// @return `true` if this owns a sampler.
  bool valid() const noexcept { return handle_.valid(); }

 private:
  Sampler() = default;

  UniqueHandle<VkSampler, vkDestroySampler> handle_;
};

}  // namespace volumetric_kit::gfx

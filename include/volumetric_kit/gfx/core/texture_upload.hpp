// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file texture_upload.hpp
/// @brief Create a sampled @ref Texture and fill it from CPU pixels in one
///        blocking transfer.

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

class Device;

/// @brief A CPU pixel buffer plus the options for uploading it into a sampled
///        texture.
///
/// @ref pixels is tightly packed, row-major, `extent.width * extent.height *
/// texel_size(format)` bytes, for an uncompressed single-plane color @ref
/// format. The renderer's CPU image model (`assets::Image`) is deliberately
/// GPU-API-free, so choosing the concrete @ref format -- and any RGB->RGBA
/// expansion the GPU needs, since most GPUs do not sample three-channel 8-bit
/// formats -- happens at the call site, keeping this the format-agnostic core
/// seam.
struct ImageUploadDesc {
  VkExtent2D extent{};  ///< Width/height in texels.
  VkFormat format =
      VK_FORMAT_UNDEFINED;       ///< Texel format (uncompressed color).
  const void* pixels = nullptr;  ///< Source pixels (see @ref ImageUploadDesc).
  VkDeviceSize size = 0;         ///< Byte length of @ref pixels.
  /// Build a full mip chain by halving linear blits down from mip 0. Requires a
  /// format that supports linear blit and linear filtering, or the upload
  /// returns @ref Status::Code::Unsupported. When false, a single mip is
  /// uploaded.
  bool generate_mips = false;
};

/// @brief Upload @p desc.pixels into a new device-local, shader-sampled @ref
///        Texture, returning once the copy has completed on the GPU.
///
/// Stages the pixels through a host-visible buffer and records the copy (plus,
/// when @ref ImageUploadDesc::generate_mips, the mip-chain blits) on the
/// device's graphics queue via @ref Device::submit_single_time, blocking on a
/// fence so the transient staging buffer is freed only after the GPU is done.
///
/// @param device     The device whose graphics queue runs the one-time
///                   transfer; must outlive the returned texture.
/// @param allocator  Allocates the staging buffer and the destination image;
///                   must outlive the returned texture (see @ref Texture).
/// @param desc       Source pixels, extent, format, and mip option.
/// @return The texture -- sampled-ready in `SHADER_READ_ONLY_OPTIMAL`, with a
///         default view spanning all mips -- or a non-OK @ref Status: @ref
///         Status::Code::InvalidArgument for a zero extent,
///         `VK_FORMAT_UNDEFINED`, null pixels, or a `desc.size` that is not
///         `width * height * texel_size(format)`; @ref
///         Status::Code::Unsupported for a
///         compressed/multi-planar/depth-stencil format, or `generate_mips` on
///         a format that cannot be linear-blitted; otherwise a Vulkan-domain
///         Status from the staging-buffer, image, or submit step.
/// @note Blocking and queue-serializing -- a setup/load-time path, never the
///       per-frame one (see @ref Device::submit_single_time).
VG_CORE_API Result<Texture> upload_texture(const Device& device,
                                           Allocator& allocator,
                                           const ImageUploadDesc& desc);

}  // namespace volumetric_kit::gfx

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file texture_upload.hpp
/// @brief Create sampled @ref Texture objects and fill them from CPU pixels:
///        one blocking transfer per texture (@ref upload_texture) or many
///        textures per submit (@ref TextureUploadBatch).

#include <cstdint>
#include <vector>

#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

class Device;

/// @brief A CPU pixel buffer plus the options for uploading it into a sampled
///        texture (2D, 2D array, or cubemap; single-mip, pre-mipped, or
///        GPU-mipped).
///
/// @ref pixels is tightly packed with no row padding, for an uncompressed
/// single-plane color @ref format. Subresources are ordered mip-major, then
/// layer: all of mip 0's layers (layer `0 .. array_layers-1`), then all of
/// mip 1's, and so on. Mip `m` of a `w x h` image is
/// `max(1, w >> m) x max(1, h >> m)` texels, so @ref size must equal the sum of
/// every (mip, layer) subresource:
/// `array_layers * sum_m(mip_width(m) * mip_height(m)) * texel_size(format)`.
///
/// The renderer's CPU image model (`assets::Image`) is deliberately
/// GPU-API-free, so choosing the concrete @ref format -- and any RGB->RGBA
/// expansion the GPU needs, since most GPUs do not sample three-channel 8-bit
/// formats -- happens at the call site, keeping this the format-agnostic core
/// seam.
struct ImageUploadDesc {
  VkExtent2D extent{};  ///< Width/height of mip 0, in texels.
  VkFormat format =
      VK_FORMAT_UNDEFINED;       ///< Texel format (uncompressed color).
  const void* pixels = nullptr;  ///< Source pixels (see @ref ImageUploadDesc).
  VkDeviceSize size = 0;         ///< Byte length of @ref pixels.
  /// Array layers (cube faces when @ref cube). A value > 1 uploads a 2D array
  /// texture whose default view is the matching array view.
  uint32_t array_layers = 1;
  /// Create a cubemap: requires `array_layers == 6` and a square @ref extent
  /// (faces are Vulkan layers ordered +X, -X, +Y, -Y, +Z, -Z); the default view
  /// is a `VK_IMAGE_VIEW_TYPE_CUBE` view.
  bool cube = false;
  /// Mip levels supplied in @ref pixels (see the packing contract above). Must
  /// not exceed the full chain for @ref extent, and is mutually exclusive with
  /// @ref generate_mips.
  uint32_t mip_levels = 1;
  /// Build a full mip chain by halving linear blits down from mip 0. Requires a
  /// format that supports linear blit and linear filtering, or the upload
  /// returns @ref Status::Code::Unsupported; only available for a single-layer,
  /// single-source-mip image (`array_layers == 1 && mip_levels == 1`). When
  /// false, exactly the supplied @ref mip_levels are uploaded.
  bool generate_mips = false;
};

/// @brief Batches texture uploads into one command buffer and one blocking
///        queue submission, replacing N serial CPU-GPU round trips with one.
///
/// @ref begin opens a one-time command buffer on the device's graphics pool;
/// each @ref add validates its @ref ImageUploadDesc, creates the destination
/// @ref Texture plus a staging @ref Buffer, and records the copies and layout
/// transitions into that open buffer; @ref finish submits everything at once
/// and blocks on a fence, after which every added texture is sampled-ready in
/// `SHADER_READ_ONLY_OPTIMAL`. A batch is one-shot: after @ref finish
/// (successful or not) it is empty (`valid()` is false) and cannot be reused --
/// @ref begin a new one. A default-constructed batch is likewise empty and safe
/// to move-assign into.
///
/// @warning The @p device and @p allocator passed to @ref begin must outlive
///          the batch (and, per @ref Texture, every texture it produced).
///          Destroying a batch without calling @ref finish discards the
///          pending uploads: nothing is submitted, the staging buffers and
///          command buffer are freed, and the textures returned by @ref add
///          hold undefined contents.
///
/// @code
/// Result<TextureUploadBatch> batch =
///     TextureUploadBatch::begin(device, allocator);
/// if (!batch) return batch.status();
/// Result<Texture> albedo = batch.value().add(albedo_desc);
/// if (!albedo) return albedo.status();
/// Result<Texture> normal = batch.value().add(normal_desc);
/// if (!normal) return normal.status();
/// VG_TRY(batch.value().finish());  // one submit; textures now sampled-ready
/// @endcode
class VG_CORE_API TextureUploadBatch {
 public:
  /// @brief Construct an empty batch (owns nothing; `valid()` is false).
  TextureUploadBatch() noexcept = default;

  /// @brief Open a batch: allocate a primary command buffer from @p device's
  ///        graphics pool and start recording.
  /// @param device     Supplies the command pool and, at @ref finish, the
  ///                   graphics queue; must outlive the batch.
  /// @param allocator  Allocates each @ref add's staging buffer and texture;
  ///                   must outlive the batch and the returned textures.
  /// @return The open batch, or a Vulkan-domain @ref Status if the command
  ///         buffer could not be allocated or begun.
  /// @note Not internally synchronized: like @ref Device::submit_single_time,
  ///       the batch records on the device's shared graphics pool and submits
  ///       on its queue, both of which Vulkan requires be externally
  ///       synchronized. Serialize batches against other users of that
  ///       pool/queue.
  static Result<TextureUploadBatch> begin(const Device& device,
                                          Allocator& allocator);

  ~TextureUploadBatch();
  TextureUploadBatch(TextureUploadBatch&& other) noexcept;
  TextureUploadBatch& operator=(TextureUploadBatch&& other) noexcept;
  TextureUploadBatch(const TextureUploadBatch&) = delete;
  TextureUploadBatch& operator=(const TextureUploadBatch&) = delete;

  /// @brief Create a texture for @p desc and record its upload into the open
  ///        command buffer.
  ///
  /// Validation and resource creation happen before any recording, so a failed
  /// add leaves the batch open and unchanged, with nothing recorded.
  /// @param desc  Source pixels, extent, format, and layer/mip options; see
  ///              @ref upload_texture for the validation rules.
  /// @return The created texture -- usable only after @ref finish returns OK --
  ///         or a non-OK @ref Status: @ref Status::Code::InvalidArgument when
  ///         the batch is empty (not begun, moved-from, or already finished),
  ///         plus everything @ref upload_texture rejects.
  /// @warning Keep the returned texture alive at least until @ref finish
  ///          returns: the recorded upload writes into it, so destroying it
  ///          earlier would submit against a freed image.
  Result<Texture> add(const ImageUploadDesc& desc);

  /// @brief Submit the batch and block until the GPU completes it.
  ///
  /// Ends the command buffer, submits it once on the device's graphics queue,
  /// waits on a fence, then frees the staging buffers and the command buffer.
  /// The batch is empty afterwards -- on success and on failure alike (a failed
  /// batch's uploads are discarded, and its textures hold undefined contents).
  /// @return OK once every added texture is sampled-ready, @ref
  ///         Status::Code::InvalidArgument if the batch is empty, or a
  ///         Vulkan-domain @ref Status from the end/submit/wait step.
  Status finish();

  /// @return `true` while the batch holds an open command buffer (begun, not
  ///         yet finished or moved-from).
  bool valid() const noexcept { return cmd_.valid(); }

 private:
  const Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;
  CommandBuffer cmd_;
  // Each add()'s staging buffer, kept alive until finish()'s fence proves the
  // GPU is done reading them (or until an unfinished batch discards them).
  std::vector<Buffer> staging_;
};

/// @brief Upload @p desc.pixels into a new device-local, shader-sampled @ref
///        Texture, returning once the copy has completed on the GPU.
///
/// A one-texture @ref TextureUploadBatch (begin + add + finish): stages the
/// pixels through a host-visible buffer and records one
/// `VkBufferImageCopy` per mip level -- each spanning all
/// @ref ImageUploadDesc::array_layers layers, which the packing contract keeps
/// contiguous per mip -- plus, when @ref ImageUploadDesc::generate_mips, the
/// mip-chain blits, then submits once and blocks on a fence. Uploading many
/// textures? Share one batch instead of paying a round trip each.
///
/// @param device     The device whose graphics queue runs the one-time
///                   transfer; must outlive the returned texture.
/// @param allocator  Allocates the staging buffer and the destination image;
///                   must outlive the returned texture (see @ref Texture).
/// @param desc       Source pixels, extent, format, and layer/mip options.
/// @return The texture -- sampled-ready in `SHADER_READ_ONLY_OPTIMAL`, with a
///         default view spanning all mips and layers -- or a non-OK @ref
///         Status: @ref Status::Code::InvalidArgument for a zero extent,
///         `VK_FORMAT_UNDEFINED`, null pixels, zero
///         `array_layers`/`mip_levels`, a `cube` that is not a square six-layer
///         image, more `mip_levels` than the extent's full chain,
///         `generate_mips` combined with `array_layers > 1` or `mip_levels >
///         1`, or a `desc.size` that does not match the packing contract (see
///         @ref ImageUploadDesc); @ref Status::Code::Unsupported for an extent
///         beyond the device's `maxImageDimension2D` (`maxImageDimensionCube`
///         for cubes), more layers than `maxImageArrayLayers`, a
///         compressed/multi-planar/depth-stencil format, a format that cannot
///         be sampled with optimal tiling, or `generate_mips` on a format that
///         cannot be linear-blitted; otherwise a Vulkan-domain Status from the
///         staging-buffer, image, or submit step.
/// @note Blocking and queue-serializing -- a setup/load-time path, never the
///       per-frame one (see @ref Device::submit_single_time).
VG_CORE_API Result<Texture> upload_texture(const Device& device,
                                           Allocator& allocator,
                                           const ImageUploadDesc& desc);

}  // namespace volumetric_kit::gfx

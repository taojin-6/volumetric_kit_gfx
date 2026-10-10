// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_pipeline.hpp
/// @brief 2D images on screen: a source image converted to display color and
///        mip-mapped on the GPU (@ref ImageTexture, filled by
///        @ref ImagePipeline::record_update), then drawn into rectangles of a
///        render target (@ref ImagePipeline::submit).

#include <array>
#include <cstdint>
#include <optional>

#include <glm/vec2.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/unique_handle.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::core {
class Allocator;
class Buffer;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

/// @brief How a source's texels become display color.
enum class ImageMapping : uint32_t {
  /// RGBA texels shown as their RGB (alpha is ignored; images draw opaque).
  Color = 0,
  /// The first channel, shown as grey.
  Grey = 1,
  /// The first channel through a color ramp between
  /// @ref ImageUpdate::ramp_min and @ref ImageUpdate::ramp_max: depth, error
  /// and coverage maps.
  Ramp = 2,
  /// Two-plane 8-bit Y'CbCr 4:2:0 (NV12), as video decoders leave a picture:
  /// plane 0 the luma (`R8_UNORM`), plane 1 the interleaved chroma
  /// (`R8G8_UNORM`, Cb first) at half the size, rounded up.
  Nv12 = 3,
  // TODO: I420 (three planes), as FFmpeg's software decoder leaves a picture.
};

/// @brief The transfer function a source's color is encoded with.
enum class ImageEncoding : uint32_t {
  /// sRGB-encoded (gamma), as camera and 8-bit image bytes are.
  Srgb = 0,
  /// Linear light.
  Linear = 1,
};

/// @brief Where an NV12 picture's chroma samples sit relative to its luma.
enum class ChromaSiting : uint32_t {
  /// Between the luma samples on both axes: JPEG (and so MJPEG).
  Center = 0,
  /// Level with the left luma sample of each pair, between the rows: H.264
  /// and H.265.
  Left = 1,
};

/// @brief How a draw magnifies its image: when zoomed in past one texel per
///        pixel. Minifying always filters through the mip chain.
enum class ImageFilter : uint32_t {
  /// Each texel a crisp square: for inspecting pixels.
  Nearest = 0,
  /// Bilinear: for a smooth preview.
  Linear = 1,
};

/// @brief What an @ref ImageTexture is made for: its size and the kind of
///        source it converts.
///
/// Fixed for the texture's life: a source of another size, mapping or format
/// needs another texture.
struct ImageTextureDesc {
  /// The picture's size in texels (the luma's, for NV12); non-zero.
  VkExtent2D extent{};
  /// How the source becomes display color.
  ImageMapping mapping = ImageMapping::Color;
  /// Plane 0's format:
  /// - @ref ImageMapping::Color: `R8G8B8A8` or `B8G8R8A8`, `_UNORM` or
  ///   `_SRGB`;
  /// - @ref ImageMapping::Grey and @ref ImageMapping::Ramp: `R8_UNORM`,
  ///   `R16_UNORM` or `R32_SFLOAT`;
  /// - @ref ImageMapping::Nv12: ignored, as the planes' formats are fixed.
  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
};

/// @brief One plane of a source picture: a device image, or rows of a device
///        buffer -- exactly one.
///
/// The plane is copied from its top-left corner, so an image may be larger
/// than the picture, as a decoder's padded surface is. The copy reads it on
/// the queue the update is submitted to, so:
/// - its writes must come **before** the copy: recorded earlier on that
///   queue -- in the same command buffer or an earlier submission, which the
///   update's barrier waits for -- or done on another queue: a fence waited
///   on, or a semaphore the update's submission waits on at the `TRANSFER`
///   stage (under the frame loop, a `windowing::Frame::waits` entry);
/// - it must be readable from that queue's family: created `CONCURRENT`
///   across the producer's and the renderer's families, or written on the
///   renderer's family (MoltenVK gives two libraries different families);
/// - and it must outlive the update's execution, not merely its recording.
///
/// Every command recorded on the queue after the update waits for the copy,
/// so a producer on that queue may write the next picture into the plane
/// with no barrier of its own.
struct ImagePlane {
  /// An image: 2D, single-sample, with `TRANSFER_SRC` usage, in the layout
  /// its `layout()` records -- `GENERAL` or `TRANSFER_SRC_OPTIMAL` -- and a
  /// format of the plane's texel size and channel order (`_UNORM` and
  /// `_SRGB` are interchangeable).
  const core::Image* image = nullptr;
  /// Or a buffer with `TRANSFER_SRC` usage, holding the plane's rows.
  const core::Buffer* buffer = nullptr;
  /// The buffer's byte offset of the first row: a multiple of the texel
  /// size.
  VkDeviceSize offset = 0;
  /// The buffer's row length in texels; 0 for tightly packed rows.
  uint32_t row_length = 0;
};

/// @brief One update of an @ref ImageTexture: the source's planes, and how to
///        read them.
///
/// @code
/// pipelines::ImageUpdate update;  // a decoder's NV12 picture
/// update.planes[0].image = &luma;
/// update.planes[1].image = &chroma;
/// update.kr = 0.299f;  // BT.601, full range, as MJPEG
/// update.kb = 0.114f;
/// update.chroma_siting = pipelines::ChromaSiting::Center;
/// @endcode
struct ImageUpdate {
  /// The planes: plane 0 always; plane 1 for @ref ImageMapping::Nv12 only.
  std::array<ImagePlane, 2> planes{};
  /// What the values encode, for @ref ImageMapping::Color,
  /// @ref ImageMapping::Grey and @ref ImageMapping::Nv12 (whose matrix gives
  /// encoded R'G'B'). Must be @ref ImageEncoding::Srgb for an `_SRGB` format.
  /// The ramp's colors are its own, so @ref ImageMapping::Ramp ignores it.
  ImageEncoding encoding = ImageEncoding::Srgb;
  /// The stored value at the ramp's first color: the integer for a `_UNORM`
  /// format (0..255, 0..65535), the value itself for `R32_SFLOAT`.
  float ramp_min = 0.0f;
  /// The stored value at the ramp's last color; must differ from
  /// @ref ramp_min (a smaller value reverses the ramp).
  float ramp_max = 255.0f;
  /// Show a stored 0 as black rather than as the ramp's color for it, as
  /// "no data" in a depth map. An `R32_SFLOAT` NaN is shown black either way
  /// (and so is one shown through @ref ImageMapping::Grey).
  bool zero_is_empty = true;
  /// The Y'CbCr matrix's red weight: BT.601 0.299, BT.709 0.2126.
  float kr = 0.299f;
  /// The Y'CbCr matrix's blue weight: BT.601 0.114, BT.709 0.0722.
  float kb = 0.114f;
  /// Y in 0..255 rather than 16..235 (and chroma in 0..255 rather than
  /// 16..240): JPEG is full range, broadcast video limited.
  bool full_range = true;
  /// Where the chroma samples sit.
  ChromaSiting chroma_siting = ChromaSiting::Center;
};

class ImagePipeline;

/// @brief One image ready to draw: a source picture converted to display
///        color, with its full mip chain.
///
/// Holds a copy of each source plane (the image @ref ImagePipeline converts
/// from, so a source need only be readable by a transfer) and the display
/// image: `R8G8B8A8_SRGB`, with every mip level down to 1 x 1, which
/// @ref ImagePipeline::record_update rewrites and @ref ImagePipeline::submit
/// samples. Its descriptor sets are made once, so a texture updated every
/// frame allocates nothing per frame.
///
/// One update and the draws after it may be in flight at once: each update
/// records barriers that wait for every earlier draw of the texture on the
/// same queue, so one texture per stream suffices, with no ring per frame in
/// flight. A texture used from two queues needs the caller's own
/// synchronization.
///
/// A texture holds about 4/3 of four bytes per texel for its display image,
/// plus its source planes (4K NV12: 44 MB and 12 MB).
///
/// @warning The @p pipeline passed to @ref create must outlive the texture,
///          whose descriptor sets name the pipeline's samplers. The device
///          the @p allocator allocates on must outlive it too, and the GPU
///          must be done with it -- every frame that updated or drew it --
///          before it is destroyed: retire it through a @ref RetireQueue or
///          after `core::Device::wait_idle`.
///
/// @code
/// pipelines::ImageTextureDesc desc;
/// desc.extent = {3840, 2160};
/// desc.mapping = pipelines::ImageMapping::Nv12;
/// VKC_ASSIGN(pipelines::ImageTexture camera,
///            pipelines::ImageTexture::create(pipeline, allocator, desc));
/// // each new picture, before the frame's rendering scope:
/// VKC_TRY(pipeline.record_update(cmd, camera, update));
/// @endcode
// TODO: a ui-tier helper that registers display() with ImGui
// (ImGui_ImplVulkan_AddTexture), so a panel can show a texture.
class VG_PIPELINES_API ImageTexture {
 public:
  /// @brief Construct an empty texture (owns nothing; `valid()` is false).
  ImageTexture() = default;

  /// @brief Create a texture for sources @p desc describes, ready for
  ///        @p pipeline to update and draw.
  /// @param pipeline   The pipeline that will update and draw it.
  /// @param allocator  Allocates its images, device-only.
  /// @param desc       Its size and the source it converts.
  /// @return The texture, holding no picture until its first update; or
  ///         `core::Status::Code::InvalidArgument` for an empty @p pipeline,
  ///         a zero extent, an extent beyond the device's
  ///         `maxImageDimension2D`, `maxFramebufferWidth` /
  ///         `maxFramebufferHeight` or `maxViewportDimensions` (the update
  ///         renders the whole picture), or a format the mapping does not
  ///         take;
  ///         `core::Status::Code::Unsupported` when the device cannot sample
  ///         or copy into a plane's format; or a backend failure.
  static core::Result<ImageTexture> create(const ImagePipeline& pipeline,
                                           core::Allocator& allocator,
                                           const ImageTextureDesc& desc);

  ~ImageTexture() = default;
  ImageTexture(ImageTexture&& other) noexcept;
  ImageTexture& operator=(ImageTexture&& other) noexcept;
  ImageTexture(const ImageTexture&) = delete;
  ImageTexture& operator=(const ImageTexture&) = delete;

  /// @return The picture's size in texels (`{0, 0}` when empty).
  VkExtent2D extent() const noexcept { return desc_.extent; }
  /// @return How it converts its source.
  ImageMapping mapping() const noexcept { return desc_.mapping; }
  /// @return The display image's mip levels (0 when empty).
  uint32_t mip_levels() const noexcept {
    return display_.valid() ? display_.mip_levels() : 0;
  }
  /// @return The display image: `R8G8B8A8_SRGB`, every mip level, in
  ///         `SHADER_READ_ONLY_OPTIMAL` once updated. Each update waits only
  ///         for fragment-shader reads of it on the same queue before
  ///         overwriting it: a reader in another stage (a compute or vertex
  ///         shader) or on another queue must be ordered before the next
  ///         update by the caller.
  const core::Image& display() const noexcept { return display_; }
  /// @return Whether an update has been recorded -- recorded, not executed:
  ///         see @ref ImagePipeline::record_update -- so draws have a picture.
  bool has_picture() const noexcept { return has_picture_; }
  /// @return `true` if this owns a texture.
  bool valid() const noexcept { return display_.valid(); }

 private:
  friend class ImagePipeline;

  ImageTextureDesc desc_{};
  std::array<core::Image, 2> planes_{};  // the source planes' copies
  core::Image display_;
  // Level 0 alone, the convert pass's color attachment.
  core::UniqueHandle<VkImageView, vkDestroyImageView> level0_view_;
  core::DescriptorPool pool_;
  core::DescriptorSet convert_set_;  // the plane copies
  // The display image, magnified nearest (0) or linear (1).
  std::array<core::DescriptorSet, 2> draw_sets_{};
  bool has_picture_ = false;
};

/// @brief One image drawn into a rectangle of the target.
///
/// The image is placed by @ref origin and @ref scale -- target pixel =
/// @ref origin + image coordinate * @ref scale -- and clipped to
/// @ref viewport. Image coordinates put texel `(i, j)` over
/// `[i, i + 1) x [j, j + 1)`, its center at `(i + 0.5, j + 0.5)`; OpenCV's
/// convention, which puts centers on integers, is half a texel off it.
/// `camera::ImageView2D` keeps @ref origin and @ref scale for a pan-and-zoom
/// view. Target pixels are the render target's, so on a high-DPI display they
/// are framebuffer pixels, not window coordinates.
struct ImageDraw {
  /// The texture; a null, empty or never-updated texture draws nothing.
  const ImageTexture* texture = nullptr;
  /// The rectangle the image is drawn within, in target pixels.
  VkRect2D viewport{};
  /// Where image coordinate (0, 0) lands, in target pixels.
  glm::vec2 origin{0.0f, 0.0f};
  /// Target pixels per texel; positive.
  float scale = 1.0f;
  /// How the image is magnified when @ref scale exceeds 1.
  ImageFilter magnify = ImageFilter::Nearest;
};

/// @brief Everything @ref ImagePipeline::submit records for one frame.
///
/// The draw list is borrowed for the duration of the call; the textures it
/// names must outlive the frame's execution.
struct ImageFrame {
  VkExtent2D extent{};               ///< The target's size, in pixels.
  const ImageDraw* draws = nullptr;  ///< The draw list.
  uint32_t draw_count = 0;           ///< The number of @ref draws.
};

/// @brief The 2D image technique: converts source pictures into
///        @ref ImageTexture s and draws them, mip-mapped, into a render
///        target.
///
/// Two stages, each a pipeline of the library's own embedded GLSL:
/// 1. @ref record_update, outside a rendering scope: copies the source's
///    planes into the texture, converts them to display color in level 0 of
///    its `_SRGB` display image -- the sRGB decode, the Y'CbCr matrix or the
///    color ramp -- and builds the mip chain by halving blits, which average
///    in linear light because the image is `_SRGB`. Converting before
///    filtering is what keeps a shrunken image's colors right: every mapping
///    is nonlinear, so the average of converted texels is not the conversion
///    of averaged ones.
/// 2. @ref submit, inside the caller's rendering scope: draws each texture
///    into its rectangle. Its sampler minifies through the chain (trilinear),
///    the level chosen per pixel from the drawn size, so it follows window
///    resizes, zoom and display scaling by itself; it magnifies with the
///    draw's @ref ImageFilter.
///
/// A source updated every frame is updated and drawn in the same command
/// buffer; a still image is updated once and drawn every frame. Images draw
/// opaque, depth-untested, in draw-list order. A draw writes the image's own
/// sRGB bytes to a `_UNORM` target as to an `_SRGB` one, and linear light to
/// a float target, which holds it for a later tonemap or encode.
///
/// @warning The @p device passed to @ref create must outlive the pipeline
///          and every texture made for it.
///
/// @code
/// VKC_ASSIGN(pipelines::ImagePipeline images,
///            pipelines::ImagePipeline::create(device, target.layout()));
/// // per frame, for a live source:
/// VKC_TRY(images.record_update(cmd, camera, update));
/// target.begin(cmd, begin_info);
/// pipelines::ImageDraw draw;
/// draw.texture = &camera;
/// draw.viewport = {{0, 0}, extent};
/// draw.origin = view.origin();
/// draw.scale = view.scale();
/// images.submit(cmd, {extent, &draw, 1});
/// target.end(cmd);
/// @endcode
// TODO: blending -- a Color source's alpha, or a draw's opacity -- so an
// image can be laid over the scene; images draw opaque until then.
class VG_PIPELINES_API ImagePipeline {
 public:
  /// @brief Construct an empty pipeline (owns nothing; `valid()` is false).
  ImagePipeline() = default;

  /// @brief Build the pipelines for drawing into targets of @p layout.
  /// @param device  The device; it must have enabled the renderer's
  ///                requirements (@ref device_requirements).
  /// @param layout  The target's format/sample signature: exactly one color
  ///                attachment, single-sample. A depth format is allowed and
  ///                left untested.
  /// @return The pipeline, or `core::Status::Code::InvalidArgument` for a
  ///         layout with no color attachment or more than one, or with
  ///         multisampling;
  ///         `core::Status::Code::Unsupported` for a device without the
  ///         renderer's requirements or that cannot render, blit and filter
  ///         `R8G8B8A8_SRGB`; or a backend failure.
  static core::Result<ImagePipeline> create(const core::Device& device,
                                            const RenderTargetLayout& layout);

  ~ImagePipeline() = default;
  // TODO: an aggregate (DECISIONS.md, 2026-10-10): delete the move pair and
  // hand it out by std::unique_ptr, as windowing::FrameLoop is.
  ImagePipeline(ImagePipeline&& other) noexcept;
  ImagePipeline& operator=(ImagePipeline&& other) noexcept;
  ImagePipeline(const ImagePipeline&) = delete;
  ImagePipeline& operator=(const ImagePipeline&) = delete;

  /// @return `true` if this owns its pipelines.
  bool valid() const noexcept { return draw_.valid(); }

  /// @brief Record the copy of @p update's planes into @p texture, their
  ///        conversion to display color, and its mip chain.
  ///
  /// Records into @p cmd and submits nothing; draws recorded after it, in
  /// this or a later submission on the same queue, show the new picture. On
  /// an error nothing is recorded and the texture keeps its last picture.
  ///
  /// The texture takes the picture as recorded, so @p cmd must run before
  /// any later draw of the texture does: if it is reset or dropped instead,
  /// record another update before drawing the texture again.
  /// @param cmd      A command buffer in the recording state, outside a
  ///                 rendering scope, for a graphics queue of the device.
  /// @param texture  The texture to fill; made for this pipeline.
  /// @param update   The source's planes and how to read them; see
  ///                 @ref ImagePlane for what each plane must satisfy.
  /// @return OK once recorded; or `core::Status::Code::InvalidArgument` for an
  ///         empty pipeline or texture, a missing or extra plane, a plane
  ///         naming both or neither of an image and a buffer, an image of the
  ///         wrong format, size, usage or layout, a buffer too small or
  ///         misaligned for the plane, @ref ImageEncoding::Linear for an
  ///         `_SRGB` format, equal ramp bounds, or matrix weights outside
  ///         `kr > 0, kb > 0, kr + kb < 1`.
  core::Status record_update(VkCommandBuffer cmd, ImageTexture& texture,
                             const ImageUpdate& update) const;

  /// @brief Record @p frame's draws into @p cmd.
  /// @param cmd    A command buffer in the recording state, inside a
  ///               rendering scope whose target matches the layout
  ///               @ref create was given.
  /// @param frame  The target's size and the draws.
  /// @pre `valid()`; an empty pipeline records nothing. Draws without a
  ///      picture, with a non-positive scale, or clipped away entirely are
  ///      skipped.
  void submit(VkCommandBuffer cmd, const ImageFrame& frame) const;

 private:
  friend class ImageTexture;

  const core::Device* device_ = nullptr;
  GraphicsPipeline convert_;  // source planes -> display level 0
  GraphicsPipeline draw_;     // display image -> the caller's target
  std::optional<Sampler> texel_sampler_;    // the planes, read by texelFetch
  std::optional<Sampler> chroma_sampler_;   // NV12's chroma, bilinear
  std::optional<Sampler> nearest_sampler_;  // the display, magnified nearest
  std::optional<Sampler> linear_sampler_;   // the display, magnified linear
};

}  // namespace volumetric_kit::gfx::pipelines

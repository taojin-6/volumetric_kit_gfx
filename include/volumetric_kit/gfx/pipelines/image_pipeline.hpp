// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_pipeline.hpp
/// @brief 2D images on screen: a picture written in regions from source
///        images, converted to display color and mip-mapped on the GPU
///        (@ref ImageTexture, filled by @ref ImagePipeline::record_write and
///        @ref ImagePipeline::record_convert, or both at once by
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
#include "volumetric_kit/gfx/camera/image_view_2d.hpp"
#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/ycbcr.hpp"

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
  /// @ref ImageConversion::ramp_min and @ref ImageConversion::ramp_max:
  /// depth, error and coverage maps.
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
/// Fixed for the texture's life: a source of another mapping or format needs
/// another texture, and a source smaller than the picture is written into a
/// region of it (@ref ImageWrite).
struct ImageTextureDesc {
  /// The picture's size in texels (the luma's, for NV12); non-zero, and at
  /// most @ref ImagePipeline::max_texture_extent.
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
/// A write copies the plane's top-left corner -- its region's extent, or
/// half of it for NV12's chroma -- so an image may be larger than the region,
/// as a decoder's padded surface is. The copy reads it on the queue the write
/// is submitted to, so:
/// - its writes must come **before** the copy: recorded earlier on that
///   queue -- in the same command buffer or an earlier submission, which the
///   write's barrier waits for -- or done on another queue: a fence waited
///   on, or a semaphore the write's submission waits on at the `TRANSFER`
///   stage (under the frame loop, a `windowing::Frame::waits` entry whose
///   value is already reached before `end_frame`);
/// - it must be readable from that queue's family: created `CONCURRENT`
///   across the producer's and the renderer's families, or written on the
///   renderer's family (MoltenVK gives two libraries different families);
/// - and it must outlive the write's execution, not merely its recording.
///
/// Every command recorded on the queue after the write waits for the copy,
/// so a producer on that queue may write the next picture into the plane
/// with no barrier of its own.
struct ImagePlane {
  /// An image: 2D, single-sample, with `TRANSFER_SRC` usage, in the layout
  /// its `layout()` records -- `GENERAL` or `TRANSFER_SRC_OPTIMAL` -- and a
  /// format of the plane's texel size and channel order (`_UNORM` and
  /// `_SRGB` are interchangeable).
  const core::Image* image = nullptr;
  /// Or a buffer with `TRANSFER_SRC` usage, holding every copied texel of
  /// the plane's rows, including the offset and any row padding.
  const core::Buffer* buffer = nullptr;
  /// The buffer's byte offset of the first row: a multiple of the texel
  /// size.
  VkDeviceSize offset = 0;
  /// The buffer's row length in texels; 0 for tightly packed rows. A nonzero
  /// value must cover the region's width and fit in 2^31 - 1 bytes per row.
  uint32_t row_length = 0;
};

/// @brief How an @ref ImageTexture's planes become display color: what
///        @ref ImagePipeline::record_convert applies to the whole picture.
///
/// One conversion serves every region of the picture, so the regions share
/// one coding: one transfer function, ramp, Y'CbCr matrix, range and siting.
/// The @ref YcbcrCoding it derives from is @ref ImageMapping::Nv12 's.
///
/// @code
/// pipelines::ImageConversion mjpeg;  // BT.601, full range, centered chroma
/// mjpeg.kr = 0.299f;
/// mjpeg.kb = 0.114f;
/// mjpeg.chroma_siting = pipelines::ChromaSiting::Center;
/// @endcode
struct ImageConversion : YcbcrCoding {
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
};

/// @brief One update of an @ref ImageTexture: the whole picture's planes, and
///        how to convert them.
///
/// @code
/// pipelines::ImageUpdate update;  // a decoder's NV12 picture
/// update.planes[0].image = &luma;
/// update.planes[1].image = &chroma;
/// update.kr = 0.299f;  // BT.601, full range, as MJPEG
/// update.kb = 0.114f;
/// update.chroma_siting = pipelines::ChromaSiting::Center;
/// @endcode
struct ImageUpdate : ImageConversion {
  /// The planes: plane 0 always; plane 1 for @ref ImageMapping::Nv12 only.
  std::array<ImagePlane, 2> planes{};
};

/// @brief One region of an @ref ImageTexture's picture, written from a
///        source: each plane's top-left @ref extent texels (half as many for
///        NV12's chroma) copied to @ref offset.
///
/// The region lies inside the picture. For @ref ImageMapping::Nv12 its
/// offset is even, and so is its extent unless it reaches the picture's right
/// or bottom edge, so that each region's chroma is its own.
///
/// One picture is converted and filtered whole, which a layout of regions
/// should allow for:
/// - **Edge chroma.** NV12's chroma is upsampled bilinearly across the
///   picture, so a region's outermost luma texels take a share of the chroma
///   beyond its edge: a quarter in its top and bottom rows; a quarter in its
///   left and right columns with centered siting, and with left siting none
///   in the left and half in the right. A corner takes both shares (7/16
///   centered; 5/8 at a left-sited right corner). An edge on the picture's
///   edge takes none. Across a gutter no write covers, whose chroma is
///   neutral, the region's edge only desaturates, and the gutter texel beside
///   each edge takes the mirrored share of the region's chroma, a tint at
///   zero luma: a quarter beside its rows and, with centered siting, its
///   columns; with left siting half beside its left column and none beside
///   its right.
/// - **Mips.** When the picture's extent is a multiple of 2^k, each texel of
///   a level up to k is the average of a 2^k-texel square of level 0. So a
///   gutter of at least 2^k texels keeps two regions from mixing at any level
///   up to k. When a region's offset and extent are multiples of 2^k too, its
///   levels up to k are made of its own texels alone, as a texture of its own
///   would be, and differ from that texture's only where its level 0 does:
///   at NV12 edges.
///
/// @code
/// pipelines::ImageWrite tile;  // camera 1's picture, right of camera 0's
/// tile.planes[0].image = &luma;
/// tile.planes[1].image = &chroma;
/// tile.offset = {3840 + 16, 0};  // a 16-texel gutter
/// tile.extent = {3840, 2160};
/// @endcode
struct ImageWrite {
  /// The source's planes: plane 0 always; plane 1 for @ref ImageMapping::Nv12
  /// only. Each is checked as @ref ImagePlane says, against the region's
  /// extent.
  std::array<ImagePlane, 2> planes{};
  /// The region's top-left texel in the picture (the luma's, for NV12);
  /// non-negative.
  VkOffset2D offset{};
  /// The region's size in texels; non-zero.
  VkExtent2D extent{};
};

class ImagePipeline;

/// @brief One image ready to draw: a picture written in regions, converted to
///        display color, with its full mip chain.
///
/// Holds the picture's planes, which @ref ImagePipeline::record_write copies
/// sources into (so a source need only be readable by a transfer) and which
/// keep what was written until written again, and the display image:
/// `R8G8B8A8_SRGB`, with every mip level down to 1 x 1, which
/// @ref ImagePipeline::record_convert rebuilds from the planes and
/// @ref ImagePipeline::submit samples. Its descriptor sets are made once, so
/// a texture updated every frame allocates nothing per frame.
///
/// One update and the draws after it may be in flight at once: each write
/// and convert records barriers that wait for every earlier use of the
/// texture on the same queue, so one texture per stream suffices, with no
/// ring per frame in flight. A texture used from two queues needs the
/// caller's own synchronization.
///
/// A texture holds about 4/3 of four bytes per texel for its display image,
/// plus its planes (4K NV12: 44 MB and 12 MB).
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
/// // after the frame, if it may not have run:
/// if (!end.ok() && !windowing::swapchain_stale(end)) camera.discard();
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
  /// @return The texture, holding no picture until its first convert; or
  ///         `core::Status::Code::InvalidArgument` for an empty @p pipeline,
  ///         a zero extent, an extent beyond
  ///         @ref ImagePipeline::max_texture_extent (the message names the
  ///         limit), or a format the mapping does not take;
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
  ///         `SHADER_READ_ONLY_OPTIMAL` once converted. Each convert waits
  ///         only for fragment-shader reads of it on the same queue before
  ///         overwriting it: a reader in another stage (a compute or vertex
  ///         shader) or on another queue must be ordered before the next
  ///         convert by the caller.
  const core::Image& display() const noexcept { return display_; }
  /// @return Whether a convert has been recorded since @ref create or
  ///         @ref discard -- recorded, not executed: see
  ///         @ref ImagePipeline::record_convert -- so draws have a picture.
  bool has_picture() const noexcept { return has_picture_; }
  /// @return `true` if this owns a texture.
  bool valid() const noexcept { return display_.valid(); }

  /// @brief Forget what recorded writes and converts left, after a command
  ///        buffer that recorded them may not have run.
  ///
  /// The texture then shows no picture until the next
  /// @ref ImagePipeline::record_convert, which is refused until a
  /// @ref ImagePipeline::record_write, and that write first clears the planes
  /// as the first write after @ref create does. So the layouts and contents
  /// the texture records agree with the GPU's again, whether or not the
  /// command buffer ran. Call it when `windowing::FrameLoop::end_frame` fails
  /// and the swapchain is not stale, as for `StreamedAtlas::discard`; an
  /// empty texture ignores it.
  void discard() noexcept;

 private:
  friend class ImagePipeline;

  ImageTextureDesc desc_{};
  std::array<core::Image, 2> planes_{};  // the picture's planes
  core::Image display_;
  // Level 0 alone, the convert pass's color attachment.
  core::UniqueHandle<VkImageView, vkDestroyImageView> level0_view_;
  core::DescriptorPool pool_;
  core::DescriptorSet convert_set_;  // the planes
  // The display image, magnified nearest (0) or linear (1).
  std::array<core::DescriptorSet, 2> draw_sets_{};
  bool has_picture_ = false;
  // Whether the planes hold a write since create or discard (else their
  // contents and layout are undefined, and the next write clears them).
  bool written_ = false;
};

/// @brief One image drawn into a rectangle of the target.
///
/// The image is placed by @ref origin and @ref scale -- target pixel =
/// @ref origin + image coordinate * @ref scale -- and clipped to
/// @ref viewport. Image coordinates put texel `(i, j)` over
/// `[i, i + 1) x [j, j + 1)`, its center at `(i + 0.5, j + 0.5)`; OpenCV's
/// convention, which puts centers on integers, is half a texel off it.
/// `camera::ImageView2D` keeps @ref origin and @ref scale for a pan-and-zoom
/// view, and @ref image_draw fills a draw from one. Target pixels are the
/// render target's, so on a high-DPI display they are framebuffer pixels, not
/// window coordinates.
struct ImageDraw {
  /// The texture; a null or empty one, or one with no picture
  /// (`has_picture()`), draws nothing.
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

/// @brief The draw of @p texture through @p view: the view's viewport,
///        origin and scale.
///
/// The viewport's edges are rounded to whole target pixels and held inside
/// `[0, 2^31 - 1]` (NaN at 0), its right and bottom edges no lower than its
/// left and top: whatever the view holds, the rectangle is bounded, and
/// empty -- so nothing is drawn -- for a negative or NaN size.
/// @code
/// const pipelines::ImageDraw draw = pipelines::image_draw(texture, view);
/// images.submit(cmd, {extent, &draw, 1});
/// @endcode
/// @param texture  The texture; @p view should be of an image its size.
/// @param view     The pan-and-zoom view, in target pixels.
/// @param magnify  How the image is magnified.
/// @return The draw.
VG_PIPELINES_API ImageDraw
image_draw(const ImageTexture& texture, const camera::ImageView2D& view,
           ImageFilter magnify = ImageFilter::Nearest) noexcept;

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
/// Three stages, the first two outside a rendering scope; the convert and the
/// draw are pipelines of the library's own embedded GLSL:
/// 1. @ref record_write: copies sources' planes into regions of the
///    texture's planes -- one camera's picture or many, in one call.
/// 2. @ref record_convert: converts the whole picture to display color in
///    level 0 of its `_SRGB` display image -- the sRGB decode, the Y'CbCr
///    matrix or the color ramp -- and builds the mip chain by halving blits,
///    which average in linear light because the image is `_SRGB`. Converting
///    before filtering is what keeps a shrunken image's colors right: every
///    mapping is nonlinear, so the average of converted texels is not the
///    conversion of averaged ones.
/// 3. @ref submit, inside the caller's rendering scope: draws each texture
///    into its rectangle. Its sampler minifies through the chain (trilinear),
///    the level chosen per pixel from the drawn size, so it follows window
///    resizes, zoom and display scaling by itself; it magnifies with the
///    draw's @ref ImageFilter.
///
/// @ref record_update is a write of the whole picture and its convert. A
/// source updated every frame is updated and drawn in the same command
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
/// const pipelines::ImageDraw draw = pipelines::image_draw(camera, view);
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
  ///                attachment, at any sample count. A depth format is
  ///                allowed and left untested. Multisampled, an image's
  ///                interior draws as at one sample and its edges blend.
  /// @return The pipeline, or `core::Status::Code::InvalidArgument` for a
  ///         layout with no color attachment or more than one;
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

  /// @return The largest picture a texture made for this pipeline holds
  ///         (`{0, 0}` when empty): the device's `maxImageDimension2D`,
  ///         bounded by `maxFramebufferWidth` / `maxFramebufferHeight` and
  ///         `maxViewportDimensions`, as the convert renders the whole
  ///         picture. Regions that do not fit one texture are paged over
  ///         several.
  VkExtent2D max_texture_extent() const noexcept;

  /// @brief Record the copies of @p writes' sources into regions of
  ///        @p texture's planes.
  ///
  /// Every write is checked before anything is recorded: on an error nothing
  /// is recorded and the texture is unchanged. The display image is not
  /// touched: draws show the last converted picture until
  /// @ref record_convert. Texels no write covers keep what they held; the
  /// first write after @ref ImageTexture::create or
  /// @ref ImageTexture::discard first clears the planes to black (Y' 0 and
  /// CbCr 128, otherwise 0, which a ramp shows black only with
  /// @ref ImageConversion::zero_is_empty), unless the writes together cover
  /// the picture.
  ///
  /// The texture takes the write as recorded, so @p cmd must run before the
  /// texture's next write or convert does: if it is reset or dropped instead,
  /// call @ref ImageTexture::discard.
  ///
  /// The copies wait for every earlier write on the queue (see
  /// @ref ImagePlane) and for the previous convert's reads of the planes, and
  /// every later command waits for them.
  /// @param cmd          A command buffer in the recording state, outside a
  ///                     rendering scope, for a graphics queue of the device.
  /// @param texture      The texture to write; made for this pipeline.
  /// @param writes       The regions and their sources; any number, no two
  ///                     overlapping.
  /// @param write_count  The number of @p writes; non-zero.
  /// @return OK once recorded; or `core::Status::Code::InvalidArgument`,
  ///         naming the write, for an empty pipeline or texture, no writes, a
  ///         region of zero size or outside the picture, an NV12 region at an
  ///         odd offset or with an odd extent short of the picture's edge,
  ///         two regions that overlap, a missing or extra plane, a plane
  ///         naming both or neither of an image and a buffer, an image of the
  ///         wrong format, usage or layout or smaller than the region, or a
  ///         buffer too small or misaligned for it.
  core::Status record_write(VkCommandBuffer cmd, ImageTexture& texture,
                            const ImageWrite* writes,
                            std::uint32_t write_count) const;

  /// @brief Record the conversion of @p texture's picture to display color,
  ///        and its mip chain.
  ///
  /// Records into @p cmd and submits nothing; draws recorded after it, in
  /// this or a later submission on the same queue, show the new picture. On
  /// an error nothing is recorded and the texture keeps its last picture.
  ///
  /// The texture takes the picture as recorded, so @p cmd must run before
  /// any later draw of the texture does: if it is reset or dropped instead,
  /// call @ref ImageTexture::discard.
  /// @param cmd         A command buffer in the recording state, outside a
  ///                    rendering scope, for a graphics queue of the device.
  /// @param texture     The texture to convert; made for this pipeline.
  /// @param conversion  How its planes become display color.
  /// @return OK once recorded; or `core::Status::Code::InvalidArgument` for an
  ///         empty pipeline or texture, a texture not written since
  ///         @ref ImageTexture::create or @ref ImageTexture::discard,
  ///         @ref ImageEncoding::Linear for an `_SRGB` format, ramp bounds
  ///         that are equal or not finite, or, for NV12, matrix weights
  ///         outside `kr > 0, kb > 0, kr + kb < 1` or a siting
  ///         @ref ChromaSiting does not name.
  core::Status record_convert(VkCommandBuffer cmd, ImageTexture& texture,
                              const ImageConversion& conversion) const;

  /// @brief Record a write of @p update's planes over the whole of
  ///        @p texture's picture, and its convert.
  ///
  /// The same as @ref record_write of one region at `{0, 0}` the picture's
  /// size, then @ref record_convert, except that both are checked before
  /// either is recorded.
  /// @param cmd      As @ref record_write.
  /// @param texture  The texture to fill; made for this pipeline.
  /// @param update   The source's planes, and how to convert them.
  /// @return OK once recorded; or what @ref record_write and
  ///         @ref record_convert refuse, with nothing recorded.
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

  // Record a write or a convert that has been checked.
  void write_checked(VkCommandBuffer cmd, ImageTexture& texture,
                     const ImageWrite* writes, std::uint32_t write_count,
                     bool covers) const;
  void convert_checked(VkCommandBuffer cmd, ImageTexture& texture,
                       const ImageConversion& conversion) const;

  const core::Device* device_ = nullptr;
  GraphicsPipeline convert_;  // source planes -> display level 0
  GraphicsPipeline draw_;     // display image -> the caller's target
  std::optional<Sampler> texel_sampler_;    // the planes, read by texelFetch
  std::optional<Sampler> chroma_sampler_;   // NV12's chroma, bilinear
  std::optional<Sampler> nearest_sampler_;  // the display, magnified nearest
  std::optional<Sampler> linear_sampler_;   // the display, magnified linear
};

}  // namespace volumetric_kit::gfx::pipelines

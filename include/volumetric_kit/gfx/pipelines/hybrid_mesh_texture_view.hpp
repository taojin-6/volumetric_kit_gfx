// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file hybrid_mesh_texture_view.hpp
/// @brief A @ref HybridMeshPipeline set-0 texture drawn flat into a rectangle
///        of a render target -- the pictures a mesh samples, shown as
///        pictures -- and the pyramid of their levels that draws them small.

#include <array>
#include <cstdint>
#include <memory>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

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
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_set.hpp"

namespace volumetric_kit::core {
class Allocator;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

class HybridMeshPipeline;
class PicturePyramid;

/// @brief One picture of a set-0 texture drawn into a rectangle of the
///        target: @ref HybridMeshTextureView::submit takes a list of them.
///
/// The part of the texture @ref uv names is placed by @ref origin and
/// @ref scale -- target pixel = @ref origin + part pixel * @ref scale, as
/// `camera::ImageView2D` maps an image -- and clipped to @ref viewport. Its
/// footprint, the picture pixels under one target pixel, is 1 / @ref scale.
struct TextureViewDraw {
  /// The texture: @ref HybridMeshTextureSet::set, or a `StreamedAtlas` set,
  /// made for the pipeline the view was made over. Null draws nothing.
  VkDescriptorSet set = VK_NULL_HANDLE;
  /// Levels 1 and up of the set's pictures, built from them since they last
  /// changed (@ref PicturePyramid::record). Null, or not yet built, a draw
  /// smaller than 1/4 reads the picture alone, 5 x 5 of its pixels at most,
  /// and aliases.
  const PicturePyramid* pyramid = nullptr;
  /// The part shown, as `uv0` addresses the set: u0, v0, u1, v1, picture `p`
  /// over `v` in `[p, p + 1)`. The part lies in one picture, `floor(v0)`;
  /// one beyond the set's pictures shows its last.
  glm::vec4 uv{0.0f, 0.0f, 1.0f, 1.0f};
  /// The part's size in picture pixels; positive.
  glm::vec2 size{};
  /// The rectangle the part is drawn within, in target pixels.
  VkRect2D viewport{};
  /// Where the part's top-left lands, in target pixels.
  glm::vec2 origin{};
  /// Target pixels per picture pixel; positive.
  float scale = 1.0f;
};

/// @brief The draw of @p uv of @p set, placed at @p at in @p view's image,
///        through @p view's viewport and scale.
///
/// The viewport is rounded as `image_draw` rounds it.
/// @code
/// // Picture 1 of a set over two, the second of two stacked down the view:
/// const pipelines::TextureViewDraw draw = pipelines::texture_view_draw(
///     set, pyramid.get(), view, {0.0f, height + 16.0f}, {width, height},
///     {0.0f, 1.0f, 1.0f, 2.0f});
/// @endcode
/// @param set      The texture (@ref TextureViewDraw::set).
/// @param pyramid  Its pyramid, or null (@ref TextureViewDraw::pyramid).
/// @param view     The pan-and-zoom view, in target pixels.
/// @param at       The part's top-left in @p view's image coordinates.
/// @param size     The part's size in picture pixels, which @p view's image
///                 holds at that scale.
/// @param uv       The part (@ref TextureViewDraw::uv).
/// @return The draw.
VG_PIPELINES_API TextureViewDraw
texture_view_draw(VkDescriptorSet set, const PicturePyramid* pyramid,
                  const camera::ImageView2D& view, glm::vec2 at, glm::vec2 size,
                  glm::vec4 uv = {0.0f, 0.0f, 1.0f, 1.0f}) noexcept;

/// @brief Draws a @ref HybridMeshPipeline set-0 texture flat into rectangles
///        of a render target: what the mesh samples, shown as pictures.
///
/// Reads the set where it is -- an RGBA image, or a Y'CbCr picture's planes
/// -- through the mesh shader's own sampling, so it shows what the mesh
/// shows. Its pipeline layout's set 0 is the mesh pipeline's, so a frame's
/// own set binds as it is. The filter follows each draw's footprint F:
/// - **Magnified** (F at most 1), each target pixel shows the picture pixel
///   it lies over -- its luma, and its chroma as the mesh samples it there --
///   so a picture pixel is a sharp square.
/// - **Minified**, each target pixel is the mean, in linear light, of the
///   pixels of one level under its footprint, each weighted by its area
///   inside: the picture while F is at most 4, then the @ref PicturePyramid
///   level `L = ceil(log2(F / 4))`, where the footprint spans at most 4 of
///   its pixels. Each pixel is read at its centre, where a picture's
///   conversion is exact. On level 0, and on a level whose pixels the
///   footprint's edges fall between (1/2, 1/4, 1/8 and 1/16 at a whole
///   origin), that is the exact mean of the picture's pixels under it.
///
/// Draws are opaque and untested against depth, in list order; every read is
/// held to its picture. A `_UNORM` target is given sRGB-encoded colour, an
/// `_SRGB` or float one linear light.
///
/// @warning The @p device and @p pipeline passed to @ref create must outlive
///          the view, and the view every @ref PicturePyramid made for it.
///
/// @code
/// VKC_ASSIGN(std::unique_ptr<pipelines::HybridMeshTextureView> map,
///            pipelines::HybridMeshTextureView::create(device, mesh_pipeline,
///                                                     target.layout()));
/// // inside the frame's rendering scope, after the mesh:
/// const pipelines::TextureViewDraw draw = pipelines::texture_view_draw(
///     texture->set(), pyramid.get(), view, {0.0f, 0.0f}, picture_size);
/// map->submit(cmd, extent, &draw, 1);
/// @endcode
class VG_PIPELINES_API HybridMeshTextureView {
 public:
  /// @brief A view over @p pipeline's set-0 layout, for targets of
  ///        @p layout.
  /// @param device    The device @p pipeline was created on.
  /// @param pipeline  The mesh pipeline whose sets the view draws.
  /// @param layout    The target's format/sample signature: one color
  ///                  attachment, single-sample; a depth format is allowed
  ///                  and left untested.
  /// @return The view; or `core::Status::Code::InvalidArgument` for an empty
  ///         @p pipeline or a layout without exactly one color attachment or
  ///         with multisampling; `core::Status::Code::Unsupported` for a
  ///         device that cannot render, blit and filter `R8G8B8A8_SRGB`, a
  ///         pyramid's format; or a backend failure.
  static core::Result<std::unique_ptr<HybridMeshTextureView>> create(
      const core::Device& device, const HybridMeshPipeline& pipeline,
      const RenderTargetLayout& layout);

  ~HybridMeshTextureView();
  HybridMeshTextureView(const HybridMeshTextureView&) = delete;
  HybridMeshTextureView& operator=(const HybridMeshTextureView&) = delete;
  HybridMeshTextureView(HybridMeshTextureView&&) = delete;
  HybridMeshTextureView& operator=(HybridMeshTextureView&&) = delete;

  /// @brief Record @p count draws into @p cmd.
  ///
  /// A draw with a null set, a size or scale that is not positive and
  /// finite, or a rectangle that misses its viewport or the target records
  /// nothing.
  /// @param cmd     A command buffer in the recording state, inside a
  ///                rendering scope whose target matches the layout
  ///                @ref create was given.
  /// @param extent  The target's size, in pixels.
  /// @param draws   The draws; borrowed for the call, while the sets, their
  ///                pictures and the pyramids they name must outlive the
  ///                frame's execution.
  /// @param count   The number of @p draws.
  void submit(VkCommandBuffer cmd, VkExtent2D extent,
              const TextureViewDraw* draws, uint32_t count) const;

 private:
  friend class PicturePyramid;

  HybridMeshTextureView(const core::Device& device, GraphicsPipeline draw,
                        GraphicsPipeline build, Sampler sampler,
                        core::DescriptorPool pool,
                        core::DescriptorSet placeholder) noexcept;

  // Records `pyramid`'s level 1 from `set`'s pictures, each into its
  // image's level 0.
  void record_level1(VkCommandBuffer cmd, VkDescriptorSet set,
                     const PicturePyramid& pyramid) const;

  const core::Device& device_;
  GraphicsPipeline draw_;   // into the caller's targets
  GraphicsPipeline build_;  // into a pyramid's level 1
  Sampler sampler_;         // a pyramid's: bilinear, clamped
  core::DescriptorPool pool_;
  // Set 1 without a pyramid: the mesh pipeline's fallback texel at both
  // pictures; layouts alike in draw_ and build_.
  core::DescriptorSet placeholder_;
};

/// @brief Levels 1 and up of a set-0 texture's pictures, for a
///        @ref HybridMeshTextureView to draw them small.
///
/// One `R8G8B8A8_SRGB` image a picture, of the picture's level 1 -- half its
/// size, floored -- with every level down to 1 x 1. @ref record rebuilds
/// them from a set: level 1 is the mean, in linear light, of each 2 x 2 of
/// the picture's pixels, as the view samples them; each level after it,
/// halving linear blits of the one above (`cmd_generate_mips`). It holds
/// about 4/3 of a byte a picture pixel: 11 MB for one 3840 x 2160 picture.
///
/// Rebuild it when the shown pictures change, and only then: a still picture
/// costs nothing a frame. One pyramid serves frames in flight: @ref record
/// waits for earlier draws that read it on the same queue.
///
/// @warning The view passed to @ref create must outlive the pyramid, and the
///          GPU must be done with every frame that recorded or drew it before
///          it is destroyed: retire it through a @ref RetireQueue or after
///          `core::Device::wait_idle`.
///
/// @code
/// VKC_ASSIGN(std::unique_ptr<pipelines::PicturePyramid> pyramid,
///            pipelines::PicturePyramid::create(*map, allocator, {3840, 2160},
///                                              1));
/// // before the frame's rendering scope, when the picture has changed:
/// pyramid->record(cmd, texture->set());
/// // if that frame may not have run:
/// pyramid->discard();
/// @endcode
class VG_PIPELINES_API PicturePyramid {
 public:
  /// @brief Images for @p count pictures of @p size.
  /// @param view       The view that builds and draws the pyramid.
  /// @param allocator  Allocates the images, device-only; it may be
  ///                   destroyed before the pyramid.
  /// @param size       Each picture's size in pixels; non-zero.
  /// @param count      From 1 to @ref HybridMeshTextureSet::kMaxPictures.
  /// @return The pyramid, holding nothing until its first @ref record; or
  ///         `core::Status::Code::InvalidArgument` for a zero @p size, a
  ///         @p count outside 1 to @ref HybridMeshTextureSet::kMaxPictures,
  ///         or a level 1 beyond the device's image, framebuffer or viewport
  ///         limits; or a backend failure.
  static core::Result<std::unique_ptr<PicturePyramid>> create(
      const HybridMeshTextureView& view, core::Allocator& allocator,
      VkExtent2D size, uint32_t count);

  ~PicturePyramid();
  PicturePyramid(const PicturePyramid&) = delete;
  PicturePyramid& operator=(const PicturePyramid&) = delete;
  PicturePyramid(PicturePyramid&&) = delete;
  PicturePyramid& operator=(PicturePyramid&&) = delete;

  /// @brief Record the rebuild of every level from @p set's pictures:
  ///        picture `p` from the set's picture `p`, at `uv` `(0, p)` to
  ///        `(1, p + 1)`.
  ///
  /// The pyramid takes the build as recorded, so @p cmd must run before a
  /// draw that reads it does: if it is reset or dropped instead, call
  /// @ref discard.
  /// @param cmd  A command buffer in the recording state, outside a rendering
  ///             scope, for a graphics queue of the device, where the set's
  ///             pictures are ready for fragment-shader reads, as for a mesh
  ///             draw.
  /// @param set  The texture, as @ref TextureViewDraw::set; null records
  ///             nothing and leaves the pyramid as it was.
  void record(VkCommandBuffer cmd, VkDescriptorSet set);

  /// @brief Forget the recorded build, after a command buffer that recorded
  ///        it may not have run; draws then read the picture alone until the
  ///        next @ref record.
  void discard() noexcept;

  /// @return Each picture's size, as @ref create took it.
  VkExtent2D size() const noexcept { return size_; }
  /// @return The number of pictures.
  uint32_t count() const noexcept { return count_; }
  /// @return The levels held: levels 1 to `levels()` of each picture.
  uint32_t levels() const noexcept;
  /// @return Whether a build has been recorded since @ref create or
  ///         @ref discard -- recorded, not executed.
  bool built() const noexcept { return built_; }

 private:
  friend class HybridMeshTextureView;

  // One picture's levels, and level 1 alone, the build's color attachment.
  struct Level {
    core::Image image;
    core::UniqueHandle<VkImageView, vkDestroyImageView> level1;
  };

  PicturePyramid(const HybridMeshTextureView& view, VkExtent2D size,
                 uint32_t count) noexcept;

  const HybridMeshTextureView& view_;
  VkExtent2D size_;
  uint32_t count_;
  std::array<Level, HybridMeshTextureSet::kMaxPictures> levels_{};
  core::DescriptorPool pool_;
  core::DescriptorSet set_;  // set 1: the levels, the placeholder past count_
  bool built_ = false;
};

}  // namespace volumetric_kit::gfx::pipelines

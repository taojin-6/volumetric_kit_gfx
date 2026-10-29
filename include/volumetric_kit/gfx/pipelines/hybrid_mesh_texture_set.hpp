// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file hybrid_mesh_texture_set.hpp
/// @brief A @ref HybridMeshPipeline's set-0 descriptor set over pictures the
///        caller owns -- an RGBA image, or one or two Y'CbCr 4:2:0 pictures'
///        planes -- with its own descriptor pool, sampler and uniform.

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/ycbcr.hpp"

namespace volumetric_kit::core {
class Allocator;
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

class HybridMeshPipeline;

/// @brief How a @ref HybridMeshTextureSet's sampler reads the picture:
///        `HybridMeshTextureSet::Options`.
struct HybridMeshTextureSetOptions {
  /// Magnification and minification filter, and the blend between mip
  /// levels: `VK_FILTER_NEAREST` or `VK_FILTER_LINEAR`. A Y'CbCr picture's
  /// two planes both take it; with `NEAREST`, left-sited chroma lies midway
  /// between two samples at every other column, and a device may read either.
  VkFilter filter = VK_FILTER_LINEAR;
  /// Addressing on every axis: `REPEAT`, `MIRRORED_REPEAT`, `CLAMP_TO_EDGE`
  /// or `CLAMP_TO_BORDER` (opaque black). A set that binds a Y'CbCr picture
  /// takes `CLAMP_TO_EDGE` alone.
  VkSamplerAddressMode address = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  /// The highest mip level sampled; 0 samples the base level alone, and
  /// `VK_LOD_CLAMP_NONE` the whole chain. A Y'CbCr picture's planes are
  /// sampled at level 0 whatever it says.
  float max_lod = 0.0f;
};

/// @brief A @ref HybridMeshPipeline's set-0 descriptor set over pictures the
///        caller owns, with its own descriptor pool, sampler and uniform.
///
/// The set binds either of two forms, and records which in its uniform, so
/// one pipeline draws both:
/// - **RGBA**: an image's view, sampled as @ref Options says; an `_SRGB`
///   format decodes to linear light as it is sampled.
/// - **Y'CbCr 4:2:0** (@ref YcbcrPicture): one picture's two planes, or up to
///   @ref kMaxPictures pictures', each sampled where it is, with no copy and
///   no RGBA image. Each plane is filtered at level 0, clamped to its
///   picture; the fragment shader then converts with that picture's
///   @ref YcbcrCoding and decodes the R'G'B' as sRGB.
///
/// The pipeline samples at each fragment's `uv0`. One picture, or an RGBA
/// image, spans `uv0` from 0 to 1, and is sampled at `uv0` as it is. Over two
/// pictures, `uv0.y`'s integer part is the picture, so picture `p`'s point
/// `(x, y)`, in texels, is at `uv0 = (x / width, p + y / height)`. A triangle
/// takes its provoking vertex's picture `p`, as it takes its albedo class,
/// and samples it at `(uv0.x, uv0.y - p)`, held to its edge: where the
/// triangle's `uv0.y` leaves `[p, p + 1)`, it draws picture `p`'s edge. A
/// provoking `uv0.y` outside the pictures takes the nearest: picture 0 below
/// 0, the last picture from the count up. Each picture has its own extent and
/// coding. @ref rebind points the same set at other pictures, or at an image.
///
/// A Y'CbCr picture's coding is a uniform the host writes in place, in
/// device-local memory the host maps
/// (`PhysicalDeviceInfo::device_mapped_memory`), which unified memory and
/// most discrete GPUs have. On a device without it a set binds RGBA images
/// alone, and a picture is `Unsupported`.
///
/// **The picture's lifetime is the caller's.** The set names the views, so the
/// image or the planes stay alive, and in the layout the set was given,
/// until the last frame that sampled them has retired: keep one set per slot
/// of a ring, each holding its picture until its slot comes round and it is
/// rebound, or push the replaced picture to a @ref RetireQueue at the last
/// such frame's number.
///
/// @warning The @p device and @p pipeline passed to @ref create must outlive
///          the set, and the GPU must be done with every frame that bound it
///          before it is destroyed: retire it through a @ref RetireQueue or
///          after `core::Device::wait_idle`.
///
/// @code
/// VKC_ASSIGN(std::unique_ptr<pipelines::HybridMeshTextureSet> texture,
///            pipelines::HybridMeshTextureSet::create(device, allocator,
///                                                    pipeline, picture));
/// // each frame, inside the rendering scope, with texture->set() in the
/// // frame's set-0 field:
/// pipeline.submit(cmd, frame);
/// // once no frame that bound the set is pending, two pictures:
/// const pipelines::YcbcrPicture both[] = {picture, other_picture};
/// VKC_TRY(texture->rebind(both, 2));
/// @endcode
class VG_PIPELINES_API HybridMeshTextureSet {
 public:
  /// @brief How the pipeline samples the picture. Declared outside the class,
  ///        so its defaults are complete where @ref create takes it.
  using Options = HybridMeshTextureSetOptions;

  /// The most Y'CbCr pictures a set binds at once.
  static constexpr uint32_t kMaxPictures = 2;

  /// @brief Build the set over the RGBA image @p view, with its own pool and
  ///        sampler.
  /// @param device     The device @p pipeline was created on.
  /// @param allocator  Allocates the uniform a later Y'CbCr picture takes,
  ///                   where the device has the memory; it may be destroyed
  ///                   before the set.
  /// @param pipeline   The pipeline that draws with the set; its set-0 layout
  ///                   is allocated against.
  /// @param view       A 2D view of the caller's image, made with `SAMPLED`
  ///                   usage, in `SHADER_READ_ONLY_OPTIMAL` whenever a frame
  ///                   that binds the set runs, of a format the device can
  ///                   sample with the filter in @ref Options.
  /// @param options    The sampler's filter, addressing and mip range.
  /// @return The set; or `core::Status::Code::InvalidArgument` for an empty
  ///         @p pipeline, a null @p view, a filter or address mode other than
  ///         those @ref Options names, or a `max_lod` below 0 or NaN; or a
  ///         backend failure. Host allocation failures return `OutOfMemory`,
  ///         including diagnostic construction; a failed create owns no set.
  static core::Result<std::unique_ptr<HybridMeshTextureSet>> create(
      const core::Device& device, core::Allocator& allocator,
      const HybridMeshPipeline& pipeline, VkImageView view,
      const Options& options = {});

  /// @brief Build the set over the Y'CbCr 4:2:0 @p picture, with its own pool,
  ///        sampler and uniform.
  /// @param device     As the RGBA @ref create.
  /// @param allocator  As the RGBA @ref create.
  /// @param pipeline   As the RGBA @ref create.
  /// @param picture    The planes, their extent and their coding, as
  ///                   @ref YcbcrPicture describes them.
  /// @param options    The sampler's filter; its address mode is
  ///                   `CLAMP_TO_EDGE`.
  /// @return The set; or what the RGBA @ref create refuses, a picture
  ///         @ref rebind refuses, or `core::Status::Code::InvalidArgument`
  ///         for an address mode other than `CLAMP_TO_EDGE`.
  static core::Result<std::unique_ptr<HybridMeshTextureSet>> create(
      const core::Device& device, core::Allocator& allocator,
      const HybridMeshPipeline& pipeline, const YcbcrPicture& picture,
      const Options& options = {});

  /// @brief Build the set over @p count Y'CbCr 4:2:0 pictures, picture `p`
  ///        at `pictures[p]`, with its own pool, sampler and uniform.
  /// @param device     As the RGBA @ref create.
  /// @param allocator  As the RGBA @ref create.
  /// @param pipeline   As the RGBA @ref create.
  /// @param pictures   @p count pictures, each as @ref YcbcrPicture describes
  ///                   it; read during the call only.
  /// @param count      From 1 to @ref kMaxPictures.
  /// @param options    As the one-picture @ref create.
  /// @return The set; or what the one-picture @ref create refuses, for any
  ///         picture, naming it; or `core::Status::Code::InvalidArgument`
  ///         for a null @p pictures or a @p count outside 1 to
  ///         @ref kMaxPictures.
  static core::Result<std::unique_ptr<HybridMeshTextureSet>> create(
      const core::Device& device, core::Allocator& allocator,
      const HybridMeshPipeline& pipeline, const YcbcrPicture* pictures,
      uint32_t count, const Options& options = {});

  ~HybridMeshTextureSet();
  HybridMeshTextureSet(const HybridMeshTextureSet&) = delete;
  HybridMeshTextureSet& operator=(const HybridMeshTextureSet&) = delete;
  HybridMeshTextureSet(HybridMeshTextureSet&&) = delete;
  HybridMeshTextureSet& operator=(HybridMeshTextureSet&&) = delete;

  /// @return The set, for @ref HybridMeshFrame's set-0 field; the same set
  ///         across @ref rebind.
  VkDescriptorSet set() const noexcept { return set_.handle(); }

  /// @brief Points the set at the RGBA image @p view, after the caller
  ///        recreates its image or in place of a Y'CbCr picture.
  ///
  /// Rewrites the set in place, so no frame that bound it may still be
  /// pending -- as for destroying the old picture, the GPU is done with every
  /// such frame -- and no command buffer being recorded may have bound it.
  /// @param view  The new image's view, as @ref create takes it; non-null,
  ///              which the write checks with `VKC_CHECK`.
  void rebind(VkImageView view);

  /// @brief Points the set at the Y'CbCr 4:2:0 @p picture, in place of the
  ///        pictures or the RGBA image it holds.
  ///
  /// Rewrites the set and its uniform in place, under the RGBA
  /// @ref rebind 's conditions. On an error the set is unchanged.
  /// @param picture  The planes, their extent and their coding.
  /// @return OK once rewritten; `core::Status::Code::InvalidArgument` for a
  ///         set whose address mode is not `CLAMP_TO_EDGE`; a missing plane;
  ///         a plane that is not a single-sample, one-layer 2D image with
  ///         `SAMPLED` usage and a view; a luma not `R8_UNORM` or a chroma not
  ///         `R8G8_UNORM`; a plane in a layout other than `GENERAL` or
  ///         `SHADER_READ_ONLY_OPTIMAL`; a zero extent, a luma smaller than
  ///         it or a chroma smaller than half of it, rounded up; or matrix
  ///         weights outside `kr > 0, kb > 0, kr + kb < 1` or a siting
  ///         @ref ChromaSiting does not name. `core::Status::Code::Unsupported`
  ///         for a plane the device cannot sample, with the set's filter, in
  ///         its tiling, or on a device without host-mapped device memory.
  ///         Host allocation failures, including diagnostic construction,
  ///         return `OutOfMemory`.
  core::Status rebind(const YcbcrPicture& picture);

  /// @brief Points the set at @p count Y'CbCr 4:2:0 pictures, picture `p` at
  ///        `pictures[p]`, in place of the pictures or the RGBA image it
  ///        holds.
  ///
  /// As the one-picture @ref rebind, for each picture. On an error the set is
  /// unchanged.
  /// @param pictures  @p count pictures; read during the call only.
  /// @param count     From 1 to @ref kMaxPictures.
  /// @return What the one-picture @ref rebind returns, its refusal naming the
  ///         picture; or `core::Status::Code::InvalidArgument` for a null
  ///         @p pictures or a @p count outside 1 to @ref kMaxPictures.
  core::Status rebind(const YcbcrPicture* pictures, uint32_t count);

 private:
  HybridMeshTextureSet(const core::Device& device, const Options& options,
                       VkImageView placeholder, VkBuffer rgba_uniform,
                       Sampler sampler, core::Buffer uniform,
                       core::DescriptorPool pool,
                       core::DescriptorSet set) noexcept;

  // Builds a set of either form, bound to nothing yet; `call` names the
  // caller in a refusal.
  static core::Result<std::unique_ptr<HybridMeshTextureSet>> make(
      const char* call, const core::Device& device, core::Allocator& allocator,
      const HybridMeshPipeline& pipeline, const Options& options);

  const core::Device& device_;
  Options options_;
  // The pipeline's fallback texel, at the planes the set's form does not
  // read: an RGBA image's chroma and picture 1, and one picture's picture 1.
  VkImageView placeholder_;
  // The pipeline's uniform saying "RGBA", which an RGBA picture binds.
  VkBuffer rgba_uniform_;
  Sampler sampler_;
  // A Y'CbCr picture's form and coding, mapped; empty on a device without
  // host-mapped device memory.
  core::Buffer uniform_;
  core::DescriptorPool pool_;  // frees set_
  core::DescriptorSet set_;
};

}  // namespace volumetric_kit::gfx::pipelines

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file hybrid_mesh_texture_set.hpp
/// @brief A @ref HybridMeshPipeline's set-0 descriptor set over an image view
///        the caller owns, with its own descriptor pool and sampler.

#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"

namespace volumetric_kit::core {
class Device;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

class HybridMeshPipeline;

/// @brief How a @ref HybridMeshTextureSet's sampler reads the image:
///        `HybridMeshTextureSet::Options`.
struct HybridMeshTextureSetOptions {
  /// Magnification and minification filter, and the blend between mip
  /// levels: `VK_FILTER_NEAREST` or `VK_FILTER_LINEAR`.
  VkFilter filter = VK_FILTER_LINEAR;
  /// Addressing on every axis: `REPEAT`, `MIRRORED_REPEAT`, `CLAMP_TO_EDGE`
  /// or `CLAMP_TO_BORDER` (opaque black).
  VkSamplerAddressMode address = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  /// The highest mip level sampled; 0 samples the base level alone, and
  /// `VK_LOD_CLAMP_NONE` the whole chain.
  float max_lod = 0.0f;
};

/// @brief A @ref HybridMeshPipeline's set-0 descriptor set over an image view
///        the caller owns, with its own descriptor pool and sampler.
///
/// The caller makes, writes and destroys the image; the set names its view,
/// which the pipeline samples at each fragment's `uv0`. The image is in
/// `SHADER_READ_ONLY_OPTIMAL` whenever a frame that binds the set runs, and
/// its format can be sampled with the filter set in @ref Options. When the
/// caller recreates its image, @ref rebind points the same set at the new
/// view.
///
/// @warning The @p device and @p pipeline passed to @ref create must outlive
///          the set, and the GPU must be done with every frame that bound it
///          before it is destroyed: retire it through a @ref RetireQueue or
///          after `core::Device::wait_idle`. The view must stay alive while
///          a frame that binds the set runs.
///
/// @code
/// VKC_ASSIGN(std::unique_ptr<pipelines::HybridMeshTextureSet> texture,
///            pipelines::HybridMeshTextureSet::create(device, pipeline,
///                                                    image.view()));
/// // each frame, inside the rendering scope, with texture->set() in the
/// // frame's set-0 field:
/// pipeline.submit(cmd, frame);
/// // once the caller has recreated its image and no frame that bound the
/// // set is pending:
/// texture->rebind(image.view());
/// @endcode
class VG_PIPELINES_API HybridMeshTextureSet {
 public:
  /// @brief How the pipeline samples the image. Declared outside the class,
  ///        so its defaults are complete where @ref create takes it.
  using Options = HybridMeshTextureSetOptions;

  /// @brief Build the set over @p view, with its own pool and sampler.
  /// @param device    The device @p pipeline was created on.
  /// @param pipeline  The pipeline that draws with the set; its set-0 layout
  ///                  is allocated against.
  /// @param view      A 2D view of the caller's image, made with
  ///                  `SAMPLED` usage.
  /// @param options   The sampler's filter, addressing and mip range.
  /// @return The set; or `core::Status::Code::InvalidArgument` for an empty
  ///         @p pipeline, a null @p view, a filter or address mode other than
  ///         those @ref Options names, or a `max_lod` below 0 or NaN; or a
  ///         backend failure.
  static core::Result<std::unique_ptr<HybridMeshTextureSet>> create(
      const core::Device& device, const HybridMeshPipeline& pipeline,
      VkImageView view, const Options& options = {});

  ~HybridMeshTextureSet();
  HybridMeshTextureSet(const HybridMeshTextureSet&) = delete;
  HybridMeshTextureSet& operator=(const HybridMeshTextureSet&) = delete;
  HybridMeshTextureSet(HybridMeshTextureSet&&) = delete;
  HybridMeshTextureSet& operator=(HybridMeshTextureSet&&) = delete;

  /// @return The set, for @ref HybridMeshFrame's set-0 field; the same set
  ///         across @ref rebind.
  VkDescriptorSet set() const noexcept { return set_.handle(); }

  /// @brief Points the set at @p view, after the caller recreates its image.
  ///
  /// Rewrites the set in place, so no frame that bound it may still be
  /// pending -- as for destroying the old image, the GPU is done with every
  /// such frame -- and no command buffer being recorded may have bound it.
  /// @param view  The new image's view, as @ref create takes it; non-null,
  ///              which the write checks with `VKC_CHECK`.
  void rebind(VkImageView view);

 private:
  HybridMeshTextureSet(Sampler sampler, core::DescriptorPool pool,
                       core::DescriptorSet set) noexcept;

  Sampler sampler_;
  core::DescriptorPool pool_;  // frees set_
  core::DescriptorSet set_;
};

}  // namespace volumetric_kit::gfx::pipelines

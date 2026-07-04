// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file ibl.hpp
/// @brief CPU-baked image-based lighting for @ref PbrPipeline: convolve a
///        caller-supplied environment into the diffuse irradiance cube, the
///        roughness-prefiltered specular cube, and the BRDF integration LUT
///        that @ref PbrScene binds (set 0, bindings 1-3).

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

#include <glm/vec3.hpp>

#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/pbr_scene.hpp"

namespace volumetric_kit::gfx {
class Allocator;
class Device;
class UploadBatch;
}  // namespace volumetric_kit::gfx

namespace volumetric_kit::gfx::pipelines {

/// @brief The environment radiance @ref bake_ibl convolves: linear HDR RGB
///        arriving from the unit world-space @p direction.
///
/// The bake convolves cube faces concurrently, so the function is called from
/// multiple threads at once: it must be pure / thread-safe (an analytic sky
/// is; anything stateful needs its own synchronization).
using EnvironmentSampler = std::function<glm::vec3(const glm::vec3& direction)>;

/// @brief Resolutions and sample counts for @ref bake_ibl.
///
/// The defaults are sized for a low-frequency analytic environment: ample for
/// a smooth sky, coarse against very tight HDR features (a small bright sun
/// can band in the irradiance and alias on low prefilter mips).
struct IblBakeDesc {
  /// Face size of the diffuse irradiance cube. The cosine convolution is
  /// heavily blurred, so a small single-mip cube suffices.
  uint32_t irradiance_size = 16;
  /// Hemisphere step in radians for the irradiance convolution (both the
  /// azimuth and zenith loops); smaller = more taps per texel.
  float irradiance_sample_delta = 0.1f;
  /// Mip-0 face size of the prefiltered specular cube.
  uint32_t prefilter_size = 64;
  /// Prefilter mip count; mip `m` is convolved at roughness
  /// `m / (prefilter_mip_levels - 1)`, so the chain maps roughness 0..1. Must
  /// not exceed the full chain for @ref prefilter_size.
  uint32_t prefilter_mip_levels = 5;
  /// GGX importance samples per prefiltered texel.
  uint32_t prefilter_samples = 64;
  /// Size of the square BRDF integration LUT.
  uint32_t brdf_lut_size = 128;
  /// Importance samples per LUT texel.
  uint32_t brdf_lut_samples = 256;
};

/// @brief World direction through cube face @p face (Vulkan layer order +X,
///        -X, +Y, -Y, +Z, -Z) at face coordinates @p u, @p v in [-1, 1].
///
/// The face convention @ref bake_ibl convolves with — use it when generating
/// cube content (e.g. a skybox cubemap) that must agree with the baked maps.
/// @param face  Cube face index 0-5; out-of-range values map like face 5 (-Z).
/// @param u     Horizontal face coordinate in [-1, 1].
/// @param v     Vertical face coordinate in [-1, 1].
/// @return The normalized world-space direction.
VG_PIPELINES_API glm::vec3 cube_face_direction(int face, float u,
                                               float v) noexcept;

/// @brief The baked IBL texture set @ref bake_ibl returns: the three maps a
///        @ref PbrScene binds plus the sampler that filters them.
///
/// @ref scene_desc wires the set into a scene in one line. A
/// default-constructed `IblMaps` is empty (`valid()` is false) and safe to
/// move-assign into.
///
/// @warning The device and allocator passed to @ref bake_ibl must outlive
///          these maps (see @ref Texture / @ref Sampler).
///
/// @code
/// Result<pipelines::IblMaps> ibl = pipelines::bake_ibl(device, alloc, sky);
/// if (!ibl) return ibl.status();
/// Result<pipelines::PbrScene> scene = pipelines::PbrScene::create(
///     device.handle(), alloc, pbr.descriptor_set_layout(0),
///     ibl.value().scene_desc(), frames_in_flight);
/// // each frame:
/// scene.value().set_camera(slot, eye, ibl.value().prefilter_max_lod);
/// @endcode
struct IblMaps {
  IblMaps() = default;
  ~IblMaps() = default;
  IblMaps(IblMaps&& other) noexcept
      : irradiance(std::move(other.irradiance)),
        prefilter(std::move(other.prefilter)),
        brdf_lut(std::move(other.brdf_lut)),
        sampler(std::move(other.sampler)),
        prefilter_max_lod(other.prefilter_max_lod) {
    other.sampler.reset();  // a moved-from optional still holds a Sampler
    other.prefilter_max_lod = 0.0f;
  }
  IblMaps& operator=(IblMaps&& other) noexcept {
    if (this != &other) {
      irradiance = std::move(other.irradiance);
      prefilter = std::move(other.prefilter);
      brdf_lut = std::move(other.brdf_lut);
      sampler = std::move(other.sampler);
      other.sampler.reset();
      prefilter_max_lod = other.prefilter_max_lod;
      other.prefilter_max_lod = 0.0f;
    }
    return *this;
  }
  IblMaps(const IblMaps&) = delete;
  IblMaps& operator=(const IblMaps&) = delete;

  Texture irradiance;  ///< Diffuse irradiance cube (RGBA16F, single mip).
  Texture prefilter;   ///< Prefiltered specular cube (RGBA16F, mipped).
  Texture brdf_lut;    ///< 2D RG16F BRDF integration LUT.
  /// Filters all three maps: trilinear, `CLAMP_TO_EDGE` on every axis.
  std::optional<Sampler> sampler;
  /// The prefilter's highest mip index (mip count - 1); pass to
  /// @ref PbrScene::set_camera so the shader clamps roughness LOD to it.
  float prefilter_max_lod = 0.0f;

  /// @return `true` if every map and the sampler are live.
  bool valid() const noexcept {
    return irradiance.valid() && prefilter.valid() && brdf_lut.valid() &&
           sampler.has_value() && sampler->valid();
  }

  /// @brief The @ref PbrScene::create input for these maps.
  /// @return A @ref PbrSceneDesc naming the three views and the sampler
  ///         (null handles when empty).
  PbrSceneDesc scene_desc() const noexcept {
    PbrSceneDesc desc;
    desc.irradiance = irradiance.view();
    desc.prefilter = prefilter.view();
    desc.brdf_lut = brdf_lut.view();
    desc.sampler = sampler.has_value() ? sampler->handle() : VK_NULL_HANDLE;
    return desc;
  }
};

/// @brief Integrate the environment-independent BRDF LUT (the split sum's
///        scale/bias term per (NdotV, roughness)) and record its upload into
///        an open batch.
///
/// The standard split-sum table: environment-free, so it is bakeable without
/// an @ref EnvironmentSampler and shared by every environment.
/// @param batch    An open batch; the LUT is sampled-ready only after the
///                 caller's @ref UploadBatch::finish returns OK.
/// @param size     LUT width and height in texels (> 0).
/// @param samples  Importance samples per texel (> 0); the default matches
///                 @ref IblBakeDesc::brdf_lut_samples.
/// @return The RG16F LUT texture on success, or a non-OK @ref Status: @ref
///         Status::Code::InvalidArgument for a zero @p size or @p samples
///         (checked before anything records, leaving @p batch unchanged) or an
///         empty batch, plus everything @ref UploadBatch::add rejects.
VG_PIPELINES_API Result<Texture> bake_brdf_lut(UploadBatch& batch,
                                               uint32_t size,
                                               uint32_t samples = 256);

/// @brief Integrate the BRDF LUT and upload it in one blocking submit.
///
/// A one-texture batch around the other overload. Baking a full IBL set? Use
/// @ref bake_ibl, which shares one batch across all three maps.
/// @param device     The device whose graphics queue runs the one-time
///                   transfer; must outlive the returned texture.
/// @param allocator  Allocates the texture; must outlive it.
/// @param size       LUT width and height in texels (> 0).
/// @param samples    Importance samples per texel (> 0); the default matches
///                   @ref IblBakeDesc::brdf_lut_samples.
/// @return The RG16F LUT texture -- sampled-ready -- or a non-OK @ref Status:
///         @ref Status::Code::InvalidArgument for a null device or a zero
///         @p size or @p samples; otherwise whatever the batch's begin / add /
///         finish steps report.
VG_PIPELINES_API Result<Texture> bake_brdf_lut(const Device& device,
                                               Allocator& allocator,
                                               uint32_t size,
                                               uint32_t samples = 256);

/// @brief Convolve @p environment into the full IBL texture set: the diffuse
///        irradiance cube, the roughness-prefiltered specular chain, and the
///        BRDF LUT, uploaded through one batch (a single submit + fence wait).
///
/// CPU-side, sized for an analytic environment (a loaded HDR environment
/// would convolve on the GPU). The per-face convolutions run concurrently --
/// @p environment must be thread-safe (see @ref EnvironmentSampler) -- and the
/// result is deterministic: faces write disjoint texels with unchanged
/// per-texel math, so any thread count yields identical bytes.
/// @param device       Runs the upload submit; must outlive the maps.
/// @param allocator    Allocates the textures; must outlive the maps.
/// @param environment  The radiance function to convolve; called concurrently.
/// @param desc         Resolutions and sample counts (defaults suit an
///                     analytic sky; see @ref IblBakeDesc).
/// @pre @p device holds a live `VkDevice`, @p environment is callable, every
///      @p desc size and sample count is non-zero,
///      `desc.irradiance_sample_delta > 0`, and `desc.prefilter_mip_levels` is
///      within the full chain for `desc.prefilter_size` -- validated before
///      Vulkan is touched, otherwise a non-OK @ref Status with domain
///      @ref Status::Code::InvalidArgument.
/// @return The baked maps (with @ref IblMaps::prefilter_max_lod filled) on
///         success, or a non-OK @ref Status (a Vulkan-domain Status from the
///         sampler, upload, or submit step).
VG_PIPELINES_API Result<IblMaps> bake_ibl(const Device& device,
                                          Allocator& allocator,
                                          const EnvironmentSampler& environment,
                                          const IblBakeDesc& desc = {});

}  // namespace volumetric_kit::gfx::pipelines

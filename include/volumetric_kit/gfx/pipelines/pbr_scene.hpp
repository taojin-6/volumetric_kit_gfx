// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pbr_scene.hpp
/// @brief The per-frame set-0 binding for @ref PbrPipeline: the camera plus the
///        image-based-lighting environment maps.

#include <vector>

#include <glm/vec3.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/pipelines/export.hpp"
#include "volumetric_kit/gfx/pipelines/impl/owned_descriptor_set.hpp"

namespace volumetric_kit::core {
class Allocator;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

/// @brief The image-based-lighting maps bound into a @ref PbrScene (set 0,
///        bindings 1-3): a diffuse irradiance cube, a roughness-prefiltered
///        specular cube, and the BRDF integration LUT.
///
/// Every view and the @ref sampler must be non-null — the shader samples all
/// three unconditionally. The caller owns the images (they outlive the scene).
struct PbrSceneDesc {
  VkImageView irradiance = VK_NULL_HANDLE;  ///< Diffuse irradiance cube.
  VkImageView prefilter = VK_NULL_HANDLE;   ///< Prefiltered specular cube.
  VkImageView brdf_lut = VK_NULL_HANDLE;    ///< 2D BRDF integration LUT.
  VkSampler sampler = VK_NULL_HANDLE;  ///< Filters all three (mipped cube).
};

/// @brief Owns the set-0 descriptor resources @ref PbrPipeline binds once per
///        frame: a camera uniform buffer plus the IBL maps.
///
/// This is the per-frame "view + environment lighting" half of the technique
/// (set 1, the material, is @ref PbrMaterial). It is **not** a scene graph —
/// the library keeps none; it is just the descriptor set for frame-constant
/// data. The camera UBO and its descriptor set are **ringed per
/// frame-in-flight slot**: create the scene with the frame loop's
/// `frames_in_flight`, then each frame write and bind the acquired frame's
/// slot, so a frame's update never waits on another frame's reads. Each slot
/// owns its own one-set `core::DescriptorPool`, `core::DescriptorSet`, and
/// camera UBO in device-only memory, written by an update recorded into the
/// frame's command buffer -- which is why @ref set_camera takes that command
/// buffer, and is called before rendering begins. A default-constructed
/// `PbrScene` is empty (`valid()` is false) and safe to move-assign into.
///
/// @warning The device, plus the IBL images @ref PbrSceneDesc names, must
///          outlive the scene.
///
/// @code
/// core::Result<pipelines::PbrScene> scene = pipelines::PbrScene::create(
///     device, allocator, pbr.descriptor_set_layout(0), ibl_desc,
///     loop.frames_in_flight());
/// // each frame, with the windowing Frame f from FrameLoop::begin_frame,
/// // before f.target->begin:
/// scene.value().set_camera(f.cmd, f.slot, eye, prefilter_max_lod);
/// pbr_frame.slot = f.slot;  // PbrPipeline::submit binds that slot's set
/// @endcode
class VG_PIPELINES_API PbrScene {
 public:
  /// @brief Construct an empty scene (owns nothing; `valid()` is false).
  PbrScene() = default;

  /// @brief Build the set-0 binding from the IBL environment, one camera UBO +
  ///        descriptor set per frame-in-flight slot.
  /// @param device           The logical device that owns the pools + sets.
  /// @param allocator        Allocates the camera UBOs.
  /// @param scene_layout     The reflected set-0 layout, from
  ///                         @ref PbrPipeline::descriptor_set_layout(0).
  /// @param desc             The three IBL map views + sampler.
  /// @param frames_in_flight The frame loop's CPU-ahead depth (>= 1); one UBO
  ///                         ring slot is built per in-flight frame.
  /// @pre @p device and @p scene_layout are non-`VK_NULL_HANDLE`, every view
  ///      in @p desc plus @p desc.sampler is non-`VK_NULL_HANDLE`, and
  ///      @p frames_in_flight >= 1 — validated before Vulkan is touched,
  ///      otherwise a non-OK `core::Status` with domain
  ///      `core::Status::Code::InvalidArgument`.
  /// @return The scene on success, or a non-OK `core::Status` (a backend
  ///         Status from buffer / pool / set allocation).
  static core::Result<PbrScene> create(VkDevice device,
                                       core::Allocator& allocator,
                                       VkDescriptorSetLayout scene_layout,
                                       const PbrSceneDesc& desc,
                                       uint32_t frames_in_flight = 1);

  ~PbrScene() = default;
  PbrScene(PbrScene&&) noexcept = default;
  // Hand-written to guard self-move: a defaulted move-assign hands slots_ a
  // self-move of std::vector, which is valid-but-unspecified.
  PbrScene& operator=(PbrScene&& other) noexcept {
    if (this != &other) {
      slots_ = std::move(other.slots_);
    }
    return *this;
  }
  PbrScene(const PbrScene&) = delete;
  PbrScene& operator=(const PbrScene&) = delete;

  /// @brief Record a write of the per-frame camera into slot @p slot's UBO.
  ///
  /// Recorded into @p cmd (@ref OwnedDescriptorSet::write_uniform), so it
  /// applies to the work recorded after it: two views rendered in one command
  /// buffer, each after its own call, each see their own camera. Call it
  /// before the frame's rendering begins: a recorded update is invalid inside
  /// a render pass instance.
  /// @param cmd               The frame's command buffer, recording, outside
  ///                          a render pass instance.
  /// @param slot              The acquired frame's in-flight slot (e.g. the
  ///                          windowing `Frame::slot`); its previous use has
  ///                          been fence-waited by the loop, so the GPU is not
  ///                          reading this UBO.
  /// @param eye               World-space camera position.
  /// @param prefilter_max_lod The specular prefilter's highest mip index
  ///                          (mip count - 1); the shader clamps roughness LOD
  ///                          to it.
  /// @pre @p cmd is non-`VK_NULL_HANDLE` (a contract check), `valid()`, and
  ///      @p slot < @ref frames_in_flight; an out-of-range slot is ignored
  ///      (no write), mirroring @ref descriptor_set.
  void set_camera(VkCommandBuffer cmd, uint32_t slot, const glm::vec3& eye,
                  float prefilter_max_lod) const;

  /// @return Slot @p slot's set-0 `VkDescriptorSet` to bind (`VK_NULL_HANDLE`
  ///         when empty).
  /// @pre @p slot < @ref frames_in_flight when `valid()`.
  VkDescriptorSet descriptor_set(uint32_t slot = 0) const noexcept {
    return slot < slots_.size() ? slots_[slot].descriptor_set()
                                : VK_NULL_HANDLE;
  }

  /// @return The number of UBO ring slots (the frames_in_flight it was built
  ///         for; 0 when empty).
  uint32_t frames_in_flight() const noexcept {
    return static_cast<uint32_t>(slots_.size());
  }

  /// @return `true` if this owns built scene sets.
  bool valid() const noexcept { return !slots_.empty(); }

 private:
  // Set 0 per frame-in-flight slot: pool + set + camera UBO each, so a slot's
  // UBO is rewritten only after its previous frame's fence was waited.
  std::vector<OwnedDescriptorSet> slots_;
};

}  // namespace volumetric_kit::gfx::pipelines

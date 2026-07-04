// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/pbr_scene.hpp"

#include <utility>

#include <glm/vec4.hpp>

namespace volumetric_kit::gfx::pipelines {

namespace {

// std140 per-frame parameters; mirrors the Scene block in the embedded
// model.frag: the world-space eye in .xyz, the prefilter max LOD in .w.
struct SceneUbo {
  glm::vec4 camera_pos;
};

}  // namespace

Result<PbrScene> PbrScene::create(VkDevice device, Allocator& allocator,
                                  VkDescriptorSetLayout scene_layout,
                                  const PbrSceneDesc& desc,
                                  uint32_t frames_in_flight) {
  if (device == VK_NULL_HANDLE || scene_layout == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "PbrScene::create: device and scene_layout must be non-null");
  }
  if (desc.irradiance == VK_NULL_HANDLE || desc.prefilter == VK_NULL_HANDLE ||
      desc.brdf_lut == VK_NULL_HANDLE || desc.sampler == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "PbrScene::create: all three IBL views and the sampler must be "
        "non-null");
  }
  if (frames_in_flight == 0) {
    return Status::invalid_argument(
        "PbrScene::create: frames_in_flight must be >= 1");
  }

  PbrScene scene;
  scene.slots_.reserve(frames_in_flight);
  for (uint32_t i = 0; i < frames_in_flight; ++i) {
    VG_ASSIGN(OwnedDescriptorSet resources,
              OwnedDescriptorSet::create(device, allocator, scene_layout,
                                         sizeof(SceneUbo), 3));

    // The IBL maps are frame-constant, so they live in every slot's set
    // alongside its camera (binding 0, written per frame by set_camera) at
    // bindings 1-3, matching model.frag.
    resources.set().write_combined_image_sampler(
        1, desc.irradiance, desc.sampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    resources.set().write_combined_image_sampler(
        2, desc.prefilter, desc.sampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    resources.set().write_combined_image_sampler(
        3, desc.brdf_lut, desc.sampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    scene.slots_.push_back(std::move(resources));
  }
  return scene;
}

void PbrScene::set_camera(uint32_t slot, const glm::vec3& eye,
                          float prefilter_max_lod) noexcept {
  *static_cast<SceneUbo*>(slots_[slot].mapped()) =
      SceneUbo{glm::vec4(eye, prefilter_max_lod)};
}

}  // namespace volumetric_kit::gfx::pipelines

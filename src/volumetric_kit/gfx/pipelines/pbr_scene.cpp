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
                                  const PbrSceneDesc& desc) {
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

  VG_ASSIGN(OwnedDescriptorSet resources,
            OwnedDescriptorSet::create(device, allocator, scene_layout,
                                       sizeof(SceneUbo), 3));

  // The IBL maps are frame-constant, so they live in the scene set alongside
  // the camera (binding 0, written per frame by set_camera) at bindings 1-3,
  // matching model.frag.
  resources.set().write_combined_image_sampler(
      1, desc.irradiance, desc.sampler,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  resources.set().write_combined_image_sampler(
      2, desc.prefilter, desc.sampler,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  resources.set().write_combined_image_sampler(
      3, desc.brdf_lut, desc.sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  PbrScene scene;
  scene.resources_ = std::move(resources);
  return scene;
}

void PbrScene::set_camera(const glm::vec3& eye,
                          float prefilter_max_lod) noexcept {
  *static_cast<SceneUbo*>(resources_.mapped()) =
      SceneUbo{glm::vec4(eye, prefilter_max_lod)};
}

}  // namespace volumetric_kit::gfx::pipelines

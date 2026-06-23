// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/pbr_scene.hpp"

#include <utility>

#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/core/allocator.hpp"

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

  // Camera UBO: host-mapped, refreshed each frame through set_camera.
  BufferDesc bd;
  bd.size = sizeof(SceneUbo);
  bd.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  bd.memory = MemoryUsage::HostVisible;
  bd.mapped = true;
  VG_ASSIGN(Buffer ubo, allocator.create_buffer(bd));

  // One-set pool: the camera UBO + the three IBL maps.
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}};
  VG_ASSIGN(DescriptorPool pool, DescriptorPool::create(device, sizes, 2, 1));
  VG_ASSIGN(DescriptorSet set, pool.allocate(scene_layout));

  set.write_uniform_buffer(0, ubo.handle(), 0, sizeof(SceneUbo));
  // The IBL maps are frame-constant, so they live in the scene set alongside
  // the camera at bindings 1-3, matching model.frag.
  set.write_combined_image_sampler(1, desc.irradiance, desc.sampler,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  set.write_combined_image_sampler(2, desc.prefilter, desc.sampler,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  set.write_combined_image_sampler(3, desc.brdf_lut, desc.sampler,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  PbrScene scene;
  scene.pool_ = std::move(pool);
  scene.set_ = set;
  scene.ubo_ = std::move(ubo);
  return scene;
}

void PbrScene::set_camera(const glm::vec3& eye,
                          float prefilter_max_lod) noexcept {
  *static_cast<SceneUbo*>(ubo_.mapped()) =
      SceneUbo{glm::vec4(eye, prefilter_max_lod)};
}

// The pool + UBO move themselves; null the set value too (it is a borrowed
// handle, freed with the pool) so a moved-from scene is fully empty and its
// accessors stay consistent with valid().
PbrScene::PbrScene(PbrScene&& other) noexcept
    : pool_(std::move(other.pool_)),
      set_(other.set_),
      ubo_(std::move(other.ubo_)) {
  other.set_ = DescriptorSet{};
}

PbrScene& PbrScene::operator=(PbrScene&& other) noexcept {
  if (this != &other) {
    pool_ = std::move(other.pool_);
    set_ = other.set_;
    ubo_ = std::move(other.ubo_);
    other.set_ = DescriptorSet{};
  }
  return *this;
}

}  // namespace volumetric_kit::gfx::pipelines

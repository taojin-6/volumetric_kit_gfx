// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pipelines/impl/owned_descriptor_set.hpp
/// Internal helper shared by @ref PbrScene (set 0) and @ref PbrMaterial (set
/// 1); included from their public headers, not a standalone type.

#include <utility>

#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {
class Allocator;
}  // namespace volumetric_kit::gfx

namespace volumetric_kit::gfx::pipelines {

/// @brief Owns the one-set descriptor resources the PBR set-0 / set-1 bindings
///        share: a one-set @ref DescriptorPool, the @ref DescriptorSet it
///        allocates, and a host-mapped uniform buffer bound at binding 0.
///
/// An internal building block for @ref PbrScene and @ref PbrMaterial that
/// factors out their identical ownership + move semantics. @ref create builds
/// the pool, allocates the set, and binds the UBO at binding 0; the owner then
/// writes its combined-image-samplers into @ref set and fills @ref mapped. The
/// set is a borrowed handle freed with the pool, so the move pair nulls it — a
/// moved-from object is fully empty and its accessors stay consistent with
/// @ref valid.
class OwnedDescriptorSet {
 public:
  /// @brief Construct an empty bundle (owns nothing; `valid()` is false).
  OwnedDescriptorSet() = default;

  /// @brief Build a one-set binding: a host-mapped UBO of @p ubo_size bytes at
  ///        binding 0, in a pool also sized for @p sampler_count
  ///        combined-image-samplers (the owner writes those at bindings
  ///        1..@p sampler_count).
  /// @param device        The logical device that owns the pool + set.
  /// @param allocator     Allocates the UBO; must outlive the bundle.
  /// @param layout        The reflected set layout to allocate against.
  /// @param ubo_size      Size of the host-mapped uniform buffer, in bytes.
  /// @param sampler_count Combined-image-sampler capacity to reserve (>= 1).
  /// @pre @p device and @p layout are non-`VK_NULL_HANDLE` (the typed caller
  ///      validates its own views/sampler first).
  /// @return The bundle on success, or a non-OK @ref Status from buffer / pool
  ///         / set allocation.
  static Result<OwnedDescriptorSet> create(VkDevice device,
                                           Allocator& allocator,
                                           VkDescriptorSetLayout layout,
                                           VkDeviceSize ubo_size,
                                           uint32_t sampler_count);

  ~OwnedDescriptorSet() = default;

  // Hand-written so the borrowed set_ handle is nulled on the moved-from object
  // (pool_/ubo_ null themselves); inline so the owning types keep `= default`
  // moves. See the class brief.
  OwnedDescriptorSet(OwnedDescriptorSet&& other) noexcept
      : pool_(std::move(other.pool_)),
        set_(other.set_),
        ubo_(std::move(other.ubo_)) {
    other.set_ = DescriptorSet{};
  }
  OwnedDescriptorSet& operator=(OwnedDescriptorSet&& other) noexcept {
    if (this != &other) {
      pool_ = std::move(other.pool_);
      set_ = other.set_;
      ubo_ = std::move(other.ubo_);
      other.set_ = DescriptorSet{};
    }
    return *this;
  }
  OwnedDescriptorSet(const OwnedDescriptorSet&) = delete;
  OwnedDescriptorSet& operator=(const OwnedDescriptorSet&) = delete;

  /// @return The descriptor set, for binding and for writing image samplers.
  const DescriptorSet& set() const noexcept { return set_; }

  /// @return The set's `VkDescriptorSet` handle (`VK_NULL_HANDLE` when empty).
  VkDescriptorSet descriptor_set() const noexcept { return set_.handle(); }

  /// @return The mapped UBO storage (binding 0); write the typed block here.
  void* mapped() const noexcept { return ubo_.mapped(); }

  /// @return `true` if this owns a built set.
  bool valid() const noexcept { return pool_.valid(); }

 private:
  DescriptorPool pool_;  // one-set pool that owns set_'s lifetime
  DescriptorSet set_;  // the allocated set: UBO at binding 0 + the owner's maps
  Buffer ubo_;         // host-mapped UBO the set points at
};

}  // namespace volumetric_kit::gfx::pipelines

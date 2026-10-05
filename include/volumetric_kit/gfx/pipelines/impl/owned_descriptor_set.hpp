// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pipelines/impl/owned_descriptor_set.hpp
/// Internal helpers shared by @ref PbrScene (set 0) and @ref PbrMaterial (set
/// 1); included from their public headers, not standalone types.

#include <cstdint>
#include <memory>
#include <utility>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::core {
class Allocator;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

/// @brief Allocate a uniform buffer for data rewritten every frame, by an
///        update @ref OwnedDescriptorSet::write_uniform records.
///
/// Device-only memory, which every device has: the core's placement for a
/// buffer shaders read (its DECISIONS.md, "Where memory lives"), written by a
/// `vkCmdUpdateBuffer` in the frame's own command buffer rather than through
/// a mapping, so the write behaves the same on every device.
/// @param allocator  Allocates it.
/// @param size       Bytes; non-zero, a multiple of 4 and at most 65536, the
///                   bounds of `vkCmdUpdateBuffer`.
/// @return The buffer, or `core::Status::Code::InvalidArgument` for a size
///         outside those bounds, or the allocator's failure.
core::Result<core::Buffer> make_frame_uniform_buffer(core::Allocator& allocator,
                                                     VkDeviceSize size);

/// @brief Owns the one-set descriptor resources the PBR set-0 / set-1 bindings
///        share: a one-set `core::DescriptorPool`, the `core::DescriptorSet` it
///        allocates, and a share of the uniform buffer bound at binding 0.
///
/// An internal building block for @ref PbrScene and @ref PbrMaterial that
/// factors out their identical ownership + move semantics. @ref create builds
/// the pool and allocates the set -- the fallible part, so an owner can do it
/// before queuing anything -- and @ref bind_uniform then binds a range of a
/// uniform buffer the bundle shares: a scene's per-frame buffer from
/// @ref make_frame_uniform_buffer, which the owner rewrites with
/// @ref write_uniform, or one block of the factor buffer every material of a
/// @ref PbrMaterial::create_all call shares. The owner then writes its
/// combined-image-samplers into @ref set. The set is a borrowed handle freed
/// with the pool, so the move pair empties it, nulls its cached handle and
/// zeroes the bound range — a moved-from object is fully empty and its
/// accessors stay consistent with @ref valid.
///
/// @code
/// VKC_ASSIGN(OwnedDescriptorSet owned,
///            OwnedDescriptorSet::create(device, layout, 3));
/// VKC_ASSIGN(core::Buffer ubo,
///            make_frame_uniform_buffer(allocator, sizeof(Block)));
/// owned.bind_uniform(std::make_shared<core::Buffer>(std::move(ubo)), 0,
///                    sizeof(Block));
/// owned.write_uniform(cmd, &block, sizeof(block));  // before rendering
/// @endcode
class OwnedDescriptorSet {
 public:
  /// @brief Construct an empty bundle (owns nothing; `valid()` is false).
  OwnedDescriptorSet() = default;

  /// @brief Build a one-set pool sized for one uniform buffer plus
  ///        @p sampler_count combined-image-samplers, and allocate the set.
  ///
  /// The owner then binds the uniform buffer with @ref bind_uniform and
  /// writes the samplers at bindings 1..@p sampler_count.
  /// @param device        The logical device that owns the pool + set.
  /// @param layout        The reflected set layout to allocate against.
  /// @param sampler_count Combined-image-sampler capacity to reserve (>= 1).
  /// @pre @p device and @p layout are non-`VK_NULL_HANDLE` (the typed caller
  ///      validates its own views/sampler first).
  /// @return The bundle on success, or a non-OK `core::Status` from pool / set
  ///         allocation.
  static core::Result<OwnedDescriptorSet> create(VkDevice device,
                                                 VkDescriptorSetLayout layout,
                                                 uint32_t sampler_count);

  ~OwnedDescriptorSet() = default;

  // Hand-written so the borrowed set_, its cached handle and the bound range
  // are cleared on the moved-from object (pool_/ubo_ null themselves); inline
  // so the owning types keep `= default` moves. See the class brief.
  OwnedDescriptorSet(OwnedDescriptorSet&& other) noexcept
      : pool_(std::move(other.pool_)),
        set_(std::exchange(other.set_, core::DescriptorSet{})),
        handle_(std::exchange(other.handle_, VK_NULL_HANDLE)),
        ubo_(std::move(other.ubo_)),
        offset_(std::exchange(other.offset_, 0)),
        range_(std::exchange(other.range_, 0)) {}
  OwnedDescriptorSet& operator=(OwnedDescriptorSet&& other) noexcept {
    if (this != &other) {
      pool_ = std::move(other.pool_);
      set_ = std::exchange(other.set_, core::DescriptorSet{});
      handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
      ubo_ = std::move(other.ubo_);
      offset_ = std::exchange(other.offset_, 0);
      range_ = std::exchange(other.range_, 0);
    }
    return *this;
  }
  OwnedDescriptorSet(const OwnedDescriptorSet&) = delete;
  OwnedDescriptorSet& operator=(const OwnedDescriptorSet&) = delete;

  /// @brief Bind @p range bytes of @p ubo, from @p offset, at binding 0, and
  ///        keep @p ubo alive with the bundle.
  /// @param ubo     The uniform buffer, shared with whatever else binds it.
  /// @param offset  Where the range starts; a multiple of the device's
  ///                `minUniformBufferOffsetAlignment` (any multiple of 256
  ///                is).
  /// @param range   Its length.
  /// @pre `valid()`, @p ubo is non-null and valid, and the range lies within
  ///      it.
  void bind_uniform(std::shared_ptr<const core::Buffer> ubo,
                    VkDeviceSize offset, VkDeviceSize range);

  /// @brief Record a write of @p size bytes from @p data to the start of the
  ///        bound range.
  ///
  /// Records into @p cmd a barrier ordering the write after earlier work in
  /// the queue that reads or writes the range, a `vkCmdUpdateBuffer`, and a
  /// barrier making it visible to the vertex and fragment shaders' uniform
  /// reads. So each write applies to the work recorded after it: two passes
  /// in one command buffer, each after its own write, read their own data.
  /// @param cmd   A command buffer in the recording state, outside a render
  ///              pass instance, as `vkCmdUpdateBuffer` requires.
  /// @param data  The bytes.
  /// @param size  Their length; a multiple of 4, within the bound range.
  /// @pre @p cmd is non-`VK_NULL_HANDLE`, `valid()` with a bound range, and
  ///      no work on another queue still reads it. A null @p cmd or a size
  ///      the range cannot take fails the contract check.
  void write_uniform(VkCommandBuffer cmd, const void* data,
                     VkDeviceSize size) const;

  /// @return The descriptor set, for binding and for writing image samplers.
  const core::DescriptorSet& set() const noexcept { return set_; }

  /// @return The set's `VkDescriptorSet` handle (`VK_NULL_HANDLE` when empty),
  ///         without asking @ref set whether its pool is alive: this owns the
  ///         pool, so the set lives exactly as long as the bundle is valid.
  VkDescriptorSet descriptor_set() const noexcept { return handle_; }

  /// @return `true` if this owns a built set.
  bool valid() const noexcept { return pool_.valid(); }

 private:
  core::DescriptorPool pool_;  // one-set pool that owns set_'s lifetime
  core::DescriptorSet set_;    // the allocated set: UBO at binding 0 + the maps
  // set_'s handle, cached: binding it per draw then skips set_'s check of a
  // pool this bundle owns.
  VkDescriptorSet handle_ = VK_NULL_HANDLE;
  std::shared_ptr<const core::Buffer> ubo_;  // the buffer binding 0 reads
  VkDeviceSize offset_ = 0;  // where binding 0's range starts in ubo_
  VkDeviceSize range_ = 0;   // binding 0's length; 0 until bound
};

}  // namespace volumetric_kit::gfx::pipelines

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file pipelines/impl/owned_descriptor_set.hpp
/// Internal helpers shared by @ref PbrScene (set 0) and @ref PbrMaterial (set
/// 1); included from their public headers, not standalone types.

#include <cstdint>
#include <utility>

#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::core {
class Allocator;
}  // namespace volumetric_kit::core

namespace volumetric_kit::gfx::pipelines {

/// @brief Where @ref make_frame_uniform_buffer puts a uniform buffer the host
///        rewrites every frame.
enum class FrameUniformMemory {
  /// Device-mapped memory where the device has it and it has room, else
  /// device-only memory.
  Prefer,
  /// Device-only memory, written by a recorded update even where the device
  /// has device-mapped memory: the fallback path, on any device.
  DeviceOnly,
};

/// @brief Allocate a uniform buffer for data the host rewrites every frame.
///
/// The core's placement for per-frame uniforms (its DECISIONS.md, "Where
/// memory lives"): `MemoryUsage::DeviceMapped`, which the host writes and
/// shaders read in place -- a discrete GPU's BAR window, or the one pool of
/// unified memory. That placement never falls back to host memory, so where
/// the device has no such memory, or its window is full, the buffer is
/// `MemoryUsage::DeviceOnly` instead, written by a `vkCmdUpdateBuffer` that
/// @ref OwnedDescriptorSet::write_uniform records.
/// @param allocator  Allocates it.
/// @param size       Bytes; non-zero, a multiple of 4 and at most 65536, the
///                   bounds of `vkCmdUpdateBuffer`, so either placement can be
///                   written.
/// @param memory     Where to put it; @ref FrameUniformMemory::DeviceOnly
///                   exercises the fallback where the device would not need
///                   it.
/// @return The buffer -- mapped when device-mapped -- or
///         @ref Status::Code::InvalidArgument for a size outside those bounds,
///         or the allocator's failure for the device-only buffer.
Result<core::Buffer> make_frame_uniform_buffer(
    core::Allocator& allocator, VkDeviceSize size,
    FrameUniformMemory memory = FrameUniformMemory::Prefer);

/// @brief Owns the one-set descriptor resources the PBR set-0 / set-1 bindings
///        share: a one-set @ref DescriptorPool, the @ref DescriptorSet it
///        allocates, and the uniform buffer bound at binding 0.
///
/// An internal building block for @ref PbrScene and @ref PbrMaterial that
/// factors out their identical ownership + move semantics. @ref create builds
/// the pool, allocates the set, and binds the uniform buffer it is given --
/// a material's factors uploaded through an @ref UploadBatch, or a scene's
/// per-frame buffer from @ref make_frame_uniform_buffer, which the owner
/// rewrites with @ref write_uniform. The owner then writes its
/// combined-image-samplers into @ref set. The set is a borrowed handle freed
/// with the pool, so the move pair nulls it — a moved-from object is fully
/// empty and its accessors stay consistent with @ref valid.
///
/// @code
/// VG_ASSIGN(core::Buffer ubo,
///           make_frame_uniform_buffer(allocator, sizeof(Block)));
/// VG_ASSIGN(OwnedDescriptorSet owned,
///           OwnedDescriptorSet::create(device, std::move(ubo), layout, 3));
/// owned.write_uniform(cmd, &block, sizeof(block));  // before rendering
/// @endcode
class OwnedDescriptorSet {
 public:
  /// @brief Construct an empty bundle (owns nothing; `valid()` is false).
  OwnedDescriptorSet() = default;

  /// @brief Build a one-set binding: @p ubo at binding 0, in a pool also sized
  ///        for @p sampler_count combined-image-samplers (the owner writes
  ///        those at bindings 1..@p sampler_count).
  /// @param device        The logical device that owns the pool + set.
  /// @param ubo           The uniform buffer to own and bind, whole.
  /// @param layout        The reflected set layout to allocate against.
  /// @param sampler_count Combined-image-sampler capacity to reserve (>= 1).
  /// @pre @p device and @p layout are non-`VK_NULL_HANDLE` and @p ubo is valid
  ///      (the typed caller validates its own views/sampler first).
  /// @return The bundle on success, or a non-OK @ref Status from pool / set
  ///         allocation; @p ubo is freed with it on failure.
  static Result<OwnedDescriptorSet> create(VkDevice device, core::Buffer ubo,
                                           VkDescriptorSetLayout layout,
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

  /// @brief Write @p size bytes from @p data to the start of the uniform
  ///        buffer.
  ///
  /// Through the mapping when the buffer has one (device-mapped memory),
  /// written now; otherwise recorded into @p cmd as a `vkCmdUpdateBuffer` and
  /// a barrier making it visible to the vertex and fragment shaders' uniform
  /// reads. Either way the shaders of work submitted afterwards see it.
  /// @param cmd   A command buffer in the recording state, outside a render
  ///              pass instance, as `vkCmdUpdateBuffer` requires.
  /// @param data  The bytes.
  /// @param size  Their length; a multiple of 4, within the buffer.
  /// @pre `valid()`, and no submitted work still reads the buffer (a ring slot
  ///      whose previous frame's fence was waited). A size the buffer cannot
  ///      take fails the contract check.
  void write_uniform(VkCommandBuffer cmd, const void* data,
                     VkDeviceSize size) const;

  /// @return The descriptor set, for binding and for writing image samplers.
  const DescriptorSet& set() const noexcept { return set_; }

  /// @return The set's `VkDescriptorSet` handle (`VK_NULL_HANDLE` when empty).
  VkDescriptorSet descriptor_set() const noexcept { return set_.handle(); }

  /// @return The uniform buffer bound at binding 0.
  const core::Buffer& uniform_buffer() const noexcept { return ubo_; }

  /// @return `true` if this owns a built set.
  bool valid() const noexcept { return pool_.valid(); }

 private:
  DescriptorPool pool_;  // one-set pool that owns set_'s lifetime
  DescriptorSet set_;  // the allocated set: UBO at binding 0 + the owner's maps
  core::Buffer ubo_;   // the uniform buffer the set points at
};

}  // namespace volumetric_kit::gfx::pipelines

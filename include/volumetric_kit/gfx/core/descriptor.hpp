// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file descriptor.hpp
/// @brief Descriptor-set layout, pool, and set — how shader resources (uniform
///        buffers, textures) are described, allocated, and bound.

#include <cstdint>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

class DescriptorSet;

/// @brief Owns a `VkDescriptorSetLayout`: the binding signature of one
///        descriptor set (which bindings exist, of what type, in which stages).
///
/// A pipeline builds one of these per descriptor set its shaders declare (from
/// reflection), and a @ref DescriptorPool allocates @ref DescriptorSet objects
/// against it. A default-constructed `DescriptorSetLayout` is empty (`valid()`
/// is false) and safe to move-assign into.
///
/// @warning The @p device passed to @ref create must outlive the layout: the
///          destructor frees through it, so destroying the device first is
///          undefined behavior.
///
/// @code
/// VkDescriptorSetLayoutBinding ubo{};
/// ubo.binding = 0;
/// ubo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
/// ubo.descriptorCount = 1;
/// ubo.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
/// Result<DescriptorSetLayout> layout =
///     DescriptorSetLayout::create(device, &ubo, 1);
/// @endcode
class VG_CORE_API DescriptorSetLayout {
 public:
  /// @brief Construct an empty layout (owns nothing; `valid()` is false).
  DescriptorSetLayout() = default;

  /// @brief Create a layout from @p count binding descriptions.
  /// @param device   The logical device that owns the layout.
  /// @param bindings Pointer to @p count binding descriptions, or `nullptr`
  ///                 with @p count `0` for an empty set.
  /// @param count    Number of @p bindings.
  /// @pre @p device is non-`VK_NULL_HANDLE`, and @p bindings is non-null when
  ///      @p count is non-zero — validated before Vulkan is touched, otherwise
  ///      a non-OK @ref Status with domain @ref Status::Code::InvalidArgument.
  /// @return The layout on success, or a non-OK @ref Status.
  static Result<DescriptorSetLayout> create(
      VkDevice device, const VkDescriptorSetLayoutBinding* bindings,
      uint32_t count);

  ~DescriptorSetLayout() = default;
  DescriptorSetLayout(DescriptorSetLayout&&) noexcept = default;
  DescriptorSetLayout& operator=(DescriptorSetLayout&&) noexcept = default;
  DescriptorSetLayout(const DescriptorSetLayout&) = delete;
  DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

  /// @return The underlying `VkDescriptorSetLayout` (`VK_NULL_HANDLE` when
  ///         empty).
  VkDescriptorSetLayout handle() const noexcept { return layout_.get(); }

  /// @return `true` if this owns a layout.
  bool valid() const noexcept { return layout_.valid(); }

 private:
  UniqueHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout> layout_;
};

/// @brief Owns a `VkDescriptorPool` and allocates @ref DescriptorSet objects
///        from it.
///
/// Sized at creation from per-type counts + a maximum set count. Allocated sets
/// are owned by the pool: they are all freed when the pool is destroyed (this
/// kit does not free sets individually), so retire the pool only once the GPU
/// is done with every set drawn from it. A default-constructed `DescriptorPool`
/// is empty (`valid()` is false) and safe to move-assign into.
///
/// @warning The @p device passed to @ref create must outlive the pool: the
///          destructor frees through it, so destroying the device first is
///          undefined behavior.
///
/// @code
/// VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
/// Result<DescriptorPool> pool = DescriptorPool::create(device, &size, 1, 1);
/// if (!pool) return pool.status();
/// Result<DescriptorSet> set = pool.value().allocate(layout);
/// @endcode
class VG_CORE_API DescriptorPool {
 public:
  /// @brief Construct an empty pool (owns nothing; `valid()` is false).
  DescriptorPool() = default;

  /// @brief Create a pool with capacity for @p max_sets sets drawing on the
  ///        per-type counts in @p sizes.
  /// @param device     The logical device that owns the pool.
  /// @param sizes      Pointer to @p size_count per-type capacities.
  /// @param size_count Number of @p sizes (must be non-zero).
  /// @param max_sets   Maximum number of sets allocatable (must be non-zero).
  /// @pre @p device is non-`VK_NULL_HANDLE`, @p sizes is non-null, and
  ///      @p size_count and @p max_sets are non-zero — validated before Vulkan
  ///      is touched, otherwise a non-OK @ref Status with domain
  ///      @ref Status::Code::InvalidArgument.
  /// @return The pool on success, or a non-OK @ref Status.
  static Result<DescriptorPool> create(VkDevice device,
                                       const VkDescriptorPoolSize* sizes,
                                       uint32_t size_count, uint32_t max_sets);

  ~DescriptorPool() = default;
  DescriptorPool(DescriptorPool&&) noexcept = default;
  DescriptorPool& operator=(DescriptorPool&&) noexcept = default;
  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;

  /// @brief Allocate one descriptor set with the binding signature of @p
  /// layout.
  /// @param layout  The set layout to allocate against — e.g.
  ///                @ref DescriptorSetLayout::handle, or a
  ///                @ref GraphicsPipeline::descriptor_set_layout. Non-null.
  /// @pre `valid()` and @p layout is non-`VK_NULL_HANDLE`.
  /// @return The set on success, or a non-OK @ref Status — e.g. the pool's
  ///         capacity is exhausted (`VK_ERROR_OUT_OF_POOL_MEMORY`).
  Result<DescriptorSet> allocate(VkDescriptorSetLayout layout);

  /// @return The underlying `VkDescriptorPool` (`VK_NULL_HANDLE` when empty).
  VkDescriptorPool handle() const noexcept { return pool_.get(); }

  /// @return `true` if this owns a pool.
  bool valid() const noexcept { return pool_.valid(); }

 private:
  VkDevice device_ = VK_NULL_HANDLE;  // borrowed; for allocation + set writes
  UniqueHandle<VkDescriptorPool, vkDestroyDescriptorPool> pool_;
};

/// @brief A `VkDescriptorSet` plus the writes that bind resources into it.
///
/// Unlike the handle-owning core types, `DescriptorSet` owns nothing — the
/// @ref DescriptorPool it was allocated from owns its lifetime (freed when the
/// pool is destroyed), so this is a freely-copyable value bundle, like a
/// @ref RenderTarget. Bind it at draw time with `vkCmdBindDescriptorSets` using
/// the pipeline's layout.
///
/// @warning The producing @ref DescriptorPool (and the device it wraps) must
///          outlive every `DescriptorSet` drawn from it and any resource a
///          write points at must outlive the draws that read it.
///
/// @code
/// DescriptorSet set = pool.allocate(layout).value();
/// set.write_uniform_buffer(0, ubo.handle(), 0, sizeof(Mvp));
/// // ... vkCmdBindDescriptorSets(cmd, ..., pipeline.layout(), 0, 1,
/// //                             &set.handle(), 0, nullptr);
/// @endcode
class VG_CORE_API DescriptorSet {
 public:
  /// @brief Construct an empty set (`valid()` is false).
  DescriptorSet() = default;

  /// @brief Adopt a @p set allocated on @p device. Produced by
  ///        @ref DescriptorPool::allocate; rarely constructed directly.
  /// @param device  The device the set's writes update through.
  /// @param set     The allocated `VkDescriptorSet`.
  DescriptorSet(VkDevice device, VkDescriptorSet set) noexcept
      : device_(device), set_(set) {}

  /// @brief Point @p binding at a uniform-buffer range.
  /// @param binding  The binding index within the set.
  /// @param buffer   The buffer to bind.
  /// @param offset   Byte offset into @p buffer.
  /// @param range    Bytes of @p buffer visible to the shader (or
  ///                 `VK_WHOLE_SIZE`).
  /// @pre `valid()`; @p buffer is a uniform-usable buffer that outlives the
  ///      draws reading it.
  void write_uniform_buffer(uint32_t binding, VkBuffer buffer,
                            VkDeviceSize offset, VkDeviceSize range) const;

  /// @brief Point @p binding at a combined image + sampler (a `sampler2D` in
  ///        GLSL): the texture a shader samples.
  /// @param binding  The binding index within the set.
  /// @param view     The sampled image's view.
  /// @param sampler  The sampler that filters @p view.
  /// @param layout   The layout @p view's image is in when sampled (typically
  ///                 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`).
  /// @pre `valid()`; @p view and @p sampler outlive the draws reading them.
  void write_combined_image_sampler(uint32_t binding, VkImageView view,
                                    VkSampler sampler,
                                    VkImageLayout layout) const;

  /// @return The underlying `VkDescriptorSet` (`VK_NULL_HANDLE` when empty).
  VkDescriptorSet handle() const noexcept { return set_; }

  /// @return `true` if this references an allocated set.
  bool valid() const noexcept { return set_ != VK_NULL_HANDLE; }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
};

}  // namespace volumetric_kit::gfx

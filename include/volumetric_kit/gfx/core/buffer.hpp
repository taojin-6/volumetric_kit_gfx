// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file buffer.hpp
/// @brief A `VkBuffer` plus the deleter that frees it and its memory.

#include <functional>

#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/export.hpp"

namespace volumetric_kit::gfx {

/// @brief Owns a `VkBuffer` and runs a deleter that frees it and its backing
/// memory.
///
/// Produced by @ref Allocator::create_buffer (whose deleter calls into VMA);
/// the deleter keeps the allocator detail out of this type's API. A
/// default-constructed `Buffer` is empty (`valid()` is false) and safe to
/// move-assign into.
///
/// @warning The producing @ref Allocator must outlive every `Buffer` it
/// created: the
///          deleter frees through that allocator, so destroying — or
///          move-assigning over — the `Allocator` while a `Buffer` is still
///          alive is undefined behavior. Retire buffers through @ref
///          RetireQueue so their destruction is gated on GPU completion, ahead
///          of allocator teardown.
///
/// @code
/// Result<Buffer> staging = allocator.create_buffer({.size = bytes,
///                                                    .usage =
///                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
///                                                    .memory =
///                                                    MemoryUsage::HostVisible,
///                                                    .mapped = true});
/// if (!staging) return staging.status();
/// std::memcpy(staging.value().mapped(), src, bytes);
/// @endcode
class VG_API Buffer {
 public:
  /// @brief Construct an empty buffer (owns nothing; `valid()` is false).
  Buffer() noexcept = default;

  /// @brief Adopt @p handle and the @p deleter that frees it. Produced by
  ///        @ref Allocator::create_buffer; rarely constructed directly.
  /// @param handle   The buffer to take ownership of.
  /// @param size     Its size in bytes.
  /// @param mapped   Persistent mapping pointer, or `nullptr` if not mapped.
  /// @param deleter  Frees @p handle and its memory; run exactly once on
  /// destruction.
  Buffer(VkBuffer handle, VkDeviceSize size, void* mapped,
         std::function<void()> deleter) noexcept;

  ~Buffer();
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  /// @return The underlying `VkBuffer` (`VK_NULL_HANDLE` when empty).
  VkBuffer handle() const noexcept { return buffer_; }

  /// @return The buffer size in bytes.
  VkDeviceSize size() const noexcept { return size_; }

  /// @return The persistent mapping if created mapped + host-visible, else
  ///         `nullptr`. The mapping is host-coherent, so CPU writes are visible
  ///         to the GPU (and GPU writes to the CPU) without a manual
  ///         flush/invalidate.
  void* mapped() const noexcept { return mapped_; }

  /// @return `true` if this owns a buffer.
  bool valid() const noexcept { return buffer_ != VK_NULL_HANDLE; }

 private:
  void destroy() noexcept;

  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceSize size_ = 0;
  void* mapped_ = nullptr;
  std::function<void()> deleter_;
};

}  // namespace volumetric_kit::gfx

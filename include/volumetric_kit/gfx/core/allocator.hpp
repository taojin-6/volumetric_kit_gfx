// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file allocator.hpp
/// @brief GPU memory allocator: the factory for device-backed resources.

#include <memory>

#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

class Device;

/// @brief Where a resource's memory should live.
enum class MemoryUsage {
  Auto,         ///< Let the allocator choose (device-preferred).
  DeviceLocal,  ///< GPU-only memory; not host-visible.
  HostVisible,  ///< CPU-mappable memory (uploads, readback, uniforms).
};

/// @brief Parameters for @ref Allocator::create_buffer.
struct BufferDesc {
  VkDeviceSize size = 0;         ///< Size in bytes.
  VkBufferUsageFlags usage = 0;  ///< How the buffer will be used.
  MemoryUsage memory = MemoryUsage::Auto;
  bool mapped = false;  ///< Persistently map the memory; requires host-visible
                        ///< memory (not @ref MemoryUsage::DeviceLocal).
  bool exportable =
      false;  ///< Request externally-shareable memory for CUDA<->Vulkan
              ///< interop; not yet wired (currently returns
              ///< VK_ERROR_FEATURE_NOT_PRESENT). The interop tier wires it.
};

/// @brief Parameters for @ref Allocator::create_image.
struct TextureDesc {
  VkExtent2D extent{};                    ///< Width/height in texels.
  VkFormat format = VK_FORMAT_UNDEFINED;  ///< Texel format.
  VkImageUsageFlags usage = 0;            ///< How the image will be used.
  VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
  MemoryUsage memory =
      MemoryUsage::DeviceLocal;  ///< Images default to GPU-only.
  bool exportable =
      false;  ///< Request externally-shareable memory for CUDA<->Vulkan
              ///< interop; not yet wired (currently returns
              ///< VK_ERROR_FEATURE_NOT_PRESENT). The interop tier wires it.
};

/// @brief Wraps the Vulkan Memory Allocator and produces RAII resources from
/// it.
///
/// One allocator per @ref Device. Resources it creates borrow the underlying
/// allocator (and, for images, the device) for their own destruction, so the
/// `Allocator` must outlive every @ref Buffer and @ref Texture it produced.
///
/// @code
/// Result<Allocator> allocator = Allocator::create(instance.handle(), device);
/// if (!allocator) return allocator.status();
/// Result<Buffer> buffer = allocator.value().create_buffer({.size = 256,
///                                                           .usage =
///                                                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
///                                                           .memory =
///                                                           MemoryUsage::HostVisible,
///                                                           .mapped = true});
/// @endcode
class VG_CORE_API Allocator {
 public:
  /// @brief Create an allocator for @p device.
  /// @param instance  The instance @p device belongs to.
  /// @param device    The logical device whose memory this allocates.
  /// @return The allocator on success, or a non-OK @ref Status.
  static Result<Allocator> create(VkInstance instance, const Device& device);

  ~Allocator();
  Allocator(Allocator&& other) noexcept;
  Allocator& operator=(Allocator&& other) noexcept;
  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;

  /// @brief Allocate a buffer.
  /// @param desc  Size, usage, memory residence, and mapping.
  /// @return The buffer on success, or a non-OK @ref Status:
  ///         - `desc.size == 0` or `desc.usage == 0` returns
  ///           `VK_ERROR_INITIALIZATION_FAILED` (both are invalid per the
  ///           spec);
  ///         - `desc.exportable` currently returns
  ///         `VK_ERROR_FEATURE_NOT_PRESENT`
  ///           (the CUDA-interop export wiring is added by the interop tier);
  ///         - `desc.mapped` with `MemoryUsage::DeviceLocal` returns
  ///           `VK_ERROR_INITIALIZATION_FAILED` (mapping needs host-visible
  ///           memory);
  ///         - `desc.mapped` the chosen memory cannot satisfy returns
  ///           `VK_ERROR_MEMORY_MAP_FAILED`.
  ///         On success, a buffer created with `desc.mapped` has a non-null
  ///         @ref Buffer::mapped backed by host-coherent memory, so writes
  ///         through it reach the GPU without a manual flush.
  Result<Buffer> create_buffer(const BufferDesc& desc);

  /// @brief Allocate a 2D image plus a default view over it.
  /// @param desc  Extent, format, usage, tiling, and memory residence.
  /// @return The texture on success, or a non-OK @ref Status:
  ///         - a zero-area `extent`, `usage == 0`, or `VK_FORMAT_UNDEFINED`
  ///           returns `VK_ERROR_INITIALIZATION_FAILED`;
  ///         - `desc.exportable` currently returns
  ///         `VK_ERROR_FEATURE_NOT_PRESENT`. The view's aspect mask is derived
  ///         from the format (depth and/or stencil for depth formats, otherwise
  ///         color).
  Result<Texture> create_image(const TextureDesc& desc);

 private:
  Allocator() noexcept;

  struct Impl;  // hides the VmaAllocator from the public header
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::gfx

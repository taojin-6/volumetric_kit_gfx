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

/// @brief How a resource's memory may be shared with an external API.
///
/// `None` is a plain Vulkan-only resource. `OpaqueFd` reserves externally
/// shareable memory (a POSIX file descriptor) for CUDA<->Vulkan interop on
/// Linux — the desktop compute platform this kit targets (Windows is out of
/// scope). External sharing is not yet wired: any value other than `None`
/// currently returns @ref Status::Code::Unsupported. The interop tier wires it.
enum class ExternalHandleType {
  None,      ///< Vulkan-only; not shareable.
  OpaqueFd,  ///< POSIX file descriptor (Linux; CUDA interop).
};

/// @brief Parameters for @ref Allocator::create_buffer.
struct BufferDesc {
  VkDeviceSize size = 0;         ///< Size in bytes.
  VkBufferUsageFlags usage = 0;  ///< How the buffer will be used.
  MemoryUsage memory = MemoryUsage::Auto;
  bool mapped = false;  ///< Persistently map the memory; requires host-visible
                        ///< memory (not @ref MemoryUsage::DeviceLocal).
  /// Export the memory for external-API interop (see @ref ExternalHandleType).
  ExternalHandleType external = ExternalHandleType::None;
};

/// @brief Parameters for @ref Allocator::create_image.
///
/// The defaults describe a single-mip, single-layer, single-sample 2D image.
/// Set `type` + `depth` for a 3D (volume) image, `array_layers` > 1 for an
/// array (the default view becomes the matching array view), `mip_levels` for a
/// mip chain, and `samples` for multisampling.
struct TextureDesc {
  VkExtent2D extent{};  ///< Width/height in texels.
  uint32_t depth = 1;   ///< Depth in texels; > 1 requires `VK_IMAGE_TYPE_3D`.
  VkFormat format = VK_FORMAT_UNDEFINED;  ///< Texel format.
  VkImageUsageFlags usage = 0;            ///< How the image will be used.
  VkImageType type = VK_IMAGE_TYPE_2D;    ///< 1D / 2D / 3D image.
  uint32_t mip_levels = 1;                ///< Number of mip levels.
  uint32_t array_layers = 1;  ///< Array layers; > 1 yields an array view and
                              ///< must be 1 for a 3D image.
  VkSampleCountFlagBits samples =
      VK_SAMPLE_COUNT_1_BIT;  ///< MSAA sample count.
  VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
  MemoryUsage memory =
      MemoryUsage::DeviceLocal;  ///< Images default to GPU-only.
  /// Export the memory for external-API interop (see @ref ExternalHandleType).
  ExternalHandleType external = ExternalHandleType::None;
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
  ///         - `desc.size == 0`, `desc.usage == 0`, `desc.mapped` with
  ///           `MemoryUsage::DeviceLocal`, or `MemoryUsage::HostVisible`
  ///           without `desc.mapped` (no separate map() accessor exists) return
  ///           domain
  ///           @ref Status::Code::InvalidArgument (malformed/contradictory
  ///           arguments);
  ///         - `desc.external != None` returns @ref Status::Code::Unsupported
  ///           (the CUDA-interop export wiring is added by the interop tier);
  ///         - a failed allocation, or a chosen memory that cannot satisfy
  ///           `desc.mapped`, returns a Vulkan-domain @ref Status carrying the
  ///           `VkResult` (e.g. `VK_ERROR_MEMORY_MAP_FAILED`).
  ///         On success, a buffer created with `desc.mapped` has a non-null
  ///         @ref Buffer::mapped backed by host-coherent memory, so writes
  ///         through it reach the GPU without a manual flush.
  Result<Buffer> create_buffer(const BufferDesc& desc);

  /// @brief Allocate an image (1D / 2D / 3D, mipped, arrayed, multisampled per
  ///        @p desc) plus a default view over it.
  /// @param desc  Extent/depth, type, format, usage, mip/array/sample counts,
  ///              tiling, and memory residence.
  /// @return The texture on success, or a non-OK @ref Status:
  ///         - a zero `extent`/`depth`/`mip_levels`/`array_layers`, `usage ==
  ///         0`,
  ///           `VK_FORMAT_UNDEFINED`, `depth > 1` without `VK_IMAGE_TYPE_3D`, a
  ///           3D image with `array_layers > 1`, or `MemoryUsage::HostVisible`
  ///           (images have no host accessor; copy to a HostVisible buffer for
  ///           readback) return domain @ref Status::Code::InvalidArgument;
  ///         - `desc.external != None` returns @ref Status::Code::Unsupported;
  ///         - a failed image, allocation, or view creation returns a
  ///           Vulkan-domain @ref Status carrying the `VkResult`.
  ///         The default view spans all mips/layers; its type follows
  ///         `desc.type` and `array_layers` (1D/2D/3D, with the `_ARRAY`
  ///         variant when `array_layers > 1`), and its aspect follows the
  ///         format: DEPTH for depth and combined depth/stencil formats,
  ///         STENCIL for stencil-only, otherwise COLOR.
  Result<Texture> create_image(const TextureDesc& desc);

 private:
  Allocator() noexcept;

  struct Impl;  // hides the VmaAllocator from the public header
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::gfx

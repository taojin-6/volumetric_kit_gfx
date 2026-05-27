// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/allocator.hpp"

#include <utility>

#include <vk_mem_alloc.h>

#include "volumetric_kit/gfx/core/device.hpp"

namespace volumetric_kit::gfx {
namespace {

// The view aspect a format implies: depth and/or stencil for depth/stencil
// formats, else color.
VkImageAspectFlags aspect_mask_for(VkFormat format) {
  switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
      return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_S8_UINT:
      return VK_IMAGE_ASPECT_STENCIL_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
      return VK_IMAGE_ASPECT_COLOR_BIT;
  }
}

// MemoryUsage -> VMA residency preference. The HostVisible host-access flag is
// applied at the call site (it differs between buffers and images), so this
// maps the residency only.
VmaMemoryUsage vma_memory_usage(MemoryUsage memory) {
  switch (memory) {
    case MemoryUsage::Auto:
      return VMA_MEMORY_USAGE_AUTO;
    case MemoryUsage::DeviceLocal:
      return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case MemoryUsage::HostVisible:
      return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  }
  return VMA_MEMORY_USAGE_AUTO;  // unreachable; satisfies -Wreturn-type
}

}  // namespace

struct Allocator::Impl {
  VmaAllocator allocator = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;  // borrowed; for image-view create/destroy

  // Own the handle here, not in ~Allocator: the defaulted move-assignment
  // destroys the overwritten Impl via unique_ptr, so freeing in ~Impl is what
  // keeps `a = std::move(b)` from leaking a's previous allocator.
  ~Impl() {
    if (allocator != VK_NULL_HANDLE) {
      vmaDestroyAllocator(allocator);
    }
  }
};

Allocator::Allocator() noexcept = default;

Result<Allocator> Allocator::create(VkInstance instance, const Device& device) {
  // Feed VMA the loader entry points; we link them statically (see
  // vma_impl.cpp).
  VmaVulkanFunctions functions{};
  functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  // VMA must not be told a higher Vulkan version than the device actually
  // supports, or it calls core 1.1 entry points
  // (vkGetBufferMemoryRequirements2, vkBindBufferMemory2, …) on a 1.0 device
  // that does not provide them. Cap our 1.1 floor at the device's reported
  // apiVersion.
  // TODO: bump to the negotiated 1.2/1.3 version once those device features
  // land (2b-5 / 2c).
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);

  VmaAllocatorCreateInfo info{};
  info.instance = instance;
  info.physicalDevice = device.physical_device();
  info.device = device.handle();
  info.vulkanApiVersion = props.apiVersion >= VK_API_VERSION_1_1
                              ? VK_API_VERSION_1_1
                              : VK_API_VERSION_1_0;
  info.pVulkanFunctions = &functions;

  auto impl = std::make_unique<Impl>();
  VG_VK_TRY(vmaCreateAllocator(&info, &impl->allocator));
  impl->device = device.handle();

  Allocator allocator;
  allocator.impl_ = std::move(impl);
  return allocator;
}

Result<Buffer> Allocator::create_buffer(const BufferDesc& desc) {
  if (desc.size == 0) {
    return Status::error(VK_ERROR_INITIALIZATION_FAILED,
                         "buffer size must be non-zero");
  }
  if (desc.usage == 0) {
    return Status::error(
        VK_ERROR_INITIALIZATION_FAILED,
        "buffer usage must name at least one VkBufferUsageFlagBit");
  }
  if (desc.exportable) {
    // TODO: wire VkExportMemoryAllocateInfo + a VMA export pool in the interop
    // PR.
    return Status::error(VK_ERROR_FEATURE_NOT_PRESENT,
                         "exportable buffers are not yet supported");
  }
  if (desc.mapped && desc.memory == MemoryUsage::DeviceLocal) {
    // A persistent mapping needs host-visible memory; DeviceLocal asks for the
    // opposite. Reject rather than silently demote the residency.
    return Status::error(VK_ERROR_INITIALIZATION_FAILED,
                         "mapped buffers need host-visible memory; DeviceLocal "
                         "cannot be persistently mapped");
  }

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = desc.size;
  buffer_info.usage = desc.usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = vma_memory_usage(desc.memory);
  if (desc.memory == MemoryUsage::HostVisible) {
    alloc_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  }
  if (desc.mapped) {
    // Mapping needs host-visible memory; request host access + a persistent
    // map. Require HOST_COHERENT so writes through mapped() reach the GPU
    // without an explicit flush — Buffer hides the VmaAllocation, so callers
    // have no flush path. The spec guarantees a HOST_VISIBLE|HOST_COHERENT
    // memory type exists, so this requirement is always satisfiable.
    alloc_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    alloc_info.requiredFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VmaAllocationInfo result_info{};
  VG_VK_TRY(vmaCreateBuffer(impl_->allocator, &buffer_info, &alloc_info,
                            &buffer, &allocation, &result_info));

  void* mapped = desc.mapped ? result_info.pMappedData : nullptr;
  if (desc.mapped && mapped == nullptr) {
    // VMA made the buffer but couldn't satisfy the persistent mapping (the
    // chosen memory isn't host-visible). Don't return an ok() buffer whose
    // mapped() is null — free it and report the failure.
    vmaDestroyBuffer(impl_->allocator, buffer, allocation);
    return Status::error(VK_ERROR_MEMORY_MAP_FAILED,
                         "requested a persistent mapping but the allocation is "
                         "not host-visible");
  }

  // The deleter captures the (opaque to callers) VMA handles, keeping VMA out
  // of Buffer's API. Valid only while this allocator lives (see Buffer's
  // @warning).
  VmaAllocator allocator = impl_->allocator;
  return Buffer(buffer, desc.size, mapped, [allocator, buffer, allocation]() {
    vmaDestroyBuffer(allocator, buffer, allocation);
  });
}

Result<Texture> Allocator::create_image(const TextureDesc& desc) {
  if (desc.extent.width == 0 || desc.extent.height == 0) {
    return Status::error(VK_ERROR_INITIALIZATION_FAILED,
                         "image extent must be non-zero");
  }
  if (desc.usage == 0) {
    return Status::error(
        VK_ERROR_INITIALIZATION_FAILED,
        "image usage must name at least one VkImageUsageFlagBit");
  }
  if (desc.format == VK_FORMAT_UNDEFINED) {
    return Status::error(VK_ERROR_INITIALIZATION_FAILED,
                         "image format must not be UNDEFINED");
  }
  if (desc.exportable) {
    // TODO: wire VkExternalMemoryImageCreateInfo + a VMA export pool in the
    // interop PR.
    return Status::error(VK_ERROR_FEATURE_NOT_PRESENT,
                         "exportable images are not yet supported");
  }

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = desc.format;
  image_info.extent = {desc.extent.width, desc.extent.height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = desc.tiling;
  image_info.usage = desc.usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = vma_memory_usage(desc.memory);

  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VG_VK_TRY(vmaCreateImage(impl_->allocator, &image_info, &alloc_info, &image,
                           &allocation, nullptr));

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = desc.format;
  view_info.subresourceRange.aspectMask = aspect_mask_for(desc.format);
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  VkImageView view = VK_NULL_HANDLE;
  VkResult view_result =
      vkCreateImageView(impl_->device, &view_info, nullptr, &view);
  if (view_result != VK_SUCCESS) {
    vmaDestroyImage(impl_->allocator, image, allocation);
    return vk_error(view_result, "vkCreateImageView");
  }

  // Deleter destroys the view (which references the image) before the image,
  // and hides device/VMA from Texture's API. Valid only while this allocator
  // lives.
  VmaAllocator allocator = impl_->allocator;
  VkDevice device = impl_->device;
  return Texture(image, view, desc.extent, desc.format,
                 [device, view, allocator, image, allocation]() {
                   vkDestroyImageView(device, view, nullptr);
                   vmaDestroyImage(allocator, image, allocation);
                 });
}

Allocator::Allocator(Allocator&& other) noexcept = default;

Allocator& Allocator::operator=(Allocator&& other) noexcept = default;

// Destruction (and the defaulted move ctor/assign above) flow through
// unique_ptr<Impl>, whose deleter runs ~Impl -> vmaDestroyAllocator exactly
// once; a moved-from Allocator holds a null impl_ and frees nothing.
Allocator::~Allocator() = default;

}  // namespace volumetric_kit::gfx

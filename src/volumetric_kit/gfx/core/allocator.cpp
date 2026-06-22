// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/allocator.hpp"

#include <algorithm>
#include <utility>

#include <vk_mem_alloc.h>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/impl/vk_format.hpp"

namespace volumetric_kit::gfx {
namespace {

// The default view type for an image of @p type with @p array_layers layers:
// 1D/2D gain their _ARRAY variant when arrayed; 3D images are never arrayed.
VkImageViewType view_type_for(VkImageType type, uint32_t array_layers) {
  const bool arrayed = array_layers > 1;
  switch (type) {
    case VK_IMAGE_TYPE_1D:
      return arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case VK_IMAGE_TYPE_3D:
      return VK_IMAGE_VIEW_TYPE_3D;
    default:  // VK_IMAGE_TYPE_2D
      return arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
  }
}

// MemoryUsage -> VMA residency preference. The host-access flag (set only for
// mapped buffers) is applied at the call site, so this maps the residency only.
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
  if (instance == VK_NULL_HANDLE || device.handle() == VK_NULL_HANDLE ||
      device.physical_device() == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "Allocator::create: instance and device must be non-null (a "
        "moved-from Device has null handles)");
  }
  // Feed VMA the loader entry points; we link them statically (see
  // vma_impl.cpp).
  VmaVulkanFunctions functions{};
  functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  // VMA must not be told a higher Vulkan version than BOTH the instance was
  // created with and the device supports, or it calls core 1.1 entry points
  // (vkGetBufferMemoryRequirements2, vkBindBufferMemory2, …) the
  // instance/device never loaded. The instance is created at min(1.3, loader)
  // (see instance.cpp), so reconstruct that and take the min with the device's
  // apiVersion. Cap at the 1.1 floor VMA currently needs.
  // TODO: bump the 1.1 cap to the negotiated 1.2/1.3 version once those device
  // features land.
  uint32_t instance_version = VK_API_VERSION_1_0;
  if (vkEnumerateInstanceVersion(&instance_version) != VK_SUCCESS) {
    instance_version = VK_API_VERSION_1_0;
  }
  instance_version =
      std::min(instance_version, static_cast<uint32_t>(VK_API_VERSION_1_3));
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  const uint32_t effective = std::min(instance_version, props.apiVersion);

  VmaAllocatorCreateInfo info{};
  info.instance = instance;
  info.physicalDevice = device.physical_device();
  info.device = device.handle();
  info.vulkanApiVersion =
      effective >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_1 : VK_API_VERSION_1_0;
  info.pVulkanFunctions = &functions;

  auto impl = std::make_unique<Impl>();
  VG_VK_TRY(vmaCreateAllocator(&info, &impl->allocator));
  impl->device = device.handle();

  Allocator allocator;
  allocator.impl_ = std::move(impl);
  return allocator;
}

Result<Buffer> Allocator::create_buffer(const BufferDesc& desc) {
  if (impl_ == nullptr) {
    return Status::invalid_argument(
        "create_buffer: allocator is empty (moved-from)");
  }
  if (desc.size == 0) {
    return Status::invalid_argument("buffer size must be non-zero");
  }
  if (desc.usage == 0) {
    return Status::invalid_argument(
        "buffer usage must name at least one VkBufferUsageFlagBit");
  }
  if (desc.external != ExternalHandleType::None) {
    // TODO: wire VkExportMemoryAllocateInfo + a VMA export pool in the interop
    // tier (honoring the requested handle type).
    return Status::unsupported("external-memory buffers are not yet supported");
  }
  if (desc.mapped && desc.memory == MemoryUsage::DeviceLocal) {
    // A persistent mapping needs host-visible memory; DeviceLocal asks for the
    // opposite. Reject rather than silently demote the residency.
    return Status::invalid_argument(
        "mapped buffers need host-visible memory; DeviceLocal cannot be "
        "persistently mapped");
  }
  if (desc.memory == MemoryUsage::HostVisible && !desc.mapped) {
    // Buffer exposes no separate map()/unmap(); host-visible memory is only
    // reachable through the persistent mapping. Without `mapped`, a HostVisible
    // buffer is unreachable from the host — a dead end. Require mapped=true, or
    // use DeviceLocal/Auto for GPU-only memory.
    return Status::invalid_argument(
        "HostVisible buffers require mapped=true (Buffer has no separate map() "
        "accessor); use DeviceLocal or Auto for GPU-only memory");
  }

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = desc.size;
  buffer_info.usage = desc.usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = vma_memory_usage(desc.memory);
  if (desc.mapped) {
    // Mapping needs host-visible memory; request host access + a persistent
    // map. Require HOST_COHERENT so writes through mapped() reach the GPU
    // without an explicit flush — Buffer hides the VmaAllocation, so callers
    // have no flush path. The spec guarantees a HOST_VISIBLE|HOST_COHERENT
    // memory type exists, so this requirement is always satisfiable. The
    // host-access flag follows desc.host_access: SEQUENTIAL_WRITE lets VMA pick
    // write-combined memory for streaming uploads; RANDOM keeps the mapping
    // readable. (A HostVisible buffer without mapped is rejected above, so this
    // is the only path that needs a host-access flag.)
    alloc_info.flags |=
        VMA_ALLOCATION_CREATE_MAPPED_BIT |
        (desc.host_access == HostAccess::SequentialWrite
             ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
             : VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
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
  if (impl_ == nullptr) {
    return Status::invalid_argument(
        "create_image: allocator is empty (moved-from)");
  }
  if (desc.extent.width == 0 || desc.extent.height == 0) {
    return Status::invalid_argument("image extent must be non-zero");
  }
  if (desc.usage == 0) {
    return Status::invalid_argument(
        "image usage must name at least one VkImageUsageFlagBit");
  }
  if (desc.format == VK_FORMAT_UNDEFINED) {
    return Status::invalid_argument("image format must not be UNDEFINED");
  }
  if (desc.depth == 0 || desc.mip_levels == 0 || desc.array_layers == 0) {
    return Status::invalid_argument(
        "image depth, mip_levels, and array_layers must each be non-zero");
  }
  if (desc.depth > 1 && desc.type != VK_IMAGE_TYPE_3D) {
    return Status::invalid_argument(
        "image depth > 1 requires VK_IMAGE_TYPE_3D");
  }
  if (desc.type == VK_IMAGE_TYPE_3D && desc.array_layers != 1) {
    return Status::invalid_argument("3D images cannot be arrayed");
  }
  if (desc.external != ExternalHandleType::None) {
    // TODO: wire VkExternalMemoryImageCreateInfo + a VMA export pool in the
    // interop tier (honoring the requested handle type).
    return Status::unsupported("external-memory images are not yet supported");
  }
  if (desc.memory == MemoryUsage::HostVisible) {
    // Texture exposes no host accessor and this path sets no host-access flags,
    // so a host-visible image would be unmappable — a dead end. Readback goes
    // image -> host-visible Buffer via vkCmdCopyImageToBuffer; keep images
    // DeviceLocal (the default) or Auto.
    return Status::invalid_argument(
        "host-visible images are not supported (no host accessor); copy to a "
        "HostVisible buffer for readback");
  }
  if (desc.samples != VK_SAMPLE_COUNT_1_BIT &&
      (desc.type != VK_IMAGE_TYPE_2D ||
       desc.tiling != VK_IMAGE_TILING_OPTIMAL || desc.mip_levels != 1)) {
    // Vulkan permits multisampling only on single-mip, optimal-tiling 2D images
    // (VUID-VkImageCreateInfo-samples-02257/02258). Reject the contradiction
    // here so it reads as a domain error like every other malformed desc,
    // instead of an opaque VkResult out of vmaCreateImage.
    return Status::invalid_argument(
        "multisampled images must be 2D, optimal-tiling, and single-mip");
  }
  // A view is creatable only over a usage that names a view-compatible bit
  // (VUID-VkImageViewCreateInfo-image-04441). Reject a with_view/usage
  // contradiction up front; a transfer-only image must set with_view = false.
  constexpr VkImageUsageFlags kViewCompatibleUsage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
  if (desc.with_view && (desc.usage & kViewCompatibleUsage) == 0) {
    return Status::invalid_argument(
        "with_view image usage must name a view-compatible bit (SAMPLED, "
        "STORAGE, COLOR_ATTACHMENT, DEPTH_STENCIL_ATTACHMENT, or "
        "INPUT_ATTACHMENT); set with_view = false for a transfer-only image");
  }

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = desc.type;
  image_info.format = desc.format;
  image_info.extent = {desc.extent.width, desc.extent.height, desc.depth};
  image_info.mipLevels = desc.mip_levels;
  image_info.arrayLayers = desc.array_layers;
  image_info.samples = desc.samples;
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

  VkImageView view = VK_NULL_HANDLE;
  if (desc.with_view) {
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = view_type_for(desc.type, desc.array_layers);
    view_info.format = desc.format;
    view_info.subresourceRange.aspectMask = aspect_mask_for(desc.format);
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = desc.mip_levels;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = desc.array_layers;

    VkResult view_result =
        vkCreateImageView(impl_->device, &view_info, nullptr, &view);
    if (view_result != VK_SUCCESS) {
      vmaDestroyImage(impl_->allocator, image, allocation);
      return vk_error(view_result, "vkCreateImageView");
    }
  }

  // Deleter destroys the view (which references the image) before the image,
  // and hides device/VMA from Texture's API. Valid only while this allocator
  // lives. A viewless image (view == VK_NULL_HANDLE) skips the view destroy.
  VmaAllocator allocator = impl_->allocator;
  VkDevice device = impl_->device;
  return Texture(image, view, desc.extent, desc.depth, desc.mip_levels,
                 desc.format, [device, view, allocator, image, allocation]() {
                   if (view != VK_NULL_HANDLE) {
                     vkDestroyImageView(device, view, nullptr);
                   }
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

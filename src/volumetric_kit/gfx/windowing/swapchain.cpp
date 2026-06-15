// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/windowing/swapchain.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/check.hpp"
#include "volumetric_kit/gfx/core/device.hpp"

namespace volumetric_kit::gfx::windowing {

Result<Swapchain> Swapchain::create(const Device& device, VkSurfaceKHR surface,
                                    const SwapchainConfig& config) {
  if (surface == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "Swapchain::create: surface must be non-null");
  }
  if (!device.has_present()) {
    return Status::invalid_argument(
        "Swapchain::create: device has no present queue (set "
        "DeviceConfig::needs_present)");
  }
  Swapchain sc;
  sc.device_ = &device;
  sc.surface_ = surface;
  sc.requested_min_image_count_ = config.min_image_count;
  VG_TRY(sc.select_surface_properties(config));
  VG_TRY(sc.build(config.extent));
  return sc;
}

Status Swapchain::select_surface_properties(const SwapchainConfig& config) {
  VkPhysicalDevice phys = device_->physical_device();

  uint32_t format_count = 0;
  VG_VK_TRY(vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &format_count,
                                                 nullptr));
  if (format_count == 0) {
    return Status::unsupported("Swapchain: surface reports no formats");
  }
  std::vector<VkSurfaceFormatKHR> formats(format_count);
  VG_VK_TRY(vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &format_count,
                                                 formats.data()));
  // Prefer the requested format + color space; else take the first supported.
  VkSurfaceFormatKHR chosen = formats[0];
  for (const VkSurfaceFormatKHR& f : formats) {
    if (f.format == config.preferred_format &&
        f.colorSpace == config.preferred_color_space) {
      chosen = f;
      break;
    }
  }
  format_ = chosen.format;
  color_space_ = chosen.colorSpace;

  uint32_t mode_count = 0;
  VG_VK_TRY(vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface_,
                                                      &mode_count, nullptr));
  if (mode_count == 0) {
    return Status::unsupported("Swapchain: surface reports no present modes");
  }
  std::vector<VkPresentModeKHR> modes(mode_count);
  VG_VK_TRY(vkGetPhysicalDeviceSurfacePresentModesKHR(
      phys, surface_, &mode_count, modes.data()));
  // FIFO is guaranteed; upgrade to the preferred mode only if offered.
  present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  for (VkPresentModeKHR m : modes) {
    if (m == config.preferred_present_mode) {
      present_mode_ = m;
      break;
    }
  }
  return Status{};
}

Status Swapchain::build(VkExtent2D desired) {
  VkPhysicalDevice phys = device_->physical_device();
  VkDevice dev = device_->handle();

  VkSurfaceCapabilitiesKHR caps{};
  VG_VK_TRY(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface_, &caps));

  // Honor the surface's fixed currentExtent when a window manager dictates one;
  // otherwise clamp the desired size to the supported range.
  VkExtent2D extent = desired;
  if (caps.currentExtent.width != UINT32_MAX) {
    extent = caps.currentExtent;
  } else {
    extent.width = std::clamp(extent.width, caps.minImageExtent.width,
                              caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height,
                               caps.maxImageExtent.height);
  }
  if (extent.width == 0 || extent.height == 0) {
    // A minimized window reports a zero extent; nothing to build until
    // restored.
    return Status::invalid_argument(
        "Swapchain::build: surface extent is zero (window minimized?)");
  }

  uint32_t image_count = requested_min_image_count_ != 0
                             ? requested_min_image_count_
                             : caps.minImageCount + 1;
  image_count = std::max(image_count, caps.minImageCount);
  if (caps.maxImageCount != 0) {
    image_count = std::min(image_count, caps.maxImageCount);
  }

  VkSwapchainCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  info.surface = surface_;
  info.minImageCount = image_count;
  info.imageFormat = format_;
  info.imageColorSpace = color_space_;
  info.imageExtent = extent;
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  const uint32_t graphics = device_->graphics_family();
  const uint32_t present = device_->present_family();
  const std::array<uint32_t, 2> families = {graphics, present};
  if (graphics != present) {
    // Images are read by present and written by graphics on different families.
    info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    info.queueFamilyIndexCount = 2;
    info.pQueueFamilyIndices = families.data();
  } else {
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  info.preTransform = caps.currentTransform;
  info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  info.presentMode = present_mode_;
  info.clipped = VK_TRUE;
  info.oldSwapchain = VK_NULL_HANDLE;

  VG_VK_TRY(vkCreateSwapchainKHR(dev, &info, nullptr, &swapchain_));
  extent_ = extent;

  uint32_t count = 0;
  VG_VK_TRY(vkGetSwapchainImagesKHR(dev, swapchain_, &count, nullptr));
  images_.resize(count);
  VG_VK_TRY(vkGetSwapchainImagesKHR(dev, swapchain_, &count, images_.data()));

  views_.reserve(count);
  targets_.reserve(count);
  for (VkImage image : images_) {
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format_;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    VG_VK_TRY(vkCreateImageView(dev, &view_info, nullptr, &view));
    views_.push_back(view);

    const RenderTargetAttachment attachment{image, view, format_};
    targets_.emplace_back(extent_, &attachment, 1, VK_SAMPLE_COUNT_1_BIT);
  }
  return Status{};
}

Result<uint32_t> Swapchain::acquire_next_image(VkSemaphore image_available,
                                               uint64_t timeout_ns) {
  uint32_t index = 0;
  // vkAcquireNextImageKHR has several success codes, so it is checked by hand
  // (VG_VK_TRY would treat SUBOPTIMAL as a failure). SUBOPTIMAL still yields a
  // usable image; OUT_OF_DATE returns its code for the caller to recreate.
  const VkResult r =
      vkAcquireNextImageKHR(device_->handle(), swapchain_, timeout_ns,
                            image_available, VK_NULL_HANDLE, &index);
  if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
    return index;
  }
  return vk_error(r, "vkAcquireNextImageKHR");
}

Status Swapchain::present(uint32_t image_index, VkSemaphore render_finished) {
  VkPresentInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &render_finished;
  info.swapchainCount = 1;
  info.pSwapchains = &swapchain_;
  info.pImageIndices = &image_index;
  const VkResult r = vkQueuePresentKHR(device_->present_queue(), &info);
  if (r == VK_SUCCESS) {
    return Status{};
  }
  // SUBOPTIMAL / OUT_OF_DATE flow back so the caller can recreate.
  return vk_error(r, "vkQueuePresentKHR");
}

Status Swapchain::recreate(VkExtent2D extent) {
  vkDeviceWaitIdle(device_->handle());
  destroy_resources();
  return build(extent);
}

const RenderTarget& Swapchain::render_target(uint32_t image_index) const {
  VG_CHECK(image_index < targets_.size(),
           "Swapchain::render_target: image_index out of range");
  return targets_[image_index];
}

VkImage Swapchain::image(uint32_t image_index) const {
  VG_CHECK(image_index < images_.size(),
           "Swapchain::image: image_index out of range");
  return images_[image_index];
}

RenderTargetLayout Swapchain::layout() const noexcept {
  RenderTargetLayout layout;
  layout.color_formats[0] = format_;
  layout.color_count = 1;
  layout.samples = VK_SAMPLE_COUNT_1_BIT;
  return layout;
}

void Swapchain::destroy_resources() noexcept {
  if (device_ == nullptr) {
    return;
  }
  VkDevice dev = device_->handle();
  targets_.clear();
  for (VkImageView view : views_) {
    vkDestroyImageView(dev, view, nullptr);
  }
  views_.clear();
  images_.clear();  // owned by the swapchain object below; just drop our copies
  if (swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(dev, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }
}

void Swapchain::destroy() noexcept {
  destroy_resources();
  device_ = nullptr;
  surface_ = VK_NULL_HANDLE;
  extent_ = VkExtent2D{};
}

Swapchain::~Swapchain() { destroy(); }

Swapchain::Swapchain(Swapchain&& other) noexcept
    : device_(other.device_),
      surface_(other.surface_),
      swapchain_(other.swapchain_),
      images_(std::move(other.images_)),
      views_(std::move(other.views_)),
      targets_(std::move(other.targets_)),
      format_(other.format_),
      color_space_(other.color_space_),
      present_mode_(other.present_mode_),
      extent_(other.extent_),
      requested_min_image_count_(other.requested_min_image_count_) {
  other.device_ = nullptr;
  other.surface_ = VK_NULL_HANDLE;
  other.swapchain_ = VK_NULL_HANDLE;
  other.extent_ = VkExtent2D{};
}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept {
  if (this != &other) {
    destroy();
    device_ = other.device_;
    surface_ = other.surface_;
    swapchain_ = other.swapchain_;
    images_ = std::move(other.images_);
    views_ = std::move(other.views_);
    targets_ = std::move(other.targets_);
    format_ = other.format_;
    color_space_ = other.color_space_;
    present_mode_ = other.present_mode_;
    extent_ = other.extent_;
    requested_min_image_count_ = other.requested_min_image_count_;
    other.device_ = nullptr;
    other.surface_ = VK_NULL_HANDLE;
    other.swapchain_ = VK_NULL_HANDLE;
    other.extent_ = VkExtent2D{};
  }
  return *this;
}

}  // namespace volumetric_kit::gfx::windowing

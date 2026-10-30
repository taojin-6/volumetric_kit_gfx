// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/windowing/swapchain.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/core/impl/color_attachment.hpp"
#include "volumetric_kit/gfx/core/impl/depth_attachment.hpp"
#include "volumetric_kit/gfx/core/impl/vk_query.hpp"

namespace volumetric_kit::gfx::windowing {

core::Result<Swapchain> Swapchain::create(const core::Device& device,
                                          VkSurfaceKHR surface,
                                          const SwapchainConfig& config,
                                          core::Allocator& allocator) {
  if (surface == VK_NULL_HANDLE) {
    return core::Status::invalid_argument(
        "Swapchain::create: surface must be non-null");
  }
  if (!device.has_present()) {
    return core::Status::invalid_argument(
        "Swapchain::create: device has no present queue (set "
        "DeviceRequirements::needs_present)");
  }
  // A device made for another library may lack the renderer's floor, which
  // every frame recorded into this swapchain's images relies on.
  VKC_TRY(device.check_enabled(device_requirements())
              .with_context("Swapchain::create"));
  const bool has_depth = config.depth_format != VK_FORMAT_UNDEFINED;
  if (has_depth) {
    // Depth-only format check (shared with OffscreenTarget); the device-support
    // check below is swapchain-specific (it has the Device's caps to hand).
    VKC_TRY(
        validate_depth_only_format(config.depth_format, "Swapchain::create"));
    if (!device.caps().format_supports(
            config.depth_format, VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
      return core::Status::unsupported(
          "Swapchain::create: depth_format has no optimal-tiling depth-stencil "
          "attachment support on this device");
    }
  }
  // Vulkan requires kWindowSamples in both limits; a portability driver that
  // reports less is refused here rather than failing at its first pipeline.
  const VkPhysicalDeviceLimits& limits = device.caps().limits();
  if ((limits.framebufferColorSampleCounts & kWindowSamples) == 0 ||
      (has_depth &&
       (limits.framebufferDepthSampleCounts & kWindowSamples) == 0)) {
    return core::Status::unsupported(
        "Swapchain::create: the device cannot render kWindowSamples samples "
        "into a framebuffer");
  }
  Swapchain sc;
  sc.device_ = &device;
  sc.surface_ = surface;
  sc.allocator_ = &allocator;
  sc.depth_format_ = config.depth_format;
  sc.requested_min_image_count_ = config.min_image_count;
  VKC_TRY(sc.select_surface_properties(config));
  VKC_TRY(sc.build(config.extent));
  return sc;
}

core::Status Swapchain::select_surface_properties(
    const SwapchainConfig& config) {
  VkPhysicalDevice phys = device_->physical_device();

  // Enumerate via the shared vk_query idiom (count, then fill; VK_INCOMPLETE
  // tolerated) so this stays in lockstep with the instance/device enumerators.
  VKC_ASSIGN(std::vector<VkSurfaceFormatKHR> formats,
             surface_formats(phys, surface_));
  if (formats.empty()) {
    return core::Status::unsupported("Swapchain: surface reports no formats");
  }
  // Prefer the requested format + color space; else an _SRGB format in that
  // color space, which encodes on write as the default preference does (an
  // Android surface lists R8G8B8A8_UNORM first and offers no B8G8R8A8_SRGB);
  // else take the first supported.
  auto it = std::find_if(formats.begin(), formats.end(),
                         [&](const VkSurfaceFormatKHR& f) {
                           return f.format == config.preferred_format &&
                                  f.colorSpace == config.preferred_color_space;
                         });
  if (it == formats.end()) {
    it = std::find_if(formats.begin(), formats.end(),
                      [&](const VkSurfaceFormatKHR& f) {
                        return is_srgb_attachment(f.format) &&
                               f.colorSpace == config.preferred_color_space;
                      });
  }
  const VkSurfaceFormatKHR chosen = it != formats.end() ? *it : formats[0];
  format_ = chosen.format;
  color_space_ = chosen.colorSpace;

  VKC_ASSIGN(std::vector<VkPresentModeKHR> modes,
             surface_present_modes(phys, surface_));
  if (modes.empty()) {
    return core::Status::unsupported(
        "Swapchain: surface reports no present modes");
  }
  // FIFO is guaranteed; upgrade to the preferred mode only if offered.
  present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  for (VkPresentModeKHR m : modes) {
    if (m == config.preferred_present_mode) {
      present_mode_ = m;
      break;
    }
  }
  return core::Status{};
}

core::Status Swapchain::build(VkExtent2D desired) {
  VkPhysicalDevice phys = device_->physical_device();
  VkDevice dev = device_->handle();

  VkSurfaceCapabilitiesKHR caps{};
  VKC_VK_TRY(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface_, &caps));

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
    return core::Status::invalid_argument(
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

  const uint32_t graphics = device_->queue_family();
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
  // Prefer an identity transform so the compositor resolves any device
  // rotation; fall back to the surface's current transform when identity is
  // unsupported.
  // TODO: a rotation-aware present path would pre-rotate and swap the extent
  // for 90/270 transforms (Android) instead of leaning on the compositor.
  info.preTransform =
      (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
          : caps.currentTransform;
  // OPAQUE is the common case but not guaranteed (some Android/Wayland surfaces
  // expose only INHERIT/PRE_MULTIPLIED); take the first supported mode in
  // preference order. supportedCompositeAlpha always has at least one bit set.
  const VkCompositeAlphaFlagBitsKHR alpha_prefs[] = {
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
      VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
  };
  info.compositeAlpha = alpha_prefs[0];
  for (VkCompositeAlphaFlagBitsKHR a : alpha_prefs) {
    if (caps.supportedCompositeAlpha & a) {
      info.compositeAlpha = a;
      break;
    }
  }
  info.presentMode = present_mode_;
  info.clipped = VK_TRUE;
  // Hand the driver the current swapchain (if any): it can carry resources
  // across a resize, and creating the replacement *before* destroying anything
  // keeps this object presentable when the rebuild fails early (the zero-extent
  // return above leaves the old chain untouched).
  info.oldSwapchain = swapchain_;

  VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
  const VkResult created =
      vkCreateSwapchainKHR(dev, &info, nullptr, &new_swapchain);
  if (created != VK_SUCCESS) {
    // Passing oldSwapchain retires it even when creation fails, and a retired
    // chain can neither acquire nor seed a later rebuild — drop everything so
    // this object reports empty instead of dangling a retired chain
    // (destroy_resources also zeroes extent_, keeping it consistent with
    // valid()). The surface/format config survives for a later recreate retry.
    destroy_resources();
    return core::vk_error(created, "vkCreateSwapchainKHR");
  }
  // The old chain (now retired) and its views are dead; replace them.
  destroy_resources();
  swapchain_ = new_swapchain;

  // Build the image views + render targets. On any failure, roll back to an
  // empty state (valid() == false) rather than leaving a half-built swapchain
  // with a stale extent and a partial target list. extent_ is committed only
  // once everything succeeds.
  const core::Status images = create_image_resources(extent);
  if (!images.ok()) {
    destroy_resources();  // also zeroes extent_ / requested_extent_
    return images;
  }
  extent_ = extent;
  requested_extent_ = desired;
  return core::Status{};
}

core::Status Swapchain::create_image_resources(VkExtent2D extent) {
  VkDevice dev = device_->handle();

  uint32_t count = 0;
  VKC_VK_TRY(vkGetSwapchainImagesKHR(dev, swapchain_, &count, nullptr));
  images_.resize(count);
  VKC_VK_TRY(vkGetSwapchainImagesKHR(dev, swapchain_, &count, images_.data()));

  // One multisampled set for every image, not one per image: a frame's
  // samples live for its one scope, and RenderTarget::begin orders each scope
  // after the previous one on the queue, so frames in flight never need two.
  core::ImageDesc color_desc;
  color_desc.extent = extent;
  color_desc.format = format_;
  color_desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  color_desc.samples = kWindowSamples;
  VKC_ASSIGN(msaa_color_, allocator_->create_image(color_desc));
  const RenderTargetAttachment msaa{msaa_color_.handle(), msaa_color_.view(),
                                    format_};
  // RenderTarget ignores an empty-view depth, so a colour-only chain passes
  // it too.
  RenderTargetAttachment depth{};
  if (depth_format_ != VK_FORMAT_UNDEFINED) {
    VKC_ASSIGN(msaa_depth_,
               make_depth_attachment(*allocator_, extent, depth_format_,
                                     kWindowSamples));
    depth = {msaa_depth_.handle(), msaa_depth_.view(), depth_format_};
  }

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
    VKC_VK_TRY(vkCreateImageView(dev, &view_info, nullptr, &view));
    views_.push_back(view);

    // Draw into the shared set; resolve into this image.
    const RenderTargetAttachment resolve{image, view, format_};
    targets_.emplace_back(extent, &msaa, 1, kWindowSamples, &depth, &resolve);
  }
  return core::Status{};
}

core::Result<uint32_t> Swapchain::acquire_next_image(
    VkSemaphore image_available, uint64_t timeout_ns) {
  if (!valid()) {
    return core::Status::invalid_argument(
        "Swapchain::acquire_next_image on an empty swapchain");
  }
  uint32_t index = 0;
  // vkAcquireNextImageKHR has several success codes, so it is checked by hand
  // (VKC_VK_TRY would treat SUBOPTIMAL as a failure). SUBOPTIMAL still yields a
  // usable image; OUT_OF_DATE returns its code for the caller to recreate.
  const VkResult r =
      vkAcquireNextImageKHR(device_->handle(), swapchain_, timeout_ns,
                            image_available, VK_NULL_HANDLE, &index);
  if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
    return index;
  }
  return core::vk_error(r, "vkAcquireNextImageKHR");
}

core::Status Swapchain::present(uint32_t image_index,
                                VkSemaphore render_finished) {
  if (!valid()) {
    return core::Status::invalid_argument(
        "Swapchain::present on an empty swapchain");
  }
  VkPresentInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &render_finished;
  info.swapchainCount = 1;
  info.pSwapchains = &swapchain_;
  info.pImageIndices = &image_index;
  // Route through the device so a shared present queue holds the submit mutex.
  const VkResult r = device_->queue_present(info);
  if (r == VK_SUCCESS) {
    return core::Status{};
  }
  // SUBOPTIMAL / OUT_OF_DATE flow back so the caller can recreate.
  return core::vk_error(r, "vkQueuePresentKHR");
}

core::Status Swapchain::recreate(VkExtent2D extent) {
  if (device_ == nullptr) {
    // Moved-from or default-constructed: no device/surface to rebuild on.
    // A swapchain that a failed build() emptied keeps its device + surface +
    // format, so it does NOT trip this guard — recreate can rebuild it once
    // the transient failure clears (that is the whole point of this path).
    return core::Status::invalid_argument(
        "Swapchain::recreate on a moved-from or default-constructed swapchain");
  }
  if (extent.width == 0 || extent.height == 0) {
    // A minimized window: skip the device drain and leave the current
    // (out-of-date but presentable) chain in place until the window restores.
    return core::Status::invalid_argument(
        "Swapchain::recreate: extent is zero (window minimized?)");
  }
  // Drain the renderer's queues before the rebuild retires the old images;
  // surface a device-loss rather than destroying resources the GPU may still
  // reference. Queue-scoped (not device-wide) so a shared adopted device does
  // not drain a sibling library's unrelated work.
  VKC_TRY(device_->wait_idle());
  return build(extent);
}

const RenderTarget& Swapchain::render_target(uint32_t image_index) const {
  VKC_CHECK(image_index < targets_.size(),
            "Swapchain::render_target: image_index out of range");
  return targets_[image_index];
}

VkImage Swapchain::image(uint32_t image_index) const {
  VKC_CHECK(image_index < images_.size(),
            "Swapchain::image: image_index out of range");
  return images_[image_index];
}

VkImageView Swapchain::image_view(uint32_t image_index) const {
  VKC_CHECK(image_index < image_count(),
            "Swapchain::image_view: image_index out of range");
  return views_[image_index];
}

RenderTargetLayout Swapchain::layout() const noexcept {
  // Derive from a live target so the signature always matches the actual
  // attachments (including the depth format when configured); empty when none.
  return targets_.empty() ? RenderTargetLayout{} : targets_.front().layout();
}

void Swapchain::destroy_resources() noexcept {
  if (device_ == nullptr) {
    return;
  }
  VkDevice dev = device_->handle();
  targets_.clear();
  // Each frees its image + view via the allocator.
  msaa_depth_ = core::Image{};
  msaa_color_ = core::Image{};
  for (VkImageView view : views_) {
    vkDestroyImageView(dev, view, nullptr);
  }
  views_.clear();
  images_.clear();  // owned by the swapchain object below; just drop our copies
  if (swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(dev, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }
  // A torn-down swapchain has no meaningful size; zero both extents so they
  // stay consistent with valid() == false (the recurring "forgot a scalar"
  // miss).
  extent_ = VkExtent2D{};
  requested_extent_ = VkExtent2D{};
}

void Swapchain::reset_state() noexcept {
  device_ = nullptr;
  surface_ = VK_NULL_HANDLE;
  allocator_ = nullptr;
  swapchain_ = VK_NULL_HANDLE;
  extent_ = VkExtent2D{};
  requested_extent_ = VkExtent2D{};
  format_ = VK_FORMAT_UNDEFINED;
  depth_format_ = VK_FORMAT_UNDEFINED;
  color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  requested_min_image_count_ = 0;
}

void Swapchain::destroy() noexcept {
  destroy_resources();
  reset_state();
}

Swapchain::~Swapchain() { destroy(); }

Swapchain::Swapchain(Swapchain&& other) noexcept
    : device_(other.device_),
      surface_(other.surface_),
      allocator_(other.allocator_),
      swapchain_(other.swapchain_),
      images_(std::move(other.images_)),
      views_(std::move(other.views_)),
      msaa_color_(std::move(other.msaa_color_)),
      msaa_depth_(std::move(other.msaa_depth_)),
      targets_(std::move(other.targets_)),
      format_(other.format_),
      depth_format_(other.depth_format_),
      color_space_(other.color_space_),
      present_mode_(other.present_mode_),
      extent_(other.extent_),
      requested_extent_(other.requested_extent_),
      requested_min_image_count_(other.requested_min_image_count_) {
  other.reset_state();
}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept {
  if (this != &other) {
    destroy();
    device_ = other.device_;
    surface_ = other.surface_;
    allocator_ = other.allocator_;
    swapchain_ = other.swapchain_;
    images_ = std::move(other.images_);
    views_ = std::move(other.views_);
    msaa_color_ = std::move(other.msaa_color_);
    msaa_depth_ = std::move(other.msaa_depth_);
    targets_ = std::move(other.targets_);
    format_ = other.format_;
    depth_format_ = other.depth_format_;
    color_space_ = other.color_space_;
    present_mode_ = other.present_mode_;
    extent_ = other.extent_;
    requested_extent_ = other.requested_extent_;
    requested_min_image_count_ = other.requested_min_image_count_;
    other.reset_state();
  }
  return *this;
}

}  // namespace volumetric_kit::gfx::windowing

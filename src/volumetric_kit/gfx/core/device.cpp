// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/device.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/impl/vk_query.hpp"

namespace volumetric_kit::gfx {
namespace {

// VK_KHR_portability_subset's name macro lives in vulkan_beta.h (gated by
// VK_ENABLE_BETA_EXTENSIONS); the string itself is stable, so we use it
// directly.
constexpr const char* kPortabilitySubset = "VK_KHR_portability_subset";

}  // namespace

Result<Device> Device::create([[maybe_unused]] VkInstance instance,
                              VkPhysicalDevice physical,
                              const DeviceConfig& config,
                              VkSurfaceKHR surface) {
  // `instance` is accepted for symmetry with Allocator::create and to document
  // the instance-outlives-device contract; the device stores only handles, so
  // it is intentionally unused at creation.
  if (physical == VK_NULL_HANDLE) {
    return Status::invalid_argument("Device::create: physical device is null");
  }
  std::optional<uint32_t> graphics = find_graphics_family(physical);
  if (!graphics) {
    return Status::unsupported("no graphics queue family");
  }

  // Capture the physical-device capabilities once, up front: the version and
  // extension checks below read from it instead of re-querying the driver, and
  // caps_ adopts the same data on the success path (no second round-trip).
  PhysicalDeviceInfo caps = PhysicalDeviceInfo::query(physical);

  // Require a Vulkan 1.3 device: shaders target SPIR-V 1.6 (--target-env=
  // vulkan1.3, see vg_shaders.cmake), and the TimelineSemaphore plumbing uses
  // the 1.2 *core* entry points (vkSignalSemaphore, vkWaitSemaphores,
  // vkGetSemaphoreCounterValue). The instance negotiates >= 1.1, which the
  // features2 query below needs.
  if (caps.properties().apiVersion < VK_API_VERSION_1_3) {
    return Status::unsupported("device does not support Vulkan 1.3");
  }

  std::optional<uint32_t> present;
  if (config.needs_present) {
    if (surface == VK_NULL_HANDLE) {
      return Status::invalid_argument(
          "needs_present requires a surface to choose a present queue");
    }
    present = find_present_family(physical, surface);
    if (!present) {
      return Status::unsupported("no present-capable queue family");
    }
  }

  std::vector<const char*> extensions;

  auto require = [&](const char* name) -> Status {
    if (!caps.supports_device_extension(name)) {
      return Status::unsupported(
          std::string("required device extension missing: ") + name);
    }
    extensions.push_back(name);
    return Status{};
  };

  if (config.needs_present) {
    VG_TRY(require(VK_KHR_SWAPCHAIN_EXTENSION_NAME));
  }
  if (config.needs_external_memory) {
    VG_TRY(require(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME));
    VG_TRY(require(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME));
  }

  auto already_enabled = [&](const char* name) {
    return std::any_of(
        extensions.begin(), extensions.end(),
        [&](const char* e) { return std::strcmp(e, name) == 0; });
  };
  // Caller-requested extensions, validated through the same require() path and
  // de-duplicated against the ones the flags above already added.
  for (const char* name : config.extra_device_extensions) {
    if (already_enabled(name)) {
      continue;
    }
    VG_TRY(require(name));
  }

  // The spec requires enabling VK_KHR_portability_subset whenever a device
  // exposes it (e.g. MoltenVK); de-duplicate in case the caller also listed it.
  if (caps.supports_device_extension(kPortabilitySubset) &&
      !already_enabled(kPortabilitySubset)) {
    extensions.push_back(kPortabilitySubset);
  }

  // Timeline semaphores are core in Vulkan 1.2; we target the core feature (the
  // TimelineSemaphore calls use the core entry points), so we only query and
  // enable the feature here — no pre-1.2 VK_KHR_timeline_semaphore path, which
  // would need the *KHR function variants the rest of the code does not call.
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
  timeline_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  // dynamicRendering is core (and required) in Vulkan 1.3; RenderTarget records
  // vkCmdBeginRendering, so it is the renderer's only path to a draw. Query it
  // alongside timelineSemaphore and enable it unconditionally below.
  VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features{};
  dynamic_rendering_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
  timeline_features.pNext = &dynamic_rendering_features;
  VkPhysicalDeviceFeatures2 supported{};
  supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  supported.pNext = &timeline_features;
  vkGetPhysicalDeviceFeatures2(physical, &supported);
  if (!dynamic_rendering_features.dynamicRendering) {
    return Status::unsupported(
        "device does not support dynamicRendering (core in Vulkan 1.3)");
  }

  const float priority = 1.0f;
  std::set<uint32_t> unique_families = {*graphics};
  if (present) {
    unique_families.insert(*present);
  }
  std::vector<VkDeviceQueueCreateInfo> queue_infos;
  for (uint32_t family : unique_families) {
    VkDeviceQueueCreateInfo q{};
    q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q.queueFamilyIndex = family;
    q.queueCount = 1;
    q.pQueuePriorities = &priority;
    queue_infos.push_back(q);
  }

  // Enable features through VkPhysicalDeviceFeatures2 (which supersedes
  // pEnabledFeatures). Consumers add further features (dynamic rendering,
  // synchronization2, …) through config.feature_chain, appended below.
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.features = config.features;

  // Enable timelineSemaphore (1.2 core) and dynamicRendering (1.3 core) exactly
  // once each. If the caller's feature_chain already carries a struct that
  // subsumes one — a version aggregate (VkPhysicalDeviceVulkan1{2,3}Features)
  // or the standalone feature struct — raise the bit there instead of linking
  // our own: a chain holding both the aggregate and the individual struct
  // violates VUID-VkDeviceCreateInfo-pNext-02830. The const_cast is safe
  // (vkCreateDevice treats the chain as input-only); each bit is only raised
  // toward what the device reported as supported.
  bool caller_carries_timeline = false;
  bool caller_carries_dynamic_rendering = false;
  if (config.feature_chain != nullptr) {
    for (auto* node = reinterpret_cast<VkBaseOutStructure*>(
             const_cast<void*>(config.feature_chain));
         node != nullptr; node = node->pNext) {
      if (node->sType ==
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
        if (timeline_features.timelineSemaphore) {
          reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(node)
              ->timelineSemaphore = VK_TRUE;
        }
        caller_carries_timeline = true;
      } else if (
          node->sType ==
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) {
        if (timeline_features.timelineSemaphore) {
          reinterpret_cast<VkPhysicalDeviceTimelineSemaphoreFeatures*>(node)
              ->timelineSemaphore = VK_TRUE;
        }
        caller_carries_timeline = true;
      } else if (node->sType ==
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) {
        if (dynamic_rendering_features.dynamicRendering) {
          reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(node)
              ->dynamicRendering = VK_TRUE;
        }
        caller_carries_dynamic_rendering = true;
      } else if (node->sType ==
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES) {
        if (dynamic_rendering_features.dynamicRendering) {
          reinterpret_cast<VkPhysicalDeviceDynamicRenderingFeatures*>(node)
              ->dynamicRendering = VK_TRUE;
        }
        caller_carries_dynamic_rendering = true;
      }
    }
  }

  // Link the feature structs this device owns (those the caller did not bring)
  // onto the tail of features2, then append the caller's chain after them. Each
  // link_owned drops the pNext set during the support query above so the chain
  // is rebuilt cleanly. VkBaseOutStructure exposes sType/pNext generically.
  auto* features_tail = reinterpret_cast<VkBaseOutStructure*>(&features2);
  auto link_owned = [&features_tail](void* feature) {
    auto* node = reinterpret_cast<VkBaseOutStructure*>(feature);
    node->pNext = nullptr;
    features_tail->pNext = node;
    features_tail = node;
  };
  if (!caller_carries_timeline && timeline_features.timelineSemaphore) {
    link_owned(&timeline_features);
  }
  if (!caller_carries_dynamic_rendering &&
      dynamic_rendering_features.dynamicRendering) {
    link_owned(&dynamic_rendering_features);
  }
  if (config.feature_chain != nullptr) {
    features_tail->pNext = reinterpret_cast<VkBaseOutStructure*>(
        const_cast<void*>(config.feature_chain));
  }

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pNext =
      &features2;  // features come via features2, not pEnabledFeatures
  create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
  create_info.pQueueCreateInfos = queue_infos.data();
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames =
      extensions.empty() ? nullptr : extensions.data();

  Device device;
  device.physical_ = physical;
  device.graphics_family_ = *graphics;
  VG_VK_TRY(vkCreateDevice(physical, &create_info, nullptr, &device.device_));

  vkGetDeviceQueue(device.device_, *graphics, 0, &device.graphics_queue_);
  if (present) {
    device.present_family_ = *present;
    vkGetDeviceQueue(device.device_, *present, 0, &device.present_queue_);
  }

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = *graphics;
  VG_VK_TRY(vkCreateCommandPool(device.device_, &pool_info, nullptr,
                                &device.command_pool_));

  // Adopt the capabilities captured up front (success path only).
  device.caps_ = std::move(caps);

  return device;
}

Status Device::submit_single_time(
    const std::function<void(VkCommandBuffer)>& record) const {
  VkCommandBufferAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc.commandPool = command_pool_;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VG_VK_TRY(vkAllocateCommandBuffers(device_, &alloc, &cmd));

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  Status status;
  VkResult begin_result = vkBeginCommandBuffer(cmd, &begin);
  if (begin_result == VK_SUCCESS) {
    record(cmd);
    VkResult end_result = vkEndCommandBuffer(cmd);
    if (end_result != VK_SUCCESS) {
      status = vk_error(end_result, "vkEndCommandBuffer");
    }
  } else {
    status = vk_error(begin_result, "vkBeginCommandBuffer");
  }

  if (status.ok()) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VkResult fence_result =
        vkCreateFence(device_, &fence_info, nullptr, &fence);
    if (fence_result == VK_SUCCESS) {
      VkSubmitInfo submit{};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &cmd;
      VkResult submit_result =
          vkQueueSubmit(graphics_queue_, 1, &submit, fence);
      if (submit_result == VK_SUCCESS) {
        VkResult wait_result =
            vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
        if (wait_result != VK_SUCCESS) {
          // A device loss (or other failure) while waiting means the submitted
          // work did not complete; report it rather than claiming success.
          status = vk_error(wait_result, "vkWaitForFences");
        }
      } else {
        status = vk_error(submit_result, "vkQueueSubmit");
      }
      vkDestroyFence(device_, fence, nullptr);
    } else {
      status = vk_error(fence_result, "vkCreateFence");
    }
  }

  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
  return status;
}

Device::Device(Device&& other) noexcept
    : physical_(other.physical_),
      device_(other.device_),
      command_pool_(other.command_pool_),
      graphics_family_(other.graphics_family_),
      present_family_(other.present_family_),
      graphics_queue_(other.graphics_queue_),
      present_queue_(other.present_queue_),
      caps_(std::move(other.caps_)) {
  other.physical_ = VK_NULL_HANDLE;
  other.device_ = VK_NULL_HANDLE;
  other.command_pool_ = VK_NULL_HANDLE;
  other.graphics_family_ = 0;
  other.present_family_ = 0;
  other.graphics_queue_ = VK_NULL_HANDLE;
  other.present_queue_ = VK_NULL_HANDLE;
  other.caps_ = PhysicalDeviceInfo{};
}

Device& Device::operator=(Device&& other) noexcept {
  if (this != &other) {
    destroy();
    physical_ = other.physical_;
    device_ = other.device_;
    command_pool_ = other.command_pool_;
    graphics_family_ = other.graphics_family_;
    present_family_ = other.present_family_;
    graphics_queue_ = other.graphics_queue_;
    present_queue_ = other.present_queue_;
    caps_ = std::move(other.caps_);
    other.physical_ = VK_NULL_HANDLE;
    other.device_ = VK_NULL_HANDLE;
    other.command_pool_ = VK_NULL_HANDLE;
    other.graphics_family_ = 0;
    other.present_family_ = 0;
    other.graphics_queue_ = VK_NULL_HANDLE;
    other.present_queue_ = VK_NULL_HANDLE;
    other.caps_ = PhysicalDeviceInfo{};
  }
  return *this;
}

Device::~Device() { destroy(); }

void Device::destroy() noexcept {
  if (command_pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, command_pool_, nullptr);
    command_pool_ = VK_NULL_HANDLE;
  }
  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }
  physical_ = VK_NULL_HANDLE;
  graphics_family_ = 0;
  present_family_ = 0;
  graphics_queue_ = VK_NULL_HANDLE;
  present_queue_ = VK_NULL_HANDLE;
  caps_ = PhysicalDeviceInfo{};
}

}  // namespace volumetric_kit::gfx

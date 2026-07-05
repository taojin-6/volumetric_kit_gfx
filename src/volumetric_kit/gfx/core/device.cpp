// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/device.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
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

// True when every core feature bit set in @p requested is also set in @p
// supported. VkPhysicalDeviceFeatures is a contiguous block of VkBool32, so it
// compares as one — a new Vulkan core feature needs no edit here.
bool core_features_supported(const VkPhysicalDeviceFeatures& requested,
                             const VkPhysicalDeviceFeatures& supported) {
  constexpr size_t kCount = sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32);
  const auto* req = reinterpret_cast<const VkBool32*>(&requested);
  const auto* sup = reinterpret_cast<const VkBool32*>(&supported);
  for (size_t i = 0; i < kCount; ++i) {
    if (req[i] == VK_TRUE && sup[i] != VK_TRUE) {
      return false;
    }
  }
  return true;
}

// Query and require the core features the renderer depends on: the requested
// 1.0 features (@p requested_core), timelineSemaphore (1.2 core), and
// dynamicRendering (1.3 core). The last two are guaranteed by the 1.3 floor,
// but checking keeps the requirement explicit and identical for create() and
// adopt() (so neither drifts). @p timeline / @p dynamic are written through so
// create() can go on to link them into its device-create chain (their sType is
// set, and they carry the device's reported — here, required — bits).
Status require_core_features(
    VkPhysicalDevice physical, const VkPhysicalDeviceFeatures& requested_core,
    VkPhysicalDeviceTimelineSemaphoreFeatures* timeline,
    VkPhysicalDeviceDynamicRenderingFeatures* dynamic) {
  *timeline = {};
  timeline->sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  *dynamic = {};
  dynamic->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
  timeline->pNext = dynamic;
  VkPhysicalDeviceFeatures2 supported{};
  supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  supported.pNext = timeline;
  vkGetPhysicalDeviceFeatures2(physical, &supported);
  if (!timeline->timelineSemaphore) {
    return Status::unsupported(
        "device does not support timelineSemaphore (core in Vulkan 1.2)");
  }
  if (!dynamic->dynamicRendering) {
    return Status::unsupported(
        "device does not support dynamicRendering (core in Vulkan 1.3)");
  }
  if (!core_features_supported(requested_core, supported.features)) {
    return Status::unsupported(
        "device does not support a requested core feature "
        "(DeviceConfig::features)");
  }
  return Status{};
}

// Create the graphics-family command pool this wrapper owns (RESET_COMMAND_
// BUFFER_BIT), shared by create() and adopt() so the two stay identical.
VkResult create_graphics_command_pool(VkDevice device, uint32_t family,
                                      VkCommandPool* out) {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = family;
  return vkCreateCommandPool(device, &pool_info, nullptr, out);
}

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
  std::optional<GraphicsFamily> graphics = find_graphics_family(physical);
  if (!graphics) {
    return Status::unsupported("no graphics queue family");
  }
  const uint32_t graphics_family = graphics->index;

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

  // Require the core features the renderer depends on: the caller's 1.0
  // features, plus timelineSemaphore (1.2 core; the TimelineSemaphore calls use
  // the core entry points) and dynamicRendering (1.3 core; RenderTarget records
  // vkCmdBeginRendering). require_core_features fills the two feature structs
  // so they can be linked into the create chain below with their (now-required)
  // bits set — the same check adopt() runs.
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
  VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features{};
  VG_TRY(require_core_features(physical, config.features, &timeline_features,
                               &dynamic_rendering_features));

  const float priority = 1.0f;
  std::set<uint32_t> unique_families = {graphics_family};
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
  device.graphics_family_ = graphics_family;
  device.graphics_timestamp_valid_bits_ = graphics->timestamp_valid_bits;
  VG_VK_TRY(vkCreateDevice(physical, &create_info, nullptr, &device.device_));

  vkGetDeviceQueue(device.device_, graphics_family, 0, &device.graphics_queue_);
  if (present) {
    device.present_family_ = *present;
    vkGetDeviceQueue(device.device_, *present, 0, &device.present_queue_);
  }

  VG_VK_TRY(create_graphics_command_pool(device.device_, graphics_family,
                                         &device.command_pool_));

  // Resolve the VK_EXT_debug_utils device entry points. The caller threads the
  // instance's debug_utils_enabled() in through config.enable_debug_utils (the
  // device only borrows a VkInstance handle, not the Instance object); the load
  // returns an all-null/inactive table whenever that is false, so the labels
  // are a branch-to-noop without the extension.
  device.debug_utils_ =
      DebugUtilsTable::load(device.device_, config.enable_debug_utils);

  // Adopt the capabilities captured up front (success path only).
  device.caps_ = std::move(caps);

  return device;
}

DeviceRequirements Device::requirements(const DeviceConfig& config) {
  // Mirror what create() enables, expressed as a mergeable descriptor. The
  // 1.3 / graphics / timeline / dynamic-rendering floor is the
  // DeviceRequirements default, so only config-derived fields are set here. The
  // spec-required VK_KHR_portability_subset is intentionally omitted: it is the
  // device *creator's* obligation when the device exposes it, not a caller
  // need.
  DeviceRequirements reqs;
  reqs.needs_present = config.needs_present;
  reqs.features = config.features;
  reqs.feature_chain = config.feature_chain;

  // De-duplicate exactly like create(): a name implied by a flag (e.g.
  // VK_KHR_swapchain) that the caller also lists must appear once, or an
  // embedder feeding device_extensions to vkCreateDevice trips the
  // unique-extension-names VUID.
  auto add = [&](const char* name) {
    if (std::none_of(
            reqs.device_extensions.begin(), reqs.device_extensions.end(),
            [&](const char* e) { return std::strcmp(e, name) == 0; })) {
      reqs.device_extensions.push_back(name);
    }
  };
  if (config.needs_present) {
    add(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  if (config.needs_external_memory) {
    add(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    add(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  }
  for (const char* name : config.extra_device_extensions) {
    add(name);
  }
  return reqs;
}

Result<Device> Device::adopt(const AdoptedDevice& adopted,
                             const DeviceConfig& config) {
  if (adopted.instance == VK_NULL_HANDLE ||
      adopted.physical_device == VK_NULL_HANDLE ||
      adopted.device == VK_NULL_HANDLE ||
      adopted.graphics_queue == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "Device::adopt: instance, physical device, device, and graphics queue "
        "must all be non-null");
  }
  // has_present must carry a real queue: create() sets present family+queue
  // together, so reject the index-without-queue state adopt could otherwise
  // reach (present_family() would then read a bogus-but-set index).
  if (adopted.has_present && adopted.present_queue == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "Device::adopt: has_present set without a present queue");
  }
  if (config.needs_present && !adopted.has_present) {
    return Status::invalid_argument(
        "Device::adopt: needs_present requires a present queue in "
        "AdoptedDevice");
  }

  // Verify against the renderer's own published requirements rather than
  // re-deriving the constants here, so requirements() stays the single source.
  const DeviceRequirements reqs = requirements(config);

  PhysicalDeviceInfo caps = PhysicalDeviceInfo::query(adopted.physical_device);
  if (caps.properties().apiVersion < reqs.api_version) {
    return Status::unsupported("adopted device does not support Vulkan 1.3");
  }

  // The assigned queue family's capabilities are a physical-device query, so
  // this stays authoritative even though a logical device's enabled state is
  // not queryable.
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(adopted.physical_device,
                                           &family_count, nullptr);
  if (adopted.graphics_family >= family_count) {
    return Status::invalid_argument(
        "Device::adopt: graphics_family out of range for the physical device");
  }
  if (adopted.has_present && adopted.present_family >= family_count) {
    return Status::invalid_argument(
        "Device::adopt: present_family out of range for the physical device");
  }
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(adopted.physical_device,
                                           &family_count, families.data());
  const VkQueueFamilyProperties& gfx_family = families[adopted.graphics_family];
  if ((gfx_family.queueFlags & reqs.queue_flags) != reqs.queue_flags) {
    return Status::unsupported(
        "Device::adopt: assigned queue family lacks the required capabilities");
  }

  // Every extension the renderer needs must be supported by the physical device
  // AND in the creator's declared enabled set — Vulkan cannot be asked what a
  // logical device enabled, so the declaration stands in for that query while
  // physical support stays authoritative (as create() checks it).
  if (!reqs.device_extensions.empty() &&
      adopted.enabled_device_extensions == nullptr) {
    return Status::unsupported(
        "Device::adopt: adopted device did not declare enabled extensions");
  }
  auto is_enabled = [&](const char* name) {
    for (uint32_t i = 0; i < adopted.enabled_device_extension_count; ++i) {
      const char* enabled = adopted.enabled_device_extensions[i];
      if (enabled != nullptr && std::strcmp(enabled, name) == 0) {
        return true;
      }
    }
    return false;
  };
  for (const char* name : reqs.device_extensions) {
    if (!caps.supports_device_extension(name)) {
      return Status::unsupported(
          std::string("Device::adopt: required extension not supported by the "
                      "adopted physical device: ") +
          name);
    }
    if (!is_enabled(name)) {
      return Status::unsupported(
          std::string("Device::adopt: required extension not enabled on the "
                      "adopted device: ") +
          name);
    }
  }

  // Require the same core features create() enables (config.features,
  // timelineSemaphore, dynamicRendering). On an adopted device this verifies
  // physical-device *support*; that they were actually *enabled* on the logical
  // device rests on the creator honoring requirements(). The filled feature
  // structs are unused here (adopt creates no device).
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
  VkPhysicalDeviceDynamicRenderingFeatures dynamic{};
  VG_TRY(require_core_features(adopted.physical_device, reqs.features,
                               &timeline, &dynamic));

  Device device;
  device.owns_device_ = false;  // borrowed — the dtor must not destroy it
  device.physical_ = adopted.physical_device;
  device.device_ = adopted.device;
  device.graphics_family_ = adopted.graphics_family;
  device.graphics_timestamp_valid_bits_ = gfx_family.timestampValidBits;
  device.graphics_queue_ = adopted.graphics_queue;
  device.submit_mutex_ = adopted.submit_mutex;
  if (adopted.has_present) {
    device.present_family_ = adopted.present_family;
    device.present_queue_ = adopted.present_queue;
  }

  // The command pool is this wrapper's own resource on the shared device — it
  // is created here (and destroyed in destroy()) even though the device is
  // borrowed.
  VG_VK_TRY(create_graphics_command_pool(
      device.device_, adopted.graphics_family, &device.command_pool_));

  device.debug_utils_ =
      DebugUtilsTable::load(device.device_, config.enable_debug_utils);
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
    status = submit_and_wait(cmd);
  }

  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
  return status;
}

Status Device::submit_and_wait(VkCommandBuffer cmd) const {
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateFence(device_, &fence_info, nullptr, &fence));

  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;

  Status status;
  // queue_submit holds the shared-queue mutex when the queue is borrowed and
  // shared with another library (a no-op lock otherwise).
  VkResult submit_result = queue_submit(1, &submit, fence);
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
  return status;
}

VkResult Device::queue_submit(uint32_t submit_count,
                              const VkSubmitInfo* submits,
                              VkFence fence) const {
  // Serialize submits on a shared queue (a device obtained through adopt with a
  // non-null submit_mutex). Vulkan requires queue submits be externally
  // synchronized; the lock is a no-op when the queue is exclusively ours.
  std::unique_lock<std::mutex> lock;
  if (submit_mutex_ != nullptr) {
    lock = std::unique_lock<std::mutex>(*submit_mutex_);
  }
  return vkQueueSubmit(graphics_queue_, submit_count, submits, fence);
}

VkResult Device::queue_present(const VkPresentInfoKHR& present_info) const {
  // The submit_mutex guards the shared (graphics) queue; hold it for a present
  // only when the present queue IS that queue. A distinct present queue is a
  // different queue outside this mutex's scope.
  std::unique_lock<std::mutex> lock;
  if (submit_mutex_ != nullptr && present_queue_ == graphics_queue_) {
    lock = std::unique_lock<std::mutex>(*submit_mutex_);
  }
  return vkQueuePresentKHR(present_queue_, &present_info);
}

Status Device::wait_idle() const {
  // Wait only on the queues the renderer was assigned, holding the shared-queue
  // mutex — never vkDeviceWaitIdle, which would idle every queue on the device
  // (a sibling library's included, on an adopted device) and cannot be
  // externally synchronized against submit_mutex_.
  std::unique_lock<std::mutex> lock;
  if (submit_mutex_ != nullptr) {
    lock = std::unique_lock<std::mutex>(*submit_mutex_);
  }
  VG_VK_TRY(vkQueueWaitIdle(graphics_queue_));
  if (present_queue_ != VK_NULL_HANDLE && present_queue_ != graphics_queue_) {
    VG_VK_TRY(vkQueueWaitIdle(present_queue_));
  }
  return Status{};
}

Device::Device(Device&& other) noexcept
    : physical_(other.physical_),
      device_(other.device_),
      command_pool_(other.command_pool_),
      owns_device_(other.owns_device_),
      submit_mutex_(other.submit_mutex_),
      graphics_family_(other.graphics_family_),
      graphics_timestamp_valid_bits_(other.graphics_timestamp_valid_bits_),
      present_family_(other.present_family_),
      graphics_queue_(other.graphics_queue_),
      present_queue_(other.present_queue_),
      caps_(std::move(other.caps_)),
      debug_utils_(other.debug_utils_) {
  other.physical_ = VK_NULL_HANDLE;
  other.device_ = VK_NULL_HANDLE;
  other.command_pool_ = VK_NULL_HANDLE;
  other.owns_device_ = true;
  other.submit_mutex_ = nullptr;
  other.graphics_family_ = 0;
  other.graphics_timestamp_valid_bits_ = 0;
  other.present_family_ = 0;
  other.graphics_queue_ = VK_NULL_HANDLE;
  other.present_queue_ = VK_NULL_HANDLE;
  other.caps_ = PhysicalDeviceInfo{};
  other.debug_utils_ = DebugUtilsTable{};
}

Device& Device::operator=(Device&& other) noexcept {
  if (this != &other) {
    destroy();
    physical_ = other.physical_;
    device_ = other.device_;
    command_pool_ = other.command_pool_;
    owns_device_ = other.owns_device_;
    submit_mutex_ = other.submit_mutex_;
    graphics_family_ = other.graphics_family_;
    graphics_timestamp_valid_bits_ = other.graphics_timestamp_valid_bits_;
    present_family_ = other.present_family_;
    graphics_queue_ = other.graphics_queue_;
    present_queue_ = other.present_queue_;
    caps_ = std::move(other.caps_);
    debug_utils_ = other.debug_utils_;
    other.physical_ = VK_NULL_HANDLE;
    other.device_ = VK_NULL_HANDLE;
    other.command_pool_ = VK_NULL_HANDLE;
    other.owns_device_ = true;
    other.submit_mutex_ = nullptr;
    other.graphics_family_ = 0;
    other.graphics_timestamp_valid_bits_ = 0;
    other.present_family_ = 0;
    other.graphics_queue_ = VK_NULL_HANDLE;
    other.present_queue_ = VK_NULL_HANDLE;
    other.caps_ = PhysicalDeviceInfo{};
    other.debug_utils_ = DebugUtilsTable{};
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
    // Only destroy a device this wrapper created; an adopted one belongs to its
    // owner (the shared bootstrap), which outlives us.
    if (owns_device_) {
      vkDestroyDevice(device_, nullptr);
    }
    device_ = VK_NULL_HANDLE;
  }
  owns_device_ = true;
  submit_mutex_ = nullptr;
  physical_ = VK_NULL_HANDLE;
  graphics_family_ = 0;
  graphics_timestamp_valid_bits_ = 0;
  present_family_ = 0;
  graphics_queue_ = VK_NULL_HANDLE;
  present_queue_ = VK_NULL_HANDLE;
  caps_ = PhysicalDeviceInfo{};
  debug_utils_ = DebugUtilsTable{};
}

}  // namespace volumetric_kit::gfx

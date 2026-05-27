// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/instance.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

#include "volumetric_kit/gfx/core/impl/vk_query.hpp"
#include "volumetric_kit/gfx/core/log.hpp"

namespace volumetric_kit::gfx {
namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

bool validation_layer_available() {
  const std::vector<VkLayerProperties> layers = instance_layers();
  return std::any_of(layers.begin(), layers.end(),
                     [](const VkLayerProperties& l) {
                       return std::strcmp(l.layerName, kValidationLayer) == 0;
                     });
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
               VkDebugUtilsMessageTypeFlagsEXT,
               const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  LogLevel level = LogLevel::Info;
  if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    level = LogLevel::Error;
  } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    level = LogLevel::Warning;
  }
  if (data != nullptr && data->pMessage != nullptr) {
    log_message(level, data->pMessage);
  }
  return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT make_messenger_info() {
  VkDebugUtilsMessengerCreateInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  info.pfnUserCallback = debug_callback;
  return info;
}

// Score by device class so we prefer a real GPU but still accept a software/CPU
// device (lavapipe) or Apple GPU (MoltenVK) — unlike the legacy NVIDIA-only
// logic.
int device_type_score(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return 1;
    default:
      return 0;
  }
}

}  // namespace

Result<Instance> Instance::create(const InstanceConfig& config) {
  const std::vector<VkExtensionProperties> available = instance_extensions();

  std::vector<const char*> extensions = config.extra_instance_extensions;

  const bool want_validation =
      config.enable_validation && validation_layer_available();
  if (config.enable_validation && !want_validation) {
    log_message(LogLevel::Warning,
                "validation requested but VK_LAYER_KHRONOS_validation is "
                "unavailable; disabling");
  }
  // The validation layer works without VK_EXT_debug_utils; routing its messages
  // through our callback (and placing a messenger in the instance pNext chain)
  // requires the extension to actually be enabled.
  const bool debug_utils =
      want_validation &&
      has_extension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if (debug_utils) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  // MoltenVK and other portability drivers are only enumerated when this flag
  // is set. The extension's name macro is absent from older Vulkan headers
  // (e.g. Ubuntu 22.04's libvulkan-dev), so guard on it: a platform old enough
  // to lack the macro also has no portability driver to enumerate.
  VkInstanceCreateFlags flags = 0;
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
  if (has_extension(available, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
#endif

  // Request 1.3, but never ask for more than the loader supports.
  uint32_t api_version = VK_API_VERSION_1_3;
  uint32_t loader_version = 0;
  if (vkEnumerateInstanceVersion(&loader_version) == VK_SUCCESS &&
      loader_version < api_version) {
    api_version = loader_version;
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = config.app_name.c_str();
  app_info.pEngineName = "volumetric_kit_gfx";
  app_info.apiVersion = api_version;

  VkDebugUtilsMessengerCreateInfoEXT messenger_info = make_messenger_info();

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.flags = flags;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames =
      extensions.empty() ? nullptr : extensions.data();
  if (want_validation) {
    create_info.enabledLayerCount = 1;
    create_info.ppEnabledLayerNames = &kValidationLayer;
  }
  if (debug_utils) {
    // Valid only with VK_EXT_debug_utils enabled; also captures messages from
    // instance creation/destruction itself.
    create_info.pNext = &messenger_info;
  }

  Instance instance;
  VG_VK_TRY(vkCreateInstance(&create_info, nullptr, &instance.instance_));

  if (debug_utils) {
    auto create_messenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance.instance_,
                                  "vkCreateDebugUtilsMessengerEXT"));
    if (create_messenger != nullptr) {
      VkResult messenger_result = create_messenger(
          instance.instance_, &messenger_info, nullptr, &instance.messenger_);
      if (messenger_result != VK_SUCCESS) {
        // Validation is best-effort: the layer is still enabled, but its
        // messages won't reach our sink. Don't fail instance creation; report
        // the degraded state and leave messenger_ null so validation_enabled()
        // stays truthful.
        instance.messenger_ = VK_NULL_HANDLE;
        log_message(LogLevel::Warning,
                    "validation layer enabled but debug messenger creation "
                    "failed; messages will not reach the log handler");
      }
    }
  }

  return instance;
}

Result<VkPhysicalDevice> Instance::select_physical_device(
    VkSurfaceKHR surface) const {
  const std::vector<VkPhysicalDevice> devices = physical_devices(instance_);
  if (devices.empty()) {
    return Status::error(VK_ERROR_INITIALIZATION_FAILED,
                         "no Vulkan physical devices found");
  }

  VkPhysicalDevice best = VK_NULL_HANDLE;
  int best_score = -1;
  for (VkPhysicalDevice device : devices) {
    if (!find_graphics_family(device)) {
      continue;
    }
    if (surface != VK_NULL_HANDLE && !find_present_family(device, surface)) {
      continue;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device, &props);
    const int score = device_type_score(props.deviceType);
    if (score > best_score) {
      best_score = score;
      best = device;
    }
  }

  if (best == VK_NULL_HANDLE) {
    return Status::error(VK_ERROR_FEATURE_NOT_PRESENT,
                         "no physical device has the required queue families");
  }
  return best;
}

Instance::Instance(Instance&& other) noexcept
    : instance_(other.instance_), messenger_(other.messenger_) {
  other.instance_ = VK_NULL_HANDLE;
  other.messenger_ = VK_NULL_HANDLE;
}

Instance& Instance::operator=(Instance&& other) noexcept {
  if (this != &other) {
    destroy();
    instance_ = other.instance_;
    messenger_ = other.messenger_;
    other.instance_ = VK_NULL_HANDLE;
    other.messenger_ = VK_NULL_HANDLE;
  }
  return *this;
}

Instance::~Instance() { destroy(); }

void Instance::destroy() noexcept {
  if (messenger_ != VK_NULL_HANDLE) {
    auto destroy_messenger =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_,
                                  "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy_messenger != nullptr) {
      destroy_messenger(instance_, messenger_, nullptr);
    }
    messenger_ = VK_NULL_HANDLE;
  }
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }
}

}  // namespace volumetric_kit::gfx

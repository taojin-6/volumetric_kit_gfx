// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Temporary measurement (draft PR, not for merging): how long the Vulkan calls
// every GPU test makes on its way in and out take -- vkCreateInstance,
// vkEnumeratePhysicalDevices, vkCreateDevice, vkDestroyDevice,
// vkDestroyInstance -- and whether running several processes at once slows
// them down. No layers: this is the driver's own cost.
//
//   vg_vk_create_bench [cycles]
//       one process, `cycles` create/destroy cycles (default 20)
//   vg_vk_create_bench --copies K [cycles]
//       K processes at once, each running `cycles` cycles; prints the
//       machine-wide rate of create/destroy cycles per second

#include <spawn.h>
#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "volumetric_kit/core/vulkan/vulkan.hpp"

extern char** environ;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double median(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

bool has_extension(const std::vector<VkExtensionProperties>& props,
                   const char* name) {
  return std::any_of(props.begin(), props.end(),
                     [name](const VkExtensionProperties& p) {
                       return std::strcmp(p.extensionName, name) == 0;
                     });
}

struct Phases {
  std::vector<double> instance, enumerate, device, destroy_device,
      destroy_instance;
};

// One cycle: everything a GPU test's fixture does to get a device and give
// it back. Returns false (after printing why) on a Vulkan failure.
bool one_cycle(Phases& t) {
  uint32_t count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> instance_exts(count);
  vkEnumerateInstanceExtensionProperties(nullptr, &count, instance_exts.data());
  // MoltenVK is a portability driver: the loader lists it only when asked.
  const bool portability = has_extension(
      instance_exts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  const char* portability_ext = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "vg_vk_create_bench";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;
  if (portability) {
    ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = &portability_ext;
  }

  Clock::time_point start = Clock::now();
  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
    std::fprintf(stderr, "bench: vkCreateInstance failed\n");
    return false;
  }
  t.instance.push_back(ms_since(start));

  start = Clock::now();
  count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  std::vector<VkPhysicalDevice> gpus(count);
  vkEnumeratePhysicalDevices(instance, &count, gpus.data());
  if (gpus.empty()) {
    std::fprintf(stderr, "bench: no Vulkan device\n");
    vkDestroyInstance(instance, nullptr);
    return false;
  }
  VkPhysicalDevice gpu = gpus.front();
  for (VkPhysicalDevice candidate : gpus) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(candidate, &props);
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      gpu = candidate;
      break;
    }
  }
  count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
  uint32_t family = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
      family = i;
      break;
    }
  }
  count = 0;
  vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> device_exts(count);
  vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count,
                                       device_exts.data());
  t.enumerate.push_back(ms_since(start));

  // A portability driver requires its subset extension to be enabled.
  const char* subset_ext = "VK_KHR_portability_subset";
  const bool subset = has_extension(device_exts, subset_ext);
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{};
  queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue.queueFamilyIndex = family;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &queue;
  if (subset) {
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &subset_ext;
  }

  start = Clock::now();
  VkDevice device = VK_NULL_HANDLE;
  if (vkCreateDevice(gpu, &dci, nullptr, &device) != VK_SUCCESS) {
    std::fprintf(stderr, "bench: vkCreateDevice failed\n");
    vkDestroyInstance(instance, nullptr);
    return false;
  }
  t.device.push_back(ms_since(start));

  start = Clock::now();
  vkDestroyDevice(device, nullptr);
  t.destroy_device.push_back(ms_since(start));

  start = Clock::now();
  vkDestroyInstance(instance, nullptr);
  t.destroy_instance.push_back(ms_since(start));
  return true;
}

void print_phases(const char* what, const Phases& t, size_t from) {
  auto tail = [from](const std::vector<double>& v) {
    return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(from),
                               v.end());
  };
  std::printf(
      "bench: %-22s instance %7.1f ms | enumerate %6.1f | device %7.1f | "
      "destroy device %6.1f | destroy instance %6.1f\n",
      what, median(tail(t.instance)), median(tail(t.enumerate)),
      median(tail(t.device)), median(tail(t.destroy_device)),
      median(tail(t.destroy_instance)));
}

int run_cycles(int cycles, const char* label) {
  Phases t;
  for (int i = 0; i < cycles; ++i) {
    if (!one_cycle(t)) {
      return 1;
    }
  }
  // The first cycle also pays for loading the loader's drivers, as a test
  // process's only cycle does; later ones show the steady cost.
  if (label == nullptr) {
    print_phases("first cycle", t, 0);
    if (cycles > 1) {
      print_phases("median of later cycles", t, 1);
    }
  } else {
    print_phases(label, t, 0);
  }
  std::fflush(stdout);
  return 0;
}

int run_copies(const char* self, int copies, int cycles) {
  const std::string cycles_arg = std::to_string(cycles);
  std::vector<pid_t> children;
  const Clock::time_point start = Clock::now();
  for (int i = 0; i < copies; ++i) {
    std::string label = "copy " + std::to_string(i + 1) + " (median)";
    char* argv[] = {const_cast<char*>(self), const_cast<char*>("--child"),
                    const_cast<char*>(cycles_arg.c_str()),
                    const_cast<char*>(label.c_str()), nullptr};
    pid_t pid = 0;
    if (posix_spawn(&pid, self, nullptr, nullptr, argv, environ) != 0) {
      std::fprintf(stderr, "bench: posix_spawn failed\n");
      return 1;
    }
    children.push_back(pid);
  }
  int failures = 0;
  for (pid_t pid : children) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ++failures;
    }
  }
  const double seconds = ms_since(start) / 1000.0;
  std::printf(
      "bench: %d copies x %d cycles: %.2f s wall, %.1f create/destroy "
      "cycles per second machine-wide%s\n",
      copies, cycles, seconds, copies * cycles / seconds,
      failures != 0 ? " (SOME COPIES FAILED)" : "");
  return failures != 0 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && std::strcmp(argv[1], "--child") == 0) {
    return run_cycles(std::atoi(argv[2]), argv[3]);
  }
  if (argc >= 3 && std::strcmp(argv[1], "--copies") == 0) {
    const int cycles = argc >= 4 ? std::atoi(argv[3]) : 10;
    return run_copies(argv[0], std::atoi(argv[2]), cycles);
  }
  return run_cycles(argc >= 2 ? std::atoi(argv[1]) : 20, nullptr);
}

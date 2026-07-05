// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/ui/imgui_overlay.hpp"

#include <mutex>
#include <string>

#include "volumetric_kit/gfx/core/check.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/log.hpp"

// Third-party renderer backend. The platform backend (imgui_impl_glfw) is
// deliberately not pulled in here — it lives in the consumer/example, keeping
// this tier window-system-agnostic like the windowing tier.
#include "imgui.h"
#include "imgui_impl_vulkan.h"

namespace volumetric_kit::gfx::ui {
namespace {

// Descriptors the backend's internal pool is sized for: the font atlas plus
// headroom for user textures registered via ImGui_ImplVulkan_AddTexture (e.g. a
// debug UI previewing a render target). Passed as InitInfo::DescriptorPoolSize,
// which asks the backend to create and own the pool — so this tier holds no
// VkDescriptorPool of its own.
constexpr uint32_t kDescriptorPoolSize = 64;

// Routes the Vulkan backend's internal VkResult failures into the kit's
// diagnostic sink (where the Vulkan debug messenger already reports). Installed
// as InitInfo::CheckVkResultFn, so it must be a plain function pointer.
void log_vk_result(VkResult err) {
  if (err == VK_SUCCESS) {
    return;
  }
  log_message(LogLevel::Error,
              "imgui vulkan backend: " + std::string(to_string(err)));
}

}  // namespace

Result<ImGuiOverlay> ImGuiOverlay::create(const Device& device,
                                          VkInstance instance,
                                          const ImGuiOverlayConfig& config) {
  if (instance == VK_NULL_HANDLE) {
    return Status::invalid_argument("ImGuiOverlay::create needs an instance");
  }
  if (device.handle() == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "ImGuiOverlay::create needs a valid device");
  }
  if (config.layout.color_count == 0) {
    return Status::invalid_argument(
        "ImGuiOverlay::create needs a layout with a color attachment");
  }
  if (static_cast<uint32_t>(config.layout.samples) == 0) {
    return Status::invalid_argument(
        "ImGuiOverlay::create needs a layout with a non-zero sample count");
  }
  if (config.min_image_count < 2) {
    return Status::invalid_argument(
        "ImGuiOverlay::create needs min_image_count >= 2");
  }
  if (config.image_count < config.min_image_count) {
    return Status::invalid_argument(
        "ImGuiOverlay::create needs image_count >= min_image_count");
  }

  // The ImGui pipeline is created (during Init) for dynamic rendering, so it
  // bakes the target's color/depth formats + sample count. The backend deep-
  // copies pColorAttachmentFormats, so the layout need not outlive this call.
  VkPipelineRenderingCreateInfo rendering{};
  rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
  rendering.colorAttachmentCount = config.layout.color_count;
  rendering.pColorAttachmentFormats = config.layout.color_formats.data();
  rendering.depthAttachmentFormat = config.layout.depth_format;

  ImGui_ImplVulkan_InitInfo init{};
  init.ApiVersion = VK_API_VERSION_1_3;  // selects the core vkCmdBeginRendering
  init.Instance = instance;
  init.PhysicalDevice = device.physical_device();
  init.Device = device.handle();
  init.QueueFamily = device.graphics_family();
  // The backend submits on this queue itself when it (re)creates a texture (the
  // font atlas on first render): a private command buffer + blocking submit,
  // not the caller's. render() takes the device's submit_mutex around that call
  // so a shared (adopted) queue stays externally synchronized.
  init.Queue = device.graphics_queue();
  init.DescriptorPoolSize = kDescriptorPoolSize;  // backend creates+owns it
  init.MinImageCount = config.min_image_count;
  init.ImageCount = config.image_count;
  // TODO: a swapchain recreate that changes the image count is not propagated
  // to the backend (no ImGui_ImplVulkan_SetMinImageCount call); the overlay
  // assumes a stable image count. Expose an update path when a target needs a
  // varying one.
  init.UseDynamicRendering = true;
  init.PipelineInfoMain.MSAASamples = config.layout.samples;
  init.PipelineInfoMain.PipelineRenderingCreateInfo = rendering;
  init.CheckVkResultFn = &log_vk_result;

  IMGUI_CHECKVERSION();
  ImGuiContext* context = ImGui::CreateContext();
  ImGui::SetCurrentContext(context);
  // A library overlay must not write imgui.ini into the consumer's working
  // directory; a consumer that wants persisted layout re-enables it via
  // context().
  ImGui::GetIO().IniFilename = nullptr;

  if (!ImGui_ImplVulkan_Init(&init)) {
    ImGui::DestroyContext(context);
    return Status::error(VK_ERROR_INITIALIZATION_FAILED,
                         "ImGui_ImplVulkan_Init failed");
  }

  ImGuiOverlay overlay;
  overlay.context_ = context;
  // Borrow the device's shared-queue mutex (null on an exclusively-owned queue)
  // so render() can serialize the backend's internal texture-upload submit.
  overlay.submit_mutex_ = device.submit_mutex();
  return overlay;
}

ImGuiOverlay::~ImGuiOverlay() { destroy(); }

ImGuiOverlay::ImGuiOverlay(ImGuiOverlay&& other) noexcept
    : context_(other.context_), submit_mutex_(other.submit_mutex_) {
  other.context_ = nullptr;
  other.submit_mutex_ = nullptr;
}

ImGuiOverlay& ImGuiOverlay::operator=(ImGuiOverlay&& other) noexcept {
  if (this != &other) {
    destroy();
    context_ = other.context_;
    submit_mutex_ = other.submit_mutex_;
    other.context_ = nullptr;
    other.submit_mutex_ = nullptr;
  }
  return *this;
}

void ImGuiOverlay::new_frame() {
  VG_CHECK(context_ != nullptr, "new_frame on an empty ImGuiOverlay");
  ImGui::SetCurrentContext(context_);
  ImGui_ImplVulkan_NewFrame();
  ImGui::NewFrame();
}

void ImGuiOverlay::render(VkCommandBuffer cmd) {
  VG_CHECK(context_ != nullptr, "render on an empty ImGuiOverlay");
  ImGui::SetCurrentContext(context_);
  ImGui::Render();
  // RenderDrawData may (re)create a texture — the font atlas on first render —
  // which the backend uploads via its own command buffer + a blocking submit on
  // the graphics queue. Hold the shared-queue mutex across it so a borrowed
  // queue shared with another library stays externally synchronized (no-op lock
  // on an exclusively-owned queue).
  std::unique_lock<std::mutex> lock;
  if (submit_mutex_ != nullptr) {
    lock = std::unique_lock<std::mutex>(*submit_mutex_);
  }
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void ImGuiOverlay::destroy() noexcept {
  if (context_ != nullptr) {
    // Shutdown frees the backend's pipeline + descriptor pool; the caller's
    // @warning requires the device be idle first.
    ImGui::SetCurrentContext(context_);
    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext(context_);
    context_ = nullptr;
  }
  submit_mutex_ = nullptr;
}

}  // namespace volumetric_kit::gfx::ui

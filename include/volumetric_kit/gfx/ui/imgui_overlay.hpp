// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file imgui_overlay.hpp
/// @brief A Dear ImGui debug overlay on the Vulkan renderer backend
///        (`imgui_impl_vulkan`), drawn into a @ref RenderTarget via dynamic
///        rendering.

#include <cstdint>

#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/ui/export.hpp"

struct ImGuiContext;

namespace volumetric_kit::gfx {

class Device;

namespace ui {

/// @brief Parameters for @ref ImGuiOverlay::create.
struct ImGuiOverlayConfig {
  /// The format + sample signature of the target the overlay renders into. Its
  /// color formats are baked into the ImGui pipeline (built for dynamic
  /// rendering), so a single overlay draws into any @ref RenderTarget whose
  /// layout is @ref RenderTargetLayout::compatible_with this one — typically a
  /// @ref windowing::Swapchain's `layout()`. Must carry at least one color
  /// attachment.
  RenderTargetLayout layout;
  /// Number of frames ImGui keeps buffered — the depth of its internal
  /// vertex/index buffer ring; must be >= 2. Use the swapchain's image count
  /// (or the @ref windowing::FrameLoop's `frames_in_flight`).
  uint32_t min_image_count = 2;
  /// Number of frames ImGui may have in flight; must be >= @ref
  /// min_image_count. Use the swapchain's image count.
  uint32_t image_count = 2;
};

/// @brief Owns a Dear ImGui context and its `imgui_impl_vulkan` renderer
///        backend, and records the built UI into a command buffer.
///
/// This is the *renderer* half of an ImGui integration only: it wires
/// `imgui_impl_vulkan` to a @ref Device and draws `ImDrawData` into a @ref
/// RenderTarget's dynamic-rendering scope. The *platform* half — input and
/// window sizing, e.g. `imgui_impl_glfw` — is the caller's, mirroring how the
/// windowing tier takes a `VkSurfaceKHR` and leaves GLFW to the consumer. Each
/// frame the caller runs its platform backend's new-frame (which sets
/// `io.DisplaySize`), then @ref new_frame, builds UI with raw `ImGui::` calls,
/// and inside a @ref RenderTarget::begin / @ref RenderTarget::end scope calls
/// @ref render. The backend owns its own descriptor pool (sized for the font
/// atlas plus user textures); the font atlas uploads itself on first @ref
/// render.
///
/// Each overlay owns a distinct `ImGuiContext`, but ImGui's *current* context
/// is a global that @ref new_frame and @ref render rebind (see @ref context),
/// so one overlay's frames must not be interleaved with another's. A
/// default-constructed `ImGuiOverlay` is empty (`valid()` is false) and safe to
/// move-assign into.
///
/// @warning The @p instance and @p device passed to @ref create must outlive
///          the overlay (it borrows both). Idle the device
///          (`vkDeviceWaitIdle`) before destroying the overlay while frames it
///          rendered are still in flight — teardown frees the ImGui pipeline
///          and descriptor pool.
///
/// @code
/// auto overlay = ui::ImGuiOverlay::create(
///     device, instance.handle(), {.layout = swapchain.layout(),
///                                 .min_image_count = swapchain.image_count(),
///                                 .image_count = swapchain.image_count()});
/// if (!overlay) return overlay.status();
/// ImGui_ImplGlfw_InitForVulkan(window, true);  // platform half: the caller's
/// // ... per frame:
/// ImGui_ImplGlfw_NewFrame();                   // sets io.DisplaySize + input
/// overlay.value().new_frame();
/// ImGui::ShowDemoWindow();
/// frame.target->begin(frame.cmd, {});
/// overlay.value().render(frame.cmd);           // inside the rendering scope
/// frame.target->end(frame.cmd);
/// @endcode
class VG_UI_API ImGuiOverlay {
 public:
  /// @brief Construct an empty overlay (owns nothing; `valid()` is false).
  ImGuiOverlay() = default;

  /// @brief Create an ImGui context + Vulkan backend for @p device.
  /// @param device    A device with a graphics queue; its handles back the
  ///                  backend and must outlive the overlay.
  /// @param instance  The instance @p device was created on (ImGui needs it for
  ///                  function/extension resolution); must outlive the overlay.
  /// @param config    The render-target layout to build the ImGui pipeline for,
  ///                  and the in-flight frame counts.
  /// @return The overlay on success, or a non-OK @ref Status: @ref
  ///         Status::Code::InvalidArgument for a null @p instance / moved-from
  ///         @p device, a @ref RenderTargetLayout with no color attachment or a
  ///         zero sample count, a @ref ImGuiOverlayConfig::min_image_count
  ///         below 2, or an @ref ImGuiOverlayConfig::image_count below it; a
  ///         Vulkan-domain @ref Status if backend initialization fails.
  static Result<ImGuiOverlay> create(const Device& device, VkInstance instance,
                                     const ImGuiOverlayConfig& config);

  ~ImGuiOverlay();
  ImGuiOverlay(ImGuiOverlay&& other) noexcept;
  ImGuiOverlay& operator=(ImGuiOverlay&& other) noexcept;
  ImGuiOverlay(const ImGuiOverlay&) = delete;
  ImGuiOverlay& operator=(const ImGuiOverlay&) = delete;

  /// @brief Begin an ImGui frame: `ImGui_ImplVulkan_NewFrame` +
  ///        `ImGui::NewFrame`. Build UI with `ImGui::` calls after this.
  /// @pre `valid()`, and the caller's platform backend new-frame (or the caller
  ///      directly) has set `ImGui::GetIO().DisplaySize` to a non-zero size
  ///      this frame — `ImGui::NewFrame` requires it. Makes @ref context the
  ///      current ImGui context.
  /// @note Each `new_frame` must be paired with exactly one @ref render before
  ///       the next `new_frame`; ImGui forbids two un-rendered frames.
  void new_frame();

  /// @brief Finalize the frame (`ImGui::Render`) and record its draw data into
  ///        @p cmd (`ImGui_ImplVulkan_RenderDrawData`).
  /// @param cmd  A command buffer in the recording state, inside a @ref
  ///             RenderTarget::begin / @ref RenderTarget::end scope whose
  ///             layout is compatible with @ref ImGuiOverlayConfig::layout.
  /// @pre `valid()`, and @ref new_frame was called this frame. Makes @ref
  ///      context the current ImGui context.
  /// @note On the first call the font atlas uploads through the backend's own
  ///       command buffer (a blocking queue submit), not @p cmd, so it is safe
  ///       inside the rendering scope.
  void render(VkCommandBuffer cmd);

  /// @return The owned `ImGuiContext` (`nullptr` when empty). Pass to
  ///         `ImGui::SetCurrentContext` to bind a platform backend or build UI
  ///         outside @ref new_frame / @ref render; the overlay re-binds it on
  ///         each of those calls.
  ImGuiContext* context() const noexcept { return context_; }

  /// @return `true` if this owns an ImGui context + backend.
  bool valid() const noexcept { return context_ != nullptr; }

 private:
  void destroy() noexcept;

  ImGuiContext* context_ = nullptr;  ///< Owns the backend via its BackendData.
};

}  // namespace ui
}  // namespace volumetric_kit::gfx

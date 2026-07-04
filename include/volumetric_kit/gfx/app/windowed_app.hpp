// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file windowed_app.hpp
/// @brief One-call windowed bring-up: the instance → surface → device →
///        allocator → swapchain → frame-loop chain as a single owner.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/gfx/app/export.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/swapchain.hpp"

namespace volumetric_kit::gfx::app {

/// @brief Parameters for @ref WindowedApp::create.
struct WindowedAppConfig {
  /// Application name reported to the driver (see @ref
  /// InstanceConfig::app_name).
  std::string app_name = "volumetric_kit_gfx";
  /// Enable the validation layer + debug messenger when available (see @ref
  /// InstanceConfig::enable_validation).
  bool enable_validation = false;
  /// Surface/platform instance extensions the window system requires — e.g.
  /// the array returned by `glfwGetRequiredInstanceExtensions`. Like the
  /// windowing tier, this tier is window-system-agnostic, so the consumer
  /// supplies them.
  std::vector<const char*> instance_extensions;
  /// Swapchain preferences: extent, color/depth formats, present mode, image
  /// count. With @ref windowing::SwapchainConfig::depth_format set, the app
  /// passes its allocator through so the swapchain owns a depth attachment per
  /// image.
  windowing::SwapchainConfig swapchain{};
  /// CPU-ahead depth of the frame loop (>= 1; see @ref
  /// windowing::FrameLoop::create).
  uint32_t frames_in_flight = 2;
};

/// @brief Owns the whole windowed bring-up chain — @ref Instance,
///        @ref windowing::Surface, @ref Device, @ref Allocator,
///        @ref windowing::Swapchain, @ref windowing::FrameLoop — created in
///        one call and destroyed in reverse order, with the surface threaded
///        consistently through device selection, `DeviceConfig::needs_present`,
///        and device creation.
///
/// The consumer keeps its window system: it passes the required instance
/// extensions in @ref WindowedAppConfig::instance_extensions and a
/// @ref SurfaceFactory that creates the `VkSurfaceKHR` (e.g. via
/// `glfwCreateWindowSurface`), so this tier stays GLFW-free like windowing.
/// Simple consumers never touch the members: @ref begin_frame / @ref end_frame
/// drive the loop's windowed protocol directly.
///
/// @note Resources created *after* the app — pipelines, overlays, scenes built
///       on @ref device / @ref allocator — destruct before it, while the app's
///       frame loop may still have frames in flight that reference them. Call
///       @ref wait_idle after the render loop, before returning, so their
///       destruction is safe; a profiler attached via @ref set_profiler is
///       likewise destroyed first, so detach it (`set_profiler(nullptr)`)
///       after idling.
///
/// @code
/// uint32_t ext_count = 0;
/// const char** exts = glfwGetRequiredInstanceExtensions(&ext_count);
/// app::WindowedAppConfig config;
/// config.app_name = "viewer";
/// config.instance_extensions.assign(exts, exts + ext_count);
/// config.swapchain.extent = {1280, 720};
/// auto app = app::WindowedApp::create(
///     config, [&](VkInstance instance) -> Result<VkSurfaceKHR> {
///       VkSurfaceKHR surface = VK_NULL_HANDLE;
///       const VkResult r =
///           glfwCreateWindowSurface(instance, window, nullptr, &surface);
///       if (r != VK_SUCCESS) return vk_error(r, "glfwCreateWindowSurface");
///       return surface;
///     });
/// if (!app) return app.status();
/// while (running) {
///   auto frame = app.value().begin_frame(window_extent());
///   if (!frame) return fail(frame.status());          // hard error only
///   if (!frame.value()) { wait_events(); continue; }  // minimized
///   const windowing::Frame& f = *frame.value();
///   f.target->begin(f.cmd, clear);
///   // ... bind pipeline, set viewport/scissor, draw ...
///   f.target->end(f.cmd);
///   Status end = app.value().end_frame(f);
///   if (!end.ok() && !windowing::swapchain_stale(end)) return fail(end);
/// }
/// app.value().wait_idle();  // locals created after the app die before it
/// @endcode
class VG_APP_API WindowedApp {
 public:
  /// @brief Creates the `VkSurfaceKHR` on the instance the app just built.
  ///        Called once by @ref create; the app adopts (and later destroys)
  ///        the returned handle. Return a non-OK @ref Status when the window
  ///        system fails to create one.
  using SurfaceFactory = std::function<Result<VkSurfaceKHR>(VkInstance)>;

  /// @brief Construct an empty app (owns nothing; `valid()` is false).
  WindowedApp() = default;

  /// @brief Run the whole bring-up chain: instance (app name / validation /
  ///        @p config extensions) → @p create_surface → present-capable
  ///        physical-device selection → device (`needs_present`) → allocator →
  ///        swapchain (with per-image depth when
  ///        `config.swapchain.depth_format` is set) → frame loop.
  /// @param config          App identity, instance extensions, swapchain
  ///                        preferences, and frames-in-flight depth.
  /// @param create_surface  Window-system callback producing the surface;
  ///                        must be callable.
  /// @return The app on success, or the first failing step's @ref Status:
  ///         @ref Status::Code::InvalidArgument for a null @p create_surface,
  ///         a factory that returns `VK_NULL_HANDLE`, or a zero
  ///         `config.frames_in_flight`; @ref Status::Code::Unsupported when no
  ///         present-capable device qualifies; otherwise the propagated
  ///         failure.
  static Result<WindowedApp> create(const WindowedAppConfig& config,
                                    const SurfaceFactory& create_surface);

  ~WindowedApp() = default;
  WindowedApp(WindowedApp&& other) noexcept = default;
  WindowedApp& operator=(WindowedApp&& other) noexcept = default;
  WindowedApp(const WindowedApp&) = delete;
  WindowedApp& operator=(const WindowedApp&) = delete;

  /// @brief Begin the next frame via the loop's windowed protocol (rebuilds
  ///        the swapchain on resize/staleness, skips ticks while minimized).
  /// @param current_extent  The window's current framebuffer extent.
  /// @return As @ref windowing::FrameLoop::begin_frame; @ref
  ///         Status::Code::InvalidArgument on an empty app.
  Result<std::optional<windowing::Frame>> begin_frame(
      VkExtent2D current_extent);

  /// @brief Submit + present the frame from @ref begin_frame.
  /// @param frame  The frame returned by @ref begin_frame this iteration.
  /// @return As @ref windowing::FrameLoop::end_frame (classify staleness with
  ///         @ref windowing::swapchain_stale); @ref
  ///         Status::Code::InvalidArgument on an empty app.
  Status end_frame(const windowing::Frame& frame);

  /// @brief Register the loop's post-rebuild hook (see @ref
  ///        windowing::FrameLoop::set_recreate_callback). No-op on an empty
  ///        app.
  /// @param callback  Receives the rebuilt swapchain's extent; whatever it
  ///                  captures must stay alive while attached.
  void set_recreate_callback(std::function<Status(VkExtent2D)> callback);

  /// @brief Attach a profiler the loop drives automatically, or detach with
  ///        `nullptr` (see @ref windowing::FrameLoop::set_profiler). No-op on
  ///        an empty app.
  /// @param profiler  Borrowed and nullable. A profiler is created on
  ///                  @ref device and therefore destroyed before the app —
  ///                  detach it (after @ref wait_idle) before it goes out of
  ///                  scope.
  void set_profiler(Profiler* profiler) noexcept;

  /// @brief Block until the device is idle — the library-blessed teardown
  ///        wait: call it after the render loop so resources created after
  ///        the app (which destruct before it) are no longer GPU-referenced.
  /// @return OK once idle; @ref Status::Code::InvalidArgument on an empty
  ///         app, or the failed `VkResult`.
  Status wait_idle() const;

  /// @return The owned instance. @pre @ref valid.
  Instance& instance() noexcept { return *state_->instance; }
  /// @copydoc instance
  const Instance& instance() const noexcept { return *state_->instance; }
  /// @return The owned device. @pre @ref valid.
  Device& device() noexcept { return *state_->device; }
  /// @copydoc device
  const Device& device() const noexcept { return *state_->device; }
  /// @return The owned allocator. @pre @ref valid.
  Allocator& allocator() noexcept { return *state_->allocator; }
  /// @copydoc allocator
  const Allocator& allocator() const noexcept { return *state_->allocator; }
  /// @return The owned swapchain. @pre @ref valid.
  windowing::Swapchain& swapchain() noexcept { return state_->swapchain; }
  /// @copydoc swapchain
  const windowing::Swapchain& swapchain() const noexcept {
    return state_->swapchain;
  }
  /// @return The owned frame loop. @pre @ref valid.
  windowing::FrameLoop& frame_loop() noexcept { return state_->frame_loop; }
  /// @copydoc frame_loop
  const windowing::FrameLoop& frame_loop() const noexcept {
    return state_->frame_loop;
  }

  /// @return `true` if this owns a created chain (`false` when
  ///         default-constructed or moved-from).
  bool valid() const noexcept { return state_ != nullptr; }

 private:
  // The chain lives behind one pointer because the later members borrow the
  // earlier ones by address (the loop points at the swapchain and device, the
  // swapchain at the device and allocator): a member-wise move would re-seat
  // the objects and dangle those borrows, whereas moving the pointer keeps
  // every address stable. Declaration order is the teardown contract — members
  // destruct in reverse, so the loop drains its in-flight frames first and the
  // instance dies last. std::optional stands in where a type has no public
  // default constructor.
  struct State {
    std::optional<Instance> instance;
    windowing::Surface surface;
    std::optional<Device> device;
    std::optional<Allocator> allocator;
    windowing::Swapchain swapchain;
    windowing::FrameLoop frame_loop;
  };
  std::unique_ptr<State> state_;
};

}  // namespace volumetric_kit::gfx::app

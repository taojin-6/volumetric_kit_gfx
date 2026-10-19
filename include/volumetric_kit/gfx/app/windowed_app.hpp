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
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/core/vulkan/shared_device.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/gfx/app/export.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/windowing/frame_loop.hpp"
#include "volumetric_kit/gfx/windowing/surface.hpp"
#include "volumetric_kit/gfx/windowing/surface_source.hpp"
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
  /// Surface/platform instance extensions the window system requires: a
  /// @ref windowing::SurfaceSource's. Like the windowing tier, this tier is
  /// window-system-agnostic, so the consumer supplies them.
  std::vector<const char*> instance_extensions;
  /// Swapchain preferences: extent, color/depth formats, present mode, image
  /// count. With @ref windowing::SwapchainConfig::depth_format set, the app
  /// passes its allocator through so the swapchain owns a depth attachment per
  /// image.
  windowing::SwapchainConfig swapchain{};
  /// CPU-ahead depth of the frame loop (>= 1; see @ref
  /// windowing::FrameLoop::create).
  uint32_t frames_in_flight = 2;
  /// What the device must provide: the renderer's floor
  /// (@ref device_requirements) plus any features, extensions or feature chain
  /// the application adds -- the escape hatch for anything the facade does not
  /// surface directly (e.g. `features.fillModeNonSolid` for a wireframe
  /// @ref GraphicsPipeline).
  /// @note Both @ref WindowedApp::create and @ref WindowedApp::adopt merge the
  ///       floor in whatever this holds, so a field here can only add to it,
  ///       and set `needs_present`: a windowed app always presents. Labels and
  ///       object names follow the instance -- the one @ref WindowedApp::create
  ///       builds, or `core::AdoptedDevice::enabled_debug_utils` for an adopted
  ///       device.
  core::DeviceRequirements device = device_requirements();
};

// TODO: the aggregate rule (AGENTS.md, "RAII resource types"): create and adopt
// return a std::unique_ptr, with no empty state, across consumers.
/// @brief Owns the whole windowed bring-up chain — `core::Instance`,
///        @ref windowing::Surface, `core::Device`, `core::Allocator`,
///        @ref windowing::Swapchain, @ref windowing::FrameLoop — created in
///        one call and destroyed in reverse order, with the surface threaded
///        consistently through device selection,
///        `DeviceRequirements::needs_present`, and device creation.
///
/// The consumer keeps its window system: it passes the required instance
/// extensions in @ref WindowedAppConfig::instance_extensions and a
/// @ref SurfaceFactory that creates the `VkSurfaceKHR`, so this tier stays
/// GLFW-free like windowing. For a GLFW window, the `glfw` tier supplies both
/// (as below).
/// Simple consumers never touch the members: @ref begin_frame / @ref end_frame
/// drive the loop's windowed protocol directly.
///
/// @note Resources created *after* the app — pipelines, overlays, scenes built
///       on @ref device / @ref allocator — destruct before it, while the app's
///       frame loop may still have frames in flight that reference them. Call
///       @ref wait_idle after the render loop, before returning, so their
///       destruction is safe; a profiler attached via @ref set_profiler is
///       likewise destroyed first, so detach it (`set_profiler(nullptr)`)
///       after idling. Moving the app keeps @ref device, @ref allocator,
///       @ref swapchain and @ref frame_loop where they are, so what was
///       built on them stays valid.
///
/// @code
/// VKC_ASSIGN(std::vector<const char*> extensions,
///            glfw::instance_extensions());
/// app::WindowedAppConfig config;
/// config.app_name = "viewer";
/// config.instance_extensions = extensions;
/// config.swapchain.extent = glfw::framebuffer_extent(window);
/// auto app =
///     app::WindowedApp::create(config, glfw::glfw_surface_factory(window));
/// if (!app) return app.status();
/// while (running) {
///   auto frame = app.value().begin_frame(glfw::framebuffer_size(window));
///   if (!frame) return fail(frame.status());          // hard error only
///   if (!frame.value()) { wait_events(); continue; }  // minimized
///   const windowing::Frame& f = *frame.value();
///   f.target->begin(f.cmd, clear);
///   // ... bind pipeline, set viewport/scissor, draw ...
///   f.target->end(f.cmd);
///   core::Status end = app.value().end_frame(f);
///   if (!end.ok() && !windowing::swapchain_stale(end)) return fail(end);
/// }
/// VKC_TRY(app.value().wait_idle());  // locals made after the app die first
/// @endcode
class VG_APP_API WindowedApp {
 public:
  /// @brief Creates the `VkSurfaceKHR` on the instance the app just built
  ///        (@ref windowing::SurfaceFactory). Called once by @ref create; the
  ///        app adopts (and later destroys) the returned handle. Return a
  ///        non-OK `core::Status` when the window system fails to create one.
  using SurfaceFactory = windowing::SurfaceFactory;

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
  /// @return The app on success, or the first failing step's `core::Status`:
  ///         `core::Status::Code::InvalidArgument` for a null
  ///         @p create_surface, a factory that returns `VK_NULL_HANDLE`, or a
  ///         zero `config.frames_in_flight`; `core::Status::Code::Unsupported`
  ///         when no present-capable device qualifies; otherwise the propagated
  ///         failure.
  static core::Result<WindowedApp> create(const WindowedAppConfig& config,
                                          const SurfaceFactory& create_surface);

  /// @brief Run the same bring-up chain on a `VkDevice` the embedder created,
  ///        instead of building an instance and device of our own.
  ///
  /// The windowed counterpart to `core::Device::adopt`. An embedder that runs
  /// the renderer alongside another Vulkan library builds **one** device from
  /// the union of both libraries' requirements (`merge` of each one's, gfx's
  /// from @ref device_requirements) and hands it to each; this is how the
  /// renderer takes its share and still gets a swapchain and a frame loop.
  /// Sharing one device is what lets the other library's `VkBuffer`/`VkImage`
  /// be drawn directly, with no cross-device copy or external-memory import.
  ///
  /// Everything before the surface is borrowed: the instance, physical device,
  /// logical device, and queues in @p adopted are **not** owned and are never
  /// destroyed here. Everything after is built and owned as in @ref create --
  /// the surface (from @p create_surface), the allocator, the swapchain, and
  /// the frame loop, all on the borrowed device.
  ///
  /// @param adopted  The embedder's device and queues. `instance_api_version`
  ///                 must be the version the instance was created with.
  ///                 `queue` is the renderer's graphics queue, and
  ///                 `has_present` must be set with a valid `present_queue`: a
  ///                 windowed app must present, so unlike `core::Device::adopt`
  ///                 this cannot be a compute-only share. Set `submit_mutex`
  ///                 (and `present_mutex`) when a queue is shared with another
  ///                 library. The share is held to the renderer's floor merged
  ///                 with `config.device`.
  /// @param config   As @ref create, except `app_name`, `enable_validation`
  ///                 and `instance_extensions` are ignored -- the embedder
  ///                 already created the instance those configure.
  /// @param create_surface  As @ref create; called with `adopted.instance`.
  /// @return The app on success, or the first failing step's `core::Status`:
  ///         `core::Status::Code::InvalidArgument` for a null handle, an unset
  ///         `instance_api_version` or no present queue in @p adopted -- all
  ///         found before @p create_surface runs -- a null @p create_surface,
  ///         a factory returning `VK_NULL_HANDLE`, or a zero
  ///         `config.frames_in_flight`; `core::Status::Code::Unsupported` when
  ///         `present_family` cannot actually present to the surface
  ///         @p create_surface returned (@ref create chooses that family *for*
  ///         its surface and so cannot hit this; a device built before any
  ///         window existed picked it blind); otherwise the propagated failure
  ///         (including `core::Device::adopt`'s verification that the device
  ///         carries what the renderer needs).
  ///
  /// @warning `adopted.instance`, `physical_device`, `device`, its queues,
  ///          `submit_mutex` and `present_mutex` must all outlive the returned
  ///          app.
  static core::Result<WindowedApp> adopt(const core::AdoptedDevice& adopted,
                                         const WindowedAppConfig& config,
                                         const SurfaceFactory& create_surface);

  /// @brief @ref adopt on @p shared's graphics payload, presenting to the
  ///        surface @p shared made, which this takes with
  ///        `core::SharedDevice::release_surface()`.
  ///
  /// @code
  /// VKC_ASSIGN(std::unique_ptr<core::SharedDevice> shared,
  ///            app::make_shared_device(surface, compute, device_config));
  /// VKC_ASSIGN(app::WindowedApp app,
  ///            app::WindowedApp::adopt(*shared, config));
  /// @endcode
  /// @param shared  The device the renderer shares with a compute library,
  ///                from @ref make_shared_device.
  /// @param config  As the other @ref adopt.
  /// @return As the other @ref adopt; `core::Status::Code::InvalidArgument`
  ///         once the surface was taken, or where @p shared made none. An
  ///         adopt that fails once it took the surface -- in
  ///         `core::Device::adopt`, the swapchain or the frame loop -- destroys
  ///         it, so a retry needs a new `core::SharedDevice`.
  /// @warning @p shared must outlive the returned app.
  static core::Result<WindowedApp> adopt(core::SharedDevice& shared,
                                         const WindowedAppConfig& config);

  ~WindowedApp() = default;
  WindowedApp(WindowedApp&& other) noexcept = default;
  WindowedApp& operator=(WindowedApp&& other) noexcept = default;
  WindowedApp(const WindowedApp&) = delete;
  WindowedApp& operator=(const WindowedApp&) = delete;

  /// @brief Begin the next frame via the loop's windowed protocol (rebuilds
  ///        the swapchain on resize/staleness, skips ticks while minimized).
  /// @param current_extent  The window's current framebuffer extent.
  /// @return As @ref windowing::FrameLoop::begin_frame;
  ///         `core::Status::Code::InvalidArgument` on an empty app.
  core::Result<std::optional<windowing::Frame>> begin_frame(
      VkExtent2D current_extent);

  /// @brief Submit + present the frame from @ref begin_frame.
  /// @param frame  The frame returned by @ref begin_frame this iteration.
  /// @return As @ref windowing::FrameLoop::end_frame (classify staleness with
  ///         @ref windowing::swapchain_stale);
  ///         `core::Status::Code::InvalidArgument` on an empty app.
  core::Status end_frame(const windowing::Frame& frame);

  /// @brief Register the loop's post-rebuild hook (see @ref
  ///        windowing::FrameLoop::set_recreate_callback). No-op on an empty
  ///        app.
  /// @param callback  Receives the rebuilt swapchain's extent; whatever it
  ///                  captures must stay alive while attached.
  void set_recreate_callback(std::function<core::Status(VkExtent2D)> callback);

  /// @brief Attach a profiler the loop drives automatically, or detach with
  ///        `nullptr` (see @ref windowing::FrameLoop::set_profiler). No-op on
  ///        an empty app.
  /// @param profiler  Borrowed and nullable. A profiler is created on
  ///                  @ref device and therefore destroyed before the app —
  ///                  detach it (after @ref wait_idle) before it goes out of
  ///                  scope.
  void set_profiler(Profiler* profiler) noexcept;

  /// @brief Block until the renderer's queues are idle — the library-blessed
  ///        teardown wait: call it after the render loop so resources created
  ///        after the app (which destruct before it) are no longer
  ///        GPU-referenced. Queue-scoped, not device-wide (see
  ///        `core::Device::wait_idle`), so it is safe on a shared adopted
  ///        device.
  /// @return OK once idle; `core::Status::Code::InvalidArgument` on an empty
  ///         app, or the failed `VkResult`.
  core::Status wait_idle() const;

  /// @return The owned instance. @pre @ref valid, **and** this app came from
  ///         @ref create -- an app from @ref adopt borrows its instance and
  ///         holds no `core::Instance` object, so asking one for it is a
  ///         programmer error and aborts (`VKC_CHECK`) rather than handing back
  ///         a reference to nothing. Use @ref instance_handle when the app may
  ///         be either, or when only the handle is wanted.
  core::Instance& instance() noexcept {
    VKC_CHECK(state_->instance.has_value(),
              "WindowedApp::instance: this app was adopted and borrows its "
              "instance; use instance_handle()");
    return *state_->instance;
  }
  /// @copydoc instance
  const core::Instance& instance() const noexcept {
    VKC_CHECK(state_->instance.has_value(),
              "WindowedApp::instance: this app was adopted and borrows its "
              "instance; use instance_handle()");
    return *state_->instance;
  }
  /// @return The `VkInstance` this app renders on, however it was obtained --
  ///         the one it created, or the embedder's. Valid for both @ref create
  ///         and @ref adopt. @pre @ref valid.
  VkInstance instance_handle() const noexcept {
    return state_->instance_handle;
  }
  /// @return The owned device. @pre @ref valid.
  core::Device& device() noexcept { return *state_->device; }
  /// @copydoc device
  const core::Device& device() const noexcept { return *state_->device; }
  /// @return The owned allocator. @pre @ref valid.
  core::Allocator& allocator() noexcept { return *state_->allocator; }
  /// @copydoc allocator
  const core::Allocator& allocator() const noexcept {
    return *state_->allocator;
  }
  /// @return The owned swapchain. @pre @ref valid.
  windowing::Swapchain& swapchain() noexcept { return state_->swapchain; }
  /// @copydoc swapchain
  const windowing::Swapchain& swapchain() const noexcept {
    return state_->swapchain;
  }
  /// @return The owned frame loop. @pre @ref valid.
  windowing::FrameLoop& frame_loop() noexcept { return *state_->frame_loop; }
  /// @copydoc frame_loop
  const windowing::FrameLoop& frame_loop() const noexcept {
    return *state_->frame_loop;
  }

  /// @return `true` if this owns a created chain (`false` when
  ///         default-constructed or moved-from).
  bool valid() const noexcept { return state_ != nullptr; }

 private:
  struct State;

  // Shared bring-up steps, so create() and adopt() cannot drift: make_surface
  // binds the app to an instance and runs the caller's factory on it,
  // finish_bring_up builds everything downstream of the device (allocator ->
  // swapchain -> frame loop). The instance is a parameter rather than something
  // make_surface reads back out of a half-built `state`, so there is no order
  // in which a caller can reach it before it is set.
  static core::Status make_surface(State& state, VkInstance instance,
                                   const SurfaceFactory& create_surface,
                                   std::string_view who);
  static core::Status finish_bring_up(State& state,
                                      const WindowedAppConfig& config);

  // The chain lives behind one pointer because the later members borrow the
  // earlier ones by address (the loop points at the swapchain and device, the
  // swapchain at the device and allocator): a member-wise move would re-seat
  // the objects and dangle those borrows, whereas moving the pointer keeps
  // every address stable. Declaration order is the teardown contract — members
  // destruct in reverse, so the loop drains its in-flight frames first and the
  // instance dies last. std::optional stands in where a type has no public
  // default constructor, and std::unique_ptr where it cannot move either (the
  // loop).
  struct State {
    // Set only on the create path; empty when the instance is the embedder's.
    std::optional<core::Instance> instance;
    // The instance in use either way, so accessors and the surface work
    // without branching on which path built this app.
    VkInstance instance_handle = VK_NULL_HANDLE;
    windowing::Surface surface;
    std::optional<core::Device> device;
    std::optional<core::Allocator> allocator;
    windowing::Swapchain swapchain;
    std::unique_ptr<windowing::FrameLoop> frame_loop;
  };
  std::unique_ptr<State> state_;
};

}  // namespace volumetric_kit::gfx::app

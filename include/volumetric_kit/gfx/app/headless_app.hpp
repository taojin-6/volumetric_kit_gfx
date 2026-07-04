// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file headless_app.hpp
/// @brief One-call headless bring-up: instance → device → allocator, no
///        surface — for offscreen rendering, bakes, and CI.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/gfx/app/export.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/instance.hpp"
#include "volumetric_kit/gfx/core/result.hpp"

namespace volumetric_kit::gfx::app {

/// @brief Parameters for @ref HeadlessApp::create.
struct HeadlessAppConfig {
  /// Application name reported to the driver (see @ref
  /// InstanceConfig::app_name).
  std::string app_name = "volumetric_kit_gfx";
  /// Enable the validation layer + debug messenger when available (see @ref
  /// InstanceConfig::enable_validation).
  bool enable_validation = false;
  /// Extra instance extensions to enable (see @ref
  /// InstanceConfig::extra_instance_extensions). None are needed for plain
  /// offscreen rendering.
  std::vector<const char*> instance_extensions;
};

/// @brief Owns the headless bring-up chain — @ref Instance, @ref Device,
///        @ref Allocator — created in one call and destroyed in reverse
///        order. No surface, no present queue: it runs wherever a Vulkan
///        device exists (CI, batch jobs).
///
/// Render targets stay consumer-side: what to render into (e.g. an
/// @ref OffscreenTarget with readback) is app policy, built on @ref allocator.
///
/// @note Resources created *after* the app — targets, pipelines, uploads built
///       on @ref device / @ref allocator — destruct before it; make sure the
///       GPU is done with them first (e.g. @ref Device::submit_single_time
///       already blocks until completion).
///
/// @code
/// auto app = app::HeadlessApp::create({.app_name = "bake"});
/// if (!app) return app.status();
/// OffscreenTargetDesc desc;
/// desc.extent = {512, 512};
/// desc.color_format = VK_FORMAT_R8G8B8A8_SRGB;
/// auto target = OffscreenTarget::create(app.value().allocator(), desc);
/// @endcode
class VG_APP_API HeadlessApp {
 public:
  /// @brief Construct an empty app (owns nothing; `valid()` is false).
  HeadlessApp() = default;

  /// @brief Run the headless bring-up chain: instance (app name / validation /
  ///        @p config extensions) → physical-device selection (no surface) →
  ///        device (no present queue) → allocator.
  /// @param config  App identity and optional instance extensions.
  /// @return The app on success, or the first failing step's @ref Status
  ///         (e.g. @ref Status::Code::Unsupported when no device qualifies).
  static Result<HeadlessApp> create(const HeadlessAppConfig& config);

  ~HeadlessApp() = default;
  HeadlessApp(HeadlessApp&& other) noexcept = default;
  HeadlessApp& operator=(HeadlessApp&& other) noexcept = default;
  HeadlessApp(const HeadlessApp&) = delete;
  HeadlessApp& operator=(const HeadlessApp&) = delete;

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

  /// @return `true` if this owns a created chain (`false` when
  ///         default-constructed or moved-from).
  bool valid() const noexcept { return state_ != nullptr; }

 private:
  // One pointer for the same reasons as WindowedApp::State: stable member
  // addresses across moves, reverse-order destruction (allocator before the
  // device it wraps, device before the instance), and cheap defaulted moves.
  // std::optional stands in where a type has no public default constructor.
  struct State {
    std::optional<Instance> instance;
    std::optional<Device> device;
    std::optional<Allocator> allocator;
  };
  std::unique_ptr<State> state_;
};

}  // namespace volumetric_kit::gfx::app

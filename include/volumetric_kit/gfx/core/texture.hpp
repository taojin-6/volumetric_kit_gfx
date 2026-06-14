// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file texture.hpp
/// @brief A `VkImage` + its default `VkImageView`, freed together by a deleter.

#include <cstdint>
#include <functional>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Owns a `VkImage` and a `VkImageView` over it, and runs a deleter that
/// frees
///        both plus the backing memory.
///
/// Produced by @ref Allocator::create_image (whose deleter destroys the view
/// then the image via VMA); the deleter keeps the allocator detail out of this
/// type's API. A default-constructed `Texture` is empty (`valid()` is false)
/// and safe to move-assign into.
///
/// @warning The producing @ref Allocator (and the device it wraps) must outlive
/// every
///          `Texture` it created: the deleter frees through them, so destroying
///          — or move-assigning over — the `Allocator` while a `Texture` is
///          still alive is undefined behavior. Retire textures through @ref
///          RetireQueue so their destruction is gated on GPU completion, ahead
///          of allocator teardown.
///
/// @code
/// Result<Texture> color = allocator.create_image({.extent = {1280, 720},
///                                                  .format =
///                                                  VK_FORMAT_R8G8B8A8_UNORM,
///                                                  .usage =
///                                                  VK_IMAGE_USAGE_SAMPLED_BIT
///                                                  |
///                                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT});
/// if (!color) return color.status();
/// VkImageView view = color.value().view();
/// @endcode
class VG_CORE_API Texture {
 public:
  /// @brief Construct an empty texture (owns nothing; `valid()` is false).
  Texture() noexcept = default;

  /// @brief Adopt @p image + @p view and the @p deleter that frees them.
  /// Produced by
  ///        @ref Allocator::create_image; rarely constructed directly.
  /// @param image    The image to take ownership of.
  /// @param view     The default view over @p image, or `VK_NULL_HANDLE` for a
  ///                 viewless image.
  /// @param extent   Its width/height in texels.
  /// @param depth    Its depth in texels (1 for 2D images; the slice count for
  /// a
  ///                 3D volume image).
  /// @param format   Its format.
  /// @param deleter  Frees the view, image, and memory; run exactly once on
  /// destruction.
  Texture(VkImage image, VkImageView view, VkExtent2D extent, uint32_t depth,
          VkFormat format, std::function<void()> deleter) noexcept;

  ~Texture();
  Texture(Texture&& other) noexcept;
  Texture& operator=(Texture&& other) noexcept;
  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;

  /// @return The underlying `VkImage` (`VK_NULL_HANDLE` when empty).
  VkImage image() const noexcept { return image_; }

  /// @return The default `VkImageView` over the image, or `VK_NULL_HANDLE` for
  /// a
  ///         viewless image (`TextureDesc::with_view == false`) or when empty.
  VkImageView view() const noexcept { return view_; }

  /// @return The image extent (width/height) in texels.
  VkExtent2D extent() const noexcept { return extent_; }

  /// @return The image depth in texels (1 for 2D images; the slice count for a
  ///         3D volume image).
  uint32_t depth() const noexcept { return depth_; }

  /// @return The image format.
  VkFormat format() const noexcept { return format_; }

  /// @return `true` if this owns an image.
  bool valid() const noexcept { return image_ != VK_NULL_HANDLE; }

 private:
  void destroy() noexcept;

  VkImage image_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkExtent2D extent_{};
  uint32_t depth_ = 1;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  std::function<void()> deleter_;
};

}  // namespace volumetric_kit::gfx

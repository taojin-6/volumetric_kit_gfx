// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image.hpp
/// @brief gfx's names for volumetric_kit_core's image, which replaced gfx's
///        `Texture`.
///
/// An @ref Image owns a `VkImage`, its device-only memory and its default
/// view, made by `Allocator::create_image` from an `ImageDesc`, or by
/// @ref UploadBatch::add from pixels. It records what Vulkan cannot be asked
/// afterwards -- type, samples, create flags -- and the layout its contents
/// are in, which the owner updates with `set_layout` after each transition
/// it submits; images an @ref UploadBatch makes report the layout its
/// `finish` leaves them in.
///
/// @code
/// ImageDesc desc;
/// desc.extent = {width, height};
/// desc.format = VK_FORMAT_R8G8B8A8_SRGB;
/// desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
///              VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
/// VG_ASSIGN(Image color, allocator.create_image(desc));
/// const VkExtent2D extent{color.width(), color.height()};
/// @endcode

#include "volumetric_kit/core/vulkan/image.hpp"

namespace volumetric_kit::gfx {

using core::Image;
using core::ImageInfo;

}  // namespace volumetric_kit::gfx

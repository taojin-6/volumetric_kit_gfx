// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file allocator.hpp
/// @brief gfx's names for volumetric_kit_core's allocator and the placements
///        it puts memory in.
///
/// The allocator is the family's shared one (the core's
/// `volumetric_kit/core/vulkan/allocator.hpp`; DECISIONS.md, 2026-10-04,
/// "Memory comes from volumetric_kit_core"). Every buffer names where its
/// memory lives, and nothing moves to slower memory when a heap fills:
///
/// - `MemoryUsage::DeviceOnly`, the default: memory only the GPU reaches --
///   vertex and index buffers, uploaded textures, render targets. Every
///   @ref Image is device-only.
/// - `MemoryUsage::DeviceMapped`: device-local memory the host writes and
///   shaders read directly -- the per-frame scene uniforms. A device without
///   such memory refuses it (`Unsupported`).
/// - `MemoryUsage::Staging`: host memory, mapped, with copy usage only -- the
///   source of an upload or the destination of a readback.
///
/// @code
/// VG_ASSIGN(Allocator allocator,
///           Allocator::create(instance.handle(), device));
/// BufferDesc readback;
/// readback.size = bytes;
/// readback.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
/// readback.memory = MemoryUsage::Staging;
/// readback.host_access = HostAccess::Random;  // the host reads it
/// VG_ASSIGN(Buffer pixels, allocator.create_buffer(readback));
/// @endcode

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/gfx/core/buffer.hpp"
#include "volumetric_kit/gfx/core/image.hpp"

namespace volumetric_kit::gfx {

using core::Allocator;
using core::BufferDesc;
using core::HeapStats;
using core::HostAccess;
using core::ImageDesc;
using core::MemoryStats;
using core::MemoryUsage;

}  // namespace volumetric_kit::gfx

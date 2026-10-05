// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file buffer.hpp
/// @brief gfx's names for volumetric_kit_core's buffer and the memory it
///        records.
///
/// A @ref Buffer owns a `VkBuffer` and its memory, made by
/// `Allocator::create_buffer` (see `allocator.hpp` for where its memory
/// lives). It may outlive its allocator; retire it through @ref RetireQueue so
/// it is freed only once the GPU is done with it.
///
/// @code
/// BufferDesc desc;
/// desc.size = bytes;
/// desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
/// desc.memory = MemoryUsage::Staging;  // mapped, written sequentially
/// VG_ASSIGN(Buffer staging, allocator.create_buffer(desc));
/// std::memcpy(staging.mapped(), src, bytes);
/// @endcode

#include "volumetric_kit/core/vulkan/buffer.hpp"

namespace volumetric_kit::gfx {

using core::Buffer;
using core::MemoryInfo;

}  // namespace volumetric_kit::gfx

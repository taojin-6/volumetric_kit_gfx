// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

/// @file vma_impl.cpp
/// The single translation unit that instantiates the (header-only) Vulkan
/// Memory Allocator. This TU exists so VMA is compiled exactly once and so the
/// VMA + Vulkan-headers toolchain is proven to build on every platform.
///
/// VMA calls the Vulkan entry points directly through the linked loader
/// prototypes
/// (`VMA_STATIC_VULKAN_FUNCTIONS`), so no function table needs to be supplied.

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

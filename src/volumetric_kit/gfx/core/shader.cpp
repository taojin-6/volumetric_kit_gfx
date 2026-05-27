// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/shader.hpp"

namespace volumetric_kit::gfx {

// The move/destroy lifecycle lives in UniqueHandle (see unique_handle.hpp);
// here we only validate, create the handle, and hand it over.

Result<ShaderModule> ShaderModule::create(VkDevice device, const uint32_t* code,
                                          size_t size_bytes) {
  // Validate before touching Vulkan, so misuse is caught even without a device:
  // SPIR-V is a stream of 32-bit words, so the byte size must be a non-zero
  // multiple of 4.
  if (code == nullptr || size_bytes == 0 || size_bytes % 4 != 0) {
    return vk_error(VK_ERROR_INITIALIZATION_FAILED,
                    "ShaderModule::create: code must be non-null with a "
                    "non-zero, 4-byte-aligned size");
  }

  VkShaderModuleCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = size_bytes;
  info.pCode = code;

  VkShaderModule handle = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateShaderModule(device, &info, nullptr, &handle));

  ShaderModule shader;
  shader.module_ =
      UniqueHandle<VkShaderModule, vkDestroyShaderModule>(device, handle);
  return shader;
}

}  // namespace volumetric_kit::gfx

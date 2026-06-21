// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/shader.hpp"

#include <exception>
#include <string>
#include <utility>

#include "spirv_cross.hpp"

namespace volumetric_kit::gfx {

namespace {

// Maps a SPIR-V execution model to its Vulkan stage bit. Only the graphics +
// compute stages are expected; anything else falls back to VK_SHADER_STAGE_ALL
// so reflection never silently mislabels a resource's stage.
VkShaderStageFlagBits stage_from_model(spv::ExecutionModel model) {
  switch (model) {
    case spv::ExecutionModelVertex:
      return VK_SHADER_STAGE_VERTEX_BIT;
    case spv::ExecutionModelTessellationControl:
      return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case spv::ExecutionModelTessellationEvaluation:
      return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case spv::ExecutionModelGeometry:
      return VK_SHADER_STAGE_GEOMETRY_BIT;
    case spv::ExecutionModelFragment:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
    case spv::ExecutionModelGLCompute:
      return VK_SHADER_STAGE_COMPUTE_BIT;
    default:
      // TODO: map the mesh/task + ray-tracing execution models once those
      // pipelines are in scope. Until then an unrecognized model falls back to
      // VK_SHADER_STAGE_ALL -- over-broad, but never a silently-wrong single
      // stage bit.
      return VK_SHADER_STAGE_ALL;
  }
}

// The number of descriptors a binding declares: 1 for a scalar binding, the
// product of its dimensions for an array, and 0 for a runtime-sized (unbounded)
// array. spirv-cross stores each array dimension in `SPIRType::array`, but an
// entry is the literal length only when the matching `array_size_literal` flag
// is set; otherwise it is the <id> of the (specialization) constant sizing that
// dimension -- resolved through the compiler to the specialized length here,
// rather than reported as the raw id.
uint32_t descriptor_count(const spirv_cross::Compiler& comp,
                          const spirv_cross::SPIRType& type) {
  uint32_t count = 1u;
  for (size_t i = 0; i < type.array.size(); ++i) {
    count *= type.array_size_literal[i]
                 ? type.array[i]
                 : comp.evaluate_constant_u32(type.array[i]);
  }
  return count;  // empty `array` -> 1; a leading runtime-sized 0 -> 0
}

// Appends every resource in `list` as a ReflectedResource of `type`, reading
// its set/binding decorations and descriptor count from the compiler.
void collect(const spirv_cross::Compiler& comp,
             const spirv_cross::SmallVector<spirv_cross::Resource>& list,
             VkDescriptorType type, VkShaderStageFlags stage,
             std::vector<ReflectedResource>& out) {
  for (const spirv_cross::Resource& res : list) {
    ReflectedResource rr;
    rr.set = comp.get_decoration(res.id, spv::DecorationDescriptorSet);
    rr.binding = comp.get_decoration(res.id, spv::DecorationBinding);
    rr.type = type;
    rr.count = descriptor_count(comp, comp.get_type(res.type_id));
    rr.stages = stage;
    out.push_back(rr);
  }
}

}  // namespace

Result<ShaderModule> ShaderModule::create(VkDevice device, const uint32_t* code,
                                          size_t size_bytes) {
  // Validate before touching Vulkan, so misuse is caught even without a device:
  // SPIR-V is a stream of 32-bit words, so the byte size must be a non-zero
  // multiple of 4.
  if (code == nullptr || size_bytes == 0 || size_bytes % 4 != 0) {
    return Status::invalid_argument(
        "ShaderModule::create: code must be non-null with a non-zero, "
        "4-byte-aligned size");
  }

  // Reflect the descriptor interface before creating the device handle: it
  // needs no device, and a malformed module surfaces here as a clean Status
  // rather than letting a spirv-cross exception escape the call.
  std::vector<ReflectedResource> resources;
  uint32_t push_constant_size = 0;
  VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL;
  try {
    spirv_cross::Compiler comp(code, size_bytes / sizeof(uint32_t));
    stage = stage_from_model(comp.get_execution_model());
    const VkShaderStageFlags stage_flags = stage;
    const spirv_cross::ShaderResources res = comp.get_shader_resources();
    collect(comp, res.uniform_buffers, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            stage_flags, resources);
    collect(comp, res.storage_buffers, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            stage_flags, resources);
    collect(comp, res.sampled_images, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            stage_flags, resources);
    collect(comp, res.separate_images, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            stage_flags, resources);
    collect(comp, res.separate_samplers, VK_DESCRIPTOR_TYPE_SAMPLER,
            stage_flags, resources);
    collect(comp, res.storage_images, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            stage_flags, resources);
    // TODO: input attachments (res.subpass_inputs), acceleration structures,
    // and uniform/storage texel buffers aren't reflected yet -- add them when a
    // technique needs them (subpass input, ray tracing, texel buffers).
    for (const spirv_cross::Resource& pc : res.push_constant_buffers) {
      const spirv_cross::SPIRType& t = comp.get_type(pc.base_type_id);
      push_constant_size =
          static_cast<uint32_t>(comp.get_declared_struct_size(t));
    }
  } catch (const std::exception& e) {
    return Status::invalid_argument(
        std::string("ShaderModule::create: SPIR-V reflection failed: ") +
        e.what());
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
  shader.resources_ = std::move(resources);
  shader.push_constant_size_ = push_constant_size;
  shader.stage_ = stage;
  return shader;
}

// Hand-written (not defaulted) so the reflected scalars are reset on the
// source: a defaulted move copies `push_constant_size_`/`stage_`, leaving a
// moved-from module reporting stale values while `valid()` is already false.
// `module_` and `resources_` reset themselves; the scalars do not.
ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : module_(std::move(other.module_)),
      resources_(std::move(other.resources_)),
      push_constant_size_(other.push_constant_size_),
      stage_(other.stage_) {
  other.resources_.clear();
  other.push_constant_size_ = 0;
  other.stage_ = VK_SHADER_STAGE_ALL;
}

ShaderModule& ShaderModule::operator=(ShaderModule&& other) noexcept {
  if (this != &other) {
    module_ =
        std::move(other.module_);  // frees the current handle, then adopts
    resources_ = std::move(other.resources_);
    push_constant_size_ = other.push_constant_size_;
    stage_ = other.stage_;
    other.resources_.clear();
    other.push_constant_size_ = 0;
    other.stage_ = VK_SHADER_STAGE_ALL;
  }
  return *this;
}

}  // namespace volumetric_kit::gfx

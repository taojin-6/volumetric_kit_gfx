// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace volumetric_kit::gfx {

// The move/destroy lifecycle lives in UniqueHandle (see unique_handle.hpp);
// here we only validate, assemble the create-info, build the layout + pipeline,
// and hand them over.

Result<GraphicsPipeline> GraphicsPipeline::create(
    VkDevice device, const GraphicsPipelineDesc& desc) {
  // Validate before touching Vulkan, so misuse yields a clean Status instead of
  // a crash in the driver (validation off is the shipping default). The desc
  // fields are checked first because they need no device -- that keeps the
  // no-device validation tests meaningful -- and the device handle is checked
  // last, just before the first Vulkan call below.
  if (desc.vertex_shader == nullptr || desc.fragment_shader == nullptr ||
      !desc.vertex_shader->valid() || !desc.fragment_shader->valid()) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: vertex_shader and fragment_shader must be "
        "non-null, valid modules");
  }
  if (desc.layout.color_count == 0) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: layout must have at least one color "
        "attachment");
  }
  if (desc.layout.color_count > RenderTargetLayout::kMaxColorAttachments) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: layout color_count exceeds "
        "kMaxColorAttachments");
  }
  for (uint32_t i = 0; i < desc.layout.color_count; ++i) {
    if (desc.layout.color_formats[i] == VK_FORMAT_UNDEFINED) {
      return Status::invalid_argument(
          "GraphicsPipeline::create: layout color formats must be defined");
    }
  }
  if (desc.entry_point == nullptr) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: entry_point must be non-null");
  }
  // Only vertex + fragment stages exist here, so a patch-list topology -- which
  // requires tessellation stages -- could never assemble a valid pipeline.
  if (desc.topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: patch-list topology requires tessellation "
        "stages, which this pipeline does not provide");
  }
  if ((desc.vertex_binding_count > 0 && desc.vertex_bindings == nullptr) ||
      (desc.vertex_attribute_count > 0 && desc.vertex_attributes == nullptr)) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: vertex binding/attribute pointers must be "
        "non-null when their counts are non-zero");
  }
  if (desc.depth_write && !desc.depth_test) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: depth_write requires depth_test (Vulkan "
        "disables depth writes when depth testing is off)");
  }
  if (desc.depth_test && desc.layout.depth_format == VK_FORMAT_UNDEFINED) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: depth_test requires layout to carry a depth "
        "format");
  }
  if (device == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: device must be non-null");
  }

  // Build the pipeline layout from the shaders' reflected interface: merge the
  // two stages' resources by (set, binding) -- OR-ing stage flags for a binding
  // both declare -- into one descriptor-set layout per set, then a pipeline
  // layout over those + a push-constant range. A set index no resource uses
  // gets an empty layout so set numbering stays contiguous. Shaders that bind
  // nothing yield an empty layout, exactly like the procedural path before.
  std::vector<std::vector<VkDescriptorSetLayoutBinding>> bindings_per_set;
  bool binding_conflict = false;
  const auto add_resource = [&bindings_per_set,
                             &binding_conflict](const ReflectedResource& r) {
    if (r.set >= bindings_per_set.size()) {
      bindings_per_set.resize(r.set + 1);
    }
    for (VkDescriptorSetLayoutBinding& b : bindings_per_set[r.set]) {
      if (b.binding == r.binding) {
        // Same (set, binding) in both stages: only the stage mask merges. If
        // the two stages disagree on the type or array size, the first one
        // seen would silently win and the other stage would then read the
        // binding as something it is not -- record it and fail below instead.
        if (b.descriptorType != r.type || b.descriptorCount != r.count) {
          binding_conflict = true;
        }
        b.stageFlags |= r.stages;
        return;
      }
    }
    VkDescriptorSetLayoutBinding b{};
    b.binding = r.binding;
    b.descriptorType = r.type;
    b.descriptorCount = r.count;
    b.stageFlags = r.stages;
    bindings_per_set[r.set].push_back(b);
  };
  for (const ReflectedResource& r : desc.vertex_shader->resources()) {
    add_resource(r);
  }
  for (const ReflectedResource& r : desc.fragment_shader->resources()) {
    add_resource(r);
  }
  if (binding_conflict) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: the vertex and fragment shaders declare the "
        "same (set, binding) with different descriptor types or array sizes");
  }

  std::vector<DescriptorSetLayout> set_layouts;
  std::vector<VkDescriptorSetLayout> set_layout_handles;
  set_layouts.reserve(bindings_per_set.size());
  set_layout_handles.reserve(bindings_per_set.size());
  for (const std::vector<VkDescriptorSetLayoutBinding>& bindings :
       bindings_per_set) {
    auto set_layout = DescriptorSetLayout::create(
        device, bindings.data(), static_cast<uint32_t>(bindings.size()));
    if (!set_layout) {
      return set_layout.status();
    }
    set_layout_handles.push_back(set_layout.value().handle());
    set_layouts.push_back(std::move(set_layout).value());
  }

  // One push-constant range spanning the larger of the two stages' blocks
  // (shaders sharing a block declare the same size), visible to whichever
  // stage(s) declare one.
  const uint32_t push_constant_size =
      std::max(desc.vertex_shader->push_constant_size(),
               desc.fragment_shader->push_constant_size());
  VkPushConstantRange push_range{};
  push_range.size = push_constant_size;
  if (desc.vertex_shader->push_constant_size() > 0) {
    push_range.stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
  }
  if (desc.fragment_shader->push_constant_size() > 0) {
    push_range.stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
  }

  // Own the layout immediately so any early return below frees it.
  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = static_cast<uint32_t>(set_layout_handles.size());
  layout_info.pSetLayouts = set_layout_handles.data();
  if (push_constant_size > 0) {
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
  }
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout));
  UniqueHandle<VkPipelineLayout, vkDestroyPipelineLayout> owned_layout(device,
                                                                       layout);

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = desc.vertex_shader->handle();
  stages[0].pName = desc.entry_point;
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = desc.fragment_shader->handle();
  stages[1].pName = desc.entry_point;

  // Empty bindings/attributes (counts 0) drive the procedural path where the
  // vertex shader computes positions from gl_VertexIndex; a non-zero count
  // binds the vertex-buffer layout the desc describes.
  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input.vertexBindingDescriptionCount = desc.vertex_binding_count;
  vertex_input.pVertexBindingDescriptions = desc.vertex_bindings;
  vertex_input.vertexAttributeDescriptionCount = desc.vertex_attribute_count;
  vertex_input.pVertexAttributeDescriptions = desc.vertex_attributes;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = desc.topology;

  // Viewport + scissor are dynamic, so the pipeline is size-independent and the
  // caller sets the rectangle at record time via vkCmdSetViewport/Scissor.
  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = desc.polygon_mode;
  raster.cullMode = desc.cull_mode;
  // Front faces are counter-clockwise -- the winding the camera tier's
  // Y-flipped projection produces for outward-facing geometry (see camera.hpp).
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample{};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  // TODO: validate desc.layout.samples against the device's framebuffer sample
  // counts once create() has the physical-device limits; today an unsupported
  // count is caught by the driver at vkCreateGraphicsPipelines.
  multisample.rasterizationSamples = desc.layout.samples;

  // Depth/stencil state is always supplied (a null pDepthStencilState would be
  // a spec violation once the layout carries a depth format). With depth_test
  // off, depthTestEnable is VK_FALSE, so depthWriteEnable/depthCompareOp are
  // inert -- correct for a color-only target.
  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = desc.depth_test ? VK_TRUE : VK_FALSE;
  depth_stencil.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
  // Ignored unless depthTestEnable is VK_TRUE (create() rejects depth_write
  // without depth_test), so no test-gated guard is needed here.
  depth_stencil.depthCompareOp = desc.depth_compare;

  // One non-blended, fully-writable state per color attachment in the layout;
  // attachmentCount must match the color count the draw's render target
  // carries.
  VkPipelineColorBlendAttachmentState
      blend_attachments[RenderTargetLayout::kMaxColorAttachments]{};
  for (uint32_t i = 0; i < desc.layout.color_count; ++i) {
    blend_attachments[i].blendEnable = VK_FALSE;
    blend_attachments[i].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  }

  VkPipelineColorBlendStateCreateInfo color_blend{};
  color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend.attachmentCount = desc.layout.color_count;
  color_blend.pAttachments = blend_attachments;

  const VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                            VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dynamic_states;

  // Dynamic rendering: the pipeline carries the attachment formats directly
  // instead of a VkRenderPass + subpass, so it is compatible with any
  // RenderTarget whose layout matches (offscreen, swapchain, or XR view).
  VkPipelineRenderingCreateInfo rendering_info{};
  rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
  rendering_info.colorAttachmentCount = desc.layout.color_count;
  rendering_info.pColorAttachmentFormats = desc.layout.color_formats.data();
  rendering_info.depthAttachmentFormat = desc.layout.depth_format;

  VkGraphicsPipelineCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  info.pNext = &rendering_info;
  info.stageCount = 2;
  info.pStages = stages;
  info.pVertexInputState = &vertex_input;
  info.pInputAssemblyState = &input_assembly;
  info.pViewportState = &viewport_state;
  info.pRasterizationState = &raster;
  info.pMultisampleState = &multisample;
  info.pDepthStencilState = &depth_stencil;
  info.pColorBlendState = &color_blend;
  info.pDynamicState = &dynamic_state;
  info.layout = layout;
  info.renderPass = VK_NULL_HANDLE;

  VkPipeline pipeline = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                      &pipeline));

  GraphicsPipeline result;
  result.set_layouts_ = std::move(set_layouts);
  result.layout_ = std::move(owned_layout);
  result.pipeline_ =
      UniqueHandle<VkPipeline, vkDestroyPipeline>(device, pipeline);
  return result;
}

}  // namespace volumetric_kit::gfx

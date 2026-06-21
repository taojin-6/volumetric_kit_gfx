// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/graphics_pipeline.hpp"

#include <utility>

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
  if (desc.vertex_shader == VK_NULL_HANDLE ||
      desc.fragment_shader == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: vertex_shader and fragment_shader must be "
        "non-null");
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

  // Empty layout: the procedural-vertex path binds no descriptor sets and no
  // push constants. Own it immediately so any early return below frees it.
  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VG_VK_TRY(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout));
  UniqueHandle<VkPipelineLayout, vkDestroyPipelineLayout> owned_layout(device,
                                                                       layout);

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = desc.vertex_shader;
  stages[0].pName = desc.entry_point;
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = desc.fragment_shader;
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
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  // No culling: the hello-triangle should draw regardless of winding order.
  raster.cullMode = VK_CULL_MODE_NONE;
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
  result.layout_ = std::move(owned_layout);
  result.pipeline_ =
      UniqueHandle<VkPipeline, vkDestroyPipeline>(device, pipeline);
  return result;
}

}  // namespace volumetric_kit::gfx

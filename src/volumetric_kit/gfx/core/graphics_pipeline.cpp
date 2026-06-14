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
  if (desc.render_pass == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "GraphicsPipeline::create: render_pass must be non-null");
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

  // No bindings/attributes: vertices are computed from gl_VertexIndex.
  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

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
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // Depth/stencil testing is disabled. The color-only hello-triangle pass has
  // no depth attachment, so Vulkan ignores this state -- but supplying a valid,
  // zero-initialized one (instead of leaving pDepthStencilState null) keeps the
  // pipeline correct if the caller's render pass *does* carry a depth/stencil
  // attachment, where a null pointer would be a spec violation.
  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

  // Single opaque color attachment; must match the subpass's color-attachment
  // count (one) and the attachment's sample count (one).
  VkPipelineColorBlendAttachmentState blend_attachment{};
  blend_attachment.blendEnable = VK_FALSE;
  blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo color_blend{};
  color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend.attachmentCount = 1;
  color_blend.pAttachments = &blend_attachment;

  const VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                            VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dynamic_states;

  VkGraphicsPipelineCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
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
  info.renderPass = desc.render_pass;
  info.subpass = desc.subpass;

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

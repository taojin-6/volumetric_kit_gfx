// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file graphics_pipeline.hpp
/// @brief A graphics `VkPipeline` and its `VkPipelineLayout`, built from a
///        vertex + fragment stage against a render pass.

#include <cstdint>

#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Parameters for @ref GraphicsPipeline::create.
///
/// This first iteration drives *procedural-vertex* draws — the vertex shader
/// computes positions from `gl_VertexIndex`, so there is no vertex-buffer input
/// and the pipeline layout is empty (no descriptor sets, no push constants).
/// The fixed-function state is fixed at sensible hello-triangle defaults:
/// a single non-blended color attachment, no depth/stencil, no culling, and
/// dynamic viewport + scissor (set at record time). Vertex input, blending,
/// depth, multisampling, and reflection-driven descriptor layouts are added as
/// the pipelines tier grows.
struct GraphicsPipelineDesc {
  /// Vertex-stage module. Must be non-`VK_NULL_HANDLE`.
  VkShaderModule vertex_shader = VK_NULL_HANDLE;
  /// Fragment-stage module. Must be non-`VK_NULL_HANDLE`.
  VkShaderModule fragment_shader = VK_NULL_HANDLE;
  /// The render pass the pipeline executes in. Must be non-`VK_NULL_HANDLE`. It
  /// is consumed for compatibility (attachment formats/samples) at create time
  /// and need not outlive the pipeline.
  VkRenderPass render_pass = VK_NULL_HANDLE;
  /// Index of the subpass within @ref render_pass the pipeline runs in.
  uint32_t subpass = 0;
  /// How vertices are assembled into primitives.
  /// `VK_PRIMITIVE_TOPOLOGY_PATCH_LIST` is rejected — it needs tessellation
  /// stages this pipeline does not provide.
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  /// Entry-point name used for both stages. Must be non-null.
  const char* entry_point = "main";
};

/// @brief Owns a graphics `VkPipeline` and the `VkPipelineLayout` it was built
///        with, freeing both on destruction.
///
/// Produced by @ref create from two @ref ShaderModule stages and a render pass.
/// A default-constructed `GraphicsPipeline` is empty (`valid()` is false) and
/// safe to move-assign into.
///
/// @warning The @p device passed to @ref create must outlive the pipeline: the
///          destructor frees through it, so destroying the device first is
///          undefined behavior.
///
/// @code
/// GraphicsPipelineDesc desc;
/// desc.vertex_shader = vert.handle();
/// desc.fragment_shader = frag.handle();
/// desc.render_pass = render_pass;
/// Result<GraphicsPipeline> pipeline = GraphicsPipeline::create(device, desc);
/// if (!pipeline) return pipeline.status();
/// // ... vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
/// //                       pipeline.value().handle());
/// @endcode
class VG_CORE_API GraphicsPipeline {
 public:
  /// @brief Construct an empty pipeline (owns nothing; `valid()` is false).
  GraphicsPipeline() = default;

  /// @brief Create a graphics pipeline from @p desc.
  /// @param device  The logical device that owns the pipeline and its layout.
  /// @param desc    The stages, render pass, and assembly state to build from.
  /// @pre @p device is non-`VK_NULL_HANDLE`; @p desc.vertex_shader,
  ///      @p desc.fragment_shader, and @p desc.render_pass are
  ///      non-`VK_NULL_HANDLE`; @p desc.entry_point is non-null; and
  ///      @p desc.topology is not `VK_PRIMITIVE_TOPOLOGY_PATCH_LIST`. These are
  ///      validated before Vulkan is touched and otherwise yield a non-OK
  ///      @ref Status with domain @ref Status::Code::InvalidArgument.
  /// @return The pipeline on success, or a non-OK @ref Status.
  static Result<GraphicsPipeline> create(VkDevice device,
                                         const GraphicsPipelineDesc& desc);

  ~GraphicsPipeline() = default;
  GraphicsPipeline(GraphicsPipeline&&) noexcept = default;
  GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept = default;
  GraphicsPipeline(const GraphicsPipeline&) = delete;
  GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;

  /// @return The underlying `VkPipeline` (`VK_NULL_HANDLE` when empty).
  VkPipeline handle() const noexcept { return pipeline_.get(); }

  /// @return The `VkPipelineLayout` the pipeline was built with
  ///         (`VK_NULL_HANDLE` when empty).
  VkPipelineLayout layout() const noexcept { return layout_.get(); }

  /// @return `true` if this owns a pipeline.
  bool valid() const noexcept { return pipeline_.valid(); }

 private:
  // Declared layout-before-pipeline so reverse-order member destruction frees
  // the pipeline first, then the layout it was built with.
  UniqueHandle<VkPipelineLayout, vkDestroyPipelineLayout> layout_;
  UniqueHandle<VkPipeline, vkDestroyPipeline> pipeline_;
};

}  // namespace volumetric_kit::gfx

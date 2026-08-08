// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file graphics_pipeline.hpp
/// @brief A graphics `VkPipeline` and its `VkPipelineLayout`, built from a
///        vertex + fragment stage for a render-target layout.

#include <cstdint>
#include <utility>
#include <vector>

#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/export.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/core/shader.hpp"
#include "volumetric_kit/gfx/core/unique_handle.hpp"
#include "volumetric_kit/gfx/core/vulkan.hpp"

namespace volumetric_kit::gfx {

/// @brief Parameters for @ref GraphicsPipeline::create.
///
/// Drives both procedural-vertex draws (leave the vertex-input fields empty and
/// the vertex shader computes positions from `gl_VertexIndex`) and
/// vertex-buffer draws (describe @ref vertex_bindings + @ref
/// vertex_attributes). The pipeline layout is derived by reflection: the
/// descriptor-set layouts + push-constant range come from the shaders' declared
/// resources (see @ref ShaderModule::resources), and the built sets are exposed
/// via @ref GraphicsPipeline::descriptor_set_layout. The remaining fixed-
/// function state is sensible defaults: non-blended color attachments and
/// dynamic viewport + scissor (set at record time); fill mode and face culling
/// are configurable via @ref polygon_mode / @ref cull_mode (defaulting to solid
/// fill / no culling). Depth testing is enabled via @ref depth_test, which
/// requires @ref layout to carry a depth format.
/// The color/depth formats and sample count come from @ref layout — the
/// pipeline renders dynamically (`vkCmdBeginRendering`), so there is no
/// `VkRenderPass`. Blending is added as the pipelines tier grows.
struct GraphicsPipelineDesc {
  /// Vertex-stage module; the pipeline reflects its descriptor interface. Must
  /// be non-null and outlive @ref GraphicsPipeline::create.
  const ShaderModule* vertex_shader = nullptr;
  /// Fragment-stage module; reflected like @ref vertex_shader. Must be non-null
  /// and outlive @ref GraphicsPipeline::create.
  const ShaderModule* fragment_shader = nullptr;
  /// The format + sample signature of the targets this pipeline draws into,
  /// baked in via `VkPipelineRenderingCreateInfo`. Must carry at least one
  /// color attachment; the pipeline is then compatible with any @ref
  /// RenderTarget whose @ref RenderTarget::layout matches.
  RenderTargetLayout layout;
  /// Vertex-buffer input bindings (stride + input rate), or `nullptr` with a
  /// @ref vertex_binding_count of `0` for a procedural-vertex pipeline.
  const VkVertexInputBindingDescription* vertex_bindings = nullptr;
  /// Number of @ref vertex_bindings.
  uint32_t vertex_binding_count = 0;
  /// Vertex-buffer input attributes (location/binding/format/offset), paired
  /// with @ref vertex_bindings; `nullptr` with a `0` count for procedural
  /// draws.
  const VkVertexInputAttributeDescription* vertex_attributes = nullptr;
  /// Number of @ref vertex_attributes.
  uint32_t vertex_attribute_count = 0;
  /// How vertices are assembled into primitives.
  /// `VK_PRIMITIVE_TOPOLOGY_PATCH_LIST` is rejected — it needs tessellation
  /// stages this pipeline does not provide.
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  /// Polygon fill mode. `VK_POLYGON_MODE_LINE` (wireframe) / `..._POINT` need
  /// the device's `fillModeNonSolid` feature — enable it via @ref DeviceConfig.
  VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
  /// Face-culling mode. Front faces are counter-clockwise (the winding the
  /// camera tier's Y-flipped projection yields for outward geometry), so
  /// `VK_CULL_MODE_BACK_BIT` draws a convex mesh correctly without depth.
  VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
  /// Enable depth testing; requires @ref layout to carry a depth format.
  bool depth_test = false;
  /// Write passing fragments' depth to the attachment. Requires @ref depth_test
  /// — Vulkan disables depth writes when depth testing is off, so @ref create
  /// rejects @ref depth_write without it.
  bool depth_write = false;
  /// Depth comparison used when @ref depth_test is set.
  VkCompareOp depth_compare = VK_COMPARE_OP_LESS;
  /// Entry-point name used for both stages. Must be non-null.
  const char* entry_point = "main";
};

/// @brief Owns a graphics `VkPipeline` and the `VkPipelineLayout` it was built
///        with, freeing both on destruction.
///
/// Produced by @ref create from two @ref ShaderModule stages and a
/// @ref RenderTargetLayout. A default-constructed `GraphicsPipeline` is empty
/// (`valid()` is false) and safe to move-assign into.
///
/// @warning The @p device passed to @ref create must outlive the pipeline: the
///          destructor frees through it, so destroying the device first is
///          undefined behavior.
///
/// @code
/// GraphicsPipelineDesc desc;
/// desc.vertex_shader = &vert;    // ShaderModule, reflected for the layout
/// desc.fragment_shader = &frag;
/// desc.layout = offscreen.layout();  // or any RenderTarget's layout
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
  /// @param desc    The stages, target layout, and assembly state to build
  /// from.
  /// @pre @p device is non-`VK_NULL_HANDLE`; @p desc.vertex_shader and
  ///      @p desc.fragment_shader are non-null and `valid()`; @p desc.layout
  ///      has between 1 and @ref RenderTargetLayout::kMaxColorAttachments color
  ///      attachments, each with a defined format; @p desc.entry_point is
  ///      non-null; @p desc.topology is not `VK_PRIMITIVE_TOPOLOGY_PATCH_LIST`;
  ///      each non-zero vertex binding/attribute count has a non-null pointer;
  ///      @p desc.depth_write is set only with @p desc.depth_test; and
  ///      @p desc.depth_test is requested only when @p desc.layout carries a
  ///      depth format. These are validated before Vulkan is touched and
  ///      otherwise yield a non-OK @ref Status with domain
  ///      @ref Status::Code::InvalidArgument.
  /// @return The pipeline on success, or a non-OK @ref Status.
  static Result<GraphicsPipeline> create(VkDevice device,
                                         const GraphicsPipelineDesc& desc);

  ~GraphicsPipeline() = default;
  GraphicsPipeline(GraphicsPipeline&&) noexcept = default;

  // Hand-written (not defaulted) for the self-move guard. A defaulted
  // move-assign forwards to std::vector's, which frees set_layouts_' elements
  // before adopting the source -- so `p = std::move(p)` would destroy the live
  // VkDescriptorSetLayouts while the two UniqueHandle members (which do guard
  // themselves) kept valid() true and the handles intact. Nothing catches that
  // downstream: the free is premature but well-formed, so the sanitizers see
  // nothing, and destroying a set layout already consumed by a pipeline layout
  // is legal, so the validation layers stay silent too.
  GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept {
    if (this != &other) {
      // Release in reverse declaration order -- the pipeline, then its layout,
      // then the set layouts both were built from -- before adopting other's.
      pipeline_ = {};
      layout_ = {};
      set_layouts_.clear();

      set_layouts_ = std::move(other.set_layouts_);
      layout_ = std::move(other.layout_);
      pipeline_ = std::move(other.pipeline_);
    }
    return *this;
  }

  GraphicsPipeline(const GraphicsPipeline&) = delete;
  GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;

  /// @return The underlying `VkPipeline` (`VK_NULL_HANDLE` when empty).
  VkPipeline handle() const noexcept { return pipeline_.get(); }

  /// @return The `VkPipelineLayout` the pipeline was built with
  ///         (`VK_NULL_HANDLE` when empty).
  VkPipelineLayout layout() const noexcept { return layout_.get(); }

  /// @return `true` if this owns a pipeline.
  bool valid() const noexcept { return pipeline_.valid(); }

  /// @return The number of descriptor sets the pipeline's layout declares
  ///         (reflected from the shaders; 0 when they bind nothing).
  uint32_t descriptor_set_count() const noexcept {
    return static_cast<uint32_t>(set_layouts_.size());
  }

  /// @return The `VkDescriptorSetLayout` for descriptor set @p set — pass it to
  ///         @ref DescriptorPool::allocate to make a matching set — or
  ///         `VK_NULL_HANDLE` when @p set is beyond what the shaders declare.
  VkDescriptorSetLayout descriptor_set_layout(uint32_t set) const noexcept {
    return set < set_layouts_.size() ? set_layouts_[set].handle()
                                     : VK_NULL_HANDLE;
  }

 private:
  // Declared sets-before-layout-before-pipeline so reverse-order member
  // destruction frees the pipeline first, then its layout, then the descriptor
  // set layouts both were built from.
  std::vector<DescriptorSetLayout> set_layouts_;
  UniqueHandle<VkPipelineLayout, vkDestroyPipelineLayout> layout_;
  UniqueHandle<VkPipeline, vkDestroyPipeline> pipeline_;
};

}  // namespace volumetric_kit::gfx

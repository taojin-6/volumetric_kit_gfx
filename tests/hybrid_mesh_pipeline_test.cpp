// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HybridMeshPipeline: reflected-layout creation, move semantics, and an
// end-to-end offscreen draw of the reconstruction hybrid mesh. The draw test
// renders a mesh whose left half carries atlas coordinates (sampled from a 2x2
// texture whose first texel is red) and whose right half carries the (-1, -1)
// "use vertex color" sentinel with a green per-vertex color -- proving both
// shading paths route correctly in one draw -- then re-renders lit to prove the
// directional term darkens the albedo. Runs under the validation layer with
// teeth. Skips when the runner exposes no Vulkan device.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/allocator.hpp"
#include "volumetric_kit/gfx/core/command_buffer.hpp"
#include "volumetric_kit/gfx/core/command_pool.hpp"
#include "volumetric_kit/gfx/core/descriptor.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
#include "volumetric_kit/gfx/core/texture.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/gpu_mesh.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "volumetric_kit/gfx/pipelines/live_mesh.hpp"
#include "vulkan_test_fixture.hpp"

namespace {

namespace pipelines = volumetric_kit::gfx::pipelines;
namespace assets = volumetric_kit::gfx::assets;

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t kSize = 64;

// The signature the depth-tested pipeline is built for (and every offscreen
// target the draw renders into).
vg::RenderTargetLayout color_depth_layout() {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = kColorFormat;
  layout.color_count = 1;
  layout.depth_format = kDepthFormat;
  return layout;
}

// Two screen-covering quads: the left (x in [-1, 0]) samples the atlas at the
// red texel; the right (x in [0, 1]) uses the (-1, -1) sentinel + a green
// vertex color. view_proj is identity in the test, so positions are already
// clip-space. The triangle winding is chosen so the quads are FRONT-facing
// under the pipeline's counter-clockwise front-face convention (there is no
// projection Y-flip here), so gl_FrontFacing is true and the lit path exercises
// the front-facing normal (0, 0, 1) -- not the back-face fallback.
assets::Mesh make_hybrid_mesh() {
  const glm::vec2 red_uv{0.25f, 0.25f};    // centre of texel (0,0) in a 2x2
  const glm::vec2 sentinel{-1.0f, -1.0f};  // "use the vertex color"
  const glm::vec4 white{1.0f, 1.0f, 1.0f, 1.0f};
  const glm::vec4 green{0.0f, 1.0f, 0.0f, 1.0f};
  auto vert = [](float x, float y, glm::vec2 uv, glm::vec4 color) {
    assets::Vertex v;
    v.position = {x, y, 0.5f};
    v.normal = {0.0f, 0.0f, 1.0f};
    v.uv0 = uv;
    v.color = color;
    return v;
  };

  assets::Mesh mesh;
  mesh.vertices = {
      vert(-1.0f, -1.0f, red_uv, white),   // 0  left quad (atlas)
      vert(0.0f, -1.0f, red_uv, white),    // 1
      vert(0.0f, 1.0f, red_uv, white),     // 2
      vert(-1.0f, 1.0f, red_uv, white),    // 3
      vert(0.0f, -1.0f, sentinel, green),  // 4  right quad (vertex color)
      vert(1.0f, -1.0f, sentinel, green),  // 5
      vert(1.0f, 1.0f, sentinel, green),   // 6
      vert(0.0f, 1.0f, sentinel, green),   // 7
  };
  // Front-facing winding (CCW in the framebuffer under identity view_proj).
  mesh.indices = {0, 2, 1, 0, 3, 2, 4, 6, 5, 4, 7, 6};
  return mesh;
}

// --- Creation validation: no device needed -----------------------------------

TEST(HybridMeshPipelineTest, DefaultConstructedIsEmpty) {
  pipelines::HybridMeshPipeline pipeline;
  EXPECT_FALSE(pipeline.valid());
  EXPECT_EQ(pipeline.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(pipeline.layout(), VK_NULL_HANDLE);
}

// --- Creation + move semantics: needs a device -------------------------------

using HybridMeshPipelineDeviceTest = VulkanDeviceTest;

TEST_F(HybridMeshPipelineDeviceTest, CreatesWithReflectedAtlasSet) {
  auto pipeline =
      pipelines::HybridMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
  EXPECT_TRUE(pipeline.value().valid());
  EXPECT_NE(pipeline.value().handle(), VK_NULL_HANDLE);
  EXPECT_NE(pipeline.value().layout(), VK_NULL_HANDLE);
  // The embedded shaders declare exactly one set: set 0, the atlas sampler.
  EXPECT_EQ(pipeline.value().descriptor_set_count(), 1u);
  EXPECT_NE(pipeline.value().descriptor_set_layout(0), VK_NULL_HANDLE);
}

TEST_F(HybridMeshPipelineDeviceTest, RejectsLayoutWithoutDepth) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = kColorFormat;
  layout.color_count = 1;  // no depth format -> depth-tested pipeline rejected
  auto pipeline = pipelines::HybridMeshPipeline::create(device(), layout);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vg::Status::Code::InvalidArgument);
}

TEST_F(HybridMeshPipelineDeviceTest, MoveLeavesSourceEmpty) {
  auto created =
      pipelines::HybridMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(created.ok()) << created.status().message();
  pipelines::HybridMeshPipeline source = std::move(created).value();
  ASSERT_TRUE(source.valid());

  pipelines::HybridMeshPipeline moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(HybridMeshPipelineDeviceTest, MoveAssignOverLiveObjectAdoptsSource) {
  auto first =
      pipelines::HybridMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(first.ok()) << first.status().message();
  auto second =
      pipelines::HybridMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(second.ok()) << second.status().message();

  pipelines::HybridMeshPipeline dst = std::move(first).value();
  pipelines::HybridMeshPipeline src = std::move(second).value();
  const VkPipeline adopted = src.handle();

  dst = std::move(src);  // frees dst's pipeline, then adopts src's
  EXPECT_TRUE(dst.valid());
  EXPECT_EQ(dst.handle(), adopted);
  EXPECT_FALSE(src.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(HybridMeshPipelineDeviceTest, SelfMoveAssignStaysValid) {
  auto created =
      pipelines::HybridMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(created.ok()) << created.status().message();
  pipelines::HybridMeshPipeline pipeline = std::move(created).value();
  const VkPipeline before = pipeline.handle();

  // Launder through a pointer so -Wself-move does not fire under -Werror.
  pipelines::HybridMeshPipeline* alias = &pipeline;
  pipeline = std::move(*alias);
  EXPECT_TRUE(pipeline.valid());
  EXPECT_EQ(pipeline.handle(), before);
}

// --- End-to-end offscreen draw: validation-with-teeth ------------------------

class HybridMeshRenderTest : public VulkanDeviceTest {
 protected:
  bool wants_validation() const override { return true; }

  // Renders `draw` (bound with `atlas` as set 0) into a fresh offscreen target
  // with the given flags, and returns a copy of the RGBA pixels. `draw` names
  // either a static GpuMesh or a live indirect mesh -- the harness is agnostic.
  std::vector<uint8_t> render(vg::Allocator& allocator,
                              const pipelines::HybridMeshPipeline& pipeline,
                              const pipelines::HybridMeshDraw& draw,
                              VkDescriptorSet atlas, uint32_t flags) {
    vg::OffscreenTargetDesc td;
    td.extent = {kSize, kSize};
    td.color_format = kColorFormat;
    td.depth_format = kDepthFormat;
    auto target = vg::OffscreenTarget::create(allocator, td);
    EXPECT_TRUE(target.ok()) << target.status().message();
    if (!target.ok()) {
      return {};
    }

    auto pool = vg::CommandPool::create(device(), device_->graphics_family());
    EXPECT_TRUE(pool.ok()) << pool.status().message();
    if (!pool.ok()) {
      return {};
    }
    auto cmd = pool.value().allocate_primary();
    EXPECT_TRUE(cmd.ok()) << cmd.status().message();
    if (!cmd.ok()) {
      return {};
    }
    EXPECT_TRUE(
        cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT).ok());
    const VkCommandBuffer raw = cmd.value().handle();

    target.value().prepare(raw);  // attachments -> attachment layouts

    vg::RenderTargetBeginInfo begin_info;
    begin_info.clear_color.float32[3] = 1.0f;  // opaque-black clear
    const vg::RenderTarget rt = target.value().target();
    rt.begin(raw, begin_info);

    pipelines::HybridMeshFrame frame;
    frame.extent = {kSize, kSize};
    frame.view_proj = glm::mat4(1.0f);  // clip-space positions
    frame.flags = flags;
    frame.atlas = atlas;
    frame.draws = &draw;
    frame.draw_count = 1;
    pipeline.submit(raw, frame);

    rt.end(raw);
    target.value().record_readback(raw);
    EXPECT_TRUE(cmd.value().end().ok());
    submit_and_wait(raw);

    const auto* px = static_cast<const uint8_t*>(target.value().pixels());
    EXPECT_NE(px, nullptr);
    if (px == nullptr) {
      return {};
    }
    return std::vector<uint8_t>(px,
                                px + static_cast<size_t>(kSize) * kSize * 4);
  }
};

TEST_F(HybridMeshRenderTest, RoutesAtlasAndVertexColor) {
  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline =
      pipelines::HybridMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_hybrid_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  // A 2x2 atlas: texel (0,0) red, the rest blue. uv0 (0.25, 0.25) + NEAREST
  // reads the red texel, so a working atlas path shows red (and a broken
  // sentinel would leak red into the vertex-color half instead of green).
  const uint8_t atlas_px[16] = {255, 0, 0,   255, 0, 0, 255, 255,
                                0,   0, 255, 255, 0, 0, 255, 255};
  vg::ImageUploadDesc adesc;
  adesc.extent = {2, 2};
  adesc.format = kColorFormat;
  adesc.pixels = atlas_px;
  adesc.size = sizeof(atlas_px);
  auto atlas_tex = vg::upload_texture(*device_, allocator.value(), adesc);
  ASSERT_TRUE(atlas_tex.ok()) << atlas_tex.status().message();

  vg::SamplerDesc sdesc;
  sdesc.mag_filter = VK_FILTER_NEAREST;
  sdesc.min_filter = VK_FILTER_NEAREST;
  sdesc.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sdesc.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sdesc.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sdesc.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  auto sampler = vg::Sampler::create(device(), sdesc);
  ASSERT_TRUE(sampler.ok()) << sampler.status().message();

  const VkDescriptorPoolSize pool_size{
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  auto pool = vg::DescriptorPool::create(device(), &pool_size, 1, 1);
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto atlas_set =
      pool.value().allocate(pipeline.value().descriptor_set_layout(0));
  ASSERT_TRUE(atlas_set.ok()) << atlas_set.status().message();
  atlas_set.value().write_combined_image_sampler(
      0, atlas_tex.value().view(), sampler.value().handle(),
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // Unlit: albedo passes straight through, so the two halves show their raw
  // sources.
  const std::vector<uint8_t> unlit = render(
      allocator.value(), pipeline.value(),
      pipelines::HybridMeshDraw{&mesh.value()}, atlas_set.value().handle(), 0u);
  ASSERT_EQ(unlit.size(), static_cast<size_t>(kSize) * kSize * 4);
  const auto at = [&](const std::vector<uint8_t>& px, uint32_t x, uint32_t y) {
    return &px[(static_cast<size_t>(y) * kSize + x) * 4];
  };

  const uint8_t* left = at(unlit, kSize / 4, kSize / 2);  // NDC x = -0.5: atlas
  EXPECT_GT(left[0], 128) << "left half should sample the red atlas texel";
  EXPECT_LT(left[1], 128);
  EXPECT_LT(left[2], 128);

  const uint8_t* right = at(unlit, 3 * kSize / 4, kSize / 2);  // NDC +0.5: vtx
  EXPECT_LT(right[0], 128) << "right half should use the green vertex color";
  EXPECT_GT(right[1], 128);
  EXPECT_LT(right[2], 128);

  // Lit: the front-facing quads darken by a KNOWN factor --
  //   ambient + (1 - ambient) * max(dot(n, l), 0),  n = (0,0,1), ambient =
  //   0.25, l = normalize(the default light_dir).
  // Asserting that specific value (not merely "dimmer than unlit") is what
  // proves the directional term is applied: dropping it would leave the
  // ambient-only ~0.25 * unlit (~64), which "dimmer than unlit" would still
  // accept.
  const std::vector<uint8_t> lit =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&mesh.value()},
             atlas_set.value().handle(), pipelines::kHybridMeshLit);
  ASSERT_EQ(lit.size(), static_cast<size_t>(kSize) * kSize * 4);

  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  const glm::vec3 l = glm::normalize(pipelines::HybridMeshFrame{}.light_dir);
  const float factor = 0.25f + 0.75f * std::fmax(glm::dot(n, l), 0.0f);

  const uint8_t* lit_left = at(lit, kSize / 4, kSize / 2);
  EXPECT_NEAR(lit_left[0], std::lround(left[0] * factor), 4)
      << "lit red = unlit red * directional factor (ambient-only would be ~64)";
  EXPECT_LT(lit_left[0], left[0]) << "lit red darker than unlit";

  const uint8_t* lit_right = at(lit, 3 * kSize / 4, kSize / 2);
  EXPECT_NEAR(lit_right[1], std::lround(right[1] * factor), 4)
      << "lit green = unlit green * directional factor";
  EXPECT_LT(lit_right[1], right[1]) << "lit green darker than unlit";
}

// A null atlas is a violated precondition (the fragment shader samples set 0
// unconditionally). submit() must record NOTHING rather than draw against an
// unbound descriptor set -- which would be UB and, under the validation layer
// this fixture runs with teeth, a VUID-vkCmdDraw-None-* error the fixture fails
// the test on. So the readback stays the clear color everywhere.
TEST_F(HybridMeshRenderTest, NullAtlasRecordsNothingInsteadOfDrawingUnbound) {
  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline =
      pipelines::HybridMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_hybrid_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  const std::vector<uint8_t> px =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&mesh.value()}, VK_NULL_HANDLE,
             pipelines::kHybridMeshLit);
  ASSERT_EQ(px.size(), static_cast<size_t>(kSize) * kSize * 4);

  // Opaque-black clear (render() sets only alpha): nothing drawn -> all
  // cleared.
  const auto at = [&](uint32_t x, uint32_t y) {
    return &px[(static_cast<size_t>(y) * kSize + x) * 4];
  };
  for (uint32_t x : {kSize / 4, 3 * kSize / 4}) {
    const uint8_t* p = at(x, kSize / 2);
    EXPECT_EQ(p[0], 0) << "no draw -> cleared pixel at x=" << x;
    EXPECT_EQ(p[1], 0) << "no draw -> cleared pixel at x=" << x;
    EXPECT_EQ(p[2], 0) << "no draw -> cleared pixel at x=" << x;
  }
}

// The indirect live-mesh draw must produce byte-identical pixels to the direct
// static draw of the same geometry -- that equivalence is the whole point of
// the indirect path (recon supplies the geometry plus a GPU-resident index
// count; the renderer draws it with vkCmdDrawIndexedIndirect). Renders the same
// mesh both ways into identical targets and compares the readbacks.
//
// Here the vertex/index buffers are uploaded (and fence-waited) and the
// indirect command is written from the CPU before the draw submit, so the
// writes are already visible with no explicit barrier. The real recon handoff
// -- a compute pass writing these buffers in-frame -- must insert the
// producer->draw barrier itself (see LiveMesh's synchronization warning); that
// seam belongs to the streamed-app driver, not this pipeline.
TEST_F(HybridMeshRenderTest, IndirectDrawMatchesDirectDraw) {
  auto allocator = vg::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline =
      pipelines::HybridMeshPipeline::create(device(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_hybrid_mesh();

  // Direct path: the static GpuMesh (owns its buffers, fixed index count).
  auto gpu = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(gpu.ok()) << gpu.status().message();

  // Indirect path: upload the same vertices + indices into device-local buffers
  // whose handles we keep, and fill a one-element indirect command with the
  // index count -- exactly what recon would populate GPU-side.
  auto batch = vg::UploadBatch::begin(*device_, allocator.value());
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  vg::BufferUploadDesc vdesc;
  vdesc.data = mesh_cpu.vertices.data();
  vdesc.size = VkDeviceSize{mesh_cpu.vertices.size()} * sizeof(assets::Vertex);
  vdesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  auto vbuf = batch.value().add_buffer(vdesc);
  ASSERT_TRUE(vbuf.ok()) << vbuf.status().message();
  vg::BufferUploadDesc idesc;
  idesc.data = mesh_cpu.indices.data();
  idesc.size = VkDeviceSize{mesh_cpu.indices.size()} * sizeof(uint32_t);
  idesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  auto ibuf = batch.value().add_buffer(idesc);
  ASSERT_TRUE(ibuf.ok()) << ibuf.status().message();
  ASSERT_TRUE(batch.value().finish().ok());

  VkDrawIndexedIndirectCommand command{};
  command.indexCount = static_cast<uint32_t>(mesh_cpu.indices.size());
  command.instanceCount = 1;
  vg::BufferDesc cmd_desc;
  cmd_desc.size = sizeof(command);
  cmd_desc.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  cmd_desc.memory = vg::MemoryUsage::HostVisible;
  cmd_desc.mapped = true;
  auto indbuf = allocator.value().create_buffer(cmd_desc);
  ASSERT_TRUE(indbuf.ok()) << indbuf.status().message();
  std::memcpy(indbuf.value().mapped(), &command, sizeof(command));

  pipelines::LiveMesh live;
  live.vertices = vbuf.value().handle();
  live.indices = ibuf.value().handle();
  live.indirect = indbuf.value().handle();
  ASSERT_TRUE(live.valid());

  // A 1x1 white atlas: both paths sample the same set, so its content is
  // irrelevant to the comparison -- it only satisfies submit()'s always-bound
  // atlas precondition.
  const uint8_t white_px[4] = {255, 255, 255, 255};
  vg::ImageUploadDesc adesc;
  adesc.extent = {1, 1};
  adesc.format = kColorFormat;
  adesc.pixels = white_px;
  adesc.size = sizeof(white_px);
  auto atlas_tex = vg::upload_texture(*device_, allocator.value(), adesc);
  ASSERT_TRUE(atlas_tex.ok()) << atlas_tex.status().message();

  vg::SamplerDesc sdesc;
  sdesc.mag_filter = VK_FILTER_NEAREST;
  sdesc.min_filter = VK_FILTER_NEAREST;
  sdesc.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sdesc.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sdesc.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sdesc.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  auto sampler = vg::Sampler::create(device(), sdesc);
  ASSERT_TRUE(sampler.ok()) << sampler.status().message();

  const VkDescriptorPoolSize pool_size{
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  auto pool = vg::DescriptorPool::create(device(), &pool_size, 1, 1);
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  auto atlas_set =
      pool.value().allocate(pipeline.value().descriptor_set_layout(0));
  ASSERT_TRUE(atlas_set.ok()) << atlas_set.status().message();
  atlas_set.value().write_combined_image_sampler(
      0, atlas_tex.value().view(), sampler.value().handle(),
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  const std::vector<uint8_t> direct = render(
      allocator.value(), pipeline.value(),
      pipelines::HybridMeshDraw{&gpu.value()}, atlas_set.value().handle(), 0u);
  const std::vector<uint8_t> indirect =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{live}, atlas_set.value().handle(), 0u);

  ASSERT_EQ(direct.size(), static_cast<size_t>(kSize) * kSize * 4);
  ASSERT_EQ(indirect.size(), direct.size());
  // Same geometry, same pipeline, same viewport -> identical rasterization.
  EXPECT_EQ(direct, indirect);
}

}  // namespace

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HybridMeshPipeline: reflected-layout creation, move semantics, and an
// end-to-end offscreen draw of the reconstruction hybrid mesh. The draw test
// renders a mesh whose left half carries atlas coordinates (sampled from a 2x2
// texture whose first texel is red) and whose right half carries the (-1, -1)
// "use vertex color" sentinel with a green per-vertex color -- proving both
// shading paths route correctly in one draw -- then re-renders lit to prove the
// directional term darkens the albedo, and in the normal debug view to prove it
// encodes the stored normal instead. Runs under the validation layer with
// teeth. Skips when the runner exposes no Vulkan device.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/assets/mesh.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/sampler.hpp"
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
// target the draw renders into). `color` is UNORM unless a test needs sRGB.
vg::RenderTargetLayout color_depth_layout(VkFormat color = kColorFormat) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = color;
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

// make_hybrid_mesh() re-normaled for the normal debug view, so each half pins
// a different channel to its axis: the left (atlas) quad stores a NON-unit +X
// normal -- red saturates only if the view normalizes -- and the right
// (vertex-color) quad stores -Y and is re-wound to face AWAY from the camera,
// so green reads 0 only if the view shows the stored normal rather than the
// back-face flip lit shading applies.
assets::Mesh make_normals_mesh() {
  assets::Mesh mesh = make_hybrid_mesh();
  for (size_t i = 0; i < 4; ++i) {
    mesh.vertices[i].normal = {2.0f, 0.0f, 0.0f};  // left: +X, non-unit
  }
  for (size_t i = 4; i < 8; ++i) {
    mesh.vertices[i].normal = {0.0f, -1.0f, 0.0f};  // right: -Y
  }
  // The right quad is the last two triangles; swapping each one's last two
  // indices reverses its winding, so it turns back-facing.
  for (size_t t = 6; t < mesh.indices.size(); t += 3) {
    std::swap(mesh.indices[t + 1], mesh.indices[t + 2]);
  }
  return mesh;
}

// A quad over NDC [-0.5, 0.5]^2, NEARER the camera than make_hybrid_mesh()'s
// z = 0.5, carrying the sentinel uv + a blue vertex color. Drawn after the
// full-screen mesh it wins the LESS depth test over the middle of the frame and
// covers nothing else -- so a frame containing it is distinguishable from one
// that does not. Drawing the *same* geometry twice would not be: identical
// opaque geometry is rejected by the depth test and writes no color, which
// would make a mixed-draw test pass with the live branch deleted.
assets::Mesh make_center_quad() {
  // Blue: neither the atlas red nor the full-screen mesh's green, so a pixel it
  // covers is unambiguously this draw's.
  const glm::vec4 blue{0.0f, 0.0f, 1.0f, 1.0f};
  auto vert = [&](float x, float y) {
    assets::Vertex v;
    v.position = {x, y, 0.25f};  // nearer than the full-screen mesh
    v.normal = {0.0f, 0.0f, 1.0f};
    v.uv0 = {-1.0f, -1.0f};  // sentinel -> per-vertex color
    v.color = blue;
    return v;
  };

  assets::Mesh mesh;
  mesh.vertices = {
      vert(-0.5f, -0.5f),
      vert(0.5f, -0.5f),
      vert(0.5f, 0.5f),
      vert(-0.5f, 0.5f),
  };
  mesh.indices = {0, 2, 1, 0, 3, 2};  // same front-facing winding
  return mesh;
}

// True if any pixel departs from the opaque-black clear -- i.e. the mesh
// actually rasterized. Guards the direct-vs-indirect equivalence tests against
// a shared regression that clears BOTH paths and so compares equal but empty.
bool any_pixel_drawn(const std::vector<uint8_t>& px) {
  for (size_t i = 0; i + 4 <= px.size(); i += 4) {
    if (px[i] != 0 || px[i + 1] != 0 || px[i + 2] != 0) {
      return true;
    }
  }
  return false;
}

// The RGBA pixel at (x, y) of a kSize x kSize readback.
const uint8_t* pixel_at(const std::vector<uint8_t>& px, uint32_t x,
                        uint32_t y) {
  return &px[(static_cast<size_t>(y) * kSize + x) * 4];
}

// True if every pixel is exactly the opaque-black clear -- i.e. nothing drew.
bool all_pixels_cleared(const std::vector<uint8_t>& px) {
  if (px.empty()) {
    return false;
  }
  for (size_t i = 0; i + 4 <= px.size(); i += 4) {
    if (px[i] != 0 || px[i + 1] != 0 || px[i + 2] != 0 || px[i + 3] != 255) {
      return false;
    }
  }
  return true;
}

// A device-mapped buffer of `pad + size` bytes with `size` bytes of
// `data` copied in at byte offset `pad`. A non-zero pad lets a test bind the
// data at a non-zero (4-aligned) offset, exercising LiveMesh's offset fields.
vkc::Buffer host_buffer_at(vkc::Allocator& allocator, const void* data,
                           VkDeviceSize size, VkBufferUsageFlags usage,
                           VkDeviceSize pad) {
  vkc::BufferDesc desc;
  desc.size = pad + size;
  desc.usage = usage;
  desc.memory = vkc::MemoryUsage::DeviceMapped;
  auto buf = allocator.create_buffer(desc);
  if (!buf.ok()) {
    ADD_FAILURE() << buf.status().message();
    return {};
  }
  std::memcpy(static_cast<uint8_t*>(buf.value().mapped()) + pad, data, size);
  return std::move(buf).value();
}

// The three device-mapped buffers a LiveMesh borrows; the caller keeps them
// alive (the LiveMesh only names their handles).
struct LiveBuffers {
  vkc::Buffer vertices;
  vkc::Buffer indices;
  vkc::Buffer indirect;
};

// The well-formed command for `mesh`: its whole index run, one instance, no
// element offsets -- what the handoff contract requires a producer to write.
VkDrawIndexedIndirectCommand draw_command(const assets::Mesh& mesh) {
  VkDrawIndexedIndirectCommand command{};
  command.indexCount = static_cast<uint32_t>(mesh.indices.size());
  command.instanceCount = 1;
  return command;
}

// Builds a LiveMesh that draws `mesh` under `command`, backed by three
// device-mapped buffers each carrying `pad` leading bytes -- a non-zero `pad`
// (a multiple of 4) exercises the vertex/index/indirect bind offsets. The
// owning buffers land in `out`, which must outlive the draw.
pipelines::LiveMesh make_live_mesh(vkc::Allocator& allocator,
                                   const assets::Mesh& mesh, VkDeviceSize pad,
                                   const VkDrawIndexedIndirectCommand& command,
                                   LiveBuffers& out) {
  const VkDeviceSize vbytes =
      VkDeviceSize{mesh.vertices.size()} * sizeof(assets::Vertex);
  const VkDeviceSize ibytes =
      VkDeviceSize{mesh.indices.size()} * sizeof(uint32_t);

  out.vertices = host_buffer_at(allocator, mesh.vertices.data(), vbytes,
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, pad);
  out.indices = host_buffer_at(allocator, mesh.indices.data(), ibytes,
                               VK_BUFFER_USAGE_INDEX_BUFFER_BIT, pad);
  out.indirect = host_buffer_at(allocator, &command, sizeof(command),
                                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, pad);

  pipelines::LiveMesh live;
  live.vertices = out.vertices.handle();
  live.vertex_offset = pad;
  live.indices = out.indices.handle();
  live.index_offset = pad;
  live.indirect = out.indirect.handle();
  live.indirect_offset = pad;
  return live;
}

// Same, under the contract's well-formed command -- the common case.
pipelines::LiveMesh make_live_mesh(vkc::Allocator& allocator,
                                   const assets::Mesh& mesh, VkDeviceSize pad,
                                   LiveBuffers& out) {
  return make_live_mesh(allocator, mesh, pad, draw_command(mesh), out);
}

// A set-0 combined-image-sampler plus the objects it points at (texture,
// sampler, pool). The set handle is bound at draw time; the owning objects live
// in a test-body vector so they outlive the render() calls yet are destroyed
// before the allocator that produced them.
struct AtlasResources {
  vkc::Image texture;
  vg::Sampler sampler;
  vkc::DescriptorPool pool;
  VkDescriptorSet set = VK_NULL_HANDLE;
};

// --- Creation validation: no device needed -----------------------------------

TEST(HybridMeshPipelineTest, DefaultConstructedIsEmpty) {
  pipelines::HybridMeshPipeline pipeline;
  EXPECT_FALSE(pipeline.valid());
  EXPECT_EQ(pipeline.handle(), VK_NULL_HANDLE);
  EXPECT_EQ(pipeline.layout(), VK_NULL_HANDLE);
}

// --- Creation + move semantics: needs a device -------------------------------

class HybridMeshPipelineDeviceTest : public VulkanDeviceTest {
 protected:
  void SetUp() override {
    VulkanDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
    ASSERT_TRUE(allocator.ok()) << allocator.status().message();
    allocator_.emplace(std::move(allocator).value());
  }

  vkc::Result<pipelines::HybridMeshPipeline> create(
      const vg::RenderTargetLayout& layout = color_depth_layout()) {
    return pipelines::HybridMeshPipeline::create(*device_, *allocator_, layout);
  }

  std::optional<vkc::Allocator> allocator_;
};

TEST_F(HybridMeshPipelineDeviceTest, CreatesWithReflectedAtlasSet) {
  auto pipeline = create();
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
  auto pipeline = create(layout);
  ASSERT_FALSE(pipeline.ok());
  EXPECT_EQ(pipeline.status().domain(), vkc::Status::Code::InvalidArgument);
}

TEST_F(HybridMeshPipelineDeviceTest, MoveLeavesSourceEmpty) {
  auto created = create();
  ASSERT_TRUE(created.ok()) << created.status().message();
  pipelines::HybridMeshPipeline source = std::move(created).value();
  ASSERT_TRUE(source.valid());

  pipelines::HybridMeshPipeline moved(std::move(source));
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(source.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_F(HybridMeshPipelineDeviceTest, MoveAssignOverLiveObjectAdoptsSource) {
  auto first = create();
  ASSERT_TRUE(first.ok()) << first.status().message();
  auto second = create();
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
  auto created = create();
  ASSERT_TRUE(created.ok()) << created.status().message();
  pipelines::HybridMeshPipeline pipeline = std::move(created).value();
  const VkPipeline before = pipeline.handle();
  ASSERT_EQ(pipeline.descriptor_set_count(), 1u);  // set 0, the atlas sampler
  const VkDescriptorSetLayout layout_before = pipeline.descriptor_set_layout(0);
  ASSERT_NE(layout_before, VK_NULL_HANDLE);

  // Launder through a pointer so -Wself-move does not fire under -Werror.
  pipelines::HybridMeshPipeline* alias = &pipeline;
  pipeline = std::move(*alias);
  EXPECT_TRUE(pipeline.valid());
  EXPECT_EQ(pipeline.handle(), before);

  // The embedded GraphicsPipeline also owns the reflected set layout, and an
  // unguarded move-assign would free it while handle()/valid() stayed intact.
  ASSERT_EQ(pipeline.descriptor_set_count(), 1u);
  EXPECT_EQ(pipeline.descriptor_set_layout(0), layout_before);
}

// --- End-to-end offscreen draw: validation-with-teeth ------------------------

class HybridMeshRenderTest : public VulkanDeviceTest {
 protected:
  bool wants_validation() const override { return true; }

  // Renders `draws` (bound with `atlas` as set 0) into a fresh offscreen target
  // with the given flags, and returns a copy of the RGBA pixels. Each draw
  // names either a static GpuMesh or a live indirect mesh -- the harness is
  // agnostic, and a draw_count > 1 list exercises submit()'s per-draw dispatch.
  // `color_format` must match the layout `pipeline` was created for.
  std::vector<uint8_t> render(
      vkc::Allocator& allocator, const pipelines::HybridMeshPipeline& pipeline,
      const std::vector<pipelines::HybridMeshDraw>& draws,
      VkDescriptorSet atlas, uint32_t flags,
      VkFormat color_format = kColorFormat) {
    vg::OffscreenTargetDesc td;
    td.extent = {kSize, kSize};
    td.color_format = color_format;
    td.depth_format = kDepthFormat;
    auto target = vg::OffscreenTarget::create(allocator, td);
    EXPECT_TRUE(target.ok()) << target.status().message();
    if (!target.ok()) {
      return {};
    }

    auto pool = vkc::CommandPool::create(device(), device_->queue_family());
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
    frame.draws = draws.data();
    frame.draw_count = static_cast<uint32_t>(draws.size());
    pipeline.submit(raw, frame);

    rt.end(raw);
    target.value().record_readback(raw);
    EXPECT_TRUE(cmd.value().end().ok());
    submit_and_wait(raw);
    // This helper returns a value, so ASSERT_NO_FATAL_FAILURE (which expands to
    // a bare `return`) cannot be used; check explicitly instead. A failed
    // submit or fence wait leaves the GPU potentially still reading `target`,
    // so bail out rather than read back and then destroy it underneath.
    if (HasFatalFailure()) {
      return {};
    }

    const auto* px = static_cast<const uint8_t*>(target.value().pixels());
    EXPECT_NE(px, nullptr);
    if (px == nullptr) {
      return {};
    }
    return std::vector<uint8_t>(px,
                                px + static_cast<size_t>(kSize) * kSize * 4);
  }

  // Single-draw convenience -- the common case is one mesh per frame.
  std::vector<uint8_t> render(vkc::Allocator& allocator,
                              const pipelines::HybridMeshPipeline& pipeline,
                              const pipelines::HybridMeshDraw& draw,
                              VkDescriptorSet atlas, uint32_t flags,
                              VkFormat color_format = kColorFormat) {
    return render(allocator, pipeline,
                  std::vector<pipelines::HybridMeshDraw>{draw}, atlas, flags,
                  color_format);
  }

  // Uploads `pixels` (`extent`, kColorFormat, `size` bytes) as a NEAREST atlas
  // and writes it into a fresh set-0 combined-image-sampler. Appends the
  // texture, sampler, and pool to `keep` (a caller-owned local that must
  // outlive the render, but be destroyed before its allocator) and returns the
  // set handle (VK_NULL_HANDLE on failure).
  VkDescriptorSet make_atlas(vkc::Allocator& allocator,
                             const pipelines::HybridMeshPipeline& pipeline,
                             const void* pixels, VkExtent2D extent,
                             VkDeviceSize size,
                             std::vector<AtlasResources>& keep) {
    vg::ImageUploadDesc adesc;
    adesc.extent = extent;
    adesc.format = kColorFormat;
    adesc.pixels = pixels;
    adesc.size = size;
    auto texture = vg::upload_texture(*device_, allocator, adesc);
    if (!texture.ok()) {
      ADD_FAILURE() << texture.status().message();
      return VK_NULL_HANDLE;
    }

    vg::SamplerDesc sdesc;
    sdesc.mag_filter = VK_FILTER_NEAREST;
    sdesc.min_filter = VK_FILTER_NEAREST;
    sdesc.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sdesc.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sdesc.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sdesc.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    auto sampler = vg::Sampler::create(device(), sdesc);
    if (!sampler.ok()) {
      ADD_FAILURE() << sampler.status().message();
      return VK_NULL_HANDLE;
    }

    const VkDescriptorPoolSize pool_size{
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    auto pool = vkc::DescriptorPool::create(device(), &pool_size, 1, 1);
    if (!pool.ok()) {
      ADD_FAILURE() << pool.status().message();
      return VK_NULL_HANDLE;
    }
    auto set = pool.value().allocate(pipeline.descriptor_set_layout(0));
    if (!set.ok()) {
      ADD_FAILURE() << set.status().message();
      return VK_NULL_HANDLE;
    }
    set.value().write_combined_image_sampler(
        0, texture.value().view(), sampler.value().handle(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    keep.push_back(
        AtlasResources{std::move(texture).value(), std::move(sampler).value(),
                       std::move(pool).value(), set.value().handle()});
    return keep.back().set;
  }
};

TEST_F(HybridMeshRenderTest, RoutesAtlasAndVertexColor) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_hybrid_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  // A 2x2 atlas: texel (0,0) red, the rest blue. uv0 (0.25, 0.25) + NEAREST
  // reads the red texel, so a working atlas path shows red (and a broken
  // sentinel would leak red into the vertex-color half instead of green).
  const uint8_t atlas_px[16] = {255, 0, 0,   255, 0, 0, 255, 255,
                                0,   0, 255, 255, 0, 0, 255, 255};
  std::vector<AtlasResources> atlas_res;
  const VkDescriptorSet atlas =
      make_atlas(allocator.value(), pipeline.value(), atlas_px, {2, 2},
                 sizeof(atlas_px), atlas_res);
  ASSERT_NE(atlas, VK_NULL_HANDLE);

  // Unlit: albedo passes straight through, so the two halves show their raw
  // sources.
  const std::vector<uint8_t> unlit =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&mesh.value()}, atlas, 0u);
  ASSERT_EQ(unlit.size(), static_cast<size_t>(kSize) * kSize * 4);

  const uint8_t* left =
      pixel_at(unlit, kSize / 4, kSize / 2);  // NDC x = -0.5: atlas
  EXPECT_GT(left[0], 128) << "left half should sample the red atlas texel";
  EXPECT_LT(left[1], 128);
  EXPECT_LT(left[2], 128);

  const uint8_t* right =
      pixel_at(unlit, 3 * kSize / 4, kSize / 2);  // NDC +0.5: vtx
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
             pipelines::HybridMeshDraw{&mesh.value()}, atlas,
             pipelines::kHybridMeshLit);
  ASSERT_EQ(lit.size(), static_cast<size_t>(kSize) * kSize * 4);

  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  const glm::vec3 l = glm::normalize(pipelines::HybridMeshFrame{}.light_dir);
  const float factor = 0.25f + 0.75f * std::fmax(glm::dot(n, l), 0.0f);

  const uint8_t* lit_left = pixel_at(lit, kSize / 4, kSize / 2);
  EXPECT_NEAR(lit_left[0], std::lround(left[0] * factor), 4)
      << "lit red = unlit red * directional factor (ambient-only would be ~64)";
  EXPECT_LT(lit_left[0], left[0]) << "lit red darker than unlit";

  const uint8_t* lit_right = pixel_at(lit, 3 * kSize / 4, kSize / 2);
  EXPECT_NEAR(lit_right[1], std::lround(right[1] * factor), 4)
      << "lit green = unlit green * directional factor";
  EXPECT_LT(lit_right[1], right[1]) << "lit green darker than unlit";
}

// kHybridMeshNormals shows each fragment's world-space normal as
// normalize(n) * 0.5 + 0.5 in place of its albedo, on make_normals_mesh()
// (see there for what each half pins). Both halves encoding their normal (not
// red/green albedo) proves the albedo is bypassed, and the back-facing right
// quad reading exactly what a front-facing -Y would is the documented
// winding-blindness.
TEST_F(HybridMeshRenderTest, NormalsViewEncodesTheUnflippedMeshNormal) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_normals_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  const uint8_t atlas_px[4] = {255, 0, 0, 255};  // red, never shown here
  std::vector<AtlasResources> atlas_res;
  const VkDescriptorSet atlas =
      make_atlas(allocator.value(), pipeline.value(), atlas_px, {1, 1},
                 sizeof(atlas_px), atlas_res);
  ASSERT_NE(atlas, VK_NULL_HANDLE);

  const pipelines::HybridMeshDraw draw{&mesh.value()};

  // Guard against a vacuous pass: lit shading flips the right quad's -Y to +Y
  // only if it really is back-facing, and +Y then catches the default light's
  // directional term. Front-facing, -Y would face away and leave ambient only.
  const std::vector<uint8_t> lit =
      render(allocator.value(), pipeline.value(), draw, atlas,
             pipelines::kHybridMeshLit);
  ASSERT_EQ(lit.size(), static_cast<size_t>(kSize) * kSize * 4);
  const glm::vec3 l = glm::normalize(pipelines::HybridMeshFrame{}.light_dir);
  EXPECT_NEAR(pixel_at(lit, 3 * kSize / 4, kSize / 2)[1],
              std::lround(255.0f * (0.25f + 0.75f * l.y)), 4)
      << "the right quad must be back-facing for the no-flip check to bite";

  const std::vector<uint8_t> normals =
      render(allocator.value(), pipeline.value(), draw, atlas,
             pipelines::kHybridMeshNormals);
  ASSERT_EQ(normals.size(), static_cast<size_t>(kSize) * kSize * 4);

  // 0.5 lands on 127 or 128 depending on the implementation's rounding.
  const uint8_t* left =
      pixel_at(normals, kSize / 4, kSize / 2);  // +X, normalized
  EXPECT_EQ(left[0], 255) << "+X normalizes to full red";
  EXPECT_NEAR(left[1], 128, 1);
  EXPECT_NEAR(left[2], 128, 1);

  const uint8_t* right =
      pixel_at(normals, 3 * kSize / 4, kSize / 2);  // -Y, back
  EXPECT_NEAR(right[0], 128, 1);
  EXPECT_EQ(right[1], 0) << "back face shows its stored -Y, not the flip";
  EXPECT_NEAR(right[2], 128, 1);

  // The view takes precedence over the lit flag: adding it changes nothing.
  const std::vector<uint8_t> normals_lit =
      render(allocator.value(), pipeline.value(), draw, atlas,
             pipelines::kHybridMeshNormals | pipelines::kHybridMeshLit);
  EXPECT_EQ(normals_lit, normals)
      << "kHybridMeshNormals ignores kHybridMeshLit";
}

// A zero normal has no direction to encode: safe_normalize() maps it to 0, so
// the view shows mid-grey rather than the NaN a bare normalize() would write.
TEST_F(HybridMeshRenderTest, NormalsViewShowsAZeroNormalAsMidGrey) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  assets::Mesh mesh_cpu = make_hybrid_mesh();
  for (assets::Vertex& v : mesh_cpu.vertices) {
    v.normal = {0.0f, 0.0f, 0.0f};
  }
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  const uint8_t atlas_px[4] = {255, 0, 0, 255};  // red, never shown here
  std::vector<AtlasResources> atlas_res;
  const VkDescriptorSet atlas =
      make_atlas(allocator.value(), pipeline.value(), atlas_px, {1, 1},
                 sizeof(atlas_px), atlas_res);
  ASSERT_NE(atlas, VK_NULL_HANDLE);

  const std::vector<uint8_t> normals =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&mesh.value()}, atlas,
             pipelines::kHybridMeshNormals);
  ASSERT_EQ(normals.size(), static_cast<size_t>(kSize) * kSize * 4);
  for (uint32_t x : {kSize / 4, 3 * kSize / 4}) {
    const uint8_t* p = pixel_at(normals, x, kSize / 2);
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(p[c], 128, 1)
          << "zero normal -> mid-grey at x=" << x << ", channel " << c;
    }
  }
}

// On an sRGB target the view pre-decodes its encoding, so the byte stored is
// still normalize(n) * 0.5 + 0.5 -- what other normal-map tools show. Written
// as-is, the hardware's encode would lift 0.5 to ~188, so the 128s below are
// what bite.
TEST_F(HybridMeshRenderTest, NormalsViewStoresTheEncodingOnAnSrgbTarget) {
  constexpr VkFormat kSrgb = VK_FORMAT_R8G8B8A8_SRGB;
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout(kSrgb));
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_normals_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  const uint8_t atlas_px[4] = {255, 0, 0, 255};  // red, never shown here
  std::vector<AtlasResources> atlas_res;
  const VkDescriptorSet atlas =
      make_atlas(allocator.value(), pipeline.value(), atlas_px, {1, 1},
                 sizeof(atlas_px), atlas_res);
  ASSERT_NE(atlas, VK_NULL_HANDLE);

  const std::vector<uint8_t> normals =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&mesh.value()}, atlas,
             pipelines::kHybridMeshNormals, kSrgb);
  ASSERT_EQ(normals.size(), static_cast<size_t>(kSize) * kSize * 4);

  const uint8_t* left = pixel_at(normals, kSize / 4, kSize / 2);  // +X
  EXPECT_EQ(left[0], 255);
  EXPECT_NEAR(left[1], 128, 1) << "0.5 stored as-is, not sRGB-lifted";
  EXPECT_NEAR(left[2], 128, 1);

  const uint8_t* right = pixel_at(normals, 3 * kSize / 4, kSize / 2);  // -Y
  EXPECT_NEAR(right[0], 128, 1);
  EXPECT_EQ(right[1], 0);
  EXPECT_NEAR(right[2], 128, 1);
}

// Bits HybridMeshFlags does not name are reserved, and the shader tests only
// the named ones, so setting every reserved bit changes no pixel.
TEST_F(HybridMeshRenderTest, ReservedFlagBitsAreIgnored) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_hybrid_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  const uint8_t atlas_px[4] = {255, 0, 0, 255};
  std::vector<AtlasResources> atlas_res;
  const VkDescriptorSet atlas =
      make_atlas(allocator.value(), pipeline.value(), atlas_px, {1, 1},
                 sizeof(atlas_px), atlas_res);
  ASSERT_NE(atlas, VK_NULL_HANDLE);

  const pipelines::HybridMeshDraw draw{&mesh.value()};
  constexpr uint32_t kReserved =
      ~uint32_t{pipelines::kHybridMeshLit | pipelines::kHybridMeshNormals |
                pipelines::kHybridMeshVertexColor};
  const std::vector<uint8_t> lit =
      render(allocator.value(), pipeline.value(), draw, atlas,
             pipelines::kHybridMeshLit);
  ASSERT_TRUE(any_pixel_drawn(lit));
  EXPECT_EQ(render(allocator.value(), pipeline.value(), draw, atlas,
                   pipelines::kHybridMeshLit | kReserved),
            lit)
      << "a reserved bit must not switch the shading mode";
}

// make_hybrid_mesh() with its atlas half carrying a magenta vertex color --
// neither an atlas texel's color nor the fallback's white -- so a pixel there
// shows which source the shader chose.
assets::Mesh make_magenta_atlas_half_mesh() {
  assets::Mesh mesh = make_hybrid_mesh();
  for (size_t i = 0; i < 4; ++i) {
    mesh.vertices[i].color = {1.0f, 0.0f, 1.0f, 1.0f};
  }
  return mesh;
}

// A frame with no atlas binds the pipeline's fallback and draws every triangle
// in its vertex color, the atlas half's too: magenta, not the fallback's
// white. Under the validation layer with teeth, a draw against an unbound or
// undefined set would fail the test.
TEST_F(HybridMeshRenderTest, NullAtlasDrawsEveryTriangleInItsVertexColor) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_magenta_atlas_half_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  const std::vector<uint8_t> px =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&mesh.value()}, VK_NULL_HANDLE, 0u);
  ASSERT_EQ(px.size(), static_cast<size_t>(kSize) * kSize * 4);

  const uint8_t* left = pixel_at(px, kSize / 4, kSize / 2);
  EXPECT_EQ(left[0], 255) << "the atlas half draws its magenta vertex color";
  EXPECT_EQ(left[1], 0);
  EXPECT_EQ(left[2], 255);
  const uint8_t* right = pixel_at(px, 3 * kSize / 4, kSize / 2);
  EXPECT_EQ(right[0], 0) << "the vertex-color half is unchanged";
  EXPECT_EQ(right[1], 255);
  EXPECT_EQ(right[2], 0);
}

// kHybridMeshVertexColor shows the vertex color where an atlas is bound and
// uv0 would sample it; without it the same frame shows the atlas.
TEST_F(HybridMeshRenderTest, VertexColorFlagIgnoresABoundAtlas) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_magenta_atlas_half_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();

  const uint8_t red_px[4] = {255, 0, 0, 255};
  std::vector<AtlasResources> atlas_res;
  const VkDescriptorSet atlas =
      make_atlas(allocator.value(), pipeline.value(), red_px, {1, 1},
                 sizeof(red_px), atlas_res);
  ASSERT_NE(atlas, VK_NULL_HANDLE);

  const pipelines::HybridMeshDraw draw{&mesh.value()};
  const std::vector<uint8_t> textured =
      render(allocator.value(), pipeline.value(), draw, atlas, 0u);
  ASSERT_EQ(textured.size(), static_cast<size_t>(kSize) * kSize * 4);
  EXPECT_EQ(pixel_at(textured, kSize / 4, kSize / 2)[2], 0)
      << "without the flag the atlas half samples the red atlas";

  const std::vector<uint8_t> flagged =
      render(allocator.value(), pipeline.value(), draw, atlas,
             pipelines::kHybridMeshVertexColor);
  ASSERT_EQ(flagged.size(), textured.size());
  const uint8_t* left = pixel_at(flagged, kSize / 4, kSize / 2);
  EXPECT_EQ(left[0], 255) << "the flag shows the magenta vertex color";
  EXPECT_EQ(left[1], 0);
  EXPECT_EQ(left[2], 255);
}

// The fallback set travels with the pipeline: after a move-construct, a
// move-assign over a live pipeline and a self-move, a frame with no atlas
// still draws, against a defined set (validation with teeth), and the
// moved-from pipeline records nothing.
TEST_F(HybridMeshRenderTest, MovesKeepTheFallback) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  const assets::Mesh mesh_cpu = make_magenta_atlas_half_mesh();
  auto mesh = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(mesh.ok()) << mesh.status().message();
  const pipelines::HybridMeshDraw draw{&mesh.value()};

  auto created = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(created.ok()) << created.status().message();
  auto other = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(other.ok()) << other.status().message();

  pipelines::HybridMeshPipeline moved(std::move(created).value());
  pipelines::HybridMeshPipeline assigned = std::move(other).value();
  assigned = std::move(moved);
  pipelines::HybridMeshPipeline* alias = &assigned;
  assigned = std::move(*alias);

  const std::vector<uint8_t> px =
      render(allocator.value(), assigned, draw, VK_NULL_HANDLE, 0u);
  ASSERT_EQ(px.size(), static_cast<size_t>(kSize) * kSize * 4);
  EXPECT_EQ(pixel_at(px, kSize / 4, kSize / 2)[2], 255)
      << "the moved pipeline draws with its fallback";
  // NOLINTNEXTLINE(bugprone-use-after-move)
  const std::vector<uint8_t> empty =
      render(allocator.value(), moved, draw, VK_NULL_HANDLE, 0u);
  EXPECT_TRUE(all_pixels_cleared(empty))
      << "the moved-from pipeline records nothing";
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
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
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
  vkc::BufferDesc cmd_desc;
  cmd_desc.size = sizeof(command);
  cmd_desc.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  cmd_desc.memory = vkc::MemoryUsage::DeviceMapped;
  auto indbuf = allocator.value().create_buffer(cmd_desc);
  ASSERT_TRUE(indbuf.ok()) << indbuf.status().message();
  std::memcpy(indbuf.value().mapped(), &command, sizeof(command));

  pipelines::LiveMesh live;
  live.vertices = vbuf.value().handle();
  live.indices = ibuf.value().handle();
  live.indirect = indbuf.value().handle();
  ASSERT_TRUE(live.valid());

  const std::vector<uint8_t> direct =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&gpu.value()}, VK_NULL_HANDLE, 0u);
  const std::vector<uint8_t> indirect =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{live}, VK_NULL_HANDLE, 0u);

  ASSERT_EQ(direct.size(), static_cast<size_t>(kSize) * kSize * 4);
  ASSERT_EQ(indirect.size(), direct.size());
  // Same geometry, same pipeline, same viewport -> identical rasterization...
  EXPECT_EQ(direct, indirect);
  // ...and non-vacuously so: the mesh actually rasterized. A shared regression
  // that cleared both paths would still compare equal, but against an empty
  // frame -- this guards that failure mode.
  EXPECT_TRUE(any_pixel_drawn(direct));
}

// submit() must skip a draw whose geometry is empty -- an unbound (default)
// LiveMesh or a null static GpuMesh -- rather than record a draw against
// VK_NULL_HANDLE buffers, which the validation-with-teeth this fixture runs
// would fail on. With every draw empty, the readback stays cleared.
TEST_F(HybridMeshRenderTest, SkipsEmptyGeometryInsteadOfDrawingUnbound) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  // A default LiveMesh (no buffers) and a null static GpuMesh -- both fail
  // their valid() gate, so the loop records nothing for either.
  const std::vector<pipelines::HybridMeshDraw> draws = {
      pipelines::HybridMeshDraw{pipelines::LiveMesh{}},
      pipelines::HybridMeshDraw{
          static_cast<const pipelines::GpuMesh*>(nullptr)},
  };
  const std::vector<uint8_t> px =
      render(allocator.value(), pipeline.value(), draws, VK_NULL_HANDLE, 0u);
  ASSERT_EQ(px.size(), static_cast<size_t>(kSize) * kSize * 4);
  EXPECT_TRUE(all_pixels_cleared(px)) << "empty draws must record nothing";
}

// The three byte offsets on LiveMesh are its sub-allocation seam: a producer
// packs many meshes into shared pools and binds each at a non-zero offset. Draw
// the mesh from buffers whose data sits past a non-zero (4-aligned) pad and
// require the result to match the direct draw -- if record_draw ignored any
// offset it would fetch the zeroed pad instead of the mesh, and differ.
TEST_F(HybridMeshRenderTest, IndirectDrawHonorsBufferOffsets) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_hybrid_mesh();
  auto gpu = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(gpu.ok()) << gpu.status().message();

  // 256 leading bytes on every buffer -- a multiple of 4, so it meets the
  // index/indirect (and MoltenVK vertex) offset alignment the contract states.
  LiveBuffers buffers;
  const pipelines::LiveMesh live =
      make_live_mesh(allocator.value(), mesh_cpu, /*pad=*/256, buffers);
  ASSERT_TRUE(live.valid());
  ASSERT_NE(live.vertex_offset, 0u);

  const std::vector<uint8_t> direct =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&gpu.value()}, VK_NULL_HANDLE, 0u);
  const std::vector<uint8_t> offset =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{live}, VK_NULL_HANDLE, 0u);

  ASSERT_EQ(direct.size(), offset.size());
  EXPECT_EQ(direct, offset) << "non-zero bind offsets must be honored";
  EXPECT_TRUE(any_pixel_drawn(offset));
}

// One frame can carry both a static GpuMesh and a live LiveMesh; submit()
// dispatches per draw. The two draws must be distinguishable in the readback
// for that to be provable, so the live mesh is a *different* mesh -- a blue
// centre quad nearer the camera -- rather than a second copy of the static one
// (which the LESS depth test would reject, leaving a frame identical to the
// static draw alone and a test that passes with the live branch deleted).
// Render the static mesh alone, then both, and require each draw to own its
// region: blue in the centre only when the live draw ran, and the static mesh's
// own shading unchanged outside it.
TEST_F(HybridMeshRenderTest, MixedStaticAndLiveInOneFrame) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_hybrid_mesh();
  auto gpu = pipelines::upload_mesh(*device_, allocator.value(), mesh_cpu);
  ASSERT_TRUE(gpu.ok()) << gpu.status().message();

  const assets::Mesh quad_cpu = make_center_quad();
  LiveBuffers buffers;
  const pipelines::LiveMesh live =
      make_live_mesh(allocator.value(), quad_cpu, /*pad=*/0, buffers);
  ASSERT_TRUE(live.valid());

  const std::vector<uint8_t> single =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{&gpu.value()}, VK_NULL_HANDLE, 0u);
  const std::vector<pipelines::HybridMeshDraw> mixed = {
      pipelines::HybridMeshDraw{&gpu.value()},
      pipelines::HybridMeshDraw{live},
  };
  const std::vector<uint8_t> both =
      render(allocator.value(), pipeline.value(), mixed, VK_NULL_HANDLE, 0u);

  ASSERT_EQ(single.size(), static_cast<size_t>(kSize) * kSize * 4);
  ASSERT_EQ(both.size(), single.size());

  // The live quad's own region: it alone is blue, and only when it drew.
  const uint8_t* both_centre = pixel_at(both, kSize / 2, kSize / 2);
  EXPECT_GT(both_centre[2], 128) << "the live draw must cover the centre";
  EXPECT_LT(both_centre[0], 128);
  EXPECT_LT(both_centre[1], 128);
  EXPECT_LT(pixel_at(single, kSize / 2, kSize / 2)[2], 128)
      << "without the live draw nothing is blue";

  // Outside it, the static mesh alone shades the frame -- unchanged by the
  // extra draw (no binding state leaked) and not the clear color, which is what
  // makes dropping the static branch fail here rather than pass silently.
  for (const uint32_t x : {kSize / 8, 7 * kSize / 8}) {
    const uint8_t* s = pixel_at(single, x, kSize / 8);
    const uint8_t* b = pixel_at(both, x, kSize / 8);
    EXPECT_TRUE(s[0] != 0 || s[1] != 0 || s[2] != 0)
        << "the static mesh must shade x=" << x;
    for (int c = 0; c < 4; ++c) {
      EXPECT_EQ(s[c], b[c])
          << "the live draw must not disturb x=" << x << " channel " << c;
    }
  }
}

// The indirect command's fields belong to the producer, and this pipeline never
// reads them -- so no gfx test can fail because recon wrote `instanceCount = 0`
// or a non-zero `firstInstance`; that assertion belongs on the side that writes
// the command. What gfx owns is the consequence, pinned here: a command that
// violates the contract's `instanceCount = 1` draws nothing at all, which is
// why the contract fixes it.
TEST_F(HybridMeshRenderTest, ZeroInstanceCountCommandDrawsNothing) {
  auto allocator = vkc::Allocator::create(instance_->handle(), *device_);
  ASSERT_TRUE(allocator.ok()) << allocator.status().message();

  auto pipeline = pipelines::HybridMeshPipeline::create(
      *device_, allocator.value(), color_depth_layout());
  ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();

  const assets::Mesh mesh_cpu = make_hybrid_mesh();
  VkDrawIndexedIndirectCommand command = draw_command(mesh_cpu);
  command.instanceCount = 0;  // the one field changed from the well-formed case

  LiveBuffers buffers;
  const pipelines::LiveMesh live =
      make_live_mesh(allocator.value(), mesh_cpu, /*pad=*/0, command, buffers);
  ASSERT_TRUE(live.valid()) << "bound buffers -- submit() must reach the draw";

  const std::vector<uint8_t> px =
      render(allocator.value(), pipeline.value(),
             pipelines::HybridMeshDraw{live}, VK_NULL_HANDLE, 0u);
  ASSERT_EQ(px.size(), static_cast<size_t>(kSize) * kSize * 4);
  EXPECT_TRUE(all_pixels_cleared(px))
      << "instanceCount = 0 draws no instances -- the frame stays cleared";
}

}  // namespace

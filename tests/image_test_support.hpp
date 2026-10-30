// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file image_test_support.hpp
/// What the ImagePipeline tests share: pipelines, textures and sources made
/// on the device and allocator a fixture names, passes recorded into one
/// command buffer and read back, and a display level read back -- for a test
/// on the renderer's shared device and for one on the two-family device
/// alike.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <glm/vec2.hpp>

#include "gfx_test_support.hpp"
#include "spirv_test_util.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_buffer.hpp"
#include "volumetric_kit/core/vulkan/command_pool.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/format.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
#include "volumetric_kit/gfx/core/mip_chain.hpp"
#include "volumetric_kit/gfx/core/offscreen_target.hpp"
#include "volumetric_kit/gfx/core/render_target.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"
#include "volumetric_kit/gfx/pipelines/image_pipeline.hpp"

namespace vg_test {

namespace pipelines = volumetric_kit::gfx::pipelines;

inline constexpr VkFormat kUnorm = VK_FORMAT_R8G8B8A8_UNORM;
inline constexpr VkFormat kSrgb = VK_FORMAT_R8G8B8A8_SRGB;

inline vg::RenderTargetLayout color_layout(VkFormat format) {
  vg::RenderTargetLayout layout;
  layout.color_formats[0] = format;
  layout.color_count = 1;
  return layout;
}

// A readback: the texels, row-major, of a `width`-wide target whose texels
// are `texel` bytes -- RGBA8 unless the test asked for another format.
struct Pixels {
  std::vector<uint8_t> bytes;
  uint32_t width = 0;
  uint32_t texel = 4;

  const uint8_t* at(uint32_t x, uint32_t y) const {
    return &bytes[(static_cast<size_t>(y) * width + x) * texel];
  }
};

// The expected byte for a channel, within `tolerance`.
inline void expect_rgb(const uint8_t* px, int r, int g, int b, int tolerance,
                       const char* what) {
  EXPECT_NEAR(px[0], r, tolerance) << what << " (red)";
  EXPECT_NEAR(px[1], g, tolerance) << what << " (green)";
  EXPECT_NEAR(px[2], b, tolerance) << what << " (blue)";
}

// A producer on the renderer's queue: a compute pass (write_words.comp)
// filling the first 32-bit words of a storage buffer with a value, in a stage
// none of the image pipeline's other barriers reach.
struct WordWriter {
  vkc::ComputeKernel kernel;  // at a fixed address from make_word_writer on
  vkc::DescriptorPool pool;

  // Records the dispatch that writes `value` into the first `words` words.
  void record(VkCommandBuffer cmd, uint32_t value, uint32_t words) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      kernel.pipeline.handle());
    const VkDescriptorSet set = kernel.set.handle();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            kernel.pipeline.layout(), 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, kernel.pipeline.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(value), &value);
    vkCmdDispatch(cmd, words, 1, 1);
  }
};

// The writer of `target`, a buffer with STORAGE_BUFFER usage; null, with a
// failure added, if it cannot be built.
inline std::unique_ptr<WordWriter> make_word_writer(const vkc::Device& device,
                                                    const vkc::Buffer& target) {
  const std::vector<uint32_t> spv =
      load_spirv(spirv_path("write_words.comp.spv"));
  EXPECT_FALSE(spv.empty());
  if (spv.empty()) {
    return nullptr;
  }
  auto writer = std::make_unique<WordWriter>();
  vkc::KernelSetBuilder builder(device);
  const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(uint32_t)};
  const vkc::Status added =
      builder.add(writer->kernel, "write_words",
                  reinterpret_cast<const unsigned char*>(spv.data()),
                  spv.size() * sizeof(uint32_t), 1, &push);
  EXPECT_TRUE(added.ok()) << added.message();
  if (!added.ok()) {
    return nullptr;
  }
  auto pool = builder.build();
  EXPECT_TRUE(pool.ok()) << pool.status().message();
  if (!pool.ok()) {
    return nullptr;
  }
  writer->pool = std::move(pool).value();
  writer->kernel.set.write_storage_buffer(0, target.handle(), 0, VK_WHOLE_SIZE);
  return writer;
}

// The helpers, on the device and allocator the fixture names.
class ImageHelpers {
 protected:
  virtual ~ImageHelpers() = default;

  virtual vkc::Device& image_device() = 0;
  virtual vkc::Allocator& image_allocator() = 0;

  pipelines::ImagePipeline make_pipeline(VkFormat target = kUnorm) {
    auto pipeline =
        pipelines::ImagePipeline::create(image_device(), color_layout(target));
    EXPECT_TRUE(pipeline.ok()) << pipeline.status().message();
    return pipeline.ok() ? std::move(pipeline).value()
                         : pipelines::ImagePipeline{};
  }

  pipelines::ImageTexture make_texture(const pipelines::ImagePipeline& pipeline,
                                       VkExtent2D extent,
                                       pipelines::ImageMapping mapping,
                                       VkFormat format) {
    pipelines::ImageTextureDesc desc;
    desc.extent = extent;
    desc.mapping = mapping;
    desc.format = format;
    auto texture =
        pipelines::ImageTexture::create(pipeline, image_allocator(), desc);
    EXPECT_TRUE(texture.ok()) << texture.status().message();
    return texture.ok() ? std::move(texture).value()
                        : pipelines::ImageTexture{};
  }

  // A device-only buffer holding `bytes`, readable by a transfer (and usable
  // as `usage` says).
  vkc::Buffer source_buffer(
      const std::vector<uint8_t>& bytes,
      VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
    vg::BufferUploadDesc desc;
    desc.data = bytes.data();
    desc.size = bytes.size();
    desc.usage = usage;
    auto buffer = vg::upload_buffer(image_device(), image_allocator(), desc);
    EXPECT_TRUE(buffer.ok()) << buffer.status().message();
    return buffer.ok() ? std::move(buffer).value() : vkc::Buffer{};
  }

  // A device image holding `bytes`, left SHADER_READ_ONLY_OPTIMAL by the
  // upload; ready_for_copy moves it to TRANSFER_SRC_OPTIMAL in a command
  // buffer, as a producer would leave a picture for a copy to read.
  vkc::Image source_image(VkFormat format, VkExtent2D extent,
                          const std::vector<uint8_t>& bytes) {
    vg::ImageUploadDesc desc;
    desc.extent = extent;
    desc.format = format;
    desc.pixels = bytes.data();
    desc.size = bytes.size();
    auto image = vg::upload_texture(image_device(), image_allocator(), desc);
    EXPECT_TRUE(image.ok()) << image.status().message();
    if (!image.ok()) {
      return vkc::Image{};
    }
    image.value().set_layout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return std::move(image).value();
  }

  static void ready_for_copy(VkCommandBuffer cmd, vkc::Image& image) {
    vg::ImageBarrierDesc b;
    b.image = image.handle();
    b.src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    b.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    b.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
    b.old_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vg::cmd_image_barrier(cmd, b);
    image.set_layout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  }

  // Records `record` into one command buffer, submits it and waits.
  vkc::Status submit(const std::function<void(VkCommandBuffer)>& record) {
    auto pool = vkc::CommandPool::create(image_device().handle(),
                                         image_device().queue_family());
    if (!pool.ok()) {
      return pool.status();
    }
    auto cmd = pool.value().allocate_primary();
    if (!cmd.ok()) {
      return cmd.status();
    }
    VKC_TRY(cmd.value().begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));
    record(cmd.value().handle());
    VKC_TRY(cmd.value().end());
    return image_device().submit_and_wait(cmd.value().handle());
  }

  // One step of a recorded command buffer: work before the render (updates),
  // then the draws into a fresh `extent` target of `format`, read back.
  struct Pass {
    std::function<void(VkCommandBuffer)> before;
    std::vector<pipelines::ImageDraw> draws;
  };

  // Records every pass into one command buffer -- each into its own target of
  // `samples` -- submits it, and returns each target's pixels.
  std::vector<Pixels> render_passes(
      const pipelines::ImagePipeline& pipeline, const std::vector<Pass>& passes,
      VkExtent2D extent, VkFormat format = kUnorm,
      VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
    std::vector<vg::OffscreenTarget> targets;
    for (size_t i = 0; i < passes.size(); ++i) {
      vg::OffscreenTargetDesc desc;
      desc.extent = extent;
      desc.color_format = format;
      desc.samples = samples;
      auto target = vg::OffscreenTarget::create(image_allocator(), desc);
      EXPECT_TRUE(target.ok()) << target.status().message();
      if (!target.ok()) {
        return {};
      }
      targets.push_back(std::move(target).value());
    }

    const vkc::Status submitted = submit([&](VkCommandBuffer raw) {
      for (size_t i = 0; i < passes.size(); ++i) {
        if (passes[i].before) {
          passes[i].before(raw);
        }
        targets[i].prepare(raw);
        vg::RenderTargetBeginInfo begin;
        begin.clear_color.float32[2] = 1.0f;  // opaque blue: "nothing drawn"
        begin.clear_color.float32[3] = 1.0f;
        const vg::RenderTarget rt = targets[i].target();
        rt.begin(raw, begin);
        pipeline.submit(raw, {extent, passes[i].draws.data(),
                              static_cast<uint32_t>(passes[i].draws.size())});
        rt.end(raw);
        targets[i].record_readback(raw);
      }
    });
    if (!submitted.ok()) {
      ADD_FAILURE() << submitted.message();
      return {};
    }

    const uint32_t texel = vkc::texel_bytes(format);
    std::vector<Pixels> out;
    for (const vg::OffscreenTarget& target : targets) {
      const auto* px = static_cast<const uint8_t*>(target.pixels());
      out.push_back(
          {std::vector<uint8_t>(px, px + static_cast<size_t>(extent.width) *
                                             extent.height * texel),
           extent.width, texel});
    }
    return out;
  }

  // The single-pass case: update `texture` from `update`, then draw `draws`.
  Pixels update_and_draw(
      const pipelines::ImagePipeline& pipeline,
      pipelines::ImageTexture& texture, const pipelines::ImageUpdate& update,
      const std::vector<pipelines::ImageDraw>& draws, VkExtent2D extent,
      VkFormat format = kUnorm,
      const std::function<void(VkCommandBuffer)>& prepare = nullptr) {
    Pass pass;
    pass.before = [&](VkCommandBuffer cmd) {
      if (prepare) {
        prepare(cmd);
      }
      const vkc::Status s = pipeline.record_update(cmd, texture, update);
      EXPECT_TRUE(s.ok()) << s.message();
    };
    pass.draws = draws;
    std::vector<Pixels> out = render_passes(pipeline, {pass}, extent, format);
    return out.empty() ? Pixels{} : std::move(out.front());
  }

  static pipelines::ImageDraw draw_of(const pipelines::ImageTexture& texture,
                                      VkExtent2D target, float scale,
                                      glm::vec2 origin = {0.0f, 0.0f}) {
    pipelines::ImageDraw draw;
    draw.texture = &texture;
    draw.viewport = {{0, 0}, target};
    draw.origin = origin;
    draw.scale = scale;
    return draw;
  }

  // Level `level` of a converted texture's display image, read back: its
  // sRGB-encoded RGBA8 bytes, as stored. Records `before` first, in the same
  // command buffer; the display is left SHADER_READ_ONLY_OPTIMAL for draws.
  Pixels read_level(
      const pipelines::ImageTexture& texture, uint32_t level,
      const std::function<void(VkCommandBuffer)>& before = nullptr) {
    const VkExtent2D size = vg::mip_level_extent(texture.extent(), level);
    vkc::BufferDesc desc;
    desc.size = static_cast<VkDeviceSize>(size.width) * size.height * 4;
    desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memory = vkc::MemoryUsage::Staging;
    desc.host_access = vkc::HostAccess::Random;
    auto readback = image_allocator().create_buffer(desc);
    EXPECT_TRUE(readback.ok()) << readback.status().message();
    if (!readback.ok()) {
      return {};
    }
    const vkc::Status s = submit([&](VkCommandBuffer cmd) {
      if (before) {
        before(cmd);
      }
      vg::ImageBarrierDesc b;
      b.image = texture.display().handle();
      b.src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      b.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      b.src_access = VK_ACCESS_MEMORY_WRITE_BIT;
      b.dst_access = VK_ACCESS_TRANSFER_READ_BIT;
      b.old_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      b.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      b.base_mip = level;
      b.mip_count = 1;
      vg::cmd_image_barrier(cmd, b);
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
      region.imageExtent = {size.width, size.height, 1};
      vkCmdCopyImageToBuffer(cmd, texture.display().handle(),
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback.value().handle(), 1, &region);
      vg::ImageBarrierDesc back = b;
      back.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      back.dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      back.src_access = 0;
      back.dst_access = VK_ACCESS_SHADER_READ_BIT;
      back.old_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      back.new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      vg::cmd_image_barrier(cmd, back);
      // The copy's writes, for the host to read once the fence is waited.
      VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                           VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
                           0, nullptr);
    });
    EXPECT_TRUE(s.ok()) << s.message();
    if (!s.ok()) {
      return {};
    }
    const auto* px = static_cast<const uint8_t*>(readback.value().mapped());
    return {std::vector<uint8_t>(px, px + desc.size), size.width, 4};
  }
};

}  // namespace vg_test

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/core/texture_upload.hpp"

#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/format.hpp"
#include "volumetric_kit/gfx/core/buffer_barrier.hpp"
#include "volumetric_kit/gfx/core/device_requirements.hpp"
#include "volumetric_kit/gfx/core/image_barrier.hpp"
#include "volumetric_kit/gfx/core/mip_chain.hpp"

namespace volumetric_kit::gfx {
namespace {

// Shader stages that may sample the finished texture. Moving each level to
// SHADER_READ makes the upload visible to vertex-texture fetch as well as
// fragment sampling -- both graphics-queue stages, which the upload submits
// on. (Cross-submission readers are additionally ordered by the fence the
// submit waits on, so this dst scope only bites within the submission, which
// records no sampling of its own.)
constexpr VkPipelineStageFlags kSampleStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

// The stages and accesses that may consume an uploaded buffer, per its usage
// flags -- the dst scope of the copy -> consume barrier that carries the bytes
// across to the later submission that reads them (the buffer analogue of the
// image path's SHADER_READ transition). This barrier -- NOT the finish() fence
// -- is what makes the copy visible to that consumer: a host fence gives
// execution + host-domain visibility only, so the precise dst scope is
// load-bearing for correctness, not merely perf.
//
// The mapped shader usages resolve to the graphics sampling stages
// (kSampleStages = VERTEX|FRAGMENT). A consumer in a stage the upload's
// graphics queue does not run -- compute (the library keeps compute on
// CUDA/Metal, but the usage bit is public), geometry/tessellation
// (feature-gated, so naming their stages unconditionally here would be an
// invalid dstStageMask), or a different queue -- is outside this scope and must
// synchronize itself. An unmapped usage bit widens to a full ALL_COMMANDS scope
// rather than leaving its consumer outside the barrier.
struct BufferConsumeScope {
  VkPipelineStageFlags stages = 0;
  VkAccessFlags access = 0;
};

BufferConsumeScope buffer_consume_scope(VkBufferUsageFlags usage) {
  constexpr VkBufferUsageFlags kMappedUsages =
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
      VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  BufferConsumeScope scope;
  if ((usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0) {
    scope.stages |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    scope.access |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
  }
  if ((usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0) {
    scope.stages |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    scope.access |= VK_ACCESS_INDEX_READ_BIT;
  }
  if ((usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0) {
    scope.stages |= kSampleStages;
    scope.access |= VK_ACCESS_UNIFORM_READ_BIT;
  }
  if ((usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) != 0) {
    scope.stages |= kSampleStages;
    scope.access |= VK_ACCESS_SHADER_READ_BIT;
  }
  if ((usage & (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)) != 0) {
    scope.stages |= kSampleStages;
    scope.access |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  }
  if ((usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0) {
    scope.stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    scope.access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  }
  if ((usage & (VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT)) != 0) {
    scope.stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    scope.access |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  }
  if ((usage & ~kMappedUsages) != 0) {
    scope.stages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    scope.access |= VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  }
  return scope;
}

// A staging buffer -- host memory, mapped -- holding @p size bytes copied from
// @p src, written once front-to-back (write-combined where the device has it)
// -- the single recipe both add() (pixels) and add_buffer() (bytes) stage their
// source through.
core::Result<core::Buffer> make_staging(core::Allocator& allocator,
                                        const void* src, VkDeviceSize size) {
  core::BufferDesc desc;
  desc.size = size;
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = core::MemoryUsage::Staging;
  desc.host_access = core::HostAccess::SequentialWrite;
  VKC_ASSIGN(core::Buffer staging, allocator.create_buffer(desc));
  std::memcpy(staging.mapped(), src, size);
  return staging;
}

// What plan_upload derives from a validated ImageUploadDesc.
struct UploadPlan {
  uint32_t texel = 0;       // bytes per texel of desc.format
  uint32_t image_mips = 1;  // mip count of the created image
};

// Validate `desc` against the device and fill `plan` -- everything that must
// hold before any staging buffer or image is created.
core::Status plan_upload(const core::Device& device,
                         const ImageUploadDesc& desc, UploadPlan* plan) {
  if (desc.extent.width == 0 || desc.extent.height == 0) {
    return core::Status::invalid_argument(
        "upload_texture: extent must be non-zero");
  }
  if (desc.array_layers == 0 || desc.mip_levels == 0) {
    return core::Status::invalid_argument(
        "upload_texture: array_layers and mip_levels must be non-zero");
  }
  if (desc.cube &&
      (desc.array_layers != 6 || desc.extent.width != desc.extent.height)) {
    return core::Status::invalid_argument(
        "upload_texture: a cube upload needs array_layers == 6 and a square "
        "extent");
  }
  // TODO: generate mip chains for array/cube images (a per-layer blit loop),
  // and on top of caller-supplied base mips; today generation is single-layer
  // and single-source-mip only.
  if (desc.generate_mips && (desc.array_layers > 1 || desc.mip_levels > 1)) {
    return core::Status::invalid_argument(
        "upload_texture: generate_mips requires array_layers == 1 and "
        "mip_levels == 1");
  }
  const VkPhysicalDeviceLimits& limits = device.caps().limits();
  const uint32_t max_dim =
      desc.cube ? limits.maxImageDimensionCube : limits.maxImageDimension2D;
  if (desc.extent.width > max_dim || desc.extent.height > max_dim) {
    // Bounds the image well below INT32_MAX too, so the int32 mip-extent math
    // in the blit path never sees a negative dimension.
    return core::Status::unsupported(
        desc.cube ? "upload_texture: extent exceeds the device's "
                    "maxImageDimensionCube limit"
                  : "upload_texture: extent exceeds the device's "
                    "maxImageDimension2D limit");
  }
  if (desc.array_layers > limits.maxImageArrayLayers) {
    return core::Status::unsupported(
        "upload_texture: array_layers exceeds the device's "
        "maxImageArrayLayers limit");
  }
  if (desc.format == VK_FORMAT_UNDEFINED) {
    return core::Status::invalid_argument(
        "upload_texture: format must not be VK_FORMAT_UNDEFINED");
  }
  if (desc.pixels == nullptr) {
    return core::Status::invalid_argument(
        "upload_texture: pixels must not be null");
  }
  const uint32_t texel = core::texel_bytes(desc.format);
  if (texel == 0) {
    // texel_bytes returns 0 for formats a flat per-texel copy cannot size:
    // compressed, multi-planar, subsampled, or depth/stencil -- and those of
    // extensions other than KHR, which the core's table does not cover.
    return core::Status::unsupported(
        "upload_texture: format must be an uncompressed, single-plane color "
        "format");
  }
  if (core::format_needs_ycbcr_conversion(desc.format)) {
    // The RGBA 4PACK16 formats are sized, but the sampled view the upload
    // makes would need a sampler Y'CbCr conversion (see create_image).
    return core::Status::unsupported(
        "upload_texture: format needs a sampler Y'CbCr conversion to be "
        "sampled");
  }
  if (desc.mip_levels > mip_level_count(desc.extent)) {
    return core::Status::invalid_argument(
        "upload_texture: mip_levels exceeds the full mip chain for extent");
  }
  // The packing contract (see ImageUploadDesc): tightly packed subresources,
  // mip-major then layer, per-mip extents halved with a floor of 1.
  VkDeviceSize texels_per_layer = 0;
  for (uint32_t m = 0; m < desc.mip_levels; ++m) {
    const VkExtent2D e = mip_level_extent(desc.extent, m);
    texels_per_layer += VkDeviceSize{e.width} * e.height;
  }
  const VkDeviceSize expected = texels_per_layer * texel * desc.array_layers;
  if (desc.size != expected) {
    return core::Status::invalid_argument(
        "upload_texture: size must equal the tightly packed mip-major, "
        "layer-minor pixel total (see ImageUploadDesc)");
  }

  // The destination is always created with SAMPLED usage + optimal tiling, so
  // the format must support being sampled there. Reject up front with a clean
  // Unsupported rather than letting create_image trip a validation error. The
  // image also gets TRANSFER_SRC/DST usage (staging copy + mip blits); those
  // features are not screened separately because Vulkan 1.1 /
  // VK_KHR_maintenance1 guarantees any optimal-tiling format reporting
  // SAMPLED_IMAGE also reports TRANSFER_SRC/DST, and the device floor is 1.3 --
  // so this SAMPLED gate implies them. Re-check here if that gate is relaxed.
  if (!device.caps().format_supports(desc.format, VK_IMAGE_TILING_OPTIMAL,
                                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
    return core::Status::unsupported(
        "upload_texture: format does not support sampling (SAMPLED_IMAGE) with "
        "optimal tiling");
  }

  plan->texel = texel;
  plan->image_mips =
      desc.generate_mips ? mip_level_count(desc.extent) : desc.mip_levels;

  // Mip generation blits with a linear filter, so the format must additionally
  // support both blit endpoints and linear filtering; otherwise the blit is
  // invalid use.
  if (desc.generate_mips && plan->image_mips > 1) {
    constexpr VkFormatFeatureFlags kBlitNeeded =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if (!device.caps().format_supports(desc.format, VK_IMAGE_TILING_OPTIMAL,
                                       kBlitNeeded)) {
      return core::Status::unsupported(
          "upload_texture: generate_mips needs a format that supports linear "
          "blit (BLIT_SRC | BLIT_DST | SAMPLED_IMAGE_FILTER_LINEAR)");
    }
  }
  return core::Status{};
}

// Copy staging into every supplied (mip, layer), generate the mip chain when
// asked, and leave the whole image in SHADER_READ_ONLY_OPTIMAL.
void record_upload(VkCommandBuffer cmd, VkImage image, VkBuffer staging,
                   const ImageUploadDesc& desc, const UploadPlan& plan) {
  // 1. Every (mip, layer) UNDEFINED -> TRANSFER_DST for the copy and blit
  //    writes; the ImageBarrierDesc defaults span the whole image.
  ImageBarrierDesc to_dst;
  to_dst.image = image;
  to_dst.src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  to_dst.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  to_dst.dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  cmd_image_barrier(cmd, to_dst);

  // 2. Copy the supplied pixels: one region per mip level spanning all layers
  //    at once -- the mip-major packing keeps a mip's layers contiguous, and
  //    bufferRowLength/bufferImageHeight of 0 mean tightly packed.
  std::vector<VkBufferImageCopy> copies(desc.mip_levels);
  VkDeviceSize offset = 0;
  for (uint32_t m = 0; m < desc.mip_levels; ++m) {
    const VkExtent2D e = mip_level_extent(desc.extent, m);
    copies[m].bufferOffset = offset;
    copies[m].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0,
                                  desc.array_layers};
    copies[m].imageExtent = {e.width, e.height, 1};
    offset += VkDeviceSize{e.width} * e.height * plan.texel * desc.array_layers;
  }
  vkCmdCopyBufferToImage(cmd, staging, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, desc.mip_levels,
                         copies.data());

  // 3. Mip generation (validated single-layer, single-source-mip): blit the
  //    chain down from the copied mip 0; every level ends in SHADER_READ.
  if (desc.generate_mips && plan.image_mips > 1) {
    MipChainDesc mips;
    mips.image = image;
    mips.extent = desc.extent;
    mips.mip_levels = plan.image_mips;
    mips.dst_stages = kSampleStages;
    cmd_generate_mips(cmd, mips);
    return;
  }

  // 4. No generation: every level holds its supplied pixels and is still
  //    TRANSFER_DST -- move the whole image to SHADER_READ in one barrier.
  ImageBarrierDesc to_read;
  to_read.image = image;
  to_read.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  to_read.dst_stage = kSampleStages;
  to_read.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_read.dst_access = VK_ACCESS_SHADER_READ_BIT;
  to_read.old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_read.new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  cmd_image_barrier(cmd, to_read);
}

}  // namespace

core::Result<UploadBatch> UploadBatch::begin(const core::Device& device,
                                             core::Allocator& allocator) {
  // A device made for another library may lack what the batch's barriers and
  // the renderer's later reads of its resources need -- a graphics queue
  // above all. Nothing is allocated until an add, nor recorded until finish().
  VKC_TRY(device.check_enabled(device_requirements())
              .with_context("UploadBatch::begin"));

  UploadBatch batch;
  batch.device_ = &device;
  batch.allocator_ = &allocator;
  return batch;
}

UploadBatch::~UploadBatch() = default;

UploadBatch::UploadBatch(UploadBatch&& other) noexcept
    : device_(other.device_),
      allocator_(other.allocator_),
      records_(std::move(other.records_)),
      staging_(std::move(other.staging_)),
      poisoned_(other.poisoned_) {
  other.device_ = nullptr;
  other.allocator_ = nullptr;
  other.records_.clear();
  other.staging_.clear();
  other.poisoned_ = false;
}

UploadBatch& UploadBatch::operator=(UploadBatch&& other) noexcept {
  if (this != &other) {
    // The member moves free this batch's queued commands and staging buffers
    // (nothing was submitted, so freeing them is always safe).
    device_ = other.device_;
    allocator_ = other.allocator_;
    records_ = std::move(other.records_);
    staging_ = std::move(other.staging_);
    poisoned_ = other.poisoned_;
    other.device_ = nullptr;
    other.allocator_ = nullptr;
    other.records_.clear();
    other.staging_.clear();
    other.poisoned_ = false;
  }
  return *this;
}

core::Result<core::Image> UploadBatch::add(const ImageUploadDesc& desc) {
  if (!valid()) {
    return core::Status::invalid_argument(
        "UploadBatch::add on an empty batch (begin one first; a batch is "
        "one-shot after finish)");
  }
  UploadPlan plan;
  VKC_TRY(plan_upload(*device_, desc, &plan));

  VKC_ASSIGN(core::Buffer staging,
             make_staging(*allocator_, desc.pixels, desc.size));

  // Destination: device-local sampled image. SAMPLED to read it in shaders,
  // TRANSFER_DST for the staging copy, and TRANSFER_SRC so the mip-chain blits
  // can read earlier levels -- and, uniformly for the non-blit cases too, so
  // the finished texture stays copyable/blittable (readback, screenshots,
  // re-upload) rather than that capability hinging on whether mips were asked
  // for.
  core::ImageDesc image_desc;
  image_desc.extent = desc.extent;
  image_desc.format = desc.format;
  image_desc.mip_levels = plan.image_mips;
  image_desc.array_layers = desc.array_layers;
  image_desc.cube = desc.cube;
  image_desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  // It records UNDEFINED, the layout its contents are in until finish()
  // submits the transitions; the caller records SHADER_READ_ONLY_OPTIMAL then.
  VKC_ASSIGN(core::Image texture, allocator_->create_image(image_desc));

  // Everything that can fail has, so a failed add above leaves the batch
  // unchanged and usable. The handles and the desc's shape are copied: the
  // pixels are already staged, and the caller keeps the texture alive.
  records_.emplace_back([image = texture.handle(), src = staging.handle(), desc,
                         plan](VkCommandBuffer cmd) {
    record_upload(cmd, image, src, desc, plan);
  });
  staging_.push_back(std::move(staging));
  return texture;
}

core::Result<core::Buffer> UploadBatch::add_buffer(
    const BufferUploadDesc& desc) {
  if (!valid()) {
    return core::Status::invalid_argument(
        "UploadBatch::add_buffer on an empty batch (begin one first; a batch "
        "is one-shot after finish)");
  }
  if (desc.data == nullptr) {
    return core::Status::invalid_argument(
        "upload_buffer: data must not be null");
  }
  if (desc.size == 0) {
    return core::Status::invalid_argument(
        "upload_buffer: size must be non-zero");
  }
  if (desc.usage == 0) {
    return core::Status::invalid_argument(
        "upload_buffer: usage must name at least one buffer usage");
  }

  VKC_ASSIGN(core::Buffer staging,
             make_staging(*allocator_, desc.data, desc.size));

  // Destination: device-only memory, TRANSFER_DST for the staging copy plus
  // the caller's usage.
  core::BufferDesc dst_desc;
  dst_desc.size = desc.size;
  dst_desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | desc.usage;
  dst_desc.memory = core::MemoryUsage::DeviceOnly;
  VKC_ASSIGN(core::Buffer buffer, allocator_->create_buffer(dst_desc));

  // Everything that can fail has, so a failed add above leaves the batch
  // unchanged and usable.
  records_.emplace_back([src = staging.handle(), dst = buffer.handle(),
                         size = desc.size,
                         usage = desc.usage](VkCommandBuffer cmd) {
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);

    // Make the copy visible to the usage-implied consumers in the later
    // submission that reads them (see buffer_consume_scope). Within this
    // submission the scope never bites -- the batch records no consumers of
    // its own. For the shader / vertex-input usages the dst scope stays off
    // the TRANSFER stage, so the batch's copies overlap; a TRANSFER_SRC/DST
    // (or unmapped) destination does include TRANSFER and so serializes later
    // copies behind this barrier -- fine for those rarer cases.
    const BufferConsumeScope scope = buffer_consume_scope(usage);
    BufferBarrierDesc barrier;
    barrier.buffer = dst;
    barrier.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    barrier.dst_stage = scope.stages;
    barrier.src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dst_access = scope.access;
    cmd_buffer_barrier(cmd, barrier);
  });

  staging_.push_back(std::move(staging));
  return buffer;
}

void UploadBatch::poison() noexcept { poisoned_ = true; }

core::Status UploadBatch::finish() {
  if (!valid()) {
    return core::Status::invalid_argument(
        "UploadBatch::finish on an empty batch");
  }
  if (poisoned_) {
    // A caller dropped a resource an earlier add queued a copy into (see
    // poison()): submitting would reference freed memory. Discard the queued
    // work instead of submitting it -- moving into a temporary frees the
    // commands + staging on return, and never submits.
    UploadBatch discard(std::move(*this));
    return core::Status::invalid_argument(
        "UploadBatch::finish on a poisoned batch: an added resource was "
        "dropped "
        "before finish, so a recorded copy would reference freed memory; begin "
        "a new batch");
  }
  // Move the owned state into locals first: whatever happens below, the batch
  // ends empty (one-shot).
  const core::Device* device = device_;
  std::vector<std::function<void(VkCommandBuffer)>> records =
      std::move(records_);
  auto staging =
      std::make_shared<std::vector<core::Buffer>>(std::move(staging_));
  device_ = nullptr;
  allocator_ = nullptr;
  records_.clear();
  staging_.clear();

  // One command buffer on a pool no other submit holds, one submit under the
  // device's queue lock, one fence wait. The device holds the staging buffers
  // until the work completes: past a failed wait, which may leave the work
  // running, until the device is destroyed. Each buffer keeps its allocator's
  // state alive, so freeing them after the allocator is gone is safe.
  return device->submit_single_time(
      [&records](VkCommandBuffer cmd) {
        for (const auto& record : records) {
          record(cmd);
        }
      },
      std::move(staging));
}

core::Result<core::Image> upload_texture(const core::Device& device,
                                         core::Allocator& allocator,
                                         const ImageUploadDesc& desc) {
  // The one-texture batch: exactly the shared validate/record path, one
  // submit, one fence wait. A failed add leaves the batch to its destructor,
  // which discards the never-submitted command buffer.
  VKC_ASSIGN(UploadBatch batch, UploadBatch::begin(device, allocator));
  VKC_ASSIGN(core::Image texture, batch.add(desc));
  VKC_TRY(batch.finish());
  texture.set_layout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  return texture;
}

core::Result<core::Buffer> upload_buffer(const core::Device& device,
                                         core::Allocator& allocator,
                                         const BufferUploadDesc& desc) {
  // The one-buffer batch: shared validate/record path, one submit, one fence
  // wait (see upload_texture).
  VKC_ASSIGN(UploadBatch batch, UploadBatch::begin(device, allocator));
  VKC_ASSIGN(core::Buffer buffer, batch.add_buffer(desc));
  VKC_TRY(batch.finish());
  return buffer;
}

}  // namespace volumetric_kit::gfx

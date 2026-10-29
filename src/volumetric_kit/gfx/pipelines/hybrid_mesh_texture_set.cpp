// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/hybrid_mesh_texture_set.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "hybrid_mesh_set0.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/gfx/pipelines/hybrid_mesh_pipeline.hpp"
#include "ycbcr_check.hpp"

namespace volumetric_kit::gfx::pipelines {

namespace {

// The core address modes; MIRROR_CLAMP_TO_EDGE needs a feature gfx does not
// enable.
bool core_address_mode(VkSamplerAddressMode address) {
  switch (address) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT:
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
      return true;
    default:
      return false;
  }
}

// Checks plane `name` of a picture: an image the set can sample, of `format`,
// covering `extent`.
core::Status check_plane(const std::string& call, const char* name,
                         const core::Image* plane, VkFormat format,
                         VkExtent2D extent, const core::Device& device,
                         VkFilter filter) {
  const std::string which = call + ": the " + name;
  if (plane == nullptr || !plane->valid()) {
    return core::Status::invalid_argument(which + " plane is missing");
  }
  if (plane->type() != VK_IMAGE_TYPE_2D ||
      plane->samples() != VK_SAMPLE_COUNT_1_BIT || plane->array_layers() != 1) {
    return core::Status::invalid_argument(
        which + " plane is not a single-sample, one-layer 2D image");
  }
  if ((plane->usage() & VK_IMAGE_USAGE_SAMPLED_BIT) == 0 ||
      plane->view() == VK_NULL_HANDLE) {
    return core::Status::invalid_argument(
        which + " plane lacks SAMPLED usage or a view");
  }
  if (plane->format() != format) {
    return core::Status::invalid_argument(
        which + (format == VK_FORMAT_R8_UNORM ? " plane is not R8_UNORM"
                                              : " plane is not R8G8_UNORM"));
  }
  if (plane->layout() != VK_IMAGE_LAYOUT_GENERAL &&
      plane->layout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    return core::Status::invalid_argument(
        which + " plane is not in GENERAL or SHADER_READ_ONLY_OPTIMAL");
  }
  if (plane->width() < extent.width || plane->height() < extent.height) {
    return core::Status::invalid_argument(
        which + " plane is smaller than the picture's");
  }
  VkFormatFeatureFlags features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
  if (filter == VK_FILTER_LINEAR) {
    features |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  }
  if (!device.caps().format_supports(format, plane->tiling(), features)) {
    return core::Status::unsupported(
        which +
        " plane's format cannot be sampled, with the set's filter, "
        "in its tiling");
  }
  return core::Status{};
}

// Checks `picture` for a set sampling with `options`.
core::Status check_picture(const std::string& call, const YcbcrPicture& picture,
                           const core::Device& device,
                           const HybridMeshTextureSet::Options& options) {
  const VkExtent2D extent = picture.extent;
  if (extent.width == 0 || extent.height == 0) {
    return core::Status::invalid_argument(call +
                                          ": the picture's extent is zero");
  }
  // The chroma is half the luma on each axis, rounded up.
  const VkExtent2D chroma{(extent.width + 1) / 2, (extent.height + 1) / 2};
  VKC_TRY(check_plane(call, "luma", picture.luma, VK_FORMAT_R8_UNORM, extent,
                      device, options.filter));
  VKC_TRY(check_plane(call, "chroma", picture.chroma, VK_FORMAT_R8G8_UNORM,
                      chroma, device, options.filter));
  return detail::check_ycbcr_coding(call, picture.coding);
}

// Checks `count` pictures at `pictures` for a set sampling with `options`;
// a refusal names the picture when there are several.
core::Status check_pictures(const char* call, const YcbcrPicture* pictures,
                            uint32_t count, const core::Device& device,
                            const HybridMeshTextureSet::Options& options) {
  const std::string name = call;
  if (pictures == nullptr) {
    return core::Status::invalid_argument(name + ": the pictures are null");
  }
  if (count == 0 || count > HybridMeshTextureSet::kMaxPictures) {
    return core::Status::invalid_argument(
        name + ": " + std::to_string(count) + " pictures, not 1 to " +
        std::to_string(HybridMeshTextureSet::kMaxPictures));
  }
  if (options.address != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) {
    return core::Status::invalid_argument(
        name + ": a Y'CbCr picture is sampled CLAMP_TO_EDGE alone");
  }
  for (uint32_t p = 0; p < count; ++p) {
    VKC_TRY(check_picture(
        count == 1 ? name : name + ": picture " + std::to_string(p),
        pictures[p], device, options));
  }
  return core::Status{};
}

// A picture's uniform takes device-local memory the host maps.
core::Status no_picture_uniform(const std::string& call) {
  return core::Status::unsupported(
      call +
      ": the device has no host-mapped device memory for a picture's "
      "uniform");
}

}  // namespace

HybridMeshTextureSet::HybridMeshTextureSet(
    const core::Device& device, const Options& options, VkImageView placeholder,
    VkBuffer rgba_uniform, Sampler sampler, core::Buffer uniform,
    core::DescriptorPool pool, core::DescriptorSet set) noexcept
    : device_(device),
      options_(options),
      placeholder_(placeholder),
      rgba_uniform_(rgba_uniform),
      sampler_(std::move(sampler)),
      uniform_(std::move(uniform)),
      pool_(std::move(pool)),
      set_(std::move(set)) {}

HybridMeshTextureSet::~HybridMeshTextureSet() = default;

core::Result<std::unique_ptr<HybridMeshTextureSet>> HybridMeshTextureSet::make(
    const char* call, const core::Device& device, core::Allocator& allocator,
    const HybridMeshPipeline& pipeline, const Options& options) {
  const std::string name = call;
  if (!pipeline.valid()) {
    return core::Status::invalid_argument(name + ": empty pipeline");
  }
  if (options.filter != VK_FILTER_NEAREST &&
      options.filter != VK_FILTER_LINEAR) {
    return core::Status::invalid_argument(
        name + ": the filter is not NEAREST or LINEAR");
  }
  if (!core_address_mode(options.address)) {
    return core::Status::invalid_argument(
        name +
        ": the address mode is not REPEAT, MIRRORED_REPEAT, CLAMP_TO_EDGE "
        "or CLAMP_TO_BORDER");
  }

  // One filter for both directions and between levels, one address mode on
  // every axis. Sampler::create refuses a max_lod below 0 or NaN.
  SamplerDesc sampler_desc;
  sampler_desc.mag_filter = options.filter;
  sampler_desc.min_filter = options.filter;
  sampler_desc.mipmap_mode = options.filter == VK_FILTER_LINEAR
                                 ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                 : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampler_desc.address_mode_u = options.address;
  sampler_desc.address_mode_v = options.address;
  sampler_desc.address_mode_w = options.address;
  sampler_desc.max_lod = options.max_lod;
  const VkDevice vk = device.handle();
  VKC_ASSIGN(Sampler sampler, Sampler::create(vk, sampler_desc));

  // A picture's form and coding, which the host writes at each rebind that no
  // pending frame has bound, so one mapped copy serves. Made wherever the
  // device has that memory, so a set made over an RGBA image may take a
  // picture later; an RGBA image binds the pipeline's device-only uniform.
  core::Buffer uniform;
  if (device.caps().device_mapped_memory()) {
    core::BufferDesc uniform_desc;
    uniform_desc.size = sizeof(detail::PictureUniform);
    uniform_desc.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    uniform_desc.memory = core::MemoryUsage::DeviceMapped;
    uniform_desc.host_access = core::HostAccess::SequentialWrite;
    VKC_ASSIGN(uniform, allocator.create_buffer(uniform_desc));
  }

  const auto pool_sizes = detail::set0_pool_sizes(1);
  VKC_ASSIGN(
      core::DescriptorPool pool,
      core::DescriptorPool::create(
          vk, pool_sizes.data(), static_cast<uint32_t>(pool_sizes.size()), 1));
  VKC_ASSIGN(core::DescriptorSet set,
             pool.allocate(pipeline.descriptor_set_layout(0)));

  // The constructor is private, so no std::make_unique.
  return std::unique_ptr<HybridMeshTextureSet>(new HybridMeshTextureSet(
      device, options, pipeline.fallback_.view(),
      pipeline.rgba_picture_.handle(), std::move(sampler), std::move(uniform),
      std::move(pool), std::move(set)));
}

core::Result<std::unique_ptr<HybridMeshTextureSet>>
HybridMeshTextureSet::create(const core::Device& device,
                             core::Allocator& allocator,
                             const HybridMeshPipeline& pipeline,
                             VkImageView view, const Options& options) try {
  constexpr const char* kCall = "HybridMeshTextureSet::create";
  if (view == VK_NULL_HANDLE) {
    return core::Status::invalid_argument(std::string(kCall) +
                                          ": the view is null");
  }
  VKC_ASSIGN(std::unique_ptr<HybridMeshTextureSet> texture,
             make(kCall, device, allocator, pipeline, options));
  texture->rebind(view);
  return texture;
} catch (...) {
  return core::exception_status("HybridMeshTextureSet::create",
                                core::Status::Code::OutOfMemory);
}

core::Result<std::unique_ptr<HybridMeshTextureSet>>
HybridMeshTextureSet::create(const core::Device& device,
                             core::Allocator& allocator,
                             const HybridMeshPipeline& pipeline,
                             const YcbcrPicture& picture,
                             const Options& options) {
  return create(device, allocator, pipeline, &picture, 1, options);
}

core::Result<std::unique_ptr<HybridMeshTextureSet>>
HybridMeshTextureSet::create(const core::Device& device,
                             core::Allocator& allocator,
                             const HybridMeshPipeline& pipeline,
                             const YcbcrPicture* pictures, uint32_t count,
                             const Options& options) try {
  constexpr const char* kCall = "HybridMeshTextureSet::create";
  // Checked before anything is made, so a refused picture costs nothing.
  VKC_TRY(check_pictures(kCall, pictures, count, device, options));
  if (!device.caps().device_mapped_memory()) {
    return no_picture_uniform(kCall);
  }
  VKC_ASSIGN(std::unique_ptr<HybridMeshTextureSet> texture,
             make(kCall, device, allocator, pipeline, options));
  VKC_TRY(texture->rebind(pictures, count));
  return texture;
} catch (...) {
  return core::exception_status("HybridMeshTextureSet::create",
                                core::Status::Code::OutOfMemory);
}

void HybridMeshTextureSet::rebind(VkImageView view) {
  detail::Set0Planes planes;
  planes.fill({placeholder_});
  planes[0] = {view};
  detail::write_set0(set_, sampler_.handle(), planes, rgba_uniform_);
}

core::Status HybridMeshTextureSet::rebind(const YcbcrPicture& picture) {
  return rebind(&picture, 1);
}

core::Status HybridMeshTextureSet::rebind(const YcbcrPicture* pictures,
                                          uint32_t count) try {
  static_assert(kMaxPictures == detail::kSet0Pictures,
                "set 0 binds kMaxPictures pictures");
  constexpr const char* kCall = "HybridMeshTextureSet::rebind";
  VKC_TRY(check_pictures(kCall, pictures, count, device_, options_));
  if (!uniform_.valid()) {
    return no_picture_uniform(kCall);
  }
  detail::PictureUniform coded{};
  coded.pictures = count;
  detail::Set0Planes planes;
  planes.fill({placeholder_});
  for (uint32_t p = 0; p < count; ++p) {
    const YcbcrPicture& picture = pictures[p];
    detail::PictureCoding& c = coded.coding[p];
    c.full_range = picture.coding.full_range ? 1u : 0u;
    c.kr = picture.coding.kr;
    c.kb = picture.coding.kb;
    c.chroma_siting = static_cast<uint32_t>(picture.coding.chroma_siting);
    c.extent[0] = static_cast<float>(picture.extent.width);
    c.extent[1] = static_cast<float>(picture.extent.height);
    planes[2 * p] = {picture.luma->view(), picture.luma->layout()};
    planes[2 * p + 1] = {picture.chroma->view(), picture.chroma->layout()};
  }
  std::memcpy(uniform_.mapped(), &coded, sizeof(coded));
  detail::write_set0(set_, sampler_.handle(), planes, uniform_.handle());
  return core::Status{};
} catch (...) {
  return core::exception_status("HybridMeshTextureSet::rebind",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::gfx::pipelines

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/pipelines/ibl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/packing.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "volumetric_kit/gfx/core/device.hpp"
#include "volumetric_kit/gfx/core/texture_upload.hpp"

namespace volumetric_kit::gfx::pipelines {

namespace {

constexpr float kPi = 3.14159265359f;

// Hammersley low-discrepancy 2D sample (van der Corput radical inverse).
glm::vec2 hammersley(uint32_t i, uint32_t n) {
  uint32_t bits = i;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  const float radical = static_cast<float>(bits) * 2.3283064365386963e-10f;
  return glm::vec2(static_cast<float>(i) / static_cast<float>(n), radical);
}

// GGX-importance-sampled half-vector around `n` for `roughness`.
glm::vec3 importance_sample_ggx(glm::vec2 xi, const glm::vec3& n,
                                float roughness) {
  const float a = roughness * roughness;
  const float phi = 2.0f * kPi * xi.x;
  const float cos_t = std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
  const float sin_t = std::sqrt(1.0f - cos_t * cos_t);
  const glm::vec3 h(std::cos(phi) * sin_t, std::sin(phi) * sin_t, cos_t);
  const glm::vec3 up =
      std::abs(n.z) < 0.999f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
  const glm::vec3 tangent = glm::normalize(glm::cross(up, n));
  const glm::vec3 bitangent = glm::cross(n, tangent);
  return glm::normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

// Smith geometry with the IBL roughness remap (k = roughness^2 / 2 -- the
// material roughness, per Karis; not the squared GGX alpha).
float geometry_smith_ibl(float n_dot_v, float n_dot_l, float roughness) {
  const float k = roughness * roughness / 2.0f;
  const float gv = n_dot_v / (n_dot_v * (1.0f - k) + k);
  const float gl = n_dot_l / (n_dot_l * (1.0f - k) + k);
  return gv * gl;
}

// Cosine-weighted hemisphere integral of the environment around `n`: the
// diffuse irradiance for that normal. PI is folded in (LearnOpenGL form), so
// the shader's diffuse term is just irradiance * albedo. `delta` is the
// hemisphere step in radians (~63 x 16 taps at the 0.1 default); coarse vs a
// tight HDR sun (mild diffuse banding), but adequate for the low-frequency
// diffuse fill.
glm::vec4 irradiance_at(const EnvironmentSampler& env, const glm::vec3& n,
                        float delta) {
  glm::vec3 up =
      std::abs(n.y) < 0.999f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
  const glm::vec3 right = glm::normalize(glm::cross(up, n));
  up = glm::cross(n, right);
  glm::vec3 sum(0.0f);
  int samples = 0;
  for (float phi = 0.0f; phi < 2.0f * kPi; phi += delta) {
    for (float theta = 0.0f; theta < 0.5f * kPi; theta += delta) {
      const float st = std::sin(theta);
      const glm::vec3 tan_sample(st * std::cos(phi), st * std::sin(phi),
                                 std::cos(theta));
      const glm::vec3 world =
          right * tan_sample.x + up * tan_sample.y + n * tan_sample.z;
      sum += env(world) * std::cos(theta) * st;
      ++samples;
    }
  }
  return glm::vec4(kPi * sum / static_cast<float>(samples), 1.0f);
}

// GGX-prefiltered specular radiance from `r` at `roughness`: the environment
// blurred for that gloss level (the split sum's L_i term).
glm::vec4 prefilter_at(const EnvironmentSampler& env, const glm::vec3& r,
                       float roughness, uint32_t samples) {
  const glm::vec3 n = r;
  const glm::vec3 v = r;
  glm::vec3 sum(0.0f);
  float weight = 0.0f;
  for (uint32_t i = 0; i < samples; ++i) {
    const glm::vec3 h =
        importance_sample_ggx(hammersley(i, samples), n, roughness);
    const glm::vec3 l = glm::normalize(2.0f * glm::dot(v, h) * h - v);
    const float n_dot_l = glm::dot(n, l);
    if (n_dot_l > 0.0f) {
      sum += env(l) * n_dot_l;
      weight += n_dot_l;
    }
  }
  return glm::vec4(weight > 0.0f ? sum / weight : env(r), 1.0f);
}

// Environment-BRDF integration (scale, bias) for the split sum, per (NdotV,
// roughness). Independent of the environment -- the standard BRDF LUT.
glm::vec2 brdf_integrate(float n_dot_v, float roughness, uint32_t samples) {
  const glm::vec3 v(std::sqrt(1.0f - n_dot_v * n_dot_v), 0.0f, n_dot_v);
  const glm::vec3 n(0.0f, 0.0f, 1.0f);
  float a = 0.0f;
  float b = 0.0f;
  for (uint32_t i = 0; i < samples; ++i) {
    const glm::vec3 h =
        importance_sample_ggx(hammersley(i, samples), n, roughness);
    const glm::vec3 l = glm::normalize(2.0f * glm::dot(v, h) * h - v);
    const float n_dot_l = std::fmax(l.z, 0.0f);
    const float n_dot_h = std::fmax(h.z, 0.0f);
    const float v_dot_h = std::fmax(glm::dot(v, h), 0.0f);
    if (n_dot_l > 0.0f) {
      const float g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
      const float g_vis = (g * v_dot_h) / std::fmax(n_dot_h * n_dot_v, 1e-6f);
      const float fc = std::pow(1.0f - v_dot_h, 5.0f);
      a += (1.0f - fc) * g_vis;
      b += fc * g_vis;
    }
  }
  return glm::vec2(a / static_cast<float>(samples),
                   b / static_cast<float>(samples));
}

// Mip count of the full chain for a square `size` (floor(log2(size)) + 1).
// Shifts the value down (not the shift amount), so it stays defined for every
// uint32_t -- `size >> 32` would be UB once the amount reached the type width.
uint32_t full_mip_chain(uint32_t size) {
  uint32_t levels = 1;
  while (size > 1) {
    size >>= 1;
    ++levels;
  }
  return levels;
}

// Run `body(i)` for every i in [0, count) across at most
// hardware_concurrency worker threads (never more than `count`), each owning a
// contiguous index range; blocks until every range finishes. Bounding the
// thread count keeps a large bake from oversubscribing the CPU or exhausting
// the OS thread limit. `body` runs on several threads at once, so it must be
// safe to call concurrently for disjoint `i` (writing only its own slot).
template <class Body>
void parallel_for(uint32_t count, const Body& body) {
  if (count == 0) {
    return;
  }
  const uint32_t threads =
      std::max(1u, std::min(std::thread::hardware_concurrency(), count));
  const uint32_t per_band = (count + threads - 1) / threads;
  std::vector<std::future<void>> bands;
  bands.reserve(threads);
  for (uint32_t begin = 0; begin < count; begin += per_band) {
    const uint32_t end = std::min(begin + per_band, count);
    bands.push_back(std::async(std::launch::async, [&body, begin, end] {
      for (uint32_t i = begin; i < end; ++i) {
        body(i);
      }
    }));
  }
  for (std::future<void>& band : bands) {
    band.get();
  }
}

// Pack per-(mip, face) RGBA float pixels from `gen` into tightly packed
// RGBA16F (2 uint32 = 8 bytes/texel), mip-major then face -- ImageUploadDesc's
// layout. 16-bit float cubes filter on the broad device set (incl.
// MoltenVK/Metal); RGBA32F linear filtering is an optional feature many GPUs
// lack. The (mip, face) blocks are convolved concurrently -- `gen` (and the
// EnvironmentSampler it closes over) runs on multiple threads -- then packed
// serially in the same mip-major order; each block's per-texel math is
// unchanged, so the bytes are identical for any thread count.
template <class Gen>
std::vector<uint32_t> pack_cube_rgba16f(uint32_t base_size, uint32_t mips,
                                        Gen gen) {
  // Enumerate the (mip, face) blocks in packing order (mip-major, then face).
  struct Block {
    uint32_t mip;
    int face;
    uint32_t size;
  };
  std::vector<Block> blocks;
  blocks.reserve(static_cast<size_t>(mips) * 6);
  for (uint32_t m = 0; m < mips; ++m) {
    const uint32_t size = (base_size >> m) > 0 ? (base_size >> m) : 1u;
    for (uint32_t f = 0; f < 6; ++f) {
      blocks.push_back({m, static_cast<int>(f), size});
    }
  }

  // Convolve the blocks concurrently into a slot each (disjoint writes), then
  // pack serially in block order -- deterministic regardless of thread count.
  std::vector<std::vector<glm::vec4>> faces(blocks.size());
  parallel_for(static_cast<uint32_t>(blocks.size()), [&](uint32_t i) {
    const Block& b = blocks[i];
    faces[i] = gen(b.mip, b.face, b.size);
  });

  size_t texels = 0;
  for (const std::vector<glm::vec4>& face : faces) {
    texels += face.size();
  }
  std::vector<uint32_t> data;
  data.reserve(texels * 2);  // 2 uint32 per RGBA16F texel
  for (const std::vector<glm::vec4>& face : faces) {
    for (const glm::vec4& px : face) {
      data.push_back(glm::packHalf2x16(glm::vec2(px.x, px.y)));
      data.push_back(glm::packHalf2x16(glm::vec2(px.z, px.w)));
    }
  }
  return data;
}

// Integrate the full LUT into packed RG16F texels (1 uint32 each), one row per
// parallel_for index: each row writes a disjoint span with unchanged per-texel
// math, so any thread count yields identical bytes.
// TODO: embed a prebaked BRDF LUT asset instead of integrating ~4M samples per
// process -- the table never changes.
std::vector<uint32_t> integrate_brdf_lut(uint32_t size, uint32_t samples) {
  std::vector<uint32_t> lut(static_cast<size_t>(size) * size);
  parallel_for(size, [&lut, size, samples](uint32_t y) {
    for (uint32_t x = 0; x < size; ++x) {
      lut[y * size + x] = glm::packHalf2x16(
          brdf_integrate((x + 0.5f) / size, (y + 0.5f) / size, samples));
    }
  });
  return lut;
}

}  // namespace

glm::vec3 cube_face_direction(int face, float u, float v) noexcept {
  switch (face) {
    case 0:
      return glm::normalize(glm::vec3(1.0f, -v, -u));
    case 1:
      return glm::normalize(glm::vec3(-1.0f, -v, u));
    case 2:
      return glm::normalize(glm::vec3(u, 1.0f, v));
    case 3:
      return glm::normalize(glm::vec3(u, -1.0f, -v));
    case 4:
      return glm::normalize(glm::vec3(u, -v, 1.0f));
    default:
      return glm::normalize(glm::vec3(-u, -v, -1.0f));
  }
}

Result<Texture> bake_brdf_lut(UploadBatch& batch, uint32_t size,
                              uint32_t samples) {
  if (size == 0) {
    return Status::invalid_argument("bake_brdf_lut: size must be non-zero");
  }
  if (samples == 0) {
    return Status::invalid_argument("bake_brdf_lut: samples must be non-zero");
  }
  if (!batch.valid()) {  // before integrating, not after
    return Status::invalid_argument("bake_brdf_lut: batch must be open");
  }
  // The concurrent integration can throw (std::async on thread exhaustion, or
  // a worker allocation) -- convert to a Status so the whole API stays on the
  // Result/Status error model rather than letting an exception escape.
  std::vector<uint32_t> lut;
  try {
    lut = integrate_brdf_lut(size, samples);
  } catch (const std::exception& e) {
    return Status::out_of_memory(
        std::string("bake_brdf_lut: concurrent integration failed: ") +
        e.what());
  }
  ImageUploadDesc desc;
  desc.extent = {size, size};
  desc.format = VK_FORMAT_R16G16_SFLOAT;
  desc.pixels = lut.data();
  desc.size = lut.size() * sizeof(uint32_t);
  return batch.add(desc);
}

Result<Texture> bake_brdf_lut(const Device& device, Allocator& allocator,
                              uint32_t size, uint32_t samples) {
  if (device.handle() == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "bake_brdf_lut: device must hold a live VkDevice");
  }
  if (size == 0 || samples == 0) {  // before Vulkan is touched
    return Status::invalid_argument(
        "bake_brdf_lut: size and samples must be non-zero");
  }
  if (size > device.caps().limits().maxImageDimension2D) {
    // From device caps, before the ~size*size-sample integration runs.
    return Status::unsupported(
        "bake_brdf_lut: size exceeds the device's maxImageDimension2D limit");
  }
  VG_ASSIGN(UploadBatch batch, UploadBatch::begin(device, allocator));
  VG_ASSIGN(Texture lut, bake_brdf_lut(batch, size, samples));
  VG_TRY(batch.finish());
  return lut;
}

Result<IblMaps> bake_ibl(const Device& device, Allocator& allocator,
                         const EnvironmentSampler& environment,
                         const IblBakeDesc& desc) {
  if (device.handle() == VK_NULL_HANDLE) {
    return Status::invalid_argument(
        "bake_ibl: device must hold a live VkDevice");
  }
  if (!environment) {
    return Status::invalid_argument("bake_ibl: environment must be callable");
  }
  if (desc.irradiance_size == 0 || desc.prefilter_size == 0 ||
      desc.brdf_lut_size == 0) {
    return Status::invalid_argument(
        "bake_ibl: every texture size must be non-zero");
  }
  // Reject over-limit sizes up front from the device caps -- otherwise the
  // whole CPU convolution (and a size*size LUT allocation) runs before the
  // upload's own limit check finally rejects the extent.
  const VkPhysicalDeviceLimits& limits = device.caps().limits();
  if (desc.irradiance_size > limits.maxImageDimensionCube ||
      desc.prefilter_size > limits.maxImageDimensionCube) {
    return Status::unsupported(
        "bake_ibl: irradiance_size / prefilter_size exceeds the device's "
        "maxImageDimensionCube limit");
  }
  if (desc.brdf_lut_size > limits.maxImageDimension2D) {
    return Status::unsupported(
        "bake_ibl: brdf_lut_size exceeds the device's maxImageDimension2D "
        "limit");
  }
  if (!(desc.irradiance_sample_delta > 0.0f)) {
    return Status::invalid_argument(
        "bake_ibl: irradiance_sample_delta must be positive");
  }
  if (desc.prefilter_samples == 0 || desc.brdf_lut_samples == 0) {
    return Status::invalid_argument(
        "bake_ibl: every sample count must be non-zero");
  }
  if (desc.prefilter_mip_levels == 0 ||
      desc.prefilter_mip_levels > full_mip_chain(desc.prefilter_size)) {
    return Status::invalid_argument(
        "bake_ibl: prefilter_mip_levels must fit prefilter_size's mip chain");
  }

  IblMaps out;

  SamplerDesc sd;
  sd.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sd.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sd.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VG_ASSIGN(Sampler sampler, Sampler::create(device.handle(), sd));
  out.sampler.emplace(std::move(sampler));

  // One batch for all three textures: one submit + fence wait instead of a
  // blocking round trip each. A failure below returns without finishing; the
  // batch (and its pending copies into any dropped textures) is discarded,
  // never submitted.
  VG_ASSIGN(UploadBatch batch, UploadBatch::begin(device, allocator));

  // Queue an RGBA16F cube upload (sampled-ready once the batch finishes).
  const auto add_cube =
      [&batch](uint32_t base_size, uint32_t mips,
               const std::vector<uint32_t>& data) -> Result<Texture> {
    ImageUploadDesc d;
    d.extent = {base_size, base_size};
    d.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    d.pixels = data.data();
    d.size = data.size() * sizeof(uint32_t);
    d.array_layers = 6;
    d.cube = true;
    d.mip_levels = mips;
    return batch.add(d);
  };

  // Convolve both cubes on the CPU first (concurrent per-face). This is the
  // only part that can throw -- std::async on thread exhaustion, or a worker
  // allocation -- so wrap it and convert to a Status rather than letting the
  // exception escape the Result contract.
  const uint32_t mips = desc.prefilter_mip_levels;
  std::vector<uint32_t> irradiance;
  std::vector<uint32_t> prefilter;
  try {
    // Diffuse irradiance: a small single-mip cube -- ample for the
    // low-frequency, heavily-blurred cosine convolution.
    irradiance = pack_cube_rgba16f(
        desc.irradiance_size, 1,
        [&environment, &desc](uint32_t, int face, uint32_t size) {
          std::vector<glm::vec4> px(static_cast<size_t>(size) * size);
          for (uint32_t y = 0; y < size; ++y) {
            for (uint32_t x = 0; x < size; ++x) {
              const float u = (x + 0.5f) / size * 2.0f - 1.0f;
              const float v = (y + 0.5f) / size * 2.0f - 1.0f;
              px[y * size + x] =
                  irradiance_at(environment, cube_face_direction(face, u, v),
                                desc.irradiance_sample_delta);
            }
          }
          return px;
        });
    // Prefiltered specular: mip -> roughness 0..1 over the chain (a tight HDR
    // feature can alias on low mips at the default sample count -- accepted).
    prefilter = pack_cube_rgba16f(
        desc.prefilter_size, mips,
        [&environment, &desc, mips](uint32_t mip, int face, uint32_t size) {
          const float roughness =
              mips > 1 ? static_cast<float>(mip) / static_cast<float>(mips - 1)
                       : 0.0f;
          std::vector<glm::vec4> px(static_cast<size_t>(size) * size);
          for (uint32_t y = 0; y < size; ++y) {
            for (uint32_t x = 0; x < size; ++x) {
              const float u = (x + 0.5f) / size * 2.0f - 1.0f;
              const float v = (y + 0.5f) / size * 2.0f - 1.0f;
              px[y * size + x] =
                  prefilter_at(environment, cube_face_direction(face, u, v),
                               roughness, desc.prefilter_samples);
            }
          }
          return px;
        });
  } catch (const std::exception& e) {
    return Status::out_of_memory(
        std::string("bake_ibl: concurrent convolution failed: ") + e.what());
  }

  VG_ASSIGN(Texture irradiance_tex,
            add_cube(desc.irradiance_size, 1, irradiance));
  out.irradiance = std::move(irradiance_tex);
  VG_ASSIGN(Texture prefilter_tex,
            add_cube(desc.prefilter_size, mips, prefilter));
  out.prefilter = std::move(prefilter_tex);
  out.prefilter_max_lod = static_cast<float>(mips - 1);

  // BRDF integration LUT, environment-independent.
  VG_ASSIGN(Texture lut,
            bake_brdf_lut(batch, desc.brdf_lut_size, desc.brdf_lut_samples));
  out.brdf_lut = std::move(lut);

  // One submit + fence wait for all three IBL textures.
  VG_TRY(batch.finish());
  return out;
}

}  // namespace volumetric_kit::gfx::pipelines

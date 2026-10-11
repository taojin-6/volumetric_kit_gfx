// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The GLSL helpers the pipelines' shaders share (shaders/common/), run on the
// device by a probe compute shader (shaders/common_probe.comp) and checked
// against reference implementations here: the sRGB curve at every 8-bit code
// and across [0, 1], at known values and round-tripped; the ACES tonemap; and
// the full-screen triangle's corners. Skips when the runner exposes no Vulkan
// device.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "gfx_test_support.hpp"
#include "spirv_test_util.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"

namespace {

// The reference curves, in double precision.

// IEC 61966-2-1's decode.
double srgb_decode(double e) {
  return e <= 0.04045 ? e / 12.92 : std::pow((e + 0.055) / 1.055, 2.4);
}

// IEC 61966-2-1's encode.
double srgb_encode(double l) {
  return l <= 0.0031308 ? l * 12.92 : 1.055 * std::pow(l, 1.0 / 2.4) - 0.055;
}

// Narkowicz's ACES fit.
double tonemap_aces(double x) {
  const double y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
  return std::clamp(y, 0.0, 1.0);
}

// How far a device result may stray from the reference, relative to the
// reference. GLSL's pow need only be as precise as exp2(y * log2(x)), which
// the Vulkan spec lets stray by up to about 6e-6 relative over these curves
// (MoltenVK measures under 1e-6); a mistyped constant, a dropped linear toe or
// a pow(x, 2.2) stand-in misses by orders of magnitude more. The absolute
// floor is for a reference of zero.
constexpr double kTolerance = 1e-5;
constexpr double kZeroTolerance = 1e-8;

bool near(double got, double want, double tolerance) {
  return std::abs(got - want) <= kZeroTolerance + tolerance * std::abs(want);
}

// One input's results, as the probe writes them (std430, 80 bytes).
struct Outputs {
  std::array<float, 4> decoded;
  std::array<float, 4> encoded;
  std::array<float, 4> encoded_decoded;
  std::array<float, 4> decoded_encoded;
  std::array<float, 4> tonemapped;
};
static_assert(sizeof(Outputs) == 80, "mirrors common_probe.comp");

// What the probe computed from one value.
struct Sample {
  float input = 0.0f;
  float decoded = 0.0f;
  float encoded = 0.0f;
  float encoded_decoded = 0.0f;
  float decoded_encoded = 0.0f;
  float tonemapped = 0.0f;
};

// The 256 8-bit codes, as a UNORM texel reads them, then [0, 1] in 4096
// equal steps.
std::vector<float> unit_values() {
  std::vector<float> values;
  for (int code = 0; code < 256; ++code) {
    values.push_back(static_cast<float>(code) / 255.0f);
  }
  for (int i = 0; i <= 4096; ++i) {
    values.push_back(static_cast<float>(i) / 4096.0f);
  }
  return values;
}

class ShaderCommonTest : public vg_test::RendererDeviceTest {
 protected:
  vkc::test::Validation validation() const override {
    return vkc::test::Validation::On;
  }

  void SetUp() override {
    RendererDeviceTest::SetUp();
    if (base_setup_incomplete()) {
      return;
    }
    const std::vector<uint32_t> spv =
        vg_test::load_spirv(vg_test::spirv_path("common_probe.comp.spv"));
    ASSERT_FALSE(spv.empty());
    vkc::KernelSetBuilder builder(device());
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(uint32_t)};
    const vkc::Status added =
        builder.add(probe_, "common_probe",
                    reinterpret_cast<const unsigned char*>(spv.data()),
                    spv.size() * sizeof(uint32_t), 3, &push);
    ASSERT_TRUE(added.ok()) << added.message();
    auto pool = builder.build();
    ASSERT_TRUE(pool.ok()) << pool.status().message();
    pool_.emplace(std::move(pool).value());
  }

  // Runs the probe over `values`, each in every channel, and returns one
  // sample per value and channel; `corners` receives the full-screen
  // triangle's, x then y of each.
  std::vector<Sample> run(const std::vector<float>& values,
                          std::array<float, 6>* corners = nullptr) {
    const auto count = static_cast<uint32_t>(values.size());
    // Rotate the values through the channels, so each lands in every one.
    std::vector<std::array<float, 4>> inputs(count);
    for (uint32_t i = 0; i < count; ++i) {
      for (uint32_t c = 0; c < 3; ++c) {
        inputs[i][c] = values[(i + c * count / 3) % count];
      }
      inputs[i][3] = 0.0f;
    }
    const VkDeviceSize input_bytes = inputs.size() * sizeof(inputs[0]);
    const VkDeviceSize output_bytes = count * sizeof(Outputs);

    auto in = vkc::device_storage_buffer(allocator(), input_bytes);
    auto out = vkc::device_storage_buffer(allocator(), output_bytes);
    std::array<float, 6> got_corners{};
    auto corner_buf =
        vkc::device_storage_buffer(allocator(), sizeof(got_corners));
    EXPECT_TRUE(in.ok() && out.ok() && corner_buf.ok());
    if (!in.ok() || !out.ok() || !corner_buf.ok()) {
      return {};
    }
    probe_.set.write_storage_buffer(0, in.value().handle(), 0, VK_WHOLE_SIZE);
    probe_.set.write_storage_buffer(1, out.value().handle(), 0, VK_WHOLE_SIZE);
    probe_.set.write_storage_buffer(2, corner_buf.value().handle(), 0,
                                    VK_WHOLE_SIZE);

    std::vector<Outputs> results(count);
    vkc::CommandBatch batch(device(), allocator());
    // A failed call poisons the batch, so submit() reports the first.
    (void)batch.upload(in.value(), 0, inputs.data(), input_bytes);
    (void)batch.dispatch(probe_, &count, sizeof(count),
                         vkc::group_count(count, 64),
                         device().caps().limits().maxComputeWorkGroupCount[0]);
    (void)batch.readback(out.value(), 0, output_bytes, results.data());
    (void)batch.readback(corner_buf.value(), 0, sizeof(got_corners),
                         got_corners.data());
    const vkc::Status submitted = batch.submit();
    EXPECT_TRUE(submitted.ok()) << submitted.message();
    if (!submitted.ok()) {
      return {};
    }
    if (corners != nullptr) {
      *corners = got_corners;
    }

    std::vector<Sample> samples;
    for (uint32_t i = 0; i < count; ++i) {
      const Outputs& r = results[i];
      for (uint32_t c = 0; c < 3; ++c) {
        samples.push_back({inputs[i][c], r.decoded[c], r.encoded[c],
                           r.encoded_decoded[c], r.decoded_encoded[c],
                           r.tonemapped[c]});
      }
    }
    return samples;
  }

  // The first sample of `value`, which run() was given.
  static Sample find(const std::vector<Sample>& samples, float value) {
    for (const Sample& s : samples) {
      if (s.input == value) {
        return s;
      }
    }
    ADD_FAILURE() << "no sample of " << value;
    return {};
  }

  vkc::ComputeKernel probe_;
  std::optional<vkc::DescriptorPool> pool_;
};

// One probe returns both curves, their round trips and the triangle corners.
// Check them together so the same inputs are uploaded and dispatched only once.
TEST_F(ShaderCommonTest, SrgbCurvesRoundTripsAndFullscreenTriangle) {
  std::vector<float> values = unit_values();
  for (const float threshold : {0.04045f, 0.0031308f}) {
    values.push_back(std::nextafter(threshold, 0.0f));
    values.push_back(threshold);
    values.push_back(std::nextafter(threshold, 1.0f));
  }
  values.push_back(0.18f);  // standard mid-grey, beyond the regular grid
  std::array<float, 6> corners{};
  const std::vector<Sample> samples = run(values, &corners);
  ASSERT_EQ(samples.size(), values.size() * 3);
  for (const Sample& s : samples) {
    EXPECT_TRUE(near(s.decoded, srgb_decode(s.input), kTolerance))
        << "srgb_to_linear(" << s.input << ") = " << s.decoded << ", want "
        << srgb_decode(s.input);
    EXPECT_TRUE(near(s.encoded, srgb_encode(s.input), kTolerance))
        << "linear_to_srgb(" << s.input << ") = " << s.encoded << ", want "
        << srgb_encode(s.input);
  }
  // The standard's own numbers independently anchor the reference curves:
  // the ends, each branch's threshold, and mid-grey both ways.
  EXPECT_EQ(find(samples, 0.0f).decoded, 0.0f);
  EXPECT_EQ(find(samples, 0.0f).encoded, 0.0f);
  EXPECT_NEAR(find(samples, 1.0f).decoded, 1.0f, 1e-6f);
  EXPECT_NEAR(find(samples, 1.0f).encoded, 1.0f, 1e-6f);
  EXPECT_NEAR(find(samples, 0.5f).decoded, 0.21404114f, 1e-6f);
  EXPECT_NEAR(find(samples, 0.5f).encoded, 0.73535698f, 1e-6f);
  EXPECT_NEAR(find(samples, 0.04045f).decoded, 0.0031308050f, 1e-8f);
  EXPECT_NEAR(find(samples, 0.0031308f).encoded, 0.040449936f, 1e-7f);
  EXPECT_NEAR(find(samples, 0.18f).encoded, 0.46135613f, 1e-6f);
  // Decoding then encoding returns every 8-bit code exactly, and either order
  // returns any value in [0, 1] to within both curves' tolerance.
  for (const Sample& s : samples) {
    EXPECT_TRUE(near(s.encoded_decoded, s.input, 2 * kTolerance))
        << "linear_to_srgb(srgb_to_linear(" << s.input
        << ")) = " << s.encoded_decoded;
    EXPECT_TRUE(near(s.decoded_encoded, s.input, 2 * kTolerance))
        << "srgb_to_linear(linear_to_srgb(" << s.input
        << ")) = " << s.decoded_encoded;
  }
  for (int code = 0; code < 256; ++code) {
    const Sample s = find(samples, static_cast<float>(code) / 255.0f);
    EXPECT_EQ(std::lround(s.encoded_decoded * 255.0f), code);
  }

  // The triangle's hypotenuse passes through (1, 1), covering [-1, 1]^2.
  const std::array<float, 6> want = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
  EXPECT_EQ(corners, want);
}

// tonemap_aces follows the reference fit from black through HDR values well
// past where it saturates (about 7.24), and hits its known values.
TEST_F(ShaderCommonTest, TonemapAcesFollowsTheReference) {
  std::vector<float> values;
  for (int i = 0; i <= 1024; ++i) {
    values.push_back(static_cast<float>(i) / 16.0f);  // 0 .. 64
  }
  values.push_back(0.18f);
  const std::vector<Sample> samples = run(values);
  ASSERT_EQ(samples.size(), values.size() * 3);
  for (const Sample& s : samples) {
    EXPECT_TRUE(near(s.tonemapped, tonemap_aces(s.input), kTolerance))
        << "tonemap_aces(" << s.input << ") = " << s.tonemapped << ", want "
        << tonemap_aces(s.input);
  }
  EXPECT_EQ(find(samples, 0.0f).tonemapped, 0.0f);
  EXPECT_NEAR(find(samples, 0.18f).tonemapped, 0.26689892f, 1e-6f);
  EXPECT_NEAR(find(samples, 1.0f).tonemapped, 2.54f / 3.16f, 1e-6f);
  EXPECT_EQ(find(samples, 64.0f).tonemapped, 1.0f);
}

}  // namespace

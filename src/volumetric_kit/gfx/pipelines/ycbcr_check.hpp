// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The check of a YcbcrCoding that ImagePipeline's NV12 convert and
// HybridMeshTextureSet's two-plane picture share.

#include <string>
#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/gfx/pipelines/ycbcr.hpp"

namespace volumetric_kit::gfx::pipelines::detail {

// OK for weights a matrix can invert (kr > 0, kb > 0, kr + kb < 1; NaN fails
// each) and a siting ChromaSiting names; otherwise InvalidArgument, the
// message led by `call`.
inline core::Status check_ycbcr_coding(std::string_view call,
                                       const YcbcrCoding& coding) {
  if (!(coding.kr > 0.0f && coding.kb > 0.0f && coding.kr + coding.kb < 1.0f)) {
    return core::Status::invalid_argument(
        std::string(call) +
        ": the matrix weights need kr > 0, kb > 0 and kr + kb < 1");
  }
  if (coding.chroma_siting != ChromaSiting::Center &&
      coding.chroma_siting != ChromaSiting::Left) {
    return core::Status::invalid_argument(
        std::string(call) + ": the chroma siting is not Center or Left");
  }
  return core::Status{};
}

}  // namespace volumetric_kit::gfx::pipelines::detail

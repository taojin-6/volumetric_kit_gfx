// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Minimal downstream consumer for the `package` CI job. Building it proves the
// public headers, exported targets, and find_dependency wiring resolve and link
// (both via an installed find_package and via add_subdirectory). Running it
// exercises the linked symbols (version.cpp, result.cpp); it makes no Vulkan
// calls, so it needs no GPU.

#include <cstdio>
#include <string>

#include "volumetric_kit/gfx/core/result.hpp"
#include "volumetric_kit/gfx/version.hpp"

int main() {
  namespace vg = volumetric_kit::gfx;
  const vg::Status ok;
  const vg::Status err = vg::Status::invalid_argument("smoke");
  std::printf("volumetric_kit_gfx %s: ok=%d, err domain=%s\n",
              vg::version_string(), static_cast<int>(ok.ok()),
              std::string(vg::to_string(err.domain())).c_str());
  return (ok.ok() && !err.ok()) ? 0 : 1;
}

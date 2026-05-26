// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/gfx/version.hpp"

namespace volumetric_kit::gfx {

const char* version_string() noexcept { return VG_VERSION_STRING; }

int version_major() noexcept { return VG_VERSION_MAJOR; }

int version_minor() noexcept { return VG_VERSION_MINOR; }

int version_patch() noexcept { return VG_VERSION_PATCH; }

}  // namespace volumetric_kit::gfx

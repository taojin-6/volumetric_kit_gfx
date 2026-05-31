# volumetric_kit_gfx

[![lint](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/lint.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/lint.yml)
[![ubuntu-22.04](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-ubuntu-2204.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-ubuntu-2204.yml)
[![ubuntu-24.04](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-ubuntu-2404.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-ubuntu-2404.yml)
[![macos-26](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-macos-26.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-macos-26.yml)

A standalone, reusable **Vulkan** renderer library (MoltenVK on Apple) for volumetric
workloads.

One Vulkan path serves Linux / Windows / Android in addition to macOS / iOS, with a single
GLSL → SPIR-V shader set. See [`CLAUDE.md`](./CLAUDE.md) for naming conventions and locked
design decisions.

> **Status:** Early development. The `core` tier — instance/device bring-up, VMA
> allocator + RAII buffer/texture, sync primitives (fence, binary + timeline semaphore),
> deferred-destruction retire queue, shader modules, and the `Status`/`Result`/logging
> foundation — is implemented and tested (GoogleTest suite, run under ASan/UBSan/LSan in
> CI). The windowing/swapchain, `passes`/`pipelines`/`app` tiers, and CUDA/Metal interop
> are not yet implemented. See [`CHANGELOG.md`](./CHANGELOG.md) for what has landed and
> [`DESIGN.md`](./DESIGN.md) for the tier roadmap.

## Architecture (tiered)

`core` → `passes` → `pipelines` → `app` (+ `windowing`, `interop`, `assets`, `camera`).
Simple consumers link `volumetric_kit::gfx_app`; advanced consumers compose passes on
`volumetric_kit::gfx_core`. The dependency rule is strict: a tier may only depend on tiers
to its left.

## Prerequisites (macOS / Apple Silicon)

The platform toolchain comes from Homebrew:

```bash
brew install molten-vk glfw glm \
             vulkan-headers vulkan-loader vulkan-validationlayers vulkan-tools \
             shaderc spirv-cross
```

Sanity-check that the loader sees your GPU through MoltenVK:

```bash
vulkaninfo --summary   # expect GPU0 ... driverID = DRIVER_ID_MOLTENVK
```

## Use it in your project

The library is consumable both via `FetchContent` and an installed
`find_package`. Link the component target you need (today only `gfx_core` is
built); the umbrella alias `volumetric_kit::gfx` pulls in whatever tiers exist.

```cmake
# Option A — FetchContent (pin GIT_TAG to a release tag or commit SHA):
include(FetchContent)
FetchContent_Declare(
  volumetric_kit_gfx
  GIT_REPOSITORY https://github.com/taojin-6/volumetric_kit_gfx.git
  GIT_TAG main)
FetchContent_MakeAvailable(volumetric_kit_gfx)

# Option B — installed package:
#   find_package(volumetric_kit_gfx CONFIG REQUIRED)

target_link_libraries(your_app PRIVATE volumetric_kit::gfx_core)
```

Useful options. `VG_BUILD_TESTS`, `VG_BUILD_EXAMPLES`, and `VG_INSTALL` default
ON only when `volumetric_kit_gfx` is the top-level project (OFF when it is
consumed via FetchContent / `add_subdirectory`). `VG_WITH_GLFW` and
`VG_WARNINGS_AS_ERRORS` default ON regardless; `VG_WITH_CUDA` defaults OFF.
`VG_SANITIZE` is a semicolon list, empty (off) by default — e.g.
`-DVG_SANITIZE="address;undefined"`.

On Linux the prerequisites come from the package manager, e.g. on Ubuntu:

```bash
sudo apt-get install -y cmake libvulkan-dev glslang-tools \
                        mesa-vulkan-drivers vulkan-tools   # lavapipe for headless tests
```

## Development

```bash
pre-commit install   # clang-format + cmake-format + hygiene hooks
```

## License

Released under the [MIT License](LICENSE). Each source file carries an
`SPDX-License-Identifier: MIT` header; vendored third-party code under
`third_party/` keeps its own upstream license.

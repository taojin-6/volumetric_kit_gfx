# volumetric_kit_gfx

[![ci](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/ci.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/ci.yml)

A standalone, reusable **Vulkan** renderer library (MoltenVK on Apple) for volumetric
workloads.

One Vulkan path serves Linux / Android in addition to macOS / iOS, with a single
GLSL → SPIR-V shader set. [AGENTS.md](AGENTS.md) is the shared working guide
for Codex and Claude Code; `CLAUDE.md` imports it. [DECISIONS.md](DECISIONS.md)
records the locked choices, and [CHANGELOG.md](CHANGELOG.md) records what has
landed. The detailed `DESIGN.md` is intentionally local and gitignored.

## Architecture (tiered)

`core` → `passes` → `pipelines` → `app` (+ `windowing`, `interop`, `assets`, `io`, `camera`, `ui`).
Simple consumers link `volumetric_kit::gfx_app`; advanced consumers compose passes on
`volumetric_kit::gfx_core`. The dependency rule is strict: a tier may only depend on tiers
to its left, with `core` and `assets` as foundational roots. See
[AGENTS.md](AGENTS.md#architecture) for the dependency boundaries.

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
`find_package`. Link the component target you need; the umbrella alias
`volumetric_kit::gfx` pulls in the available tiers.

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

gfx's errors (`Status`, `Result`, `VKC_TRY` / `VKC_ASSIGN` / `VKC_CHECK`) and
log sink are [volumetric_kit_core](https://github.com/taojin-6/volumetric_kit_core)'s
base tier, and its instance, device, allocator, buffers and images, Vulkan
umbrella header, `VkResult` bridge, format helpers and shader build functions
are the core's vulkan tier, which gfx's API names as the core does
(`volumetric_kit::core::`, `vkc::` by convention); both are fetched pinned by commit and linked PUBLIC, so a
`Status`, a device or a buffer passes between gfx, recon and calib unchanged
and one `set_log_handler` routes all three. An installed gfx carries
the core beside it and re-finds it; an application that also fetches recon
declares `volumetric_kit_core` first to pick one pin for both, at or after
gfx's and with the core's vulkan tier on (`VKC_WITH_VULKAN`), which gfx checks
at configure. Build against a local core checkout with
`-DFETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE=<path>`.

gfx compiles against the system's Vulkan headers -- 1.3.204 or newer, 1.3.208
on Apple, the core's floor -- and vendors none. To build against a pinned copy,
point `Vulkan_INCLUDE_DIR` at it; the loader still comes from the system.

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

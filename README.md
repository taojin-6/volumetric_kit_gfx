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

`core` → `passes` → `pipelines` → `app` → `glfw` (+ `windowing`, `interop`, `assets`, `io`,
`camera`, `ui`).
Simple consumers link `volumetric_kit::gfx_app`; advanced consumers compose passes on
`volumetric_kit::gfx_core`. The dependency rule is strict: a tier may only depend on tiers
to its left, with `core`, `assets` and `camera` as foundational roots. See
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

The build-tree `volumetric_kit::gfx_ui` target uses 32-bit ImGui indices and
propagates `ImDrawIdx=unsigned int` to consumers, including any ImGui platform
backend or demo sources they compile. Rebuild those sources with the target's
usage requirements; all code sharing its ImGui context must use the same index
type.

gfx's errors (`Status`, `Result`, `VKC_TRY` / `VKC_ASSIGN` / `VKC_CHECK`) and
log sink are
[volumetric_kit_core](https://github.com/taojin-6/volumetric_kit_core)'s base
tier, and its instance, device, allocator, buffers and images, Vulkan umbrella
header, `VkResult` bridge, format helpers and shader build functions are the
core's vulkan tier, which gfx's API names as the core does
(`volumetric_kit::core::`, `vkc::` by convention); both are fetched pinned by
commit and linked PUBLIC, so a `Status`, a device or a buffer passes between
gfx, recon and calib unchanged and one `set_log_handler` routes all three. An
installed gfx carries the core beside it and re-finds it; an application that
also fetches recon declares `volumetric_kit_core` first to pick one pin for
both, no older than `VG_VKC_MIN_VERSION` (CMakeLists.txt) and with the core's
vulkan tier on (`VKC_WITH_VULKAN`), which gfx checks at configure with the
core's `vkc_require_core`. Build against a local core checkout with
`-DFETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE=<path>`.

gfx compiles against the system's Vulkan headers -- 1.3.275 or newer with the
pinned core -- and vendors none. To build against a pinned copy,
point `Vulkan_INCLUDE_DIR` at it; the loader still comes from the system.

Useful options. `VG_BUILD_TESTS`, `VG_BUILD_EXAMPLES`, `VG_INSTALL` and
`VG_WARNINGS_AS_ERRORS` default ON only when `volumetric_kit_gfx` is the
top-level project (OFF when it is consumed via FetchContent /
`add_subdirectory`). `VG_WITH_GLFW` (`gfx_glfw` and the examples, where GLFW
is installed) defaults ON regardless; `VG_WITH_CUDA` defaults OFF.
`VG_SANITIZE` is a semicolon list, empty (off) by default — e.g.
`-DVG_SANITIZE="address;undefined"`.

On Linux the prerequisites come from the package manager, e.g. on Ubuntu:

```bash
sudo apt-get install -y cmake libvulkan-dev glslang-tools \
                        mesa-vulkan-drivers vulkan-tools   # lavapipe for headless tests
```

## Development

The examples use core's versioned JSON run configuration:

```bash
build/examples/example_03_model --config examples/config/smoke.json
build/examples/example_03_model --help
# No display or GLFW initialization is needed for a screenshot:
build/examples/example_03_model --config build/examples/config/model-screenshot.json
```

The smoke configuration runs 30 frames. A minimal
`{"version": 1, "operation": "view"}` file opens the example with its usual
defaults; omit `run.frames` for an interactive run.
The model example shows a cube unless `source` selects a model:

```json
{
  "version": 1,
  "operation": "view",
  "source": {"kind": "model", "path": "models/Helmet.glb"},
  "window": {"width": 1024, "height": 720, "validation": true},
  "output": {"screenshot": "helmet.ppm"}
}
```

Paths are relative to the configuration file. A screenshot uses `window.width`
and `window.height` as output pixels and runs without opening a window; without
`output.screenshot`, they set the initial window size. The copied screenshot
example writes `build/examples/config/model.ppm`. `window.overlay` controls the
panels in the ImGui and model examples. Core owns the schema, defaults, strict
key/type/range checks and application compatibility checks. The command line
accepts only `--config <file>` and `--help`; the former individual flags are
replaced by file fields.

Only the example executables need `volumetric_kit::core_config`. Gfx enables
`VKC_WITH_CONFIG` when those executables build; a consuming application that
makes core available first must enable it there if it also enables gfx's
examples. Renderer libraries and their installed APIs remain independent of
the JSON configuration component.

```bash
pre-commit install   # clang-format + cmake-format + hygiene hooks
```

## License

Released under the [MIT License](LICENSE). Each source file carries an
`SPDX-License-Identifier: MIT` header; vendored third-party code under
`third_party/` keeps its own upstream license.

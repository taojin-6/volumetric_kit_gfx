# volumetric_kit_gfx

[![lint](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/lint.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/lint.yml)
[![ubuntu-22.04](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-ubuntu-2204.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-ubuntu-2204.yml)
[![ubuntu-24.04](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-ubuntu-2404.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-ubuntu-2404.yml)
[![macOS](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-macos.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_gfx/actions/workflows/build-macos.yml)

A standalone, reusable **Vulkan** renderer library (MoltenVK on Apple) for volumetric
workloads.

One Vulkan path serves Linux / Windows / Android in addition to macOS / iOS, with a single
GLSL → SPIR-V shader set. See [`CLAUDE.md`](./CLAUDE.md) for naming conventions and locked
design decisions.

> **Status:** Empty skeleton — only the repo scaffolding and pre-commit tooling are in
> place. The `core` tier, windowing/swapchain, the `passes`/`pipelines` tiers, and
> CUDA/Metal interop are not yet implemented.

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

## Development

```bash
pre-commit install   # clang-format + cmake-format + hygiene hooks
```

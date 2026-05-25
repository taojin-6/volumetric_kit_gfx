# volumetric_kit_gfx

A standalone, reusable **Vulkan** renderer library (MoltenVK on Apple) for volumetric
workloads.

One Vulkan path serves Linux / Windows / Android in addition to macOS / iOS, with a single
GLSL → SPIR-V shader set.

## Naming conventions (use these consistently)

- Package/repo: `volumetric_kit_gfx`
- Namespace: `volumetric_kit::gfx`. Internally and in docs, `vg::` abbreviates
  `volumetric_kit::gfx::`.
- Headers: `include/volumetric_kit/gfx/<tier>/…` → e.g. `#include "volumetric_kit/gfx/core/context.hpp"`
- CMake: `find_package(volumetric_kit_gfx)`; component targets `volumetric_kit::gfx_core`,
  `…_passes`, `…_pipelines`, `…_app` (+ `…_windowing`, `…_interop`, `…_assets`, `…_camera`);
  umbrella alias `volumetric_kit::gfx`.

## Architecture (tiered)

`core` → `passes` → `pipelines` → `app` (+ `windowing`, `interop`, `assets`, `camera`).
Simple consumers link `…_app`; advanced consumers compose their own passes on `…_core`.
The dependency rule is strict: a tier may only depend on tiers to its left.

## Locked decisions

- **Single rendering API: Vulkan + MoltenVK on Apple.** Chosen for cross-platform reach
  (Linux/Windows/Android + Mac/iOS) and one shader source / one renderer to maintain. A
  native-Metal iOS backend is a *fallback only*, gated on an iPad validation spike.
- **Compute stays CUDA (desktop) + Metal (Apple)** — NOT unified to Vulkan compute. The renderer
  meets compute at a thin external-memory interop layer (`vg::interop::{Cuda,Metal}ExternalMemory`).
- **One GLSL shader source per technique** (→ SPIR-V; MoltenVK consumes SPIR-V — no MSL hand-port).
- **Descriptor layouts from spirv-cross reflection.** No global frame type; `Pipeline::submit()`
  takes a per-pipeline struct. No scene graph in the library.

## Key gotchas (verified)

- ✅ MoltenVK supports `VK_KHR_fragment_shader_barycentric` (Apple6+ HW gate) — techniques
  relying on per-triangle barycentric interpolation port cleanly.
- ⚠️ MoltenVK's external-semaphore export to `MTLSharedEvent` is buggy/incomplete. Do NOT build
  compute↔render sync on it. **Double-buffer the shared resource + release via a
  `VkFence`-completion CPU token** instead.

## Where to start

The repo is currently bare scaffolding (config + pre-commit only). First milestone: a repo
skeleton (CMake + volk/VMA + third_party) and an `examples/01_triangle` hello-triangle smoke
test. Validating MoltenVK on the target Apple GPU early is worth doing — it decides whether
the single Vulkan path holds or a native-Metal fallback is needed.

## Working preferences

- Prefer plain, behavior-level tests over friend-class backdoors into private state.

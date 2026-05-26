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
- **Vulkan via the link-time loader (`Vulkan::Vulkan`), accessed through one internal umbrella
  header (`core/vulkan.hpp`, added in PR2).** No volk for now — the loader/dispatch ceremony
  (`VK_NO_PROTOTYPES`, global function pointers, per-platform init) isn't worth it while iOS/Android
  and CUDA interop are deferred. Because every call site includes only the umbrella header, adopting
  volk later (for iOS/Android loader portability or `volkLoadDevice` dispatch perf) is a non-breaking
  change to that one header + the link line — not a one-way door. VMA uses the linked Vulkan
  prototypes (`VMA_STATIC_VULKAN_FUNCTIONS`).

## Key gotchas (verified)

- ✅ MoltenVK supports `VK_KHR_fragment_shader_barycentric` (Apple6+ HW gate) — techniques
  relying on per-triangle barycentric interpolation port cleanly.
- ⚠️ MoltenVK's external-semaphore export to `MTLSharedEvent` is buggy/incomplete. Do NOT build
  compute↔render sync on it. **Double-buffer the shared resource + release via a
  `VkFence`-completion CPU token** instead.

## Where to start

The repo is currently bare scaffolding (config + pre-commit only). First milestone: a repo
skeleton (CMake + VMA + third_party) and an `examples/01_triangle` hello-triangle smoke
test. Validating MoltenVK on the target Apple GPU early is worth doing — it decides whether
the single Vulkan path holds or a native-Metal fallback is needed.

## RAII resource types (handle/deleter wrappers)

Every type that owns a Vulkan/VMA handle — or a deleter that frees one — follows the same
shape. These are the mistakes reviews keep catching, so get them right at authoring time:

- **Move-only.** `= delete` the copy ctor/assign; `= default` (or hand-write) the move pair.
  A copyable wrapper double-frees — e.g. a copied `std::function` deleter runs twice.
- **Reset *every* owned member on each ownership transfer** — in the move ctor, move
  assignment, *and* `destroy()`. Null the handle *and* zero the metadata (`size_`,
  `extent_`, `format_`, `mapped_`, …) and the deleter, so a moved-from / destroyed object
  is fully empty and its accessors stay consistent with `valid()`. Forgetting a scalar
  (e.g. `size_`/`extent_`) is the recurring miss.
- **`operator=` guards self-move** (`if (this != &other)`) and runs `destroy()` on the
  current state before adopting the source's.
- **Type-erase the backend via a `std::function<void()>` deleter** so VMA/etc. stay out of
  the public header (see `Buffer`/`Texture`). Reset the moved-from `deleter_` to `nullptr`
  explicitly — a moved-from `std::function` is valid-but-unspecified and can otherwise run
  twice. The producing owner (e.g. the `Allocator` and the device it wraps) must outlive
  the resource: state that in an `@warning` and point at `RetireQueue` for fence-gated
  destruction.
- **Validate before creating** — reject zero size/extent, `usage == 0`,
  `VK_FORMAT_UNDEFINED`, etc. with a non-OK `Status` before touching Vulkan/VMA.

**Tests for every move-only type** (not just the headline behavior):
- move-construct → assert the *moved-from source* is empty, not only the destination;
- move-assign *over a live object* — exercises the `destroy()`-then-adopt path where
  double-free/leak bugs live;
- self-move — launder through a pointer (`T* p = &x; x = std::move(*p);`) to dodge
  `-Wself-move` under `-Werror`.

The `sanitizers` CI job (ASan/UBSan/LSan, Linux) is what turns those tests into actual
leak/double-free detectors — a green normal build is necessary but not sufficient.

## Working preferences

- Prefer plain, behavior-level tests over friend-class backdoors into private state.
- Mark deferred/future work inline with a `TODO:` comment (e.g. `// TODO: adopt volk for the
  iOS static-MoltenVK path`) so it is greppable, rather than tracking it only in prose or commits.
- Document public classes/functions with full Doxygen, matching `include/volumetric_kit/gfx/core/result.hpp`:
  `@file`/`@brief` on the header, `@brief` + a `@code … @endcode` example per class, and
  `@brief`/`@param`/`@return` (`@pre` where relevant) on methods; accessors may be a single `/// @return`.
  Do **not** write "move-only" (or similar) in prose — the deleted-copy/defaulted-move declarations
  already convey it.

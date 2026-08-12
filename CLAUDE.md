# volumetric_kit_gfx

A standalone, reusable **Vulkan** renderer library (MoltenVK on Apple) for volumetric
workloads.

One Vulkan path serves Linux / Android in addition to macOS / iOS, with a single
GLSL → SPIR-V shader set.

## Naming conventions (use these consistently)

- Package/repo: `volumetric_kit_gfx`
- Namespace: `volumetric_kit::gfx`. Internally and in docs, `vg::` abbreviates
  `volumetric_kit::gfx::`.
- Headers: `include/volumetric_kit/gfx/<tier>/…` → e.g. `#include "volumetric_kit/gfx/core/context.hpp"`
- CMake: `find_package(volumetric_kit_gfx)`; component targets `volumetric_kit::gfx_core`,
  `…_passes`, `…_pipelines`, `…_app` (+ `…_windowing`, `…_interop`, `…_assets`, `…_io`,
  `…_camera`, `…_ui`); umbrella alias `volumetric_kit::gfx`.

## Architecture (tiered)

`core` → `passes` → `pipelines` → `app` (+ `windowing`, `interop`, `assets`, `io`, `camera`,
`ui`). Simple consumers link `…_app`; advanced consumers compose their own passes on `…_core`.
The dependency rule is strict: a tier may only depend on tiers to its left — with `core`
and `assets` as foundational roots (no tier dependencies of their own) that any tier may
build on (e.g. `io` → `assets`, `pipelines` → `core` + `assets`). `assets` is the
format-neutral CPU data model (header-only, glm); `io` holds the file loaders that
produce it (glTF now; OBJ/PLY/assimp later) and so depends on `assets`. `ui` is a Dear
ImGui debug overlay (depends only on `core`): it wraps ImGui's *renderer* backend
(`imgui_impl_vulkan`) and draws into a `RenderTarget` via dynamic rendering; like
`windowing` it is GLFW-free, so the *platform* backend (`imgui_impl_glfw`) stays in the
consumer/example.

## Locked decisions

- **Single rendering API: Vulkan + MoltenVK on Apple.** Chosen for cross-platform reach
  (Linux/Android + Mac/iOS) and one shader source / one renderer to maintain. A
  native-Metal iOS backend is a *fallback only*, gated on an iPad validation spike.
- **Compute stays CUDA (desktop) + Metal (Apple)** — NOT unified to Vulkan compute. The renderer
  meets compute at a thin external-memory interop layer (`vg::interop::{Cuda,Metal}ExternalMemory`).
- **A `VkDevice` may be created *or adopted*.** `core::Device` accepts a device the embedder
  already created — non-owning `Device::adopt(AdoptedDevice, DeviceConfig)` alongside
  `Device::create`, gated on a `DeviceRequirements` descriptor it verifies against the creator's
  declared enabled state (Vulkan can't be queried for a *logical* device's enabled
  features/extensions). This lets a sibling **Vulkan-compute** library (e.g. `volumetric_kit_recon`)
  share one `VkDevice` with the renderer and hand over `VkBuffer`/`VkImage` **zero-copy** — the
  same-API case that needs none of the CUDA/Metal external-memory machinery above. The embedding app
  owns the shared instance/device and merges both libraries' requirements; gfx stays standalone
  (`create` is unchanged). The indirect-draw path a *live* mesh needs has since landed
  (`pipelines::LiveMesh`, below); per-slot material/atlas ringing for a live-updated texture is
  what remains.
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
- **2026-07-06 — Hybrid mesh pipeline (the `volumetric_kit_recon` color handoff).**
  `pipelines::HybridMeshPipeline` — the "mesh pipeline" the device-adopt decision anticipated —
  renders a world-space interleaved `assets::Vertex` mesh and chooses albedo *per fragment*: the
  atlas texel (set 0, a combined-image-sampler, sampled at `uv0`) where a triangle won projective
  texturing, else the per-vertex `color` where `uv0` is the `(-1,-1)` sentinel (recon's TSDF
  vertex-color fallback); lit or unlit via a push-constant flag (`light.w`). Reuses `assets::Vertex`
  + `GpuMesh` unchanged (binds position/normal/uv0/color, not tangent), one push constant carries the
  view-projection + light (no per-frame UBO), and the sampler set is the pipeline's only descriptor
  set. This is the *static* data-path (upload a mesh + atlas, draw); proven headless via an offscreen
  draw + pixel readback under validation. A vertex-color/atlas mesh pipeline is a broadly-useful
  renderer feature, so the siblings stay independent — gfx gains a capability, not a dependency on
  recon.
- **2026-08-12 — Patch mesh pipeline (per-PRIMITIVE atlas addressing).**
  `pipelines::PatchMeshPipeline` draws the same handoff shaded from a progressive **per-triangle
  patch atlas**, which `HybridMeshPipeline` structurally cannot: a patch belongs to a triangle, so
  its three corners need three distinct coordinates, and a vertex shared between up to six triangles
  has one `uv0` slot to put them in. The atlas offset is `gl_PrimitiveID` — under an indexed indirect
  draw with `firstIndex == 0` that is exactly the producer's arena triangle slot, so no side table
  maps one to the other — and the position within a patch comes from the fragment's barycentric
  coordinate, **recovered** from the interpolated world position and the triangle's three corners
  rather than read from `VK_KHR_fragment_shader_barycentric`: that extension is supported on the
  Apple targets but is a *device feature*, and taking it would thread a requirement through every
  embedder's device creation (including the neutral two-library bootstrap) to save a 2×2 solve.
  Set 0 is three storage buffers — atlas, index run, vertices — declared as flat scalar arrays so no
  `scalarBlockLayout` is needed either. A texel no frame observed falls back to the per-vertex colour
  **per fragment** on the stored weight, so a partially observed surface fades along real patch
  boundaries instead of switching whole triangles. The atlas is decoded from canonical encoded 8-bit
  in the shader, since a storage buffer has no `_SRGB` format to do it in hardware — that, plus
  filtering and mips, is what a buffer atlas gives up against an image. `HybridMeshPipeline` is
  untouched.
- **2026-08-03 — `pipelines::LiveMesh`, the indirect-draw half of the live zero-copy path.**
  A borrowed vertex/index/indirect buffer triple drawn with `vkCmdDrawIndexedIndirect`, so a mesh
  whose index count changes per frame draws with no CPU round trip. `HybridMeshDraw::geometry` is a
  `std::variant<const GpuMesh*, LiveMesh>` — one frame mixes static and live meshes, and `submit()`
  dispatches per draw via a visitor (a new alternative is a compile error, not a silently dropped
  draw). gfx owns only the *recording*: synchronization, buffer lifetime, and the command's contents
  are the producer's, spelled out in `docs/integration/recon-live-mesh.md` — the cross-repo byte
  contract, which `hybrid_mesh_pipeline.cpp` `static_assert`s the vertex half of. Still outstanding
  for the full live path: per-slot atlas ringing, then the `app::StreamedApp` driver.

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

- Implement every change in a dedicated **git worktree**, not by `git checkout`-ing a branch
  in the primary checkout. Create it locally under `.worktrees/` (e.g. `git worktree add
  .worktrees/<branch> -b <branch>`) — never as a sibling in the parent folder — and remove it
  with `git worktree remove` once the PR merges.
- Prefer plain, behavior-level tests over friend-class backdoors into private state.
- Mark deferred/future work inline with a `TODO:` comment (e.g. `// TODO: adopt volk for the
  iOS static-MoltenVK path`) so it is greppable, rather than tracking it only in prose or commits.
- Document public classes/functions with full Doxygen, matching `include/volumetric_kit/gfx/core/result.hpp`:
  `@file`/`@brief` on the header, `@brief` + a `@code … @endcode` example per class, and
  `@brief`/`@param`/`@return` (`@pre` where relevant) on methods; accessors may be a single `/// @return`.
  Do **not** write "move-only" (or similar) in prose — the deleted-copy/defaulted-move declarations
  already convey it.

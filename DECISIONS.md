# Renderer decisions

This record preserves the locked decisions previously committed in
`CLAUDE.md`, with their original dates where recorded. [AGENTS.md](AGENTS.md)
is the concise shared working guide. Detailed local planning in the gitignored
`DESIGN.md` remains local and is not published by this change.

Implementation-status sentences in older entries describe the stage when the
entry was written. Use [CHANGELOG.md](CHANGELOG.md), the current code, and the
[live-mesh integration contract](docs/integration/recon-live-mesh.md) to establish
what has landed since then. Record amendments when a contract changes.

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
- **2026-10-04 — Vulkan headers come from the system**, as the core's do; the core's
  `format.hpp` replaces Vulkan-Utility-Libraries. See the dated entry below.
- **2026-10-04 — Error handling comes from `volumetric_kit_core`.** `Status`, `Result`,
  `VG_CHECK` and the log sink are the family's shared core's base tier; see the dated entry
  below for the `VkResult` bridge and the stages still to come.
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
  vertex-color fallback); lit or unlit via a push-constant `flags` word. Reuses `assets::Vertex`
  + `GpuMesh` unchanged (binds position/normal/uv0/color, not tangent), one push constant carries the
  view-projection + light (no per-frame UBO), and the sampler set is the pipeline's only descriptor
  set. This is the *static* data-path (upload a mesh + atlas, draw); proven headless via an offscreen
  draw + pixel readback under validation. A vertex-color/atlas mesh pipeline is a broadly-useful
  renderer feature, so the siblings stay independent — gfx gains a capability, not a dependency on
  recon.
- **2026-08-03 — `pipelines::LiveMesh`, the indirect-draw half of the live zero-copy path.**
  A borrowed vertex/index/indirect buffer triple drawn with `vkCmdDrawIndexedIndirect`, so a mesh
  whose index count changes per frame draws with no CPU round trip. `HybridMeshDraw::geometry` is a
  `std::variant<const GpuMesh*, LiveMesh>` — one frame mixes static and live meshes, and `submit()`
  dispatches per draw via a visitor (a new alternative is a compile error, not a silently dropped
  draw). gfx owns only the *recording*: synchronization, buffer lifetime, and the command's contents
  are the producer's, spelled out in `docs/integration/recon-live-mesh.md` — the cross-repo byte
  contract, which `hybrid_mesh_pipeline.cpp` `static_assert`s the vertex half of. Still outstanding
  for the full live path: per-slot atlas ringing, then the `app::StreamedApp` driver.

## 2026-10-04 — Vulkan headers come from the system

gfx compiles against the Vulkan headers `find_package(Vulkan)` finds, as the
family's core does, settling the question its error-handling move left open
(the core's DECISIONS.md, "Vulkan headers come from the system"). It no longer
vendors Vulkan-Headers or Vulkan-Utility-Libraries.

- **Only the format metadata needed them.** gfx used five `vkuFormat*`
  helpers, and nothing newer than Vulkan 1.3 core and long-standing
  extensions otherwise. Those helpers become the core's `format.hpp`:
  `aspect_mask_for` → `core::view_aspect`, `texel_size` →
  `core::texel_bytes`, and `format_has_depth` / `format_has_stencil` → the
  core's. A vendor or EXT extension's format now reads as nothing, so an
  offscreen readback or texture upload of one is refused rather than sized.
- **The oldest supported headers are the core's: 1.3.204, and 1.3.208 on
  Apple.** `core/vulkan.hpp` forwards to the core's umbrella, whose check
  refuses older headers in every gfx translation unit; the Ubuntu 22.04 leg
  builds on its system's 1.3.204.
- **gfx turns the core's vulkan tier on**, and refuses to configure where a
  parent made the core available without it, as recon does. `gfx_core` and
  `gfx_windowing` link it PRIVATE while no public gfx header names a core
  vulkan type; that changes when gfx's device moves onto the core's.
- **An application decides the headers**, not gfx: one that wants a pin
  points `Vulkan_INCLUDE_DIR` at it, for gfx, the core and every other
  library alike.

## 2026-10-04 — Error handling comes from volumetric_kit_core

gfx's `Status`, `Result`, `VG_CHECK` and log sink are volumetric_kit_core's
base tier, fetched pinned by commit (`third_party/CMakeLists.txt`), linked
PUBLIC by `gfx_core`, and re-found by the installed package at the minor it
was built against. This is the first stage of gfx's move onto the family's
shared core (the core's DECISIONS.md, "Tiers"), following recon's.

- **One error type across the family.** `vg::Status` and `vg::Result` are
  using-declarations of the core's, so a status passes between gfx, recon and
  calib unchanged, and one `set_log_handler` routes all three; gfx's messages
  carry source `"vg"`, so the default sink still prints `[vg <level>]`.
  Contract failures (`VG_CHECK`, reading an error `Result`) are the core's,
  with source `"core"`.
- **The `VkResult` bridge stays in gfx for now.** The core's base tier
  includes no GPU API: a failed Vulkan call is `Status::Code::Backend` with
  the `VkResult` as its `detail()`. gfx keeps `vk_error`, `VG_VK_TRY`,
  `to_string(VkResult)` and adds `vk_result(status)`, with the core's vulkan
  tier's names, so adopting that tier replaces them with using-declarations.
  `Status::Code::Vulkan`, `Status::error` and `Status::code()` are gone.
- **The bridge compiles beside the core's vulkan tier**, which an application
  using the core's compute tier includes too. gfx therefore names no
  `to_string` of the core's (that would take the tier's `to_string(VkResult)`,
  clashing with gfx's; argument-dependent lookup finds `to_string(Status::Code)`),
  and gfx's own calls qualify `vk_result`, which the tier also defines.
  `tests/core_vulkan_tier_test.cpp` includes the tier's header first.
- **`vk_result` cannot tell Vulkan from CUDA.** recon's CUDA failures share
  `Code::Backend`, so it reads a `cudaError_t` as an unrelated `VkResult`; gfx
  asks it only of its own statuses. It does return empty for a detail wider
  than 32 bits, which would be undefined to convert. Recording the backend in
  `Status` is the core's to decide, and its own `vk_result` has both gaps.
- **`VG_TRY` / `VG_ASSIGN` / `VG_CHECK` remain**, as object-like aliases of
  the core's `VKC_*` macros, so open branches merge cleanly and a check
  reports its condition unexpanded; a `TODO:` marks the rename, as recon has
  for its `VR_*` names.
- **The core is resolved at the top level**, like gfx's other PUBLIC
  dependencies, so a core FetchContent finds installed is visible to `src/`.
  `VG_INSTALL` turns the core's install rules on, never off: a sibling that
  installs (recon) needs them whichever project populates the core first.
- **gfx's own build fetched only the base tier**, so the core's system Vulkan
  headers never met gfx's pinned Vulkan-Headers. Superseded: gfx now builds on
  the system's headers and the core's vulkan tier ("Vulkan headers come from
  the system", above).

Still open, for the next stages: moving `gfx_core`'s instance, device, allocator, buffers, images, descriptors,
command pools, sync and query pool onto the core's vulkan tier, with gfx's
graphics-only parts (swapchain, render targets, graphics pipelines, samplers,
the frames-in-flight profiler) staying here; `Texture` becoming the core's
`Image`; and re-measuring frame times, as the core's allocator places memory
differently (`DeviceOnly` never falls back to host memory, and per-frame
uniforms become `DeviceMapped` or batch-uploaded `DeviceOnly`).

## 2026-10-02 — Shared agent guidance

`AGENTS.md` now owns the shared working rules and task-based reading map.
`CLAUDE.md` imports it with `@AGENTS.md`. This decision record keeps the
previously committed constraints available to both agents without copying the
local-only design document. Update shared rules here and in AGENTS.md as
appropriate when a decision changes; keep tool-specific configuration separate.

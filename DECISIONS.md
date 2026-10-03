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

## 2026-10-02 — Shared agent guidance

`AGENTS.md` now owns the shared working rules and task-based reading map.
`CLAUDE.md` imports it with `@AGENTS.md`. This decision record keeps the
previously committed constraints available to both agents without copying the
local-only design document. Update shared rules here and in AGENTS.md as
appropriate when a decision changes; keep tool-specific configuration separate.

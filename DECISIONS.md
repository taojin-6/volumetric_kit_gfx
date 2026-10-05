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
- **A `VkDevice` may be created *or adopted*.** The device — volumetric_kit_core's since
  2026-10-04 (below) — accepts one the embedder already created: non-owning
  `Device::adopt(AdoptedDevice, DeviceRequirements)` alongside `Device::create`, verifying the
  renderer's requirements (`device_requirements()`) against the creator's declared enabled state
  (Vulkan can't be queried for a *logical* device's enabled features/extensions). This lets a sibling **Vulkan-compute** library (e.g. `volumetric_kit_recon`)
  share one `VkDevice` with the renderer and hand over `VkBuffer`/`VkImage` **zero-copy** — the
  same-API case that needs none of the CUDA/Metal external-memory machinery above. The embedding app
  owns the shared instance/device and merges both libraries' requirements; gfx stays standalone
  (`create` is unchanged). The indirect-draw path a *live* mesh needs has since landed
  (`pipelines::LiveMesh`, below); per-slot material/atlas ringing for a live-updated texture is
  what remains.
- **2026-10-04 — gfx names the core's types as the core does** (`core::Buffer`, `vkc::Buffer`
  outside gfx), with no aliases of its own: one name per type across the family. See
  "Memory comes from volumetric_kit_core", below.
- **2026-10-04 — Memory comes from volumetric_kit_core.** `Allocator`, `Buffer` and
  `Image` (formerly gfx's `Texture`) are the core's, so every buffer names its placement --
  `DeviceOnly`, `DeviceMapped` or `Staging` -- and nothing spills into slower memory. The
  per-frame camera uniforms are device-mapped, or device-only written by a recorded update;
  material factors are uploaded. See the dated entry below.
- **2026-10-04 — The device comes from volumetric_kit_core.** `Instance`,
  `PhysicalDeviceInfo`, `Device`, `AdoptedDevice` and `DeviceRequirements` are the core's;
  gfx brings `device_requirements()`. See the dated entry below.
- **2026-10-04 — Vulkan headers come from the system**, as the core's do; gfx builds on the
  core's vulkan tier, linked PUBLIC, whose `format.hpp` replaces Vulkan-Utility-Libraries and
  whose `VkResult` bridge and shader build functions replace gfx's copies. See the dated entry
  below.
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
  change at gfx's call sites — not a one-way door. VMA — the core's since 2026-10-04 —
  uses the linked Vulkan prototypes (`VMA_STATIC_VULKAN_FUNCTIONS`). *Amended 2026-10-04:* gfx's umbrella now forwards to
  volumetric_kit_core's (`volumetric_kit/core/vulkan/vulkan.hpp`), which makes the loader choice
  for the whole family, so adopting volk is a change to the core's header plus the link lines
  (the core's `TODO` marks it); see "Vulkan headers come from the system", below.
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

## 2026-10-04 — Memory comes from volumetric_kit_core

gfx's allocator, buffers and images are the family's core's (stage 2b), named
as the core's: gfx's API takes and returns `core::Allocator`, `core::Buffer`
and `core::Image`. gfx's own VMA, its `Allocator` and `Texture`, and
`ExternalHandleType` -- a field whose every value but `None` was refused --
are gone; exported memory is the core's `create_exported_buffer`. A buffer
gfx makes is the type recon binds, and the reverse.

- **No gfx names for the core's types.** gfx first named the core's types
  through using-declarations in `vg::`, as recon does in `vr::`. That kept
  call sites unchanged but gave each type a second name per library, an alias
  header and docs to keep in step, and left a reader to discover that
  `vg::Buffer` was `vkc::Buffer`. The point of the shared core is one type
  for the whole family, so gfx spells the core's types as the core does:
  `core::Allocator` inside gfx's namespaces, `vkc::Allocator` in tests and
  examples (the family's alias for `volumetric_kit::core`), including the
  core's headers directly. The names aliased before this entry -- `Status`,
  `Result`, the `VG_*` macros, `Instance`, `Device` and the rest of the device
  entry below -- move the same way in a follow-up.

- **Every buffer names its placement, as the core's DECISIONS.md, "Where
  memory lives", sets out.** Vertex, index and uploaded buffers, textures and
  render targets are `DeviceOnly`: device-local memory the host cannot map,
  never host memory, where gfx's `DeviceLocal` only preferred device memory
  and its `Auto` let VMA put a buffer in host memory once VRAM filled.
  `UploadBatch` stages through `Staging` memory, and `OffscreenTarget` reads
  back through `Staging` memory the host reads cached (`HostAccess::Random`).
  `BufferDesc::mapped` and `ImageDesc::memory` are gone: the placement says
  whether a buffer is mapped, and every image is device-only.
- **The per-frame camera uniforms are `DeviceMapped`, or device-only written
  by a recorded update.** The host writes them each frame and shaders read
  them in place, the core's use for device-mapped memory. That placement never
  falls back to host memory, and a device may lack it or have a full BAR
  window, so `make_frame_uniform_buffer` takes device-only memory then, and
  `OwnedDescriptorSet::write_uniform` records a `vkCmdUpdateBuffer` and a
  barrier to the shaders' uniform reads. The core's DECISIONS.md names a
  `CommandBatch` for that fallback; a recorded update in the frame's own
  command buffer is the same placement without a second submit each frame.
  Because the update is a command, `PbrScene::set_camera` takes the frame's
  command buffer and is called before rendering begins, on every device, so a
  caller written on a device with device-mapped memory is right on one
  without. `PbrSceneDesc::camera_memory` forces the fallback -- to keep the
  uniforms out of a BAR window other libraries share, and so tests cover that
  path on any device.
- **Material factors are uploaded, not mapped.** They never change, so
  `PbrMaterial::create` puts them in device-only memory through the
  `UploadBatch` that uploads the model's meshes and maps, rather than taking
  device-mapped memory that per-frame data needs more. It poisons the batch if
  it fails after queuing the copy.
- **Images record their layout.** The core's `Image` records the layout its
  contents are in, for a library handed one. An `UploadBatch` image records
  `SHADER_READ_ONLY_OPTIMAL`, the layout `finish` leaves it in; render targets
  transition their attachments every frame and keep their images to
  themselves, so they record none.
- **Resources outlive their allocator.** Each holds the core's VMA state, so
  only the device must outlive gfx's buffers and images. `UploadBatch::finish`
  now hands its staging buffers to `Device::submit_single_time`, which keeps
  them past a failed wait until the device is destroyed, instead of waiting
  for the queue to drain and leaking them if it never did.
- **Frame times re-measured, as the core's DECISIONS.md asks, and unchanged.**
  A headless benchmark of the 03_model scene -- a grid of cubes, each its own
  mesh and material -- timed the move's base and head alternately, on unified
  memory (Apple M5 Max and M4) and on a discrete GPU (RTX 4090). GPU and wall
  frame times stayed within run-to-run spread; the PR that made the move
  records the numbers. The benchmark was not kept: a one-off before/after
  check did not justify maintaining a benchmark executable, an A/B script and
  CI steps.

Still open: descriptors, sync and the query pool (stage 2c).

## 2026-10-04 — The device comes from volumetric_kit_core

gfx's instance, physical-device capabilities and logical device are the family's
core's (stage 2a's second half): `Instance`, `InstanceConfig`,
`PhysicalDeviceInfo`, `Device`, `AdoptedDevice`, `EnabledFeatures`,
`DeviceRequirements`, `merge` and `check_device_support` are using-declarations
in `vg::`, as recon's are in `vr::`. A device gfx makes is the type recon adopts,
and the reverse.

- **gfx brings its requirements, not a config.** `device_requirements()`
  returns the renderer's floor -- Vulkan 1.3, a graphics queue,
  `dynamicRendering`, `timelineSemaphore` -- as the core's `DeviceRequirements`,
  which selection, creation and adoption all take; a caller sets
  `needs_present` and adds features or extensions on it. `DeviceConfig`, gfx's
  own `DeviceRequirements` and `Device::requirements` are gone. The core
  enables `VK_KHR_swapchain` for `needs_present`, and portability subset where
  the device exposes it.
- **The floor holds wherever gfx meets a device.** The app tier merges
  `device_requirements()` into `config.device`, so a config built from bare
  requirements can only add to the floor. A device gfx did not make -- one
  built for recon's requirements, say, on a compute-only queue -- is held to
  the floor by `Device::check_enabled` in `UploadBatch::begin` (and so every
  upload helper), `Swapchain::create`, `FrameLoop::create`, `Profiler::create`
  and `ImGuiOverlay::create`, rather than having graphics barriers recorded on
  it.
- **Debug labels follow the instance.** The core's instance requests
  `VK_EXT_debug_utils` by default, and its device resolves labels when the
  instance enabled it, so the app tier no longer threads a flag through.
  gfx keeps its `DebugUtilsTable`, as the core's device records neither queue
  labels nor label colors; `debug_utils(device)` loads it, gated on
  `Device::debug_labels_available`. A `TODO:` marks taking both from the core
  if another library needs them.
- **No shared command pool.** The core's device keeps its pools to itself, so
  `UploadBatch` queues its copies and records them at `finish` through
  `Device::submit_single_time`, on a pool no other submit holds -- which also
  lifts its old caveat about sharing the device's pool across threads. After a
  failed wait the device keeps the command buffer; the staging buffers, whose
  allocator is gone by the time the device is, are freed once the queue
  drains, or leaked (and logged) if it never does.
- **Borrowers ask the device for its submit mutex.** It is never null -- the
  embedder's on a shared queue, else the device's own, which does not move with
  it -- so `ImGuiOverlay` borrows the device by address, as `Swapchain` and
  `FrameLoop` do, and locks its mutex on each render.
- **The allocator reads the device's usable version** (`caps().api_version()`,
  the lower of the device's and its instance's) instead of reconstructing the
  instance's from the loader, which an adopted device's could not be.

Still open: the allocator, buffers, images (`Texture` becoming the core's
`Image`), descriptors, sync and the query pool (stages 2b and 2c), and
re-measuring frame times as the core's allocator changes placement. The memory
half has since landed ("Memory comes from volumetric_kit_core", above).

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
- **gfx turns the core's vulkan tier on and links it PUBLIC.** gfx's public
  `core/vulkan.hpp` and `core/result.hpp` include the tier's headers, so a
  consumer needs the tier whichever gfx type it names; `gfx_core` links it
  PUBLIC and the installed package refuses a core built without it.
- **gfx checks the core it got, not the one it asked for.** Its pin and
  `VKC_WITH_VULKAN` yield to a project that made the core available first, and
  FetchContent may find an installed core; the core's version does not advance
  between commits. So `vg_require_core_vulkan` (`cmake/vg_core.cmake`) refuses
  to configure unless the tier is there and has `format.hpp`, the newest header
  gfx needs, naming the pin and where the core came from; the package config
  checks the same.
- **The `VkResult` bridge and the shader build functions are the core's.**
  `vk_error`, `vk_result` and `to_string` are using-declarations of the tier's
  and `VG_VK_TRY` aliases `VKC_VK_TRY`, so an unqualified call finds one
  function however the two namespaces are brought into scope (the clash the
  error-handling entry below worked around is gone). Shaders compile and embed
  through `vkc_compile_shaders` / `vkc_embed_shaders` with `TARGET_ENV
  vulkan1.3`, gfx's device floor, and `SYMBOL_PREFIX vg_`; gfx's
  `vg_shaders.cmake`, `vg_embed.cmake` and `embed_spirv.cmake` are gone.
- **A default view refuses a format that needs a Y'CbCr conversion**
  (multi-planar, 4:2:2, RGBA 4PACK16; the core's
  `format_needs_ycbcr_conversion`), as the core's `Image` does: `create_image`
  with `with_view` and `upload_texture` return `Unsupported` instead of making
  a view the spec forbids.
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
- **The `VkResult` bridge stayed in gfx at first.** The core's base tier
  includes no GPU API: a failed Vulkan call is `Status::Code::Backend` with
  the `VkResult` as its `detail()`. gfx kept `vk_error`, `VG_VK_TRY`,
  `to_string(VkResult)` and added `vk_result(status)`, with the core's vulkan
  tier's names, so adopting that tier would replace them with
  using-declarations. `Status::Code::Vulkan`, `Status::error` and
  `Status::code()` are gone. Superseded: they are now the tier's ("Vulkan
  headers come from the system", above).
- **The bridge compiled beside the core's vulkan tier**, which an application
  using the core's compute tier includes too, by naming no `to_string` of the
  core's and qualifying gfx's `vk_result` calls. Superseded with the bridge;
  `tests/core_vulkan_tier_test.cpp` still includes the tier's header first.
- **`vk_result` cannot tell Vulkan from CUDA.** recon's CUDA failures share
  `Code::Backend`, so it reads a `cudaError_t` as an unrelated `VkResult`; gfx
  asks it only of its own statuses. It does return empty for a detail wider
  than 32 bits, which would be undefined to convert. Recording the backend in
  `Status` is the core's to decide, and its `vk_result`, now gfx's, has the
  gap.
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
uniforms become `DeviceMapped` or batch-uploaded `DeviceOnly`). The device and
memory have since moved (the dated entries above).

## 2026-10-02 — Shared agent guidance

`AGENTS.md` now owns the shared working rules and task-based reading map.
`CLAUDE.md` imports it with `@AGENTS.md`. This decision record keeps the
previously committed constraints available to both agents without copying the
local-only design document. Update shared rules here and in AGENTS.md as
appropriate when a decision changes; keep tool-specific configuration separate.

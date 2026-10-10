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
  *Amended 2026-10-05:* superseded for the family. Its compute is Vulkan compute on a `VkDevice`
  shared with the renderer (the device-adopt entry below; volumetric_kit_core's vulkan tier), with
  CUDA added per kernel where profiling shows a need and native Metal deferred. The interop layer
  remains for those; `pipelines::ImagePipeline` takes a Vulkan producer's pictures by copy.
- **A `VkDevice` may be created *or adopted*.** The device — volumetric_kit_core's since
  2026-10-04 (below) — accepts one the embedder already created: non-owning
  `Device::adopt(AdoptedDevice, DeviceRequirements)` alongside `Device::create`, verifying the
  renderer's requirements (`device_requirements()`) against the creator's declared enabled state
  (Vulkan can't be queried for a *logical* device's enabled features/extensions). This lets a sibling **Vulkan-compute** library (e.g. `volumetric_kit_recon`)
  share one `VkDevice` with the renderer and hand over `VkBuffer`/`VkImage` **zero-copy** — the
  same-API case that needs none of the CUDA/Metal external-memory machinery above. The embedding app
  owns the shared instance/device and merges both libraries' requirements; gfx stays standalone
  (`create` is unchanged). The indirect-draw path a *live* mesh needs has since landed
  (`pipelines::LiveMesh`, below), and so has per-slot atlas ringing for an atlas the frame copies
  into (`pipelines::StreamedAtlas`, 2026-10-08); a producer writing an atlas image from its own
  queue, zero-copy, is what remains.
- **2026-10-08 — The hybrid mesh's atlas is streamed, and the pipeline owns a fallback.**
  `pipelines::StreamedAtlas` rings its images on frame numbers and is updated by copies recorded
  into the frame (`record_image_update` / `record_image_upload`); a frame with no atlas draws in
  vertex color against the pipeline's own fallback set. See the dated entry below.
- **2026-10-08 — Frames are numbered on a timeline.** `windowing::FrameLoop` sets each
  frame's number on one timeline semaphore; `RetireQueue` frees on that timeline's values,
  and a frame may wait for and set other timeline values. See the dated entry below.
- **2026-10-05 — gfx writes every core name as the core does** -- `core::Status`,
  `core::Device`, `VKC_TRY` in gfx; `vkc::` in tests and examples -- and its re-export
  headers and `VG_*` macro aliases are gone (amends the device and error-handling entries).
  See the dated entry below.
- **2026-10-04 — Sync, descriptors and queries come from volumetric_kit_core.** gfx's
  fences, semaphores, descriptor objects, command pools and buffers, query pool and
  `UniqueHandle` are gone; gfx uses the core's. The profiler times frames on the core's
  `QueryPool`, and `FrameMetrics` holds the core's `StageRow`s. See the dated entry below.
- **2026-10-04 — gfx names the core's types as the core does** (`core::Buffer`, `vkc::Buffer`
  outside gfx), with no aliases of its own: one name per type across the family. See
  "Memory comes from volumetric_kit_core", below.
- **2026-10-04 — Memory comes from volumetric_kit_core.** `Allocator`, `Buffer` and
  `Image` (formerly gfx's `Texture`) are the core's, so every buffer names its placement --
  `DeviceOnly`, `DeviceMapped` or `Staging` -- and nothing spills into slower memory. The
  per-frame camera uniforms are device-only, written by a recorded update; material factors
  are uploaded, one buffer for a model's materials. See the dated entry below.
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
  A helper several shaders use (the sRGB curve, the tonemap, the full-screen triangle) is
  written once in `shaders/common/` and `#include`d, with `shaders/` as the include root.
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
  recon. *Amended 2026-10-08:* the pipeline owns a fallback atlas, so a frame needs none ("A
  streamed atlas", below).
- **2026-10-05 — 2D images are converted, then mip-mapped, then drawn.**
  `pipelines::ImagePipeline` copies a picture's planes into an `ImageTexture`, renders them to
  display color in level 0 of an `R8G8B8A8_SRGB` image (sRGB decode, NV12's matrix and siting, or
  a color ramp), rebuilds its mip chain with halving blits that average in linear light, and draws
  it trilinearly into rectangles of any target; `camera::ImageView2D` holds the pan-and-zoom
  mapping. See the dated entry below.
- **2026-08-03 — `pipelines::LiveMesh`, the indirect-draw half of the live zero-copy path.**
  A borrowed vertex/index/indirect buffer triple drawn with `vkCmdDrawIndexedIndirect`, so a mesh
  whose index count changes per frame draws with no CPU round trip. `HybridMeshDraw::geometry` is a
  `std::variant<const GpuMesh*, LiveMesh>` — one frame mixes static and live meshes, and `submit()`
  dispatches per draw via a visitor (a new alternative is a compile error, not a silently dropped
  draw). gfx owns only the *recording*: synchronization, buffer lifetime, and the command's contents
  are the producer's, spelled out in `docs/integration/recon-live-mesh.md` — the cross-repo byte
  contract, which `hybrid_mesh_pipeline.cpp` `static_assert`s the vertex half of. Still outstanding
  for the full live path: the `app::StreamedApp` driver (per-slot atlas ringing landed
  2026-10-08).

## 2026-10-08 — A streamed atlas, and the pipeline's own fallback

**The contract.** `HybridMeshPipeline::create` now takes the device and an
allocator and builds a fallback atlas -- one texel, its set and a sampler.
`submit` binds it for a frame whose `atlas` is null and sets
`kHybridMeshVertexColor`, so every triangle draws in its vertex color.
`pipelines::StreamedAtlas` is the atlas a live mesh samples: a ring of images,
each with a set written once. A frame binds `use(frame.number)`, the newest
picture's set, which marks that image used by the frame. An update -- tiles
from device buffers (`record_update`) or host pixels (`record_upload`) -- is
recorded into a frame's command buffer and copies into the least recently used
image whose last frame the timeline has reached; that image becomes the
picture. A frame that never reaches the queue is given back with
`discard(frame.number)`, which makes the picture its update replaced current
again: the atlas cannot tell a frame whose commands ran from one whose number
the frame loop's stand-in submit set. The copy and its transitions are the core
tier's `record_image_update`, and `record_image_upload` stages host pixels
through a buffer a `RetireQueue` frees at the frame's number: gfx's one
in-frame image update, beside the blocking `upload_texture` for load time.

**Why the pipeline owns the fallback.** The shader samples set 0
unconditionally, so the pipeline owns the one fallback rather than each
consumer. Vertex color, not the fallback's texel, is what a frame without an
atlas shows: every reconstruction vertex carries a fused color, while one texel
would paint each textured triangle a flat color that passes for a texture.

**Why a ring, when `ImageTexture` needs none.** Both updates wait on the GPU
for the earlier fragment reads on the queue. An atlas image is rewritten only
once the host has seen its last frame complete, but its transition waits for
the fragment stage all the same, so the queue itself orders those reads
before the copy; without that wait, validation layers that do not track host
waits on timeline semaphores (Ubuntu 24.04's 1.3.275) report a
write-after-read hazard. The ring keeps the picture an update replaces
intact, so `discard` can restore it even if the frame ran before its present
failed. That picture is excluded from updates until the next frame: the ring
requires at least two images. A frame that starts with a picture can update
at most `slots - 1` images; without a prior picture all slots are available.
The ring is also the structure a producer writing an atlas image from its own
queue will need, where no barrier reaches. When every writable image is still
used by an earlier frame, the update waits for the oldest on the host, as the
frame loop waits for a slot; with frames in flight
plus one images and one update a frame it never does. It waits only for a
frame the core's record shows submitted, so a frame that failed before its
submit is refused rather than waited for forever. Destroying an atlas waits
for its newest frame, or for the renderer's queues when that frame never
reached them, as a `RetireQueue` drains.

**Not yet.** Mips for a minified atlas; a producer writing an image from its
own queue; the `app::StreamedApp` driver.

## 2026-10-10 — An aggregate is neither copied nor moved

A type that owns several Vulkan objects and borrows others, such as
`windowing::FrameLoop`, is a thing with an identity, not a value. Nothing
needs to move one: `FrameLoop` moved only from `create` to its owner. Its
hand-written move pair still cost 57 lines that moved and then reset each of
its 13 members by hand, and every new member had to join both lists. So an
aggregate deletes copy and move and is handed out by `std::unique_ptr`, while
a handle wrapper, which owns one Vulkan object, stays a move-only value
([AGENTS.md](AGENTS.md#raii-resource-types) states the rule).

- **Not defaulted moves.** A defaulted move assignment frees the old objects
  without first draining the work that uses them, and a moved-from aggregate
  is an empty one again, with the guards that come with it.
- **recon is narrower.** It deletes both only for internal aggregates; its
  public `VoxelBlockGrid` keeps a defaulted move constructor. gfx deletes both
  for public aggregates too.
- `FrameLoop` is the first. The other types with hand-written move pairs are
  sorted into the two kinds and converted one at a time; a `TODO:` marks each
  aggregate found so far.

## 2026-10-08 — Frames are numbered on a timeline

`windowing::FrameLoop` numbers its frames on one `core::TimelineSemaphore`
(`timeline()`): `begin_frame` hands out `Frame::number`, one more than the last
frame submitted, and `end_frame`'s submit sets that value once the frame's
work completes. Every frame up to `completed()` has finished, so "free this
once the last frame that used it is done" is one comparison, made in one
place: a `RetireQueue` on the timeline.

- **The loop reuses its slots on their fences.** Through MoltenVK a timeline
  value is reached before the submission's completion handler has run, and
  timestamp queries read as unavailable until it has, so a slot's command
  buffer, semaphore and queries are reused once its fence signals. Waits on
  what consumers and the acquired image need -- that the GPU is done -- are by
  number; the per-image fence handles are gone. The acquire and the present
  keep their binary semaphores, as presentation requires.
- **A failed frame still sets its number.** A frame that fails between its
  acquire and its submit -- one whose values `end_frame` refuses among them --
  is replaced by an empty submit that consumes the acquire and sets the number
  and the slot's fence, and the frame's signals unless they were the ones
  refused. The numbers stay contiguous, the slot's semaphore is free for its
  next use without a blocking wait, and nothing waiting for those values
  hangs. The frame's image was never presented, so the next extent-taking
  `begin_frame` rebuilds the swapchain, which releases it.
- **`RetireQueue` is keyed on timeline values.** It borrows a
  `TimelineSemaphore` -- the loop's, or another producer's -- and runs a
  deleter once the value pushed with it is reached. The `VkFence` key is gone:
  the loop's fences are private, and a slot's fence, reused every `N` frames,
  still reads as signalled while a newer frame on the slot is being recorded.
- **A frame carries timeline waits and signals.** `Frame::waits` (a
  `TimelinePoint` and the stages that wait) and `Frame::signals` join
  `end_frame`'s one submit through `VkTimelineSemaphoreSubmitInfo`, so another
  queue's or library's work can feed a frame, and wait for one, on the GPU.
  `end_frame` checks them with the core's `check_timeline_points`, taking a
  wait only for a value already reached or submitted to be set
  (`TimelineWaits::Submitted`), and adds what each frame sets to the core's
  record of submitted values; `FrameLoop::end_frame` says why. A producer whose
  value is not yet submitted is gated on the host
  ([the live-mesh contract](docs/integration/recon-live-mesh.md)).
- **The core pin is core #18** (`5913731`), which makes those checks public.

## 2026-10-05 — GPU tests share a device per process

**The contract.** *Amended 2026-10-08:* the fixtures and the require-device
variable are volumetric_kit_core's. gfx's GPU tests derive from the core's test
fixtures (`volumetric_kit::core_test_support`) through
`tests/gfx_test_support.hpp`, which adds the renderer's requirements and the
headless-surface helpers. The fixtures share one instance and device among the
tests of a process rather than make them per test, as the core's DECISIONS.md
records ("One Vulkan test fixture for the family").

**Why.** Measured in CI (the draft PR #111, closed after measuring), on
NVIDIA's Linux driver 615.71.09:
- **A cost the driver serializes machine-wide.** An instance + device
  create/destroy cycle costs ~70-90 ms, nearly all of it creating and
  destroying the device and instance. The driver does those one at a time
  across the machine: ~7-9 cycles per second on an RTX 4090 or 5090 box, with
  1, 4 or 8 processes at once. With a device per test, every Linux leg's test
  time was bound by it, and running tests in parallel (#109) could not help.
  MoltenVK does the same cycle in about 1 ms, in parallel.
- **A limit per process.** After 26 instances in one process,
  `vkCreateInstance` fails with `VK_ERROR_INCOMPATIBLE_DRIVER`. The loader
  loads the driver library at each instance and unloads it at each
  `vkDestroyInstance`; each load takes glibc static TLS for `libnvidia-tls`
  that the unload evidently does not return, until the loader logs "cannot
  allocate memory in static TLS block" and finds no driver. Descriptors,
  threads and memory stay flat, with or without the validation layer; one
  instance held open throughout keeps the library loaded, and the limit never
  comes. The fixtures' shared instances do that, and the test binary's global
  environment holds one more whenever a process runs several tests, for the
  tests that make their own. An application that recreates its `VkInstance`
  many times in one process would hit this limit too.

**In CI, test binaries run as shards.** `_build.yml` configures with
`-DVG_TEST_SHARDS=4`, so CTest runs each test binary as four entries, each
running its quarter of the tests in one process (`GTEST_TOTAL_SHARDS` /
`GTEST_SHARD_INDEX`) and in shuffled order (`GTEST_SHUFFLE`), sharing the
fixture's devices. A shard passes or fails on its exit code alone: given a skip
expression, one skipped test would mark the whole shard skipped, failures
included -- the failure that hid the Ubuntu 22.04 leg's results from August to
#113. Instead CI sets `VKC_REQUIRE_VULKAN_DEVICE`, under which a fixture test
that cannot get an instance or device fails rather than skips, so no shard
passes with its tests skipped. Locally the default stays one CTest entry per
test.

## 2026-10-05 — 2D images: convert, then mip, then draw

**The technique.** `pipelines::ImagePipeline` draws 2D pictures -- camera
frames, depth and error maps -- into rectangles of any render target, beside
the mesh pipelines; it is the image-grid pipeline the roadmap planned. Per
picture an `ImageTexture` holds copies of the source's planes and an
`R8G8B8A8_SRGB` display image with its full mip chain. `record_update` copies
the planes in, renders them to display color in level 0 (the sRGB decode,
NV12's Y'CbCr matrix and chroma siting, or a color ramp), and rebuilds the
chain with halving linear blits (`cmd_generate_mips`, which `UploadBatch` now
shares); `submit` samples it trilinearly, magnifying nearest or linear per
draw. `camera::ImageView2D` holds the pan-and-zoom mapping that draws, overlays
and picking share, in framebuffer pixels.

**Why convert before the chain.** A shrunken image must show the average of
what the viewer would see at full size. Every mapping is nonlinear -- sRGB,
the Y'CbCr matrix and its clamp, the ramp -- so averaging source values and
converting the average would not give that. The display image is `_SRGB`, so
the blits, which sample and write as shaders do, decode, average in linear
light and re-encode: a one-texel black-and-white checkerboard shrinks to 188,
not the 128 of averaging encoded bytes (`MipsAverageInLinearLight`). The
level is chosen per pixel by the hardware from the drawn size, so window
resizes, zoom and display scaling need nothing from the host, and the chain is
rebuilt on every update, so a live stream's mips are never stale.

**Why copy the planes in.** A producer's picture then need only be copyable: a
decoder's imported surface may carry `TRANSFER_SRC` alone, and a buffer
(NVDEC's pictures, recon's frame prep) cannot be sampled as an image at all.
The copies give the texture descriptor sets made once, so a stream allocates
nothing per frame; and since a texture's updates and draws run on one queue,
each update's barriers order it after the earlier draws still reading it --
one texture per stream, with no ring per frame in flight. (The mesh atlas
rings: "A streamed atlas", 2026-10-08.)

**What it costs.** One device-to-device copy per update, small beside the
decode. Memory: about 4/3 x 4 bytes per texel for the display image (44 MB for
4K) plus the planes (12 MB for 4K NV12). An odd dimension's blit drops its
last row or column's share at each level, as floor halving does everywhere in
Vulkan. A producer's plane must be readable from the renderer's queue family
(`CONCURRENT`, or written on it), and its writes must come before the copy:
recorded earlier on the renderer's queue, or fenced or semaphore-waited from
another -- `ImagePlane` states it. Since the update cannot know which stage
wrote a plane, its copies wait for every earlier write on the queue, and every
later command waits for its copies, so a producer can refill a plane with no
barrier of its own; letting a plane name its writer's stage would narrow the
first wait. Each update waits only for *fragment-shader* reads of the display
image before overwriting it, so a reader in another stage synchronizes
itself (`ImageTexture::display` states it). Synchronization validation checks
the pipeline's barriers in its tests, catching a missing read-after-write or
write-after-write dependency, but it does not flag a missing write-after-read
wait before a transition from `UNDEFINED`, so those stages rest on review.

**Not yet.** A ui-tier helper that shows a texture in an ImGui panel
(`ImGui_ImplVulkan_AddTexture` over `display()`); overlays beyond ImGui's draw
lists; I420 and other planar layouts; blending, as images draw opaque.

## 2026-10-05 — gfx writes the core's types and macros under the core's names

Amends "The device comes from volumetric_kit_core" and "Error handling comes
from volumetric_kit_core", below, as recon's 2026-10-04 entry amended its own.

**The rule.** gfx names volumetric_kit_core's types, functions and macros as
the core does: `core::Status`, `core::Result`, `core::Device`,
`core::vk_error` and `VKC_TRY`, `VKC_ASSIGN`, `VKC_CHECK`, `VKC_VK_TRY` in
gfx's namespaces, and `volumetric_kit::core::` (a `vkc` alias) in tests,
examples and consumers -- the rule "Memory comes from volumetric_kit_core"
set for the allocator, now for every name. The headers that only re-exported
the core's names into `vg::` are gone -- `core/result.hpp`, `check.hpp`,
`instance.hpp`, `physical_device_info.hpp` and the `vulkan.hpp` forwarder --
and so are the four macro aliases. What stays under `core/` is gfx's own:
`device_requirements.hpp` (was `device.hpp`; only `device_requirements()`),
`log.hpp` (`log_message`, source `"vg"`), `debug_label.hpp` (the label
scopes), and the renderer's types. gfx declares no `core` namespace of its
own, so `core::Status` in gfx's namespaces finds `volumetric_kit::core`.

**Why.** The aliases kept gfx's call sites unchanged while it moved onto the
core, and its open branches merging cleanly. With stages 2a to 2c landed
nothing waits on them, and two spellings of one type cost every reader: an
application using recon and gfx met one device as `vkc::Device` and
`vg::Device`, and each re-export header was one more file to keep in step
with the core's. recon dropped its re-exports the same way (its #167).

**What it costs.** Consumers rename on their next pin bump: the four macros,
`vg::X` to `vkc::X` for each core name, and the include map in the
CHANGELOG. gfx's own names (`vg::device_requirements`, `vg::log_message`)
are unchanged. The tests of the core's types under gfx's
names (`result_test`, `core_vulkan_tier_test`) are gone with the names; the
core tests its own, and `swapchain_stale`, the one gfx function the tier test
covered, moved to `windowing_test`.

## 2026-10-04 — Sync, descriptors and queries come from volumetric_kit_core

The last of gfx's copies of the core's vulkan tier are gone (stage 2c): its
`Fence`, `Semaphore`, `TimelineSemaphore`, `DescriptorSetLayout`,
`DescriptorPool`, `DescriptorSet`, `CommandPool`, `CommandBuffer`,
`QueryPool` and `UniqueHandle`, with their headers and tests. gfx uses the
core's, named as the core names them ("Memory comes from volumetric_kit_core",
below), and the core's tests cover them. What stays in gfx is graphics-only:
the swapchain and frame loop, render targets, graphics pipelines, samplers,
uploads and the frames-in-flight profiler.

- **The core's types differ from gfx's in ways callers see.**
  - Each is default-constructible. gfx's `QueryPool` was not, so the
    profiler held it in an `std::optional`; it now holds the pool directly.
    The frame loop's `std::optional<CommandPool>` goes too, though gfx's
    `CommandPool` was default-constructible and never needed it.
  - `TimelineSemaphore::create` takes the `Device`, so it can refuse one
    that did not enable timeline semaphores.
  - A `DescriptorSet` reads as empty once its pool is destroyed, and its
    copies share a write count; gfx keeps every set beside its pool, so none
    observes a dead pool. Wrapping a raw set
    (`DescriptorSet(VkDevice, VkDescriptorSet)`) allocates that shared state,
    so the constructor is no longer `noexcept`.
  - `DescriptorSet::write_*` aborts (`VKC_CHECK`) on a null `VkBuffer` or
    `VkImageView`, where gfx's passed it to Vulkan. A null descriptor is
    valid only under `VK_EXT_robustness2`'s `nullDescriptor`, which the core
    does not enable, so clearing a binding that way no longer goes through
    the core's set; a caller that enables the feature itself writes the null
    descriptor with `vkUpdateDescriptorSets`.
  - `QueryPool::create` takes only `VK_QUERY_TYPE_TIMESTAMP` and
    `VK_QUERY_TYPE_OCCLUSION`, and refuses every other type
    (`Status::Code::Unsupported`); gfx's accepted any `VkQueryType`. A
    pipeline-statistics pool is a raw `vkCreateQueryPool` until the core
    takes the type.
- **The profiler stays gfx's, on the core's `QueryPool`.** The core's
  DECISIONS.md rebuilds the profiler on its `QueryPool` and `GpuTimer`. The
  frames-in-flight shape -- a slot's timestamps read when the slot recurs,
  after its fence -- is the profiler's own, and the core's `QueryPool`
  already gives it range-checked commands and reads, so it times on that pool
  with the core's `timestamp_delta` / `ticks_to_ms`. The `GpuTimer`, built for
  spans read the moment a blocking submit returns, would add a window per
  slot without changing what the profiler measures; a `TODO:` in
  `profiler.cpp` marks moving onto it if the core gives it such a window.
- **`FrameMetrics` holds the core's `StageRow`s.** `FrameMetrics::Section`
  had the same four fields, so `sections` is a `std::vector<core::StageRow>`
  and recon's stage timings and gfx's frame share one row type, as the core's
  DECISIONS.md asks. The frame totals (`cpu_frame_ms`, `fps`, memory) stay
  gfx's. A `StageRow`'s name is never null, and the core's `StageMetrics`
  aborts on a null one, so the profiler records a scope opened with a null
  name as `"(unnamed)"` and labels it so, rather than publish a row a
  consumer cannot fold into its stage table.
- **`ShaderModule` stays gfx's, over the core's.** The core's module declares
  no interface; gfx's holds one and adds the spirv-cross reflection its
  pipelines build their layouts from.

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
  entry below -- have since moved the same way ("gfx writes the core's types
  and macros under the core's names", above).

- **Every buffer names its placement, as the core's DECISIONS.md, "Where
  memory lives", sets out.** Vertex, index and uploaded buffers, textures and
  render targets are `DeviceOnly`: device-local memory the host cannot map,
  never host memory, where gfx's `DeviceLocal` only preferred device memory
  and its `Auto` let VMA put a buffer in host memory once VRAM filled.
  `UploadBatch` stages through `Staging` memory, and `OffscreenTarget` reads
  back through `Staging` memory the host reads cached (`HostAccess::Random`).
  `BufferDesc::mapped` and `ImageDesc::memory` are gone: the placement says
  whether a buffer is mapped, and every image is device-only.
- **The per-frame camera uniforms are device-only, written by a recorded
  update.** `make_frame_uniform_buffer` makes them `DeviceOnly`, and
  `OwnedDescriptorSet::write_uniform` records into the frame's command buffer
  a barrier after earlier reads and writes of the block, a `vkCmdUpdateBuffer`,
  and a barrier to the shaders' uniform reads. The core's DECISIONS.md names
  `DeviceMapped` memory for data the host rewrites each frame, falling back to
  device-only memory written by a `CommandBatch`; a recorded update in the
  frame's own command buffer is that fallback without a second submit. A
  first cut wrote device-mapped memory where the device had room and fell
  back to the update where it did not, but a device may lack that memory, so
  `PbrScene::set_camera` had to take the command buffer and be called before
  rendering on every device anyway. The mapped path then bought nothing but
  a second behavior: a memcpy applies to the whole submitted frame, a
  recorded update to the work after it, so a frame rendering two views
  through one slot saw the last camera on one device and each view's own on
  another. One path behaves the same everywhere -- each `set_camera` reaches
  the work recorded after it -- leaves the BAR window to libraries that need
  it, and has no fallback whose failures need sorting. Its cost is one small
  update and two barriers a frame.
- **Material factors are uploaded, not mapped, one buffer for many.** They
  never change, so `PbrMaterial::create_all` puts every material's factors in
  one device-only buffer, each at a 256-byte offset (the largest
  `minUniformBufferOffsetAlignment` Vulkan allows, so no device query), and
  uploads it on the `UploadBatch` that uploads the model's meshes and maps:
  one staging buffer and one copy, not one each per material.
  `PbrModel::create` builds its materials that way. The materials share the
  buffer, so it lives until the last of them is destroyed. The descriptor
  pools and sets are allocated before the copy is queued, so a failed create
  leaves the batch unchanged rather than poisoning it.
- **Images record their layout.** The core's `Image` records the layout its
  contents are in, for a library handed one, and asks whoever submits a
  transition to record it after. An `UploadBatch` image so records
  `UNDEFINED` until `finish` returns OK -- a batch discarded, poisoned or
  failed never transitions it -- and its owner records
  `SHADER_READ_ONLY_OPTIMAL` then, as `upload_texture`, `bake_ibl` and
  `PbrModel` do. Render targets transition their attachments every frame and
  keep their images to themselves, so they record none.
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

Still open: descriptors, sync and the query pool (stage 2c), since landed
("Sync, descriptors and queries come from volumetric_kit_core", above).

## 2026-10-04 — The device comes from volumetric_kit_core

gfx's instance, physical-device capabilities and logical device are the family's
core's (stage 2a's second half): `Instance`, `InstanceConfig`,
`PhysicalDeviceInfo`, `Device`, `AdoptedDevice`, `EnabledFeatures`,
`DeviceRequirements`, `merge` and `check_device_support` are using-declarations
in `vg::`, as recon's are in `vr::`. A device gfx makes is the type recon adopts,
and the reverse. *Amended 2026-10-05:* gfx names them `core::` and the
using-declarations are gone ("gfx writes the core's types and macros under the
core's names", above).

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
  Command-buffer labels and object names are the core's device's
  (`begin_debug_label`, `end_debug_label`, `set_object_name`): `Profiler` and
  `DebugLabelScope` record through them, so gfx's labels carry no color. The
  core's device records no queue labels, so `QueueLabelScope` looks up those
  two entry points itself, gated on `Device::debug_labels_available`, and
  calls them holding `Device::submit_mutex`: Vulkan requires the queue be
  externally synchronized, and another library may share it. A `TODO:` marks
  taking queue labels from the core.
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
re-measuring frame times as the core's allocator changes placement. Both
stages have since landed ("Memory comes from volumetric_kit_core" and "Sync,
descriptors and queries come from volumetric_kit_core", above).

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
  refuses older headers in every gfx translation unit. No gfx leg builds on
  that floor since the Ubuntu 22.04 leg went (2026-10-10): the oldest headers
  in CI are Ubuntu 24.04's 1.3.275.
- **gfx turns the core's vulkan tier on and links it PUBLIC.** gfx's public
  `core/vulkan.hpp` and `core/result.hpp` include the tier's headers, so a
  consumer needs the tier whichever gfx type it names; `gfx_core` links it
  PUBLIC and the installed package refuses a core built without it.
- **gfx checks the core it got, not the one it asked for.** Its pin and
  `VKC_WITH_VULKAN` yield to a project that made the core available first, and
  FetchContent may find an installed core. *Amended 2026-10-08:* the check is
  the core's `vkc_require_core` (the core's DECISIONS.md, "Consumers pin, and
  an application declares the core first"). On a core older than
  `VG_VKC_MIN_VERSION`, the oldest gfx builds with, or one without the vulkan
  tier, gfx's top-level CMakeLists.txt fails the configure, and its package
  config reports `volumetric_kit_gfx` not found (`PACKAGE`), so a consumer
  that can do without gfx carries on. Either way the message names where the
  core came from and how to fix it. A core older than 0.1.0 has no
  `vkc_require_core`, so the configure stops at gfx's call as an unknown
  command (the core's README, "Use it in your project").
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

*Amended 2026-10-05:* gfx names these as the core does -- `core::Status`,
`VKC_TRY`, `VKC_CHECK` -- and the `vg::` names and `VG_*` macros below are
gone ("gfx writes the core's types and macros under the core's names", above).

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
  gap. *Amended 2026-10-08:* superseded: a backend status records the backend
  that set it, and `vk_result` is empty for any but Vulkan (the core's
  DECISIONS.md, "Merging the three `Status`/`Result` types").
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
uniforms become `DeviceMapped` or batch-uploaded `DeviceOnly`). All of it has
since moved: the device, then memory, then descriptors, sync and the query pool
(the dated entries above).

## 2026-10-02 — Shared agent guidance

`AGENTS.md` now owns the shared working rules and task-based reading map.
`CLAUDE.md` imports it with `@AGENTS.md`. This decision record keeps the
previously committed constraints available to both agents without copying the
local-only design document. Update shared rules here and in AGENTS.md as
appropriate when a decision changes; keep tool-specific configuration separate.

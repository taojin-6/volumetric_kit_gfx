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
  `QueryPool` (*amended 2026-10-23*: its `GpuTimer`), and `FrameMetrics` holds the core's
  `StageRow`s. See the dated entry below.
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
  streamed atlas", below). *Amended 2026-10-26:* set 0 may instead hold a camera picture's two
  Y'CbCr 4:2:0 planes, with a uniform saying which ("A mesh samples a 4:2:0 picture", below).
  *Amended 2026-10-27:* the pipeline binds `tangent` too, as a second atlas view ("A hybrid mesh
  blends a second atlas view", below). *Amended 2026-10-28:* set 0 may hold two pictures, and a
  triangle takes its provoking vertex's picture ("A mesh samples two pictures in place", below).
- **2026-10-05 — 2D images are converted, then mip-mapped, then drawn.**
  `pipelines::ImagePipeline` copies a picture's planes into an `ImageTexture`, renders them to
  display color in level 0 of an `R8G8B8A8_SRGB` image (sRGB decode, NV12's matrix and siting, or
  a color ramp), rebuilds its mip chain with halving blits that average in linear light, and draws
  it trilinearly into rectangles of any target; `camera::ImageView2D` holds the pan-and-zoom
  mapping. See the dated entry below. *Amended 2026-10-16:* a picture is written in regions,
  from any number of sources, and converted once ("Pictures written in regions", below).
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
- **2026-10-11 — The GLFW window glue is a library tier, `glfw`.** It reverses "library
  tiers never read GLFW": `gfx_glfw` does, and no other tier. See the dated entry below.
- **2026-10-16 — Points come straight from depth buffers.** `pipelines::DepthCloudPipeline`
  reads a producer's depth and colour buffers in its vertex stage, through pinhole cameras
  the host folds into matrices. See the dated entry below.
- **2026-10-19 — Camera input is gestures.** On every platform, `camera::OrbitCamera`
  takes drags in viewport heights and zoom factors, never raw pointer events, and no
  platform-neutral tier routes them. See the dated entry below.
- **2026-10-19 — One surface source per window system.** `windowing::SurfaceSource` carries
  a window system's instance extensions and surface factory together, from one constructor
  each for a GLFW window, a `CAMetalLayer` and no window. See the dated entry below.
- **2026-10-22 — gfx is consumed by commit.** Applications fetch it pinned by commit
  (FetchContent or `add_subdirectory`), as the player and the iOS app do; gfx has no install
  rules, `VG_INSTALL` or package config. No consumer installed it, and siblings pinning the
  core at different minors could not share one install prefix. This amends every entry below
  that describes an installed package.
- **2026-10-23 — A frame's timing says where its time went.** The stage table splits host
  time into busy and blocked and says what each GPU cell holds; the frame loop reports its own
  waits, submit and present; the profiler times on the core's per-frame ranges. See the dated
  entry below.
- **2026-10-26 — A mesh samples a 4:2:0 picture where it is.** `HybridMeshTextureSet` binds
  a `YcbcrPicture`'s two plane images, or an RGBA image, and the fragment shader converts:
  no sampler Y'CbCr conversion, and nothing in the core. See the dated entry below.
- **2026-10-27 — A hybrid mesh blends a second atlas view.** A `HybridMeshPipeline` draw whose
  `HybridMeshDraw::second_view` is set reads `assets::Vertex::tangent` as `(uv_B, w_B, unused)`
  and mixes the two views of an RGBA atlas in linear light by `w_B` and their alphas; the
  `(1, 0, 0, 1)` placeholder draws as before, and a draw without the flag draws and costs as
  before. See the dated entry below.
- **2026-10-28 — A mesh samples two pictures in place.** `HybridMeshTextureSet` binds one or
  two `YcbcrPicture`s, each in its own images; a triangle takes the picture its provoking
  vertex's `uv0.y` names. One picture draws as before, byte for byte. See the dated entry
  below.
- **2026-10-30 — Multisampling: 4x, resolved in the scope.** A multisampled `RenderTarget`
  keeps no samples past its scope: `begin` transitions them, the scope resolves each colour
  into a single-sample attachment, and the caller's barriers stay those of one sample.
  `OffscreenTarget` takes 1 or `kWindowSamples` (4); a mesh's texture coordinates are
  `centroid`. See the dated entry below.

## 2026-10-30 — Multisampling: 4x, resolved in the scope, centroid mesh UVs

**The contract.** A `RenderTarget` above one sample bundles one single-sample resolve
attachment per colour, of the colour's format. `begin` moves the multisampled colour and
depth images from `UNDEFINED` to their attachment layouts in one barrier, ordered after any
earlier scope's clear, stores and resolve read on them; the scope stores them `DONT_CARE`
and resolves each colour (`AVERAGE`) into its attachment in `COLOR_ATTACHMENT_OPTIMAL`.
The caller transitions only the resolve images, exactly as it transitions a single-sample
target's, so `OffscreenTarget::prepare`, `record_readback` and the frame loop's barriers
are unchanged. `begin` refuses `LOAD` and a stored depth (`VKC_CHECK`).

**Why the scope owns the samples.** Every consumer draws a frame in one scope -- the mesh,
a panel, then the UI -- so one resolve at its end is all a frame needs; nothing reads the
samples or the depth afterwards. Never stored, they need no memory traffic on a tile-based
GPU. Because `begin` orders each scope after the last on the same images, one
multisampled set can serve several targets that resolve into different images, such as a
swapchain's. A multisampled frame over several scopes, and a depth resolve, wait for a
caller that needs them.

**Why 4x alone.** Vulkan guarantees 4 samples for depth and for every colour format that
averages when resolved, on every device, so an `OffscreenTarget` needs no device query,
which its allocator cannot make. Another count, or an integer colour (no `AVERAGE`
resolve), is `Unsupported`. The multisampled images are `TRANSIENT`; the core places no
memory in lazily allocated types, so they take device memory on every device for now.

**Why centroid texture coordinates.** At 4x a silhouette pixel whose centre lies outside
the triangle is still shaded at its centre, so its texture coordinates extrapolate past the
triangle's patch of the texture and show whatever lies beyond it. `centroid` interpolates
them at the covered samples, inside the triangle. At one sample, and inside a triangle,
nothing changes. Its derivatives are less exact on those pixels, which matters only to a
set sampling mips (`HybridMeshTextureSetOptions::max_lod` above 0); pictures are read at
level 0. `image_draw` and `texture_view` keep exact derivatives: the one selects mips, the
other holds its reads to the picture.

**What stays single-sample.** `ImagePipeline`'s convert pass and `HybridMeshTextureView`'s
pyramid write their own images; exports and passes that must hit exact texels draw into
single-sample offscreen targets.

## 2026-10-29 — A mesh's texture drawn flat, beside the scene

**The pieces.** `HybridMeshTextureView` draws what a mesh samples as
pictures; `PicturePyramid` holds their levels from 1; `ui::ImagePanel` shows
them in a `ui::DockedPanel` beside the scene, which
`HybridMeshFrame::viewport` confines the mesh to. The view's set 0 is the
mesh pipeline's, so a frame's own set binds, and its level 0 is read through
`common/hybrid_mesh_set0.glsl`, the mesh shader's own sampling.

**An area box, not bilinear taps.** A tap between pixels of a Y'CbCr picture
mixes codes before the conversion, which is not linear. Against the exact box
mean in linear light, on a test card at a sub-pixel origin, `ceil(F)` x
`ceil(F)` bilinear taps were 45 codes off at the most (11.2 on average) at
1/2 and 37 (7.4) at 1/3; reading whole pixels at their centres, each by its
area inside the footprint, is within 1 (0.05). An RGBA image's taps were
already exact, as the hardware filters an `_SRGB` image in linear light.

**A pyramid below 1/4.** The box's reads grow with the footprint, so a
smaller draw reads the level where its footprint spans 4 pixels at most,
about 5 x 5 reads. A pyramid pixel the footprint's edge cuts counts whole:
off the pyramid's grid, at 1/5 to 1/12, that is 0.4-1.4 codes off on average
and 26 at a sharp edge.

**One `_SRGB` RGBA image a picture.** The view sees a set, not its form, so
the pyramid holds what a draw shows, in one form: level 1 drawn by the view
at 1/2, then linear blits. It takes about 4/3 of a byte a picture pixel, and
is rebuilt only when the pictures change, so a paused picture costs no build.

**Docked, not floating.** The panel and the scene split the drawable, so
nothing covers the mesh; `DockedPanel` says which side it docks at.

**One docked panel for any content.** `DockedPanel` holds the layout, the
handle and the header; `ImagePanel` holds the pictures and their view, and no
dock. A program whose panel shows content of its own docks by the same rules,
and one that shows pictures in a host's panel arranges them by the same
rules, with no copy of either.

## 2026-10-28 — A mesh samples two pictures in place

**The contract.** `HybridMeshTextureSet::create` and `rebind` take 1 to `kMaxPictures` (2)
`YcbcrPicture`s. Picture `p`'s point `(x, y)` in texels is at
`uv0 = (x / width, p + y / height)`. A triangle takes the picture `p` its provoking vertex's
`uv0.y` names, as it takes its albedo class, and samples it at `(uv0.x, uv0.y - p)`, held to
its edge. A provoking `uv0.y` outside the pictures takes the nearest: picture 0 below 0, the
last from the count up. Over one picture `uv0` is used as it is. Each picture has its own
extent and coding. Set 0 gains bindings 3 and 4, picture 1's luma and chroma, and its uniform
grows to 80 bytes: the form or count, then a coding per picture. A plane the form does not
read binds the pipeline's fallback texel. The pipeline, `HybridMeshFrame` and
`StreamedAtlas`'s API do not change.

**Why separate images.** A hardware decoder (VideoToolbox, NVDEC) leaves each picture in its
own surfaces, and a set binds them where they are. A layered image or one tall image would
take a copy per picture.

**Why two fixed bindings, not an array.** Neighbouring triangles may name different
pictures, so indexing an array of samplers would need
`shaderSampledImageArrayNonUniformIndexing`, which gfx's device requirements do not include.
The shader branches between two fixed pairs of bindings instead. Each branch samples at an
explicit level, so it needs no derivatives, and a group of fragments takes both branches
only where it spans triangles of both pictures. Four combined image samplers fit every
target's per-stage limits.

**Why per triangle.** Chosen per fragment from the interpolated `uv0.y`, the picture flipped
at an integer `uv0.y`. In perspective, a mesh of picture 1 at `uv0.y = 1` drew 47 % of its
pixels from picture 0 on an RTX 4090 and 13 % on lavapipe, and an edge at 1 flipped a few
pixels on every GPU. The vertex stage decides once, beside the albedo class, in the same flat
value, so no interpolant is added.

**Why clamped, not refused.** A shader cannot refuse a coordinate, and the nearest picture's
edge is what one picture has always drawn past its own edge.

**Why two.** It meets the caller's need. A third picture is two more bindings and a branch,
added when a caller needs it.

**The cost.** Measured A/B against the shaders before, Release, 600 interleaved iterations
in 3 runs over 8192x4352 pictures, paired medians. On a 1.26 M-triangle unshared mesh at
1280x720 and 2560x1440, one picture costs +0.001-0.004 ms on an Apple M5 Max, a 0.68-0.87 ms
frame, and +0.000 ms on an RTX 4090, a 0.30-0.33 ms frame. Where 4,608 large triangles leave
the frame to the fragment stage, at 2560x1440 and 3840x2160, it costs the M5 Max
+0.001-0.003 ms, 2-5 % of a 0.04-0.12 ms frame, and the RTX 4090 -0.002 ms at 2560x1440. On
the mesh, two pictures, each read by its own region, cost +0.000-0.005 ms more than one. The
worst case, a checkerboard where nearly every group of fragments reads both, costs
+0.026-0.037 ms more than one picture on the M5 Max and +0.013-0.014 ms on the RTX 4090.

## 2026-10-27 — A hybrid mesh blends a second atlas view

**The contract.** `HybridMeshPipeline` binds `assets::Vertex::tangent` at
location 4 and, in a draw whose `HybridMeshDraw::second_view` is set, reads it
as `(uv_B.x, uv_B.y, w_B, unused)`: a second view of the same atlas,
interpolated like `uv0`. Over an RGBA set, a fragment of the atlas class shows,
in linear light,
`((1 - w) a_A c_A + w a_B c_B) / ((1 - w) a_A + w a_B)`: each view's colour
`c` and alpha `a` (the atlas's coverage, linear in an `_SRGB` format) at
`uv0` and `uv_B`, with `w = w_B` clamped to [0, 1]. `w = 0` and a zero
denominator show the `uv0` view exactly. The vertex-colour class, the flags
and a picture's two planes are unchanged. A draw without the flag, the
default, shows the `uv0` view whatever its tangents hold.

**Why the tangent slot.** The hybrid technique never read those 16 bytes, so
a second view fits with the vertex at 64 bytes and the wire layout as it is.
Their placeholder `(1, 0, 0, 1)`, which `assets::Vertex` defaults to and
recon's mesher writes, carries `w = 0`, so every producer of today draws bit
for bit as before. The alternatives were a wider vertex, which grows every
mesh for one technique, or projecting each fragment into the cameras, which
puts the producer's camera model in the renderer.

**Why the alphas weigh in.** A producer may ramp an atlas's alpha toward its
image's border. Weighting each view by it fades a view out there instead of
cutting it at the edge, and a view with alpha 0 never shows beside one that
covers the fragment.

**The cost.** One more sample of the atlas per fragment of the atlas class,
in the same uniform control flow as the first, so both take the same implicit
LOD; where `w = 0` it reads the placeholder's one texel, which stays in the
cache. And three more floats from the vertex to the fragment stage. Measured
A/B against the shaders before, Release, 600 interleaved iterations over a
7680x4320 atlas, on an unshared mesh (three vertices per triangle, as a
multi-view texturer writes) of 1.26 M triangles, with no second view and with
every fragment blending: on an RTX 4090, +0.001 and +0.036 ms at 1280x720
(a 0.32 ms frame), +0.007 and +0.040 ms at 2560x1440 (0.35 ms); on an Apple
M4, +0.40-0.60 and +0.60-0.75 ms at 1280x720 (3.5-4.0 ms), and +2.27-2.29
and +2.41-2.47 ms at 2560x1440 (5.1 ms). On the M4 the three floats, not the
sample, are the cost, even where `w = 0`, and it grows with the vertices drawn
and the target's size.

**Only a draw that says its mesh carries one pays.**
`HybridMeshDraw::second_view`, false by default, says whether the mesh's
tangents carry a second view, and `submit` binds one of two pipelines per
draw. They are built from the same SPIR-V and differ in one fragment
specialization constant: in the pipeline without the second view the fragment
stage never reads the slot's interpolant, and the driver drops it. Both bind
the slot, since Vulkan wants an attribute for every input the vertex stage
declares; fetching it costs nothing measurable. Measured as above, in three
runs against the shaders before the second view, a draw without the flag
costs from -0.024 to +0.021 ms on the M4 at 1280x720 and from -0.002 to
+0.001 ms at 2560x1440 (medians; every run's 95 % interval of the mean holds
0), and 0.000 ms on the RTX 4090; with the flag it costs what it did, within
0.007 ms. The flag is the caller's because the producer knows what it wrote,
while the weights live on the GPU, where `submit` cannot read them. A second
SPIR-V build of the same source, with the slot compiled out, was the
alternative; the measurement leaves nothing for it to save.

**The risk.** A mesh with real tangents, a glTF model's, drawn with the flag
would blend its atlas at those tangents. None is drawn that way: without the
flag the tangents are ignored.

## 2026-10-26 — A mesh samples a 4:2:0 picture where it is

**The contract.** `HybridMeshTextureSet` binds an RGBA image's view or a
`YcbcrPicture`: an 8-bit 4:2:0 picture's luma (`R8_UNORM`) and chroma
(`R8G8_UNORM`) images, its extent at their corner, and its `YcbcrCoding`.
`rebind` moves a set between the two forms. Set 0 holds two planes and a
32-byte uniform with the form and the coding. An RGBA set binds the
pipeline's fallback texel as its unread chroma, and the pipeline's
device-only uniform. A picture's uniform is the set's own, in host-mapped
device memory, as `rebind` has no command buffer to record an update in; a
device without that memory takes RGBA sets alone. For a picture,
`hybrid_mesh.frag` filters each plane at level 0, held to the picture,
converts, then decodes sRGB. The pipeline, `HybridMeshFrame` and
`StreamedAtlas`'s API do not change; the set's `create` takes an allocator
for its uniform. *Amended 2026-10-28:* set 0 holds four planes and an
80-byte uniform, for up to two pictures, and a set binds the fallback texel
at every plane its form does not read ("A mesh samples two pictures in
place").

**Why.** A camera picture arrives as two planes -- ARKit's `capturedImage`
is '420f' -- and `ImagePipeline` already draws NV12. Sampling the planes where
they are spares a staging copy and an RGBA image. The iOS scanner's keyframe
textures are the caller.

**Why two plane images, not a sampler Y'CbCr conversion.** The conversion
needs a device feature, immutable samplers and a pipeline per range and
siting. On MoltenVK, a CoreVideo picture becomes one multi-planar image only
through behaviour it calls undefined, and its linear chroma reconstruction is
wrong. A multi-planar format refuses odd sizes, which two images take, each
addressed by its own size. Off texel centres the planes are filtered, then
converted; Vulkan's explicit model converts, then filters, which differs at
hard chroma edges.

**Where the form lives.** In the set, so one pipeline draws both forms. The
form is the same across a draw, so the shader's branch does not diverge.

**One home for the conversion in gfx, none in the core.**
`shaders/common/ycbcr.glsl` holds the chroma coordinate and the matrix, for
the mesh and `ImagePipeline`'s NV12 convert. recon's sensor tier keeps its own
conversion: it prepares camera input for fusion, in a library that does not
link gfx, sharing only the published BT.601/709 formula, as each repo writes
the sRGB curve its shaders use and pins it in its own tests; a GLSL include
for two libraries' shaders is not core library code.

**The picture's lifetime is the caller's.** The set names views, so the planes
stay alive until the last frame that sampled them retires: one set per slot
of a ring, rebound when its slot comes round, or the replaced picture pushed
to a `RetireQueue`.

**Not yet.** Mips for a picture: a minified mesh aliases, as it does over a
one-level RGBA image.

## 2026-10-23 — A frame's timing says where its time went

The family's timing design (its D3 and D9, the core's 0.20.0) splits a
stage's host time into busy and blocked, and says what its device half holds.
gfx shows both, and adds the blocking only it sees.

- **Each cell says what it holds.** The GPU column's `-` stood alike for a
  host-only stage, a stage run untimed and a queue that cannot time; each has
  a cell of its own now (`ui::draw_stage_table`), and host time is busy time,
  as a wait is no work. The line above the table (`ui::StageWindow`) states
  the frames the figures cover and how many were timed, so a stream's frames
  and a display's are not read alike, then the application's note: when it
  times and how its clocks are calibrated, which only it knows. The panel's
  "ms/frame" was the recording time; the period now comes from the rate.
- **The frame loop reports its own blocking.** With a profiler attached,
  `FrameLoop` adds `frame wait` -- the slot's fence, the acquire and the
  image's previous frame -- `submit` and `present` after the caller's stages
  (`Profiler::add_rows`), each in its own frame: a frame stays unpublished
  until its slot recurs, so its present joins it. The waits are
  `core::WaitScope`s, so a scope the caller holds around the frame sees them
  as blocked time. The acquire and the present, which no core wait covers,
  report their spans themselves (`core::note_blocked`), so without a
  profiler the caller's scopes still see every wait, and only the rows are
  missing.
- **A profiler scope's wait is the core's.** Each open `Profiler::Scope` is a
  `core::StageScope` on its thread, so a core wait inside one -- a fence, a
  timeline, a blocking submit -- is that row's `wait_ms`, and its host
  column, busy time, never holds it. The scope therefore closes on the
  thread that opened it, as a core scope does.
- **The profiler times on the core's ranges** (D9). It replaces "The profiler
  stays gfx's, on the core's `QueryPool`" (2026-10-04, below). Each frame
  takes a range of one `core::GpuTimer` as its `core::GpuTiles`, holds it
  until its slot recurs, and marks it at `begin_frame`, at each GPU scope's
  open and close, and at `end_frame`. The tiles are named by index, and a
  scope's time is the sum of the tiles between its marks, so it measures what
  a timestamp pair at its ends did, nested scopes included, and two scopes of
  one name stay two rows. It costs the two marks at the frame's ends. A row
  reads `NotTimed` past the bound, on another command buffer, or where a mark
  was not yet readable, and `Unavailable` where the queue has no timestamps
  or the pool will not allocate, which no longer fails `create`. The core's
  ceiling of 4096 timestamps a pool, the most a Metal-backed pool holds,
  lowers `max_gpu_sections_per_frame` where the frames in flight need more:
  a frame of n scopes takes n + 1 pairs, so each frame in flight gets at most
  `kMaxSpans / frames_in_flight` pairs, n one fewer. No count can then pass
  32 bits, so `create` no longer refuses a bound whose product would.

## 2026-10-20 — Panels for a finger

**A frame height apart from the scale.** With `scale` alone, the 2 that made
targets big enough for a finger made a panel too wide for a phone in portrait.
`ImGuiOverlayConfig::min_frame_height` lets a phone take body-sized text (1.3)
with 44-pt targets.

**A collapsible panel is two windows.** ImGui takes input, or none, per
window, so the header's button fills a window of its own and the body is a
second one that takes none.

**Why not ImGui's own collapse.** Its arrow is a font-size square, about
17 pt at scale 1.3, too small to tap; a single tap on the title bar does not
toggle it, only a double-click does; and a window with a title bar takes
input over its whole body.

**Why `draw_stage_table`.** A panel with lines of its own, such as the
player's stats, shows its stages without a second window.

## 2026-10-19 — One surface source per window system

**The contract.** A `windowing::SurfaceSource` is a window system's surface
as the tiers above take it: its instance extensions and its factory, made by
one constructor per window system -- `glfw::surface_source`,
`windowing::metal_surface` and `windowing::headless_surface` -- so the two
cannot disagree. Each refuses what it can see is wrong before an instance
exists. `app::make_shared_device` builds the device a windowed renderer shares
with a compute library on any of them, and `WindowedApp::adopt(SharedDevice&,
...)` adopts it, taking its surface once.

**Why `metal_surface` is in windowing.** It is the Vulkan call GLFW makes for
a macOS window, so iOS presents through the MoltenVK path desktop macOS
already runs, with no Metal API in gfx. The layer is a `const void*`, so no
header names an Objective-C type, and the entry point is resolved on the
instance, so the source builds, and its refusals test, on every OS. Only a
test on Apple makes the surface.

**One headless factory.** `headless_surface()` is the one home of the
headless factory and of the loader probe, its `Unsupported`;
`Surface::headless` and the tests take it. NVIDIA's ICD offers no
`VK_EXT_headless_surface`, while Lavapipe and MoltenVK do, so on Apple the
shared-device cases run, and a skip there fails.

**Why adopt takes the `SharedDevice`.** The surface a shared device made is
the renderer's to take, once, so the overload takes it and no consumer writes
the adopt lambda over `release_surface()`. A later adopt is refused, naming
why, rather than handed a null surface.

**GLFW's init and window.** `glfw::Library` and `glfw::create_window` are the
one home of GLFW's init guard and its no-API window; gfx's examples use them.

## 2026-10-19 — Gestures drive the orbit camera

**The contract.** `camera::OrbitCamera` takes the same gestures on every
platform: drags in viewport heights (`drag_orbit`, `drag_pan`) and zoom
factors (`zoom`, for a wheel or a pinch). Each shell divides its pointer or
finger delta by the viewport's height in the same unit --
`glfw::cursor_in_framebuffer` over the viewport's height in framebuffer
pixels, UIKit's translation over the view's -- and calls the same verbs, so
the mapping from a gesture to the camera has one home.

**Why the shell routes.** Which button or finger count orbits, whether ImGui
holds the pointer, and how a wheel's notches accumulate differ per window
system, so the shell owns them: the camera takes no raw pointer events, and
no platform-neutral tier (`camera`, `ui`, `app`, `windowing`) decides where
one goes. The GLFW half of that glue -- scroll and key queues, the drag
latch, ImGui capture -- belongs in `gfx_glfw`, deferred as calib's input-glue
TODO (`calib_rig.cpp`) records.

**Why OrbitCamera stays.** It is the turntable controller, no longer waiting
to fold into `CameraRig`: its users -- calib's 3D view, and the player's --
set and read a target, a distance and two angles, and a fold would change
their code for no new behaviour.

## 2026-10-19 — ImGui's colours as authored

**The contract.** `ui::ImGuiOverlay` shows ImGui's colours -- its theme and
`IM_COL32` marks -- as authored on any target, as the pipelines show
`pack_rgba8` colours. ImGui's colours are sRGB-encoded, and upstream's
fragment shader writes them as given, which suits a UNORM target; an `_SRGB`
target, the default swapchain's, would encode them again. Where the first color
attachment takes linear light (`_SRGB` or float, the pipelines' rule),
`create` gives the backend a variant of that shader which decodes the vertex
colour first.

**Why a derived shader.** The ui build derives the variant from the fetched
`glsl_shader.frag` at configure time, as it patches the scissor ("Pictures
written in regions", below), and fails if a line it rewrites changes
upstream. It takes the curve from `shaders/common/srgb.glsl` and reaches the
backend through `CustomShaderFragCreateInfo`, so it copies neither upstream's
interface nor the curve. Rejected: a UNORM swapchain, which changes how every
pipeline draws; and decoding the draw lists on the host each frame.

**The cost.** On such a target ImGui blends in linear light, so translucent
windows and anti-aliased text edges read slightly differently from a UNORM
target.

## 2026-10-19 — ImGui on any window system

**The contract.** gfx draws its ImGui panels the same way on every platform;
only ImGui's platform half differs. `glfw::ImGuiGlfw` is GLFW's.
`ui::ImGuiTouch` is the half for a window system ImGui has no backend for
(UIKit now, Android later), fed by the app: a `DisplayFrame` each frame, and
touches in points. It reads no window system, so it sits in `ui` beside the
renderer half. `overlay_config(swapchain)` is the one home of the overlay
config that gfx's examples, recon's viewers and calib's rig app each wrote.

**Why one finger.** ImGui has one pointer, and a second finger belongs to
the app's camera gestures (a pinch, a two-finger pan). Each event follows the
touch-screen mouse source, so ImGui takes a finger's position and its press on
separate frames: the widget under a finger, which has no hover to come from,
sees it before the press.

**Why a cancel is not a lift.** UIKit cancels the first finger when a pinch
or a two-finger pan recognizes. A release where the finger rests would press
the button under it, so `touch_cancelled` clears the position before it
releases.

**Why the work inset.** ImGui has no public setting for a per-edge safe area.
`new_frame` assigns it to the main viewport's build inset
(`ImGuiViewportP::BuildWorkInsetMin` / `Max`, internal), which
`ImGui::NewFrame` moves into `WorkPos` / `WorkSize` and then zeroes.
Assigning rather than adding means a tick the app skips cannot apply it twice;
the cost is that it replaces what a main menu bar reserved the frame before.
`imgui_touch_test` pins the behaviour, so an ImGui update that changes these
internals fails there.

**Why the scale is a config field.** ImGui adds its default font on the first
frame and picks it by the size set then: the pixel font below 15 px, the
vector font from there. A setter used after the first frame would stretch the
pixel font. So `ImGuiOverlay::create` scales the fresh context's default dark
style, the font size included, before that frame.

**Why `ui` links `windowing`.** `overlay_config` takes a `Swapchain`, and
`windowing` is left of `ui` in the tier order.

## 2026-10-18 — Categorical colours

**The contract.** `pipelines::categorical_rgba8(i)` colours item i of a set
of any size: Okabe and Ito's palette, then a hue ring. A view that shows
several sources -- calib's rig cameras, as frusta and as `DepthCloudDraw`
tints -- takes its colours here rather than keeping a palette of its own.
`color.hpp` states how far apart the colours are, and the tests hold it.

**Why Okabe and Ito's palette.** It is the established palette for colour
vision deficiency (Okabe and Ito, https://jfly.uni-koeln.de/color/, 2008;
Wong, Nature Methods 8:441, 2011). As frusta and `Solid` points, under
Machado, Oliveira and Fernandes' severity-1 simulation (IEEE TVCG 15(6),
2009), its seven colours stay at least 0.076 apart in OKLab for protan,
deutan and tritan vision, where Paul Tol's bright and light schemes fall to
0.033 and 0.037 (tritan). Its black is left out: it vanishes on the views' dark
backgrounds, and as a `TintedLuma` tint it turns points black. The darkest
colour kept, blue `#0072B2`, is OKLab lightness 0.53.

**Why yellow and blue swap.** A `TintedLuma` point takes its lightness from
the scene, so only its tint's chromaticity names its camera, and Okabe and
Ito tell several colours apart mainly by lightness. At equal luma a
dichromat keeps one chromatic axis: for protan vision orange, yellow and
vermillion come within 0.008 of each other in OKLab, and no five of the
seven stay 0.05 apart. In the published order a 4-camera rig gets both
orange and yellow; with blue fourth, the first four are among the 4-sets of
the seven that stay furthest apart there for normal, protan and deutan
vision alike. The cost is that a 4-camera rig gets the darkest colour kept.

**Why the hue ring.** Later items share OKLCH lightness 0.75, the palette's
orange, and chroma 0.125: at that lightness sRGB holds chroma up to 0.1275 at
every hue, so no item needs gamut mapping. Golden-angle steps spread
consecutive hues. Item 7's hue, 12.5 degrees, is about the start that keeps
the first 16 colours furthest apart, so separation falls off gradually as a
set grows rather than holding for one size and then dropping. Hue 0 served
14 items best, but made item 7 a lighter reddish purple and put the 15th
item within one JND of sky blue; one golden step later serves 8 items best
but puts the 14th on sky blue. The ring keeps no colour-blind margin: its
one lightness takes away the cue dichromats rely on.

**Why pipelines.** The colour is the pipelines' word, which `ColorVertex` and
`DepthCloudDraw::tint` take; `ui` is ImGui's, and build-tree only.
`pack_rgba8` moved to `pipelines/color.hpp` with it, so the palette needs no
pipeline header.

## 2026-10-16 — Points from depth buffers

**The contract.** `pipelines::DepthCloudPipeline` draws a depth image as
points: one vertex per depth pixel, read in the vertex stage from the
producer's Float32 storage buffer, with no vertex buffer and no copy. An
optional buffer of R'G'B'A8 words colours each point by recon's projection
rule, and a point with no colour takes the draw's tint. A draw is one depth
source; a frame has any number. gfx knows no rig or calibration: a draw is
buffers, `camera::PinholeCamera`s and poses. The `PinholeCamera` is gfx's
own, not recon's camera model, because gfx cannot link recon, whose viewers
link gfx; it is float, carries the image size, and the caller converts to it.

**Why the host folds the cameras.** Each draw's intrinsics, poses and view
fold on the host, in double, into two matrices: depth pixel to clip space,
and to the colour camera's pixels. The vertex stage keeps only recon's pixel
rule -- in front of the camera, inside [0, w) x [0, h), and
min(floor(q + 0.5), size - 1) -- copied from recon's `nearest_pixel`, as gfx
cannot include recon's GLSL. A change to recon's rule must change this copy.
gfx's tests pin it against a host model of the rule; only a test that draws
recon's prepared frames through it, in calib, can check the two together.
Equal poses cancel exactly on the host, with intrinsics composed directly:
registered depth and colour keep an identity pixel mapping, including the
first row and column at the image boundary.

**Why vertex-stage storage buffers.** They read the producer's buffers where
they are, `CONCURRENT` across the families when the producer names both (for
recon's prepared depth, `GpuFramePrepConfig::depth_queue_families`, which
recon's #246 adds); the two-family test draws such buffers with a
producer-signalled semaphore wait covering the vertex shader. Sharing mode
alone supplies no memory visibility, and a host fence wait does not replace
that GPU dependency. The public header owns the caller's synchronization
contract; `submit` records draws inside rendering, where it cannot insert the
buffer dependency. There is no image path and no u16 depth, which no producer
needs (raw sensor depth is distorted, so it is prepared first), and no quad
fallback for wide points: without `largePoints` a point is one pixel.

**Why host-mapped parameters.** `submit` records inside a rendering scope,
where `vkCmdUpdateBuffer` cannot be, so each draw's parameters go into
`DeviceMapped` memory. Push descriptors would need an extension and a core
wrapper, and a record-before-rendering call would split each draw in two.

## 2026-10-16 — Pictures written in regions

**The contract.** `ImagePipeline::record_write` copies any number of sources
into regions of one `ImageTexture`'s planes, and `record_convert` converts the
whole picture and rebuilds its chain; `record_update` is the two for one
region the picture's size. A rig's cameras share one texture: one convert, one
chain and one draw per frame, however many cameras, and a camera with no new
picture keeps its last. The first write after `create` or `discard` clears
the planes to black, so texels no write covers -- a gutter -- are black;
after that they keep what was written.

**Why gfx records the copies.** The application picks the regions -- calib's
rig mosaic -- and gfx keeps the planes private and records the copies, so
`check_plane` and the barriers stay in one place. The copy is the one a
texture per camera made already: a decoder's picture allows transfer only.

**Why one picture, not regions clamped apart.** Convert-then-mip is
unchanged (2026-10-05, below), and its chain is of the whole picture, so
regions meet in its coarse levels and in NV12's chroma upsampling. Clamping
each region in the shader would cost a region table per texture; instead a
layout leaves a gutter, and `ImageWrite`'s comment states what it buys: the
edge chroma's share, and the levels a gutter of 2^k texels keeps apart. One
conversion serves the picture, so its regions share one coding.

**Why discard.** A texture records the layouts its commands leave, so a
command buffer that never runs (a failed `end_frame`, the swapchain not
stale) leaves them wrong. `ImageTexture::discard` forgets them, so the next
write clears from `UNDEFINED` and nothing draws until the next convert: the
picture is blank, never wrong.

**Limits.** A texture is at most `ImagePipeline::max_texture_extent`, the
device's `maxImageDimension2D` bounded by its framebuffer and viewport
limits, as the convert renders the whole picture; `ImageTexture::create`'s
refusal names it, so a caller pages its regions over several textures.

**Marks over a region.** One view shows the whole picture, so
`ui::ImageOverlay` takes an `OverlayRegion`: a camera's marks, in its own
pixels, land on its region and are clipped to it, and its caption stays in
the region's visible corner. The half-texel rule and the clip stay in the
overlay, so a caller adds no offsets of its own. ImGui's Vulkan backend
truncates a clip rectangle to its scissor, so the clip is put on whole
pixels, those that show the region, and the grid's outer borders run along
them: at any pan, a tile keeps all four, and none spills into the next.

At fractional framebuffer scales, an integer edge may project back just
above itself. After intersecting the clip, the scissor must subtract its
integer endpoints; truncating their floating-point difference instead loses
the last row or column when the far edge is clamped. The ui build applies
this two-expression correction to a generated copy of ImGui's Vulkan backend,
leaving fetched sources untouched and refusing a changed upstream expression
so dependency updates revisit it. TODO: remove the correction once the pinned
upstream backend preserves the integer span itself. GPU readback covers
framebuffer and nested draw-list edges for regions and panned whole images.

## 2026-10-11 — Core owns application run configuration

The examples use core's optional `core_config` target for the JSON format,
typed settings, defaults, validation and the `--config` / `--help` entry point.
Gfx does not register individual flags or own a second configuration parser.
The examples translate validated settings into the existing renderer structs;
driver capability checks and actual framebuffer sizes remain runtime concerns.

Only example executables link `core_config`. The renderer's public API,
headless applications and non-GLFW consumers keep their existing typed APIs
and do not acquire a JSON or GLFW dependency from configuration. The GLFW
glue remains independent of the choice of application configuration format.

## 2026-10-11 — The GLFW window glue is a library tier

`gfx_glfw` holds what each GLFW application of the renderer wrote for itself:
the instance extensions, the surface factory, the framebuffer extent, the
cursor in framebuffer pixels, `make_shared_device` (one device for the window's
renderer and a compute library, from the core's `SharedDevice`) and
`ImGuiGlfw`, which compiles `imgui_impl_glfw` once. gfx's four examples,
recon's two viewers and calib's coming rig app each had, or would have had, a
copy: the family keeps one home per function. gfx learns only a compute
library's `DeviceRequirements`, never the library.

- **What it reverses.** gfx's tiers were GLFW-free, with the glue in each
  consumer. Now `glfw` reads GLFW, and only it: `windowing`, `app` and `ui`
  still take a `VkSurfaceKHR`, a surface factory and an ImGui context, so a
  window system without GLFW (iOS's `CAMetalLayer`) uses them as before.
  The requirement union `make_shared_device` builds has no GLFW in it, so it
  lives in `app` as `shared_device_config`, for such a window system to call.
  *Amended 2026-10-19:* `make_shared_device` itself moved to `app`, over a
  `windowing::SurfaceSource`, and `glfw::make_shared_device` wraps it ("One
  surface source per window system", above).
- **Optional and build-tree only.** It is built with `VG_WITH_GLFW` where GLFW
  is installed, and is not installed or exported, like `gfx_ui`, until the
  ImGui headers are co-installed; recon and calib fetch gfx, so they can link
  it.

## 2026-10-11 — ImGui uses 32-bit indices

The ui tier builds Dear ImGui and its Vulkan backend with
`ImDrawIdx=unsigned int`. `volumetric_kit::gfx_ui` publishes the same compile
definition to every consumer, including platform backends and demo sources
compiled by an application. All sources sharing the context must be rebuilt
with that configuration; the index type is part of ImGui's ABI.

A single `ImageOverlay::polyline` can generate more than 65,535 vertices.
ImGui's vertex-offset support starts a new range between allocations, but
cannot split one oversized stroke allocation. Using 32-bit indices preserves
ImGui's joins, anti-aliasing and closed paths without a separate batching
implementation. The Vulkan backend already supports this index width. Index
storage doubles; vertex and texture storage do not change.

## 2026-10-08 — A streamed atlas, and the pipeline's own fallback

**The contract.** `HybridMeshPipeline::create` now takes the device and an
allocator and builds a fallback atlas -- one texel, its set and a sampler.
`submit` binds it for a frame whose `atlas` is null and sets
`kHybridMeshVertexColor`, so every triangle draws in its vertex color.
`pipelines::StreamedAtlas` is the atlas a live mesh samples: a ring of images,
each with a set written once. A frame binds `use(frame.number)`, the newest
picture's set, which marks that image used by the frame. An update -- tiles
from device buffers, one buffer or one a tile (`record_update`), or host
pixels (`record_upload`) -- is recorded into a frame's command buffer and
copies into the least recently used image whose last frame the timeline has
reached; that image becomes the picture. A frame that never reaches the queue is given back with
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
  wait only for a value already reached (`TimelineWaits::Reached`), and adds
  what each frame sets to the core's record of submitted values;
  `FrameLoop::end_frame` says why. A producer whose
  value is not yet reached is gated on the host
  ([the live-mesh contract](docs/integration/recon-live-mesh.md)).
- **The core's former `Submitted` spelling now means `Reached`**: a
  submitted producer alone does not prove its upstream dependencies are
  resolved. Atlas retirement still checks whether work was submitted,
  through the core's signal-value ordering check, before waiting on the
  host or releasing a discarded frame's images.

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
test. *Amended 2026-10-22:* that covers a missing device only; a shard on Apple
also fails on any other skip, and the Linux legs list theirs in the step
summary.

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
- **The profiler stays gfx's, on the core's `QueryPool`.** *Replaced
  2026-10-23:* it times on the core's `GpuTimer` ranges ("A frame's timing
  says where its time went", above). The core's
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
  `FrameLoop` do, and locks its mutex on a render that updates a texture, the
  only kind on which the backend submits.
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
- *Amended 2026-10-19:* **a top-level build finds Vulkan and glm after the
  core**, so in an iOS build the core's `cmake/vkc_ios.cmake` answers both:
  MoltenVK's static library and headers (on iOS `MoltenVK_ROOT` picks them, not
  `Vulkan_INCLUDE_DIR`) and the host's glm headers. An iOS build never fetches
  glm; without the host's, it stops with the core's message. Elsewhere the
  order changes one thing: a missing Vulkan SDK is reported by the core's find,
  not gfx's.

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

## 2026-10-26 — Deferred release requires established completion

`RetireQueue::drain` returns a `Status`. A successful wait releases the guarded
entry; device loss releases the remaining entries and is still returned to the
caller. Other wait failures preserve the unresolved entries for an explicit
retry. Destruction and move assignment cannot report that failure, so they
preserve the complete pending owner, including closure captures, without an
allocation. This deliberate leak is preferable to destroying a resource whose
GPU use remains unknown. `reclaim` remains an explicit operation for resources
known complete or a device known idle/lost.

Image upload enrolls staging with the retirement queue before recording its
copy. All image-update host allocation finishes before the first command, so
an allocation failure reports `OutOfMemory` without partial recording or a
changed image layout.

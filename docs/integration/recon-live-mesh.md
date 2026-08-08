# Live mesh handoff — `volumetric_kit_recon` → `gfx` (PR1: indirect draw)

**Status:** the draw mechanics, with the seams it depends on now answered by the
reconstruction side. This is the first of the three slices that complete the
live zero-copy path (indirect draw → per-slot atlas ringing →
`app::StreamedApp` driver).

This lands **only the draw mechanics**. The synchronization and lifetime seams
were left open here for recon to answer, and it has — see **§4**, which records
the answers rather than the questions (the sync/lifetime ones landed; the
command-authorship one is in review). Nothing in the contract below changed as a
result: recon adopted the shape this was drafted against.

## 1. What PR1 adds

`pipelines::LiveMesh` (`include/volumetric_kit/gfx/pipelines/live_mesh.hpp`) — a
borrowed, per-frame-variable mesh drawn with `vkCmdDrawIndexedIndirect`. It owns
nothing; it names three buffers the producer writes and records an indirect
indexed draw against them. The index count lives in the command buffer, so a
mesh that grew/shrank this frame draws correctly with **no CPU round trip**.

It plugs into the existing `HybridMeshPipeline` unchanged otherwise: a
`HybridMeshDraw` now carries `std::variant<const GpuMesh*, LiveMesh>`, so one
frame can mix static and live meshes, and `submit()` dispatches per draw.

## 2. Buffer contract (what recon provides)

Three buffers, all borrowed (gfx never copies, never frees). The flags below are
what the **draw** needs. Usage flags are a union, not a choice: recon adds
whatever its own write path requires on top — writing all three from a compute
pass (§4.3) means `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` as well, or
`VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` for a buffer-device-address write.
A buffer created with only the draw flag cannot be bound to recon's own
descriptors.

| Buffer | Usage the draw needs | Contents |
|---|---|---|
| `vertices` | `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` | interleaved `assets::Vertex` |
| `indices`  | `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`  | **32-bit** indices (`VK_INDEX_TYPE_UINT32`) |
| `indirect` | `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT` | one `VkDrawIndexedIndirectCommand` |

- **Vertex layout** is `assets::Vertex`, byte for byte. Declaration order is
  `position`, `normal`, **`tangent`**, `uv0`, `color` — with glm's default
  (unaligned) types that is offsets 0 / 12 / 24 / 40 / 48 at a stride of 64. The
  hybrid technique binds position/normal/uv0/color; **tangent is ignored** but
  still occupies bytes 24–39. Emitting a different field order at the same
  stride produces no size mismatch and no validation error — just every
  attribute read from the wrong bytes. `assets/mesh.hpp` is the source of truth;
  `hybrid_mesh_pipeline.cpp` `static_assert`s the stride and all five offsets,
  so a change on the gfx side breaks the build rather than the wire.
- **The albedo sentinel is `uv0.x < 0`, not the exact `(-1, -1)`.** Any negative
  x selects the per-vertex color; recon's `(-1, -1)` TSDF sentinel is simply one
  such value. So a projective-texturing uv that lands *marginally* negative —
  the kind `CLAMP_TO_EDGE` would otherwise absorb — silently drops that triangle
  to flat vertex color. Clamp atlas coordinates to `[0, 1]` before emitting
  them.
- **The albedo class is per triangle.** The choice is resolved per vertex and
  forwarded `flat`, so a triangle is shaded entirely under its **provoking
  vertex's** class (`hybrid_mesh.vert`). A triangle straddling a camera-coverage
  boundary — some vertices sentinel, some not — is therefore not split: it
  either samples the atlas at a uv interpolated toward a sentinel vertex's
  stand-in `(0, 0)`, or drops the real uvs entirely. Keep the two classes
  triangle-aligned.
- **The command**: set `instanceCount = 1` and `firstInstance = 0`
  (`firstInstance != 0` would need the `drawIndirectFirstInstance` feature).
  `indexCount`, `firstIndex`, `vertexOffset` are yours — write them GPU-side from
  the marching-cubes output. gfx never reads the command, so it can neither
  check nor report a violation (see §5).
- **Sub-allocation**: pack many meshes into shared pools either way — byte
  offsets on `LiveMesh` (`vertex_offset`/`index_offset`/`indirect_offset`) at
  bind time, or `firstIndex`/`vertexOffset` inside the command. The two compose
  **additively** on the same buffer, so set at most one per axis (using both
  double-counts and reads past the mesh).
- **Offset rules**: all three byte offsets must be 4-aligned —
  `index_offset`/`indirect_offset` per core Vulkan, `vertex_offset` per
  MoltenVK/Metal. Each must also leave room for what the draw reads: the last
  command in a ring sits at `size - sizeof(VkDrawIndexedIndirectCommand)`, *not*
  `size - 16`, and the vertex/index offsets must fall strictly inside their
  buffers. A `LiveMesh` borrows handles and never learns a size, so `valid()`
  cannot check any of this.
- **An empty slot must be said explicitly.** `valid()` gates on the three
  handles alone, so a bound `LiveMesh` always draws whatever its command
  currently holds — a stale or never-written `indexCount` fetches indices
  outside the arena, which `robustBufferAccess` does not cover (device loss, or
  a Metal fault). A slot with no geometry this frame passes a
  default-constructed `LiveMesh` or writes `indexCount = 0`.

**Device features:** a single indirect draw (`drawCount = 1`) is core Vulkan
1.0. **PR1 adds no new `DeviceRequirements`** — nothing new to merge into the
shared/adopted device's feature set.

**Clip space:** `HybridMeshFrame::view_proj` is a *caller-built* matrix, so the
projection convention is part of this contract. gfx targets Vulkan's clip space:
depth in `[0, 1]` and a Y-down framebuffer. Consumers that build the matrix with
glm must compile with `GLM_FORCE_DEPTH_ZERO_TO_ONE` — linking any gfx target
that exports glm (`gfx_assets`, and so `gfx_pipelines`; or `gfx_camera`) applies
it automatically as a usage requirement, so this only needs thought if the
matrix is built somewhere that does not link gfx. Getting it wrong produces a
`[-1, 1]` projection against a `[0, 1]` depth attachment: Vulkan clips at
`0 <= z <= w`, so the near half of the frustum silently disappears — no compile
error, no validation message. If the matrix comes from a non-glm math library,
apply the same two conventions there (and flip Y, as `camera.cpp` does, or the
image renders upside down).

## 3. Zero-copy via `Device::adopt`

The three buffers may be VMA-allocated by recon on the **same `VkDevice` the
renderer adopted** (PR #85). gfx binds their handles directly — no external-memory
import, no staging copy. This is the same-API case the device-adopt decision
anticipated (none of the CUDA/Metal interop machinery applies).

**The embedder must declare what it enabled.** `AdoptedDevice` carries an
`enabled_*` block — extensions, `enabled_features`, `enabled_timeline_semaphore`,
`enabled_dynamic_rendering` — and `Device::adopt` verifies the renderer's
requirements against *that declaration*, not against physical-device support.
Support is not evidence of enablement: every Vulkan 1.3 physical device reports
`dynamicRendering` whether or not the logical device turned it on, so a
compute-focused bootstrap that leaves `VkPhysicalDeviceVulkan13Features` zeroed
would otherwise adopt cleanly and then hit
`VUID-vkCmdBeginRendering-dynamicRendering-06446` on every frame. The fields
default to "not enabled", so an embedder that declares nothing fails `adopt`
loudly at startup instead.

## 4. Answers from recon

These were open when this was first written. recon has since answered all four:
§4.1 and §4.2 are **landed** (`volumetric_kit_recon` #47, buffer sharing +
barrier visibility; #48, the arena slot ring). §4.3 and §4.4 are recon's
**proposed** shape, in review as `volumetric_kit_recon` #49 — recon's shipped
code still takes the host path (download, then `upload_mesh` into a static
`GpuMesh`). Recorded here because the contract above only makes sense alongside
them, and flagged because #49 could still move in review; nothing in gfx depends
on it landing, since `LiveMesh` draws whatever command it is handed.

1. **Synchronization ownership: the application, gating on the host.**
   `record_draw` stays a pure recorder and inserts no barrier, as drafted -- but
   the producer→draw dependency is *not* a semaphore the draw waits on. It
   cannot be: on the shared-queue arrangement a command buffer waiting on a
   value the sibling has not signalled deadlocks against a swapchain rebuild,
   which drains the queue while holding the submit mutex. The application polls
   readiness on the host and skips a not-ready frame instead, which is what it
   already does for the host-mesh path.

   Visibility is still a barrier, and recon now emits it: its shared `dispatch()`
   widened its destination scope to `VERTEX_INPUT | DRAW_INDIRECT` with
   `VERTEX_ATTRIBUTE_READ | INDEX_READ | INDIRECT_COMMAND_READ` -- gated on the
   recording family actually supporting graphics, since naming `VERTEX_INPUT` on
   a compute-only family is invalid usage. Where recon lands on such a family the
   handoff needs a semaphore regardless, and a semaphore's signal/wait carries
   the visibility itself.

   Cross-family access needs the buffers created `VK_SHARING_MODE_CONCURRENT`,
   which recon's `BufferDesc` now takes queue families for. This matters on
   Apple, where the bootstrap hands recon and gfx queues from *different*
   families.

2. **Buffer lifetime: recon rings, the consumer reports completion.**
   `MarchingCubesConfig::slot_count` gives each outstanding extract its own
   arena, index run and command. The consumer calls
   `MarchingCubes::release_through(generation)` as its frames retire, and an
   extract only ever writes, grows or frees a released slot -- so no fence queue
   is needed inside recon and no `MTLSharedEvent` anywhere.

   Depth: **frames in flight + 1**. Extracting with every slot outstanding is
   reported as an error rather than overwriting a live draw.

3. **Who fills the command: recon, GPU-side, in full.** Its marching-cubes
   kernel counts *indices* rather than triangles, so the atomic it already had
   *is* `indexCount` -- the command is written by the extraction rather than
   assembled from a count afterwards. `instanceCount = 1` and the three offsets
   are host-written once per extract, matching this contract exactly. No count
   buffer variant is needed.

4. **Index width: `UINT32`.** Confirmed -- marching cubes emits it and there is
   no 16-bit case.

## 5. Proof (this PR)

Under the validation layer with teeth, in `tests/hybrid_mesh_pipeline_test.cpp`:

- `IndirectDrawMatchesDirectDraw` renders the same mesh as a static `GpuMesh`
  (direct `vkCmdDrawIndexed`) and as a `LiveMesh` (indirect) and asserts
  **byte-identical** framebuffers;
- `IndirectDrawHonorsBufferOffsets` repeats that from buffers whose data sits
  past a non-zero pad, so every bind offset is exercised;
- `MixedStaticAndLiveInOneFrame` draws a static mesh plus a *distinguishable*
  live one and asserts each covers its own region — both variant branches run;
- `ZeroInstanceCountCommandDrawsNothing` pins why the command contract fixes
  `instanceCount = 1`.

`tests/live_mesh_test.cpp` covers the value-type contract (`valid()` gates on
all three buffers).

**What this does not prove.** gfx never reads the command, so no gfx test can
fail because *recon* wrote `instanceCount = 0` or a non-zero `firstInstance` —
the tests above supply their own well-formed command. Asserting the values the
extraction kernel emits belongs to recon's own test, on the side that writes
them; gfx pins only the consequence of getting them wrong.

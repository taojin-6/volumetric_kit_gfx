# Live mesh handoff — `volumetric_kit_recon` → `gfx` (PR1: indirect draw)

**Status:** the draw mechanics, with the seams it depends on now settled by the
reconstruction side. This is the first of the three slices that complete the
live zero-copy path (indirect draw → per-slot atlas ringing →
`app::StreamedApp` driver).

This lands **only the draw mechanics**. The synchronization and lifetime seams
were left open here for recon to answer, and it has — see **§4**, which records
the answers rather than the questions. Nothing in the contract below changed as
a result: recon adopted the shape this was drafted against.

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

Three buffers, all borrowed (gfx never copies, never frees):

| Buffer | Usage flag | Contents |
|---|---|---|
| `vertices` | `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` | interleaved `assets::Vertex` |
| `indices`  | `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`  | **32-bit** indices (`VK_INDEX_TYPE_UINT32`) |
| `indirect` | `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT` | one `VkDrawIndexedIndirectCommand` |

- **Vertex layout** is `assets::Vertex` (position, normal, uv0, color, tangent).
  The hybrid technique binds position/normal/uv0/color; **tangent is ignored**.
  `uv0 = (-1, -1)` is the "use per-vertex color" sentinel (the TSDF fallback);
  any other `uv0` samples the atlas. This is unchanged from the static path.
- **The command**: set `instanceCount = 1` and `firstInstance = 0`
  (`firstInstance != 0` would need the `drawIndirectFirstInstance` feature).
  `indexCount`, `firstIndex`, `vertexOffset` are yours — write them GPU-side from
  the marching-cubes output.
- **Sub-allocation**: pack many meshes into shared pools either way — byte
  offsets on `LiveMesh` (`vertex_offset`/`index_offset`/`indirect_offset`) at
  bind time, or `firstIndex`/`vertexOffset` inside the command. The two compose
  **additively** on the same buffer, so set at most one per axis (using both
  double-counts and reads past the mesh). All three byte offsets must be
  4-aligned — `index_offset`/`indirect_offset` per core Vulkan, `vertex_offset`
  per MoltenVK/Metal.

**Device features:** a single indirect draw (`drawCount = 1`) is core Vulkan
1.0. **PR1 adds no new `DeviceRequirements`** — nothing new to merge into the
shared/adopted device's feature set.

## 3. Zero-copy via `Device::adopt`

The three buffers may be VMA-allocated by recon on the **same `VkDevice` the
renderer adopted** (PR #85). gfx binds their handles directly — no external-memory
import, no staging copy. This is the same-API case the device-adopt decision
anticipated (none of the CUDA/Metal interop machinery applies).

## 4. Answers from recon

These were open when this was first written. recon has since settled all four,
across `volumetric_kit_recon` #47 (buffer sharing + barrier visibility), #48
(the arena slot ring) and #49 (the indirect command). Recorded here because the
contract below only makes sense alongside them.

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

`tests/hybrid_mesh_pipeline_test.cpp::IndirectDrawMatchesDirectDraw` renders the
same mesh as a static `GpuMesh` (direct `vkCmdDrawIndexed`) and as a `LiveMesh`
(indirect), and asserts **byte-identical** framebuffers — under the validation
layer with teeth. `tests/live_mesh_test.cpp` covers the value-type contract
(`valid()` gates on all three buffers).

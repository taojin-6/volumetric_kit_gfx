# Live mesh handoff — `volumetric_kit_recon` → `gfx` (PR1: indirect draw)

**Status:** proposal for review by the reconstruction side. This is the first of
the three slices that complete the live zero-copy path (indirect draw →
per-slot atlas ringing → `app::StreamedApp` driver). PR1 lands **only the draw
mechanics**; the synchronization and lifetime seams it depends on are called out
below as open questions for recon to weigh in on before PR2/PR3.

Recon agent: the questions in **§4** are the ones your feedback most changes.

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
  bind time, or `firstIndex`/`vertexOffset` inside the command. Both honored.

**Device features:** a single indirect draw (`drawCount = 1`) is core Vulkan
1.0. **PR1 adds no new `DeviceRequirements`** — nothing new to merge into the
shared/adopted device's feature set.

## 3. Zero-copy via `Device::adopt`

The three buffers may be VMA-allocated by recon on the **same `VkDevice` the
renderer adopted** (PR #85). gfx binds their handles directly — no external-memory
import, no staging copy. This is the same-API case the device-adopt decision
anticipated (none of the CUDA/Metal interop machinery applies).

## 4. Open questions for recon (please review)

1. **Synchronization ownership.** gfx's `record_draw` inserts **no barrier** — it
   assumes the producer's writes are already visible to
   `VK_PIPELINE_STAGE_VERTEX_INPUT_BIT` (vertex+index) and
   `VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT` (the command). Where should the
   producer→draw dependency live?
   - **Same queue** (recon compute + gfx graphics on one queue): a
     `vkCmdPipelineBarrier` (`src = COMPUTE_SHADER`, `dst = VERTEX_INPUT |
     DRAW_INDIRECT`; access `SHADER_WRITE → VERTEX_ATTRIBUTE_READ | INDEX_READ |
     INDIRECT_COMMAND_READ`). Who records it — recon before handoff, or gfx at
     the top of `submit()` given a producer-stage hint?
   - **Separate queues**: a timeline semaphore. What signals it, and does recon
     want gfx's `StreamedApp` (PR3) to own the wait value, or expose the seam?

   **This is the decision that shapes PR3.** My default: `StreamedApp` owns a
   timeline-semaphore handshake and the barrier, and `LiveMesh` stays a pure
   recorder. Does that fit recon's submission model?

2. **Buffer lifetime / double-buffering.** recon must keep each frame's buffers
   alive until gfx's frame retires (its fence signals). PR2 adds per-slot
   ringing (double-buffer + `VkFence`-completion token via `RetireQueue`) so
   recon can write slot B while gfx reads slot A — **and deliberately does NOT
   use `MTLSharedEvent`** (MoltenVK's export is buggy; see CLAUDE.md). How many
   in-flight geometry slots does recon want to ring (2 minimum)?

3. **Who fills the command?** PR1 assumes recon writes the whole
   `VkDrawIndexedIndirectCommand` GPU-side (marching cubes emits `indexCount`).
   If recon would rather hand over just a count buffer, say so — gfx can adapt
   the seam.

4. **Index width.** Fixed at `UINT32` (matches `GpuMesh` and recon's current
   output). If 16-bit is ever wanted it's a one-field add — flag it now if so.

## 5. Proof (this PR)

`tests/hybrid_mesh_pipeline_test.cpp::IndirectDrawMatchesDirectDraw` renders the
same mesh as a static `GpuMesh` (direct `vkCmdDrawIndexed`) and as a `LiveMesh`
(indirect), and asserts **byte-identical** framebuffers — under the validation
layer with teeth. `tests/live_mesh_test.cpp` covers the value-type contract
(`valid()` gates on all three buffers).

# Shared repository instructions

This is the canonical working guide for Codex, Claude Code, and other agents.
Keep shared instructions here. `CLAUDE.md` imports this file with `@AGENTS.md`;
do not duplicate rules there or create a circular reference.

## Project and boundaries

`volumetric_kit_gfx` is a standalone Vulkan renderer for volumetric workloads,
using MoltenVK on Apple and one GLSL → SPIR-V shader set across platforms.

- Keep rendering on Vulkan. A native-Metal renderer is a fallback gated on
  device validation, not a parallel implementation to add by default.
- Keep the renderer independent of compute consumers. Shared Vulkan devices
  use the create/adopt seam; CUDA/Metal consumers have their separate interop
  contracts. See the [recorded decisions](DECISIONS.md#locked-decisions).
- Descriptor layouts come from spirv-cross reflection. Use per-pipeline
  submission structs; there is no library scene graph or global frame type.
- Reach Vulkan through `core/vulkan.hpp` and the link-time loader. Keep VMA
  and backend details out of public headers.
- Use volumetric_kit_core's types by their own names -- `core::Buffer` inside
  gfx, `vkc::Buffer` in tests and examples -- and include the core's headers.
  Add no `vg::` alias for a core type
  ([DECISIONS.md](DECISIONS.md#2026-10-04--memory-comes-from-volumetric_kit_core)).

## Read what the task needs

| Task | Read |
| --- | --- |
| Build, dependencies, public consumption | [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), relevant CMake files |
| Architecture and locked choices | [DECISIONS.md](DECISIONS.md) and the affected tier's public headers |
| Recon live mesh, ownership, synchronization | [Integration contract](docs/integration/recon-live-mesh.md) |
| What has landed | [CHANGELOG.md](CHANGELOG.md) and the current implementation/tests |
| Detailed local roadmap | The primary checkout's gitignored `DESIGN.md`, if available and relevant |

`DESIGN.md` is intentionally local-only. From `.worktrees/<branch>/`, read
`../../DESIGN.md` if needed; do not force-add or publish it. Public contracts
and the committed decision record remain available to fresh clones. Read the
sections relevant to the task rather than every document before every edit.

## Naming conventions

- Package/repo: `volumetric_kit_gfx`
- Namespace: `volumetric_kit::gfx`. Internally and in docs, `vg::` abbreviates
  `volumetric_kit::gfx::`.
- Headers: `include/volumetric_kit/gfx/<tier>/…` → e.g. `#include "volumetric_kit/gfx/core/context.hpp"`
- CMake: `find_package(volumetric_kit_gfx)`; component targets `volumetric_kit::gfx_core`,
  `…_passes`, `…_pipelines`, `…_app` (+ `…_windowing`, `…_interop`, `…_assets`, `…_io`,
  `…_camera`, `…_ui`); umbrella alias `volumetric_kit::gfx`.

## Architecture

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

## Key gotchas

- ✅ MoltenVK supports `VK_KHR_fragment_shader_barycentric` (Apple6+ HW gate) — techniques
  relying on per-triangle barycentric interpolation port cleanly.
- ⚠️ MoltenVK's external-semaphore export to `MTLSharedEvent` is buggy/incomplete. Do NOT build
  compute↔render sync on it. **Double-buffer the shared resource + release via a
  `VkFence`-completion CPU token** instead.

## RAII resource types

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
  the public header (as the core's `Buffer`/`Image` do). Reset the moved-from `deleter_` to
  `nullptr` explicitly — a moved-from `std::function` is valid-but-unspecified and can
  otherwise run twice. The producing owner (e.g. the device) must outlive the resource:
  state that in an `@warning` and point at `RetireQueue` for fence-gated destruction.
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

## Conventions

- C++17, with no compiler extensions; fallible APIs use `Status` / `Result<T>`,
  volumetric_kit_core's types (`core/result.hpp`); a failed Vulkan call is
  `vk_error` / `VG_VK_TRY`, read back with `vk_result`.
- Prefer plain behavior-level tests over private-state backdoors.
- Mark deferred work inline with greppable `TODO:` comments.
- Full Doxygen on public classes/functions, matching
  `include/volumetric_kit/gfx/core/sync.hpp`: `@file`/`@brief` on headers,
  `@brief` and a `@code` example per class, and `@brief`/`@param`/`@return`
  (`@pre` where needed) on methods. An accessor may use just `/// @return`.
- Deleted copy/defaulted move declarations convey ownership; do not repeat
  "move-only" in API prose. Keep first-party code warning-clean.

## Working with Git

- Implement each task in a dedicated branch and git worktree under
  `.worktrees/`; never create a sibling worktree in the parent folder.
  Concurrent Claude/Codex tasks use separate worktrees.
- Preserve unrelated local changes and other worktrees. Remove your worktree
  with `git worktree remove` after its PR merges.
- Use absolute paths for `git -C`, `cmake -S/-B`, and file operations so work
  cannot spill into a sibling checkout.
- Use Conventional Commits, e.g. `feat(core): …`, `fix(core): …`, `docs: …`.
- Assign PRs to the authenticated user (`gh pr create --assignee @me`).
- Commit shared decisions and handoff context with the project so either agent
  can continue the task.

## Build and validation

Run from the task's worktree. The build requires CMake ≥ 3.21, a C++17 compiler, and the Vulkan toolchain;
see README and CI for platform dependencies.

```sh
gfx_root="$(git rev-parse --show-toplevel)"
cmake -S "$gfx_root" -B "$gfx_root/build" \
  -DCMAKE_BUILD_TYPE=Release -DVG_BUILD_TESTS=ON
cmake --build "$gfx_root/build" --parallel
ctest --test-dir "$gfx_root/build" --output-on-failure
pre-commit run --all-files --show-diff-on-failure
git -C "$gfx_root" diff --check
```

- Use the pinned formatting tools in `.pre-commit-config.yaml`; CI runs the
  same hooks. Set the build type explicitly and report it with measurements.
- For code changes, build affected targets and run relevant tests, with
  regression coverage for changed behavior. Broaden validation when shared
  contracts or the change's scope warrant it.
- Documentation-only changes need formatting, link, and consistency checks;
  no build is needed. Hooks can be scoped with `pre-commit run --files`.
- GPU tests can skip when no Vulkan device is present. Report actual GPU
  coverage separately from host-only success. Sanitizer CI requires lavapipe.
- `VG_WITH_GLFW` gates examples, not library windowing. `VG_WITH_CUDA` is
  reserved until its interop implementation is wired; verify current CMake.

## Keeping guidance current

- Keep this guide concise (roughly 100–200 lines). Put detailed contracts and
  rationale in DECISIONS.md or the relevant committed integration document.
- Update a changed decision and its rationale in the same commit as the code,
  then update any essential rule here and affected README/contributor guidance.
  Preserve the distinction between locked choices, open questions, and plans.
- Keep local-only DESIGN.md local. `CLAUDE.md` stays an import; add only
  Claude-specific instructions below it if a concrete need arises.

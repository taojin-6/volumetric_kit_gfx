# Changelog

All notable changes to `volumetric_kit_gfx` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Build system foundation: top-level CMake (C++17), `volumetric_kit::gfx_core` library
  target + `volumetric_kit::gfx` umbrella alias, and first-class install/export so the
  library is consumable via both FetchContent and `find_package(volumetric_kit_gfx)`.
- Dependency strategy (hybrid): `VMA` and `spirv-cross` vendored via pinned FetchContent;
  `Vulkan` from the system. VMA is implemented in one translation unit and the
  Vulkan loader is linked PRIVATE, so neither leaks into the public API.
- GPU-capable CI: lavapipe (Linux software Vulkan) + MoltenVK (macOS) so headless tests
  can run on GitHub-hosted runners; GPU tests skip gracefully when no device is present.
- GoogleTest-based `tests/` with a version smoke test.
- `ShaderModule`: an RAII wrapper around `VkShaderModule`, created from a SPIR-V blob with
  validation (rejects null/empty/misaligned code) before Vulkan is touched.
- GLSL→SPIR-V build step: `vg_compile_shaders()` (in `cmake/vg_shaders.cmake`) compiles
  shaders via glslc/glslangValidator; the first shaders are `shaders/triangle.{vert,frag}`.

### Changed

- `core`: **gfx builds on the system's Vulkan headers**, as the family's core
  does, and no longer vendors Vulkan-Headers or Vulkan-Utility-Libraries. The
  core is pinned at its PR #13 with its vulkan tier on and linked PUBLIC: its
  `format.hpp` replaces gfx's internal `vkuFormat*` helpers, and its `VkResult`
  bridge (`vk_error`, `vk_result`, `to_string(VkResult)`, `VKC_VK_TRY`) and
  shader build functions (`vkc_compile_shaders` / `vkc_embed_shaders`) replace
  gfx's copies, under the same `vg::` names and `VG_VK_TRY`. Migrating:
  - Vulkan headers 1.3.204 or newer are required, 1.3.208 on Apple; older ones
    fail at configure or in `core/vulkan.hpp`, naming the version needed.
  - An application that fetches the core before gfx sets `VKC_WITH_VULKAN ON`
    first and pins the core at or after gfx's pin, or gfx refuses to configure,
    saying which. The installed package likewise refuses a core without the
    tier.
  - A texture upload or offscreen readback of a vendor or EXT extension's
    format is refused (`Unsupported`); core and KHR formats are unchanged.
  - `create_image` with `with_view`, and `upload_texture`, refuse a format
    whose view needs a sampler Y'CbCr conversion (multi-planar, 4:2:2, RGBA
    4PACK16) with `Unsupported`, instead of creating an invalid view.
  - `vg_compile_shaders` and `vg_embed_shaders` are gone; call the core's
    functions with `TARGET_ENV vulkan1.3`.
- `core`: **error handling comes from `volumetric_kit_core`**, fetched pinned by
  commit, linked PUBLIC and re-found by the installed package. `vg::Status` and
  `vg::Result` are the core's types, so they pass to recon and calib unchanged.
  Migrating:
  - `Status::Code::Vulkan` is `Status::Code::Backend`, and the `VkResult` is its
    `detail()`. Read it with `vg::vk_result(status)`, an `std::optional<VkResult>`
    that is empty for other domains, instead of `status.code()`; build one with
    `vg::vk_error(result, what)` instead of `Status::error`. An exhaustive `switch`
    over the codes also needs `Status::Code::Numerical`.
  - `vg::to_string` names the core's `to_string`, for a `Status::Code` as
    for a `VkResult`; an unqualified `to_string(status.domain())` finds it
    too, by argument-dependent lookup.
  - recon's CUDA failures are `Status::Code::Backend` too, and `vk_result` reads
    their `cudaError_t` as an unrelated `VkResult`: ask it only of a status from
    a Vulkan call.
  - A log handler takes `(level, source, message)`; gfx's messages carry source
    `"vg"`, and `vg::log_message(level, message)` is unchanged.
  - `Status` and `Result` are `[[nodiscard]]`; handle or `(void)` a dropped one
    (the examples now report a failed `wait_idle`).
  - `Status::backend_error(0, …)` and `vk_error(VK_SUCCESS, …)` abort;
    `Status::with_context` prefixes a message and keeps the domain and detail.
  - `std::move(r).value()` and `*std::move(r)` return the value, not a reference
    into `r`; `Result`'s success constructor refuses a pointer for `Result<bool>`
    and `nullptr` for a string-like `T`.
  - `set_log_handler` returns only once no other thread is still in the previous
    handler, so a handler must not wait for a thread that may call it.
  - `VG_TRY` / `VG_ASSIGN` / `VG_CHECK` are the core's `VKC_*` macros under gfx's
    names. A failed `VG_CHECK`, or reading the value of an error `Result`, logs
    with source `"core"` (`[core error] contract check failed: …`, formerly
    `[vg error]`), so a handler that keeps only `"vg"` misses them.

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

- `core`: **error handling comes from `volumetric_kit_core`**, fetched pinned by
  commit, linked PUBLIC and re-found by the installed package. `vg::Status` and
  `vg::Result` are the core's types, so they pass to recon and calib unchanged.
  Migrating:
  - `Status::Code::Vulkan` is `Status::Code::Backend`, and the `VkResult` is its
    `detail()`. Read it with `vg::vk_result(status)`, an `std::optional<VkResult>`
    that is empty for other domains, instead of `status.code()`; build one with
    `vg::vk_error(result, what)` instead of `Status::error`. An exhaustive `switch`
    over the codes also needs `Status::Code::Numerical`.
  - `vg::to_string(Status::Code)` is gone: call `to_string(status.domain())`
    unqualified, which finds the core's by argument-dependent lookup.
    `vg::to_string(VkResult)` is unchanged.
  - recon's CUDA failures are `Status::Code::Backend` too, and `vk_result` reads
    their `cudaError_t` as an unrelated `VkResult`: ask it only of a status from
    a Vulkan call. Where the core's vulkan tier is also included, qualify it as
    `vg::vk_result`, as the core's `vk_result` is found too.
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

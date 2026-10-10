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
- `core`: `cmd_buffer_barrier` / `BufferBarrierDesc`
  (`core/buffer_barrier.hpp`), the buffer counterpart of `cmd_image_barrier`.
- `pipelines`: `ImagePipeline` and `ImageTexture` (`pipelines/image_pipeline.hpp`)
  draw 2D pictures -- RGBA, grey, a color ramp, or NV12 from a decoder -- into
  rectangles of a render target. Each update copies the source's planes (images
  or buffers), converts them to display color in an `_SRGB` image and rebuilds
  its mip chain, so a picture shrunk by window scaling or zoom averages in
  linear light; draws minify trilinearly and magnify nearest or linear.
- `camera`: `ImageView2D` (`camera/image_view_2d.hpp`), a pan-and-zoom view of
  a 2D image in a viewport, mapping image coordinates to target pixels and
  back.
- `core`: `cmd_generate_mips`, `mip_level_count` and `mip_level_extent`
  (`core/mip_chain.hpp`) build a mip chain on the GPU from level 0, for an
  image filled any way; `UploadBatch`'s `generate_mips` uses them.
- `core`: `cmd_image_barriers` (`core/image_barrier.hpp`) records several
  image transitions as one `vkCmdPipelineBarrier`, under the union of their
  stage masks; `cmd_generate_mips` hands a finished chain to the samplers
  with it, in one barrier rather than one per level.
- `examples/04_image`: a zone plate drawn through `ImagePipeline`, re-converted
  every frame, with scroll-to-zoom, drag-to-pan and resize.
- `pipelines`: `PbrMaterial::create_all` builds many materials on one upload:
  their factors share one uniform buffer, each at a 256-byte-aligned offset,
  uploaded by one copy. `PbrModel::create` builds its materials this way.
- `windowing`: **frames are numbered on a timeline.** `Frame::number` counts
  from 1, and `FrameLoop::timeline()` reaches a frame's number once its work
  completes; `completed()` reads it, and `submitted()` gives the newest frame
  submitted. `Frame::waits` (`FrameWait`: a `core::TimelinePoint` and the
  stages that wait) and `Frame::signals` add timeline waits and signals to
  `end_frame`'s submit, so another queue's work can feed a frame on the GPU.
  `end_frame` checks them as the core's submits check theirs, refuses a wait
  for a value not yet submitted to be set, and adds what each frame sets to
  the core's record of submitted values.
- `core`: `record_image_update` and `record_image_upload`
  (`core/image_update.hpp`) record an image's update into a frame's command
  buffer between the transitions that order it; an upload stages its pixels
  through a buffer a `RetireQueue` frees once the frame completes.
- `pipelines`: `StreamedAtlas` (`pipelines/streamed_atlas.hpp`), the atlas a
  live `HybridMeshPipeline` mesh samples: a ring of images reused by frame
  number, updated by copies recorded into the frame -- tiles from device
  buffers (`record_update`) or host pixels (`record_upload`) -- bound with
  `use(frame.number)`, and given back with `discard(frame.number)` for a frame
  that fails. At least two slots are required. A frame that starts with a
  picture can record at most `slots - 1` updates, preserving that picture
  for discard even if the frame's commands ran.
- `pipelines`: `kHybridMeshVertexColor` draws every triangle in its vertex
  color, the atlas bound or not.

### Changed

- `pipelines`: **`HybridMeshPipeline` owns a fallback atlas**, and a frame
  with no atlas draws in vertex color instead of drawing nothing. Migrating:
  `HybridMeshPipeline::create(device.handle(), layout)` →
  `create(device, allocator, layout)`; a 1x1 set bound only to satisfy the old
  precondition can go (pass `VK_NULL_HANDLE`).
- `core`: **`RetireQueue` is keyed on timeline values**, not `VkFence`s.
  Migrating: `RetireQueue(device.handle())` → `RetireQueue(timeline)` on a
  `core::TimelineSemaphore` -- `FrameLoop::timeline()` for what frames use --
  and `push(fence, deleter)` → `push(value, deleter)`, with the value the
  guarding work sets (a `Frame::number`).
- `windowing`: `end_frame` refuses a `Frame` other than the one
  `begin_frame` handed out (another number or command buffer), and a frame
  that fails before its submit is replaced by an empty submit, so its number
  is still set; the next extent-taking `begin_frame` rebuilds the swapchain to
  release the frame's unpresented image.
- build: gfx pins volumetric_kit_core at its PR #18 (`5913731`) and refuses an
  older core at configure and in the installed package.
- `core`: **labels record through the core's device, and queue labels take
  the queue's mutex.** `QueueLabelScope` labelled the queue without
  `Device::submit_mutex`, racing any submit on another thread or from another
  library sharing the queue. Migrating:
  - `DebugLabelScope(cmd, table, name, color)` →
    `DebugLabelScope(device, cmd, name)`; `name` must outlive the scope.
  - `QueueLabelScope(queue, table, name, color)` →
    `QueueLabelScope(device, name)`, labelling `device.queue()`. Do not open
    or close one while holding `device.submit_mutex()`.
  - `vg::set_object_name(vk_device, table, type, handle, name)` →
    `device.set_object_name(type, handle, name)`.
  - `vg::debug_utils` and `DebugUtilsTable` are gone, and labels take no
    color; `device.debug_labels_available()` says whether labels emit.
- `core`: **gfx writes the core's names; its re-exports of them are gone.**
  gfx's API names volumetric_kit_core's types and macros as the core does,
  so the `vg::` names for them and the `VG_*` macros are removed. Migrating:
  - `VG_TRY` / `VG_ASSIGN` / `VG_CHECK` / `VG_VK_TRY` → `VKC_TRY` /
    `VKC_ASSIGN` / `VKC_CHECK` / `VKC_VK_TRY`.
  - `vg::Status`, `Result`, `vk_error`, `vk_result`, `to_string`,
    `Instance`, `InstanceConfig`, `PhysicalDeviceInfo`, `Device`,
    `AdoptedDevice`, `DeviceRequirements`, `EnabledFeatures`,
    `DeviceSupport`, `merge`, `check_device_support`, `LogLevel`,
    `LogHandler` and `set_log_handler` → `vkc::` (`volumetric_kit::core::`).
    gfx's own names stay: `vg::device_requirements()`, `vg::log_message`,
    `vg::kLogSource`.
  - Headers: `gfx/core/result.hpp` → `core/base/result.hpp` (and
    `core/vulkan/vk_result.hpp` for the `VkResult` bridge and `VKC_VK_TRY`);
    `gfx/core/check.hpp` → `core/base/check.hpp`; `gfx/core/instance.hpp`,
    `physical_device_info.hpp` and `vulkan.hpp` → the core's
    `core/vulkan/` headers of the same names; `gfx/core/device.hpp` →
    `core/vulkan/device.hpp`, with `device_requirements()` now in
    `gfx/core/device_requirements.hpp`. All are under
    `volumetric_kit/`.
- `core`: **sync, descriptors, commands and queries are volumetric_kit_core's.**
  gfx's `Fence`, `Semaphore`, `TimelineSemaphore`, `DescriptorSetLayout`,
  `DescriptorPool`, `DescriptorSet`, `CommandPool`, `CommandBuffer`,
  `QueryPool` and `UniqueHandle` are gone, with their headers. Migrating:
  - `vg::Fence` and the rest → `vkc::` (`volumetric_kit::core::`), from the
    core's `volumetric_kit/core/vulkan/sync.hpp`, `descriptor.hpp`,
    `command_pool.hpp`, `command_buffer.hpp`, `query_pool.hpp` and
    `unique_handle.hpp`, which replace gfx's headers of the same names.
  - `TimelineSemaphore::create(device.handle(), v)` →
    `create(device, v)`: it takes the `Device`, and refuses one that did not
    enable timeline semaphores.
  - A `DescriptorSet` reads as empty (`handle()` null) once its pool is
    destroyed; keep the pool alive while the set is used, as before.
  - `DescriptorSet::write_*` aborts on a null `VkBuffer` or `VkImageView`
    instead of passing it to Vulkan. To clear a binding under
    `VK_EXT_robustness2`'s `nullDescriptor`, call `vkUpdateDescriptorSets`
    directly. `DescriptorSet(VkDevice, VkDescriptorSet)` is no longer
    `noexcept`.
  - `QueryPool::create` accepts only `VK_QUERY_TYPE_TIMESTAMP` and
    `VK_QUERY_TYPE_OCCLUSION`; any other type (pipeline statistics, say)
    returns `Status::Code::Unsupported`. Create such a pool with
    `vkCreateQueryPool`.
  - `FrameMetrics::Section` → `vkc::StageRow` (the same four fields), from
    the core's base tier. `vg::ticks_to_ms` / `timestamp_delta` →
    `vkc::ticks_to_ms` / `timestamp_delta`, from
    `volumetric_kit/core/vulkan/gpu_timer.hpp`.
  - `Profiler::cpu_scope` / `gpu_scope` record a null name as `"(unnamed)"`,
    as a `StageRow`'s name is never null; a null-named GPU scope now carries
    that debug-utils label instead of none.
  - `Profiler::create` refuses a `ProfilerConfig` whose
    `frames_in_flight * max_gpu_sections_per_frame * 2` does not fit in
    32 bits (`Status::Code::InvalidArgument`), where it sized a wrapped,
    too-small timestamp pool.
  - `ShaderModule` and its reflection stay gfx's, now over the core's module.
- `core`: **memory is volumetric_kit_core's.** gfx's API takes and returns the
  core's `Allocator`, `Buffer` and `Image` (`volumetric_kit::core`, which the
  family aliases `vkc`), so buffers and images pass to recon unchanged; gfx
  keeps no names of its own for them, and no longer builds its own VMA.
  Migrating:
  - `vg::Allocator`, `BufferDesc`, `Buffer`, `MemoryUsage`, `HostAccess`,
    `HeapStats` and `MemoryStats` → `vkc::` (`volumetric_kit::core::`), from
    `volumetric_kit/core/vulkan/allocator.hpp`; `vg::Texture` /
    `TextureDesc` → `vkc::Image` / `vkc::ImageDesc`, from
    `volumetric_kit/core/vulkan/image.hpp`. gfx's `core/allocator.hpp`,
    `core/buffer.hpp` and `core/texture.hpp` are gone.
  - `MemoryUsage::DeviceLocal` and `Auto` → `DeviceOnly` (the default), which
    never falls back to host memory: a full heap fails the allocation with
    `VK_ERROR_OUT_OF_DEVICE_MEMORY`, and a buffer past the heap's budget is
    refused. `HostVisible` with `mapped = true` → `Staging` for a buffer only
    copied to or from (usage `TRANSFER_SRC` / `TRANSFER_DST` at most), or
    `DeviceMapped` for one the host writes and shaders read in place, which a
    device without device-mapped memory refuses (`Unsupported`;
    `device.caps().device_mapped_memory()` says beforehand).
  - `BufferDesc::mapped` is gone: `Staging` and `DeviceMapped` buffers are
    always mapped and coherent, `DeviceOnly` never. `host_access` defaults to
    `SequentialWrite`; a buffer the host reads (a readback) sets `Random`,
    which a device-mapped buffer on a discrete GPU refuses. `BufferDesc` and
    `ImageDesc` take `queue_families` for a resource another queue family
    uses (concurrent sharing).
  - `ExternalHandleType` and the `external` fields are gone; export a buffer
    with the core's `create_exported_buffer`.
  - `ImageDesc` has no `memory`. `texture.image()` →
    `handle()`; `extent()` returns a `VkExtent3D`, so a 2D extent is
    `{image.width(), image.height()}`. An image records its layout
    (`layout()` / `set_layout`). `upload_texture`, `bake_ibl`,
    `bake_brdf_lut(device, ...)` and `PbrModel` return images recording
    `SHADER_READ_ONLY_OPTIMAL`; an image from `UploadBatch::add` records
    `UNDEFINED` until its owner records `SHADER_READ_ONLY_OPTIMAL` once
    `finish` returns OK.
  - `HeapStats` adds `reserved_bytes` and `allocation_bytes`, the allocator's
    own share; with `VK_EXT_memory_budget` enabled, `usage_bytes` counts every
    allocation in the process, as `FrameMetrics::memory_used_bytes` then does.
  - Buffers and images may outlive the allocator that made them; only the
    device must outlive them.
  - `PbrScene::set_camera(slot, eye, lod)` → `set_camera(cmd, slot, eye,
    lod)`, called before the frame's rendering begins: the camera uniform
    buffer is device-only and the write is an update recorded into `cmd`, so
    it applies to the work recorded after it -- two views in one command
    buffer, each after its own call, see their own camera. A null `cmd` fails
    the contract check.
  - `PbrMaterial::create(device, allocator, layout, desc)` →
    `create(device, batch, layout, desc)`: the factors upload through an open
    `UploadBatch`, into device-only memory; keep the material alive until the
    batch has finished, and draw it after. A failed create leaves the batch
    unchanged.
- `core`: **the instance and device are volumetric_kit_core's.** `vg::Instance`,
  `InstanceConfig`, `PhysicalDeviceInfo`, `Device`, `AdoptedDevice` and
  `DeviceRequirements` name the core's types, so a device gfx makes is the type
  recon adopts. Migrating:
  - `DeviceConfig` → `DeviceRequirements`, starting from
    `vg::device_requirements()` (Vulkan 1.3, graphics queue, dynamic rendering,
    timeline semaphores): `needs_present` and `features` keep their names,
    `extra_device_extensions` → `extensions` (owned strings), and
    `needs_external_memory` → add `VK_KHR_external_memory_fd` and
    `VK_KHR_external_semaphore_fd` to `extensions`. `enable_debug_utils` is
    gone: the device follows its instance. `Device::requirements(config)` →
    `device_requirements()`; `HeadlessAppConfig::device` and
    `WindowedAppConfig::device` are `DeviceRequirements`.
  - `instance.select_physical_device(surface)` →
    `select_physical_device(reqs, surface)`, returning a `PhysicalDeviceInfo`
    for the best device meeting `reqs`; `Device::create(instance.handle(),
    physical, config, surface)` → `Device::create(instance, info, reqs,
    surface)`; `instance.query_physical_device(p)` →
    `PhysicalDeviceInfo::query(p, instance.api_version())`.
  - `InstanceConfig::extra_instance_extensions` → `extensions`, and
    `enable_debug_utils` → `request_debug_utils`, now on by default.
  - `Device::graphics_queue()` / `graphics_family()` /
    `graphics_timestamp_valid_bits()` → `queue()` / `queue_family()` /
    `timestamp_valid_bits()`. `command_pool()` is gone: make a `CommandPool` on
    `queue_family()`, or use `submit_single_time`. `device.debug_utils()` is
    gone: the label scopes take the device (above).
    `PhysicalDeviceInfo::features2()` is gone (`features()` and the
    `supports_*` flags remain).
  - `AdoptedDevice`: `graphics_family` / `graphics_queue` → `queue_family` /
    `queue`; `enabled_device_extensions` / `_count` → `enabled_extensions` /
    `_count`; `enabled_features`, `enabled_timeline_semaphore` and
    `enabled_dynamic_rendering` → `enabled_features.core`,
    `.timeline_semaphore` and `.dynamic_rendering`. Set `instance_api_version`,
    and `present_mutex` beside `submit_mutex` for a shared present queue.
    `WindowedApp::adopt` refuses an unset `instance_api_version` before it
    runs the surface factory.
  - The renderer's floor holds however a device arrives: `HeadlessApp` and
    `WindowedApp` merge `device_requirements()` into `config.device`, and
    `UploadBatch::begin` (so every upload helper), `Swapchain::create`,
    `FrameLoop::create`, `Profiler::create` and `ImGuiOverlay::create` return
    `Unsupported` for a device that did not enable it, such as one made for
    another library's requirements. `HeadlessApp::create` refuses
    `needs_present` before creating an instance.
  - `UploadBatch` records its uploads at `finish`, on a command pool of the
    device's, instead of a pool per batch. `ImGuiOverlay` borrows its device
    by address, as `Swapchain` and `FrameLoop` do: keep the device where it is
    for the overlay's lifetime.
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
- CI: **the Ubuntu 22.04 leg goes.** The Linux legs are Ubuntu 24.04 and
  26.04, so no leg builds on the core's 1.3.204 header floor; the oldest
  headers in CI are 24.04's 1.3.275.
- `windowing`: **`FrameLoop` is neither copied nor moved.**
  `FrameLoop::create` returns `core::Result<std::unique_ptr<FrameLoop>>`, and
  the default constructor and `valid()` are gone: a loop is never empty.
  Migrating: `loop.value().begin_frame(...)` → `loop.value()->begin_frame(...)`,
  and hold a loop by `std::unique_ptr` where it was held by value.
  `WindowedApp::frame_loop()` still returns a reference. `end_frame` refuses,
  with `InvalidArgument`, any frame but the one `begin_frame` last handed out,
  and one whose swapchain was rebuilt or emptied since.

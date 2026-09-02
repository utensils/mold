use crate::engine::LoadStrategy;
use crate::progress::ProgressReporter;
use mold_core::types::{GpuBackend, GpuSelection, GpuSelector};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

/// A CUDA memory observation failed before Mold could make a safe admission
/// decision.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeviceMemoryError {
    /// The CUDA context reported an asynchronous fault that makes every
    /// retained handle suspect. Callers must quarantine the owner and stop
    /// issuing CUDA calls until process restart.
    #[error("fatal CUDA error during {operation}: {message}")]
    FatalCuda {
        operation: &'static str,
        message: String,
    },
    /// The driver could not provide an authoritative current-free reading.
    /// Admission must fail closed; nominal capacity and host RAM are not
    /// substitutes for CUDA free VRAM.
    #[error("CUDA memory sample unavailable during {operation}: {message}")]
    Unavailable {
        operation: &'static str,
        message: String,
    },
}

impl DeviceMemoryError {
    pub fn is_fatal_cuda(&self) -> bool {
        matches!(self, Self::FatalCuda { .. })
    }
}

// ── Thread-local GPU ordinal guard ─────────────────────────────────────────
//
// Each GPU worker thread is pinned to a single ordinal. We stash that ordinal
// in a thread-local so cross-engine hotpaths can debug-assert the caller
// isn't drifting onto a sibling GPU's context.
//
// Threads without a bound ordinal (tokio blocking pool, tests) see `None`
// and the assert is skipped.

thread_local! {
    static THREAD_GPU_ORDINAL: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Bind the current thread to a GPU ordinal. Call once from each GPU worker
/// thread's entry point. Any subsequent device operation on this thread must
/// match `ordinal` (debug builds only).
pub fn init_thread_gpu_ordinal(ordinal: usize) {
    THREAD_GPU_ORDINAL.with(|c| c.set(Some(ordinal)));
}

/// Clear the thread's GPU binding. Not strictly needed in production (workers
/// run for the process lifetime) but useful for tests that reuse threads.
pub fn clear_thread_gpu_ordinal() {
    THREAD_GPU_ORDINAL.with(|c| c.set(None));
}

/// Returns the currently-bound ordinal, if any.
pub fn thread_gpu_ordinal() -> Option<usize> {
    THREAD_GPU_ORDINAL.with(|c| c.get())
}

thread_local! {
    static THREAD_VRAM_GRANT: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Bind the scheduler's admitted VRAM peak for the job this thread is about to
/// run.
///
/// The frozen execution plan owns the memory authority (`CLAUDE.md`: "Frozen
/// execution plans remain authoritative at worker dispatch"), but engines that
/// self-size against sampled free VRAM never saw it. Workers set this around a
/// dispatch and clear it afterwards; engines read it through
/// [`thread_vram_grant_bytes`]. `None` means no scheduler authority (CLI local
/// runs, tests) and engines keep their free-VRAM-only behaviour.
pub fn init_thread_vram_grant_bytes(bytes: u64) {
    THREAD_VRAM_GRANT.with(|c| c.set(Some(bytes)));
}

/// Clear the thread's VRAM grant. Workers call this when a dispatch ends so a
/// later job can't inherit a stale grant.
pub fn clear_thread_vram_grant_bytes() {
    THREAD_VRAM_GRANT.with(|c| c.set(None));
}

/// The scheduler-admitted VRAM peak for the job on this thread, if any.
pub fn thread_vram_grant_bytes() -> Option<u64> {
    THREAD_VRAM_GRANT.with(|c| c.get())
}

/// Panic in debug builds if `ordinal` doesn't match the thread's bound GPU.
/// A mismatch means a call site is ignoring its engine's `gpu_ordinal` and
/// reaching for another GPU's context — the SD3.5/LTX-2 crash pattern.
#[inline]
pub(crate) fn debug_assert_ordinal_matches_thread(ordinal: usize, context: &'static str) {
    if cfg!(debug_assertions) {
        if let Some(expected) = thread_gpu_ordinal() {
            assert_eq!(
                expected, ordinal,
                "{context}: ordinal {ordinal} does not match this thread's \
                 bound GPU {expected} — hardcoded ordinal regression?"
            );
        }
    }
}

// ── GPU discovery ──────────────────────────────────────────────────────────

/// Discovered GPU information for multi-GPU support.
#[derive(Debug, Clone)]
pub struct DiscoveredGpu {
    pub ordinal: usize,
    /// Opaque backend-qualified identity. CUDA IDs exist only when the driver
    /// returns a UUID; ordinals are never substituted as durable identity.
    pub stable_id: Option<String>,
    /// Exact bytes returned by `cuDeviceGetUuid_v2`.
    pub raw_cuda_uuid: Option<[u8; 16]>,
    /// CUDA kind when this is a CUDA device. `None` identifies Metal.
    pub device_kind: Option<CudaDeviceKind>,
    /// Identity/discovery failure that makes an otherwise visible device
    /// unavailable for worker startup.
    pub identity_error: Option<String>,
    pub backend: GpuBackend,
    pub name: String,
    pub compute_capability: Option<(u16, u16)>,
    pub pci_bus_id: Option<String>,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaDeviceKind {
    FullGpu,
    Mig,
    UnknownCuda,
}

/// Build the public opaque identity directly from CUDA's UUID bytes.
pub fn stable_cuda_id(uuid: [u8; 16]) -> String {
    let mut id = String::with_capacity("cuda:".len() + uuid.len() * 2);
    id.push_str("cuda:");
    for byte in uuid {
        use std::fmt::Write as _;
        write!(&mut id, "{byte:02x}").expect("writing to String cannot fail");
    }
    id
}

/// A UUID-form `CUDA_VISIBLE_DEVICES` token proves the textual CUDA kind.
///
/// Numeric entries and an unset variable still preserve UUID identity, but
/// do not prove whether CUDA returned a full GPU or a MIG compute instance.
pub fn cuda_device_kind_from_visible_token(token: Option<&str>) -> CudaDeviceKind {
    let token = token.map(str::trim);
    if token.is_some_and(|value| {
        value
            .get(..4)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("MIG-"))
    }) {
        CudaDeviceKind::Mig
    } else if token.is_some_and(|value| {
        value
            .get(..4)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("GPU-"))
    }) {
        CudaDeviceKind::FullGpu
    } else {
        CudaDeviceKind::UnknownCuda
    }
}

#[cfg(feature = "cuda")]
fn cuda_visible_token(ordinal: usize) -> Option<String> {
    std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()?
        .split(',')
        .nth(ordinal)
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
}

#[cfg(feature = "cuda")]
fn raw_cuda_uuid_v2(
    context: &candle_core::cuda_backend::cudarc::driver::CudaContext,
) -> Result<[u8; 16], candle_core::cuda_backend::cudarc::driver::DriverError> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    use std::mem::MaybeUninit;

    let mut uuid = MaybeUninit::<sys::CUuuid>::uninit();
    context.preflight_raw_call()?;
    // SAFETY: `context.cu_device()` is a live CUdevice obtained by cudarc and
    // `uuid` points to writable storage of the exact CUDA ABI type.
    unsafe {
        sys::cuDeviceGetUuid_v2(uuid.as_mut_ptr(), context.cu_device()).result()?;
        Ok(uuid.assume_init().bytes.map(|byte| byte as u8))
    }
}

#[cfg(feature = "cuda")]
fn cuda_pci_bus_id(
    context: &candle_core::cuda_backend::cudarc::driver::CudaContext,
) -> Option<String> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    use std::ffi::CStr;

    let mut buffer = [0_i8; 32];
    context.preflight_raw_call().ok()?;
    // SAFETY: CUDA writes at most `buffer.len()` bytes and the CUdevice comes
    // from the live cudarc context.
    unsafe {
        sys::cuDeviceGetPCIBusId(
            buffer.as_mut_ptr(),
            buffer.len() as i32,
            context.cu_device(),
        )
        .result()
        .ok()?;
        CStr::from_ptr(buffer.as_ptr())
            .to_str()
            .ok()
            .map(str::to_string)
    }
}

/// Discover all available GPUs on the system.
pub fn discover_gpus() -> Vec<DiscoveredGpu> {
    let mut gpus = Vec::new();

    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver;
        if candle_core::utils::cuda_is_available() {
            // `CudaContext::device_count()` calls `cuInit(0)` first, which is
            // required before any driver API — bare `result::device::get_count()`
            // returns `ErrorNotInitialized` and we'd silently see zero GPUs.
            match driver::CudaContext::device_count() {
                Ok(count) => {
                    for ordinal in 0..count as usize {
                        match driver::CudaContext::new(ordinal) {
                            Ok(ctx) => {
                                let name = ctx
                                    .name()
                                    .unwrap_or_else(|_| format!("CUDA Device {ordinal}"));
                                let compute_capability =
                                    ctx.compute_capability().ok().and_then(|(major, minor)| {
                                        Some((
                                            u16::try_from(major).ok()?,
                                            u16::try_from(minor).ok()?,
                                        ))
                                    });
                                let pci_bus_id = cuda_pci_bus_id(&ctx);
                                // The UUID call is the identity authority. A
                                // failure leaves the record visible but
                                // unavailable; no ordinal identity is minted.
                                let (stable_id, raw_cuda_uuid, identity_error) =
                                    match raw_cuda_uuid_v2(&ctx) {
                                        Ok(uuid) => (Some(stable_cuda_id(uuid)), Some(uuid), None),
                                        Err(error) => {
                                            let message = format!(
                                                "CUDA UUID lookup failed for visible device {ordinal}: {error}"
                                            );
                                            tracing::warn!("{message}");
                                            (None, None, Some(message))
                                        }
                                    };
                                // `CudaContext::new` binds the calling thread
                                // to this ordinal, so this query is not
                                // affected by physical-ordinal remapping.
                                let (free, total) = ctx.mem_get_info().unwrap_or((0, 0));
                                gpus.push(DiscoveredGpu {
                                    ordinal,
                                    stable_id,
                                    raw_cuda_uuid,
                                    device_kind: Some(cuda_device_kind_from_visible_token(
                                        cuda_visible_token(ordinal).as_deref(),
                                    )),
                                    identity_error,
                                    backend: GpuBackend::Cuda,
                                    name,
                                    compute_capability,
                                    pci_bus_id,
                                    total_vram_bytes: total as u64,
                                    free_vram_bytes: free as u64,
                                });
                            }
                            Err(error) => {
                                let message =
                                    format!("failed to open CUDA device {ordinal}: {error}");
                                tracing::warn!("{message}");
                                gpus.push(DiscoveredGpu {
                                    ordinal,
                                    stable_id: None,
                                    raw_cuda_uuid: None,
                                    device_kind: Some(cuda_device_kind_from_visible_token(
                                        cuda_visible_token(ordinal).as_deref(),
                                    )),
                                    identity_error: Some(message),
                                    backend: GpuBackend::Cuda,
                                    name: format!("CUDA Device {ordinal}"),
                                    compute_capability: None,
                                    pci_bus_id: None,
                                    total_vram_bytes: 0,
                                    free_vram_bytes: 0,
                                });
                            }
                        }
                    }
                }
                Err(e) => tracing::warn!("CUDA device count failed: {e}"),
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        if candle_core::utils::metal_is_available() {
            // Metal: single device on macOS (unified memory).
            //
            // Total is installed RAM — the fixed ceiling the GPU can address.
            // Free is free + inactive, matching `free_vram_bytes`: macOS
            // reclaims inactive pages with no I/O, so budgeting on free pages
            // alone understated a 48 GB machine as ~7 GB and made the scheduler
            // reject plans that fit comfortably.
            let total = total_system_memory_bytes()
                .or_else(available_system_memory_bytes)
                .unwrap_or(0);
            let free = available_system_memory_bytes()
                .or_else(free_system_memory_bytes)
                .unwrap_or(0);
            gpus.push(DiscoveredGpu {
                ordinal: 0,
                stable_id: Some("metal:default".to_string()),
                raw_cuda_uuid: None,
                device_kind: None,
                identity_error: None,
                backend: GpuBackend::Metal,
                name: "Apple Metal GPU".to_string(),
                compute_capability: None,
                pci_bus_id: None,
                total_vram_bytes: total,
                free_vram_bytes: free,
            });
        }
    }

    gpus
}

/// Resolve startup selectors against the current visible inventory.
///
/// UUID selectors are matched against raw CUDA identity and remain stable
/// across `CUDA_VISIBLE_DEVICES` reordering. Invalid and ambiguous selectors
/// fail instead of silently starting a different device set.
pub fn resolve_gpu_selection(
    gpus: &[DiscoveredGpu],
    selection: &GpuSelection,
) -> anyhow::Result<Vec<DiscoveredGpu>> {
    match selection {
        GpuSelection::None => Ok(Vec::new()),
        GpuSelection::All => Ok(gpus
            .iter()
            .filter(|gpu| gpu.stable_id.is_some())
            .cloned()
            .collect()),
        GpuSelection::Specific(selectors) => {
            let mut selected_ids = BTreeSet::new();
            for selector in selectors {
                let matches = match selector {
                    GpuSelector::Ordinal(ordinal) => {
                        let Some(gpu) = gpus.iter().find(|gpu| gpu.ordinal == *ordinal) else {
                            anyhow::bail!(
                                "GPU ordinal {ordinal} did not match the current visible inventory; {}",
                                visible_gpu_choices(gpus)
                            );
                        };
                        if gpu.stable_id.is_none() {
                            anyhow::bail!(
                                "GPU ordinal {ordinal} is visible but has no stable CUDA identity: {}",
                                gpu.identity_error.as_deref().unwrap_or("UUID unavailable")
                            );
                        }
                        vec![gpu]
                    }
                    GpuSelector::Identifier(identifier) => {
                        if identifier
                            .get(..6)
                            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("metal:"))
                        {
                            gpus.iter()
                                .filter(|gpu| {
                                    gpu.stable_id.as_deref().is_some_and(|stable_id| {
                                        stable_id.eq_ignore_ascii_case(identifier)
                                    })
                                })
                                .collect::<Vec<_>>()
                        } else {
                            let normalized = normalize_uuid_selector(identifier)?;
                            gpus.iter()
                                .filter(|gpu| uuid_selector_matches(gpu, &normalized))
                                .collect::<Vec<_>>()
                        }
                    }
                };

                if matches.is_empty() {
                    anyhow::bail!(
                        "GPU selector '{}' did not match any device; {}",
                        gpu_selector_display(selector),
                        visible_gpu_choices(gpus)
                    );
                }
                if matches.len() > 1 {
                    anyhow::bail!(
                        "GPU selector '{}' is ambiguous; matches: {}",
                        gpu_selector_display(selector),
                        matches
                            .iter()
                            .filter_map(|gpu| gpu.stable_id.as_deref())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                if let Some(id) = matches[0].stable_id.as_ref() {
                    selected_ids.insert(id.clone());
                }
            }

            Ok(gpus
                .iter()
                .filter(|gpu| {
                    gpu.stable_id
                        .as_ref()
                        .is_some_and(|id| selected_ids.contains(id))
                })
                .cloned()
                .collect())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NvidiaUuidKind {
    AnyCuda,
    FullGpu,
    Mig,
}

struct NormalizedUuidSelector {
    hex_prefix: String,
    kind: NvidiaUuidKind,
}

fn normalize_uuid_selector(identifier: &str) -> anyhow::Result<NormalizedUuidSelector> {
    let identifier = identifier.trim();
    let (suffix, kind) = if identifier
        .get(..5)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("cuda:"))
    {
        (&identifier[5..], NvidiaUuidKind::AnyCuda)
    } else if identifier
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("GPU-"))
    {
        (&identifier[4..], NvidiaUuidKind::FullGpu)
    } else if identifier
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("MIG-"))
    {
        (&identifier[4..], NvidiaUuidKind::Mig)
    } else {
        anyhow::bail!("invalid GPU selector '{identifier}': expected cuda:, GPU-, or MIG- UUID");
    };
    let hex_prefix = suffix
        .chars()
        .filter(|character| *character != '-')
        .collect::<String>()
        .to_ascii_lowercase();
    if hex_prefix.is_empty()
        || hex_prefix.len() > 32
        || !hex_prefix.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        anyhow::bail!(
            "invalid GPU UUID selector '{identifier}': expected 1 to 32 hexadecimal digits"
        );
    }
    Ok(NormalizedUuidSelector { hex_prefix, kind })
}

fn uuid_selector_matches(gpu: &DiscoveredGpu, selector: &NormalizedUuidSelector) -> bool {
    let Some(uuid) = gpu.raw_cuda_uuid else {
        return false;
    };
    let kind_matches = match (selector.kind, gpu.device_kind) {
        (NvidiaUuidKind::AnyCuda, _) => true,
        (NvidiaUuidKind::FullGpu, Some(CudaDeviceKind::Mig)) => false,
        (NvidiaUuidKind::Mig, Some(CudaDeviceKind::FullGpu)) => false,
        (NvidiaUuidKind::FullGpu | NvidiaUuidKind::Mig, _) => true,
    };
    kind_matches && stable_cuda_id(uuid)["cuda:".len()..].starts_with(selector.hex_prefix.as_str())
}

fn gpu_selector_display(selector: &GpuSelector) -> String {
    match selector {
        GpuSelector::Ordinal(ordinal) => ordinal.to_string(),
        GpuSelector::Identifier(identifier) => identifier.clone(),
    }
}

fn visible_gpu_choices(gpus: &[DiscoveredGpu]) -> String {
    let choices = gpus
        .iter()
        .map(|gpu| {
            format!(
                "{}={}",
                gpu.ordinal,
                gpu.stable_id.as_deref().unwrap_or("<uuid unavailable>")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("visible choices: [{}]", choices)
}

/// Select the single best GPU (most free VRAM) for local CLI use.
pub fn select_best_gpu(gpus: &[DiscoveredGpu]) -> Option<&DiscoveredGpu> {
    gpus.iter().max_by_key(|g| g.free_vram_bytes)
}

// ── Metal device identity ──────────────────────────────────────────────────
//
// `Device::new_metal(ordinal)` mints a brand-new `MetalDevice` on every call.
// candle identifies a Metal device by a process-global counter
// (`metal_backend::DeviceId::new`), not by the physical GPU — its own comment
// says "the registryID is not sufficient as it identifies the GPU rather than
// the device itself" — and `MetalDevice::new` builds a fresh command queue,
// kernel cache, buffer pools and residency set each time. Two such devices for
// one GPU therefore fail `same_device`, and `Tensor::to_device` has no
// Metal->Metal arm: it bails with "not implemented yet, self.device:
// Metal(DeviceId(3)), device: Metal(DeviceId(1))".
//
// Engines resolve each component's placement independently — Z-Image asks for
// the transformer, the VAE and the Qwen3 encoder in turn — so once the
// scheduler began materializing an explicit `DeviceRef` per component, each one
// opened its own device and the first cross-component tensor move died. Sharing
// one device per ordinal is also strictly better than deduplicating the copy:
// one buffer pool and one command queue, instead of N competing for the same
// unified memory.
//
// CUDA is deliberately not memoized. `CudaDevice::new` takes
// `context.per_thread_stream()`, so a cached device would leak one thread's
// stream to another, and `to_device` already implements Cuda->Cuda.

static METAL_DEVICES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<usize, candle_core::Device>>,
> = std::sync::OnceLock::new();

/// Insert-or-clone against a memo table, creating at most one value per key.
///
/// The lock is deliberately held across `create`: two threads racing on a cold
/// key must not both construct a device, which is the whole point of the table.
/// A failed creation is not cached, so a transient error stays retryable.
fn memoized<K, V, F>(
    cache: &std::sync::Mutex<std::collections::BTreeMap<K, V>>,
    key: K,
    create: F,
) -> anyhow::Result<V>
where
    K: Ord,
    V: Clone,
    F: FnOnce() -> anyhow::Result<V>,
{
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = guard.get(&key) {
        return Ok(existing.clone());
    }
    let created = create()?;
    guard.insert(key, created.clone());
    Ok(created)
}

/// Return pooled Metal buffers for `ordinal` to the OS.
///
/// candle's `MetalDevice` owns a caching allocator: a freed buffer stays in
/// `buffers`/`private_buffers` at `strong_count == 1` and is only released by
/// `drop_unused_buffers`, which runs solely from `wait_until_completed` /
/// `flush_and_wait_current` (`metal_backend/device.rs`) — or when the device
/// itself is dropped.
///
/// Before memoization the device was the last thing holding that pool, so
/// dropping the final engine dropped the pool with it. Now the device outlives
/// every engine, and an unload has to ask for the sweep explicitly. Without it
/// `mold model unload` frees the engine but leaves the buffers checked out,
/// which is precisely what users run it to undo — and the post-drop VRAM sample
/// would report no recovery.
///
/// Only sweeps a device this process already opened; never creates one.
pub fn release_pooled_metal_memory(ordinal: usize) {
    let Some(cache) = METAL_DEVICES.get() else {
        return;
    };
    // Clone out from under the lock: `synchronize` waits on the GPU and must
    // not hold up an unrelated ordinal's device lookup.
    let device = {
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.get(&ordinal).cloned()
    };
    let Some(device) = device else {
        return;
    };
    if let Err(error) = device.synchronize() {
        tracing::warn!(
            ordinal,
            %error,
            "could not sweep the Metal buffer pool after unload; freed memory stays cached"
        );
    }
}

/// Open Metal `ordinal`, reusing this process's existing device for that GPU.
///
/// Every Metal device construction in Mold must go through here. Calling
/// `Device::new_metal` directly reintroduces the split-identity bug above.
pub fn metal_device(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    memoized(METAL_DEVICES.get_or_init(Default::default), ordinal, || {
        candle_core::Device::new_metal(ordinal)
            .map_err(|error| anyhow::anyhow!("failed to open Metal device {ordinal}: {error}"))
    })
}

// ── Device creation ────────────────────────────────────────────────────────

/// Create a device on the specified GPU ordinal.
/// Use ordinal 0 for single-GPU setups.
/// Reports device selection via the progress reporter.
pub fn create_device(
    ordinal: usize,
    progress: &ProgressReporter,
) -> anyhow::Result<candle_core::Device> {
    use candle_core::Device;
    // MOLD_DEVICE=cpu forces CPU inference (for debugging Metal issues)
    let force_cpu = crate::runtime_env::value("MOLD_DEVICE")
        .map(|v| v.eq_ignore_ascii_case("cpu"))
        .unwrap_or(false);
    if force_cpu {
        progress.info("CPU forced via MOLD_DEVICE=cpu");
        tracing::info!("CPU forced via MOLD_DEVICE=cpu");
        return Ok(Device::Cpu);
    }
    debug_assert_ordinal_matches_thread(ordinal, "create_device");
    if candle_core::utils::cuda_is_available() {
        progress.info(&format!("Using CUDA device {ordinal}"));
        tracing::info!("Using CUDA device {ordinal}");
        Ok(Device::new_cuda(ordinal)?)
    } else if candle_core::utils::metal_is_available() {
        progress.info(&format!("Using Metal device {ordinal}"));
        tracing::info!("Using Metal device {ordinal}");
        metal_device(ordinal)
    } else {
        progress.info("No GPU detected, using CPU");
        tracing::warn!("No GPU detected, falling back to CPU");
        Ok(Device::Cpu)
    }
}

/// Construct exactly the backend and ordinal selected by an admitted plan.
///
/// Unlike [`create_device`], this function deliberately ignores `MOLD_DEVICE`
/// and never falls back to CPU or another backend. Callers must propagate an
/// error so the scheduler can explicitly replan.
pub(crate) fn create_exact_gpu_device(
    backend: mold_core::GpuBackend,
    ordinal: usize,
    progress: &ProgressReporter,
) -> anyhow::Result<candle_core::Device> {
    debug_assert_ordinal_matches_thread(ordinal, "create_exact_gpu_device");
    match backend {
        mold_core::GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                progress.info(&format!("Using exact CUDA device {ordinal}"));
                candle_core::Device::new_cuda(ordinal).map_err(|error| {
                    anyhow::anyhow!("failed to open exact CUDA device {ordinal}: {error}")
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = progress;
                anyhow::bail!(
                    "exact CUDA device {ordinal} requested but this build has no CUDA support"
                )
            }
        }
        mold_core::GpuBackend::Metal => {
            #[cfg(feature = "metal")]
            {
                progress.info(&format!("Using exact Metal device {ordinal}"));
                metal_device(ordinal)
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = progress;
                anyhow::bail!(
                    "exact Metal device {ordinal} requested but this build has no Metal support"
                )
            }
        }
    }
}

/// Headroom above model size for activation memory during encoding.
pub const T5_ACTIVATION_HEADROOM: u64 = 2_000_000_000; // 2GB

/// Compute VRAM threshold for a T5 model of a given size.
/// The model needs its own weight size plus headroom for activations.
pub fn t5_vram_threshold(model_size_bytes: u64) -> u64 {
    model_size_bytes + T5_ACTIVATION_HEADROOM
}

/// Minimum free VRAM (bytes) required to place FP16 T5-XXL on GPU.
/// Kept for backward compatibility — equivalent to `t5_vram_threshold(9_200_000_000)`.
pub const T5_VRAM_THRESHOLD: u64 = 16_000_000_000;
/// Minimum free VRAM (bytes) required to place CLIP-L on GPU: ~246MB model + 500MB headroom.
pub const CLIP_VRAM_THRESHOLD: u64 = 800_000_000;
/// Minimum free VRAM (bytes) required to place CLIP-G on GPU: ~1.39GB model + ~1.4GB headroom.
pub const CLIPG_VRAM_THRESHOLD: u64 = 2_800_000_000;

/// Compute VRAM threshold for a Qwen3 text encoder of a given size.
/// Uses the same headroom formula as T5 (model size + 2GB activations).
pub fn qwen3_vram_threshold(model_size_bytes: u64) -> u64 {
    model_size_bytes + T5_ACTIVATION_HEADROOM
}

/// Compute VRAM threshold for a Qwen2.5-VL text encoder of a given size.
/// Uses the same headroom formula as T5/Qwen3 (model size + 2GB activations).
pub fn qwen2_vram_threshold(model_size_bytes: u64) -> u64 {
    model_size_bytes + T5_ACTIVATION_HEADROOM
}

/// Headroom above the expand LLM weights for activations + KV cache.
/// The expander generates short sequences (<= 512 tokens) so 2 GB is generous.
/// Matches `T5_ACTIVATION_HEADROOM` convention (decimal GB) for easy comparison.
pub const EXPAND_ACTIVATION_HEADROOM: u64 = 2_000_000_000;

// ── Activation-aware memory budget ───────────────────────────────────────────
//
// ComfyUI computes a per-arch `memory_required(input_shape)` (see
// `comfy/model_base.py:387-409`) instead of a fixed inference headroom. The
// fixed 3 GB / 5 GB heuristics below over-budget at small resolutions (forcing
// unneeded offload at 768²) and under-budget at large ones (causing the OOMs
// that motivated the pre-VAE force-drop in `1c276c6`). The helper here scales
// the activation budget with `area × dtype × batch × per_arch_factor`.

/// Architecture family for activation-budget purposes.
///
/// Mirrors the engine families in `crates/mold-inference/src/`. The factors
/// in [`activation_bytes`] are calibrated empirically per arch — flash /
/// memory-efficient attention reshapes the constant by a substantial factor
/// because peak attention workspace stops scaling as `B × H × N × N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFamily {
    /// FLUX v1 dit (dev / schnell / krea / kontext / fill).
    FluxDit,
    /// Flux.2 dit (Klein / Pro).
    Flux2Dit,
    /// SD3 / SD3.5 MMDiT.
    Sd3Mmdit,
    /// SDXL UNet (CFG-batched + cross-attn KV cache).
    SdxlUnet,
    /// Qwen-Image / Qwen-Image-Edit dit.
    QwenImageDit,
    /// Z-Image dit.
    ZImageDit,
    /// Wuerstchen v2 cascade (Stage C/B decoder).
    Wuerstchen,
    /// Hunyuan3D shape DiT + shape VAE. Present so the family can be NAMED in
    /// a hint; its numbers do not come from the pixel-area model at all —
    /// `mold-server/src/hunyuan3d_admission.rs` is the authority, because a
    /// mesh has no canvas and the peak is set by token counts and the decode
    /// chunk size, neither of which appears in an area.
    Hunyuan3dShape,
    /// T5 / CLIP / Qwen3 / Gemma text encoder workspace.
    SmallTransformer,
    /// LTX-Video (0.9.6 / 0.9.8 2B or 13B) video transformer. Loads the
    /// entire transformer into VRAM at generation time (not streamed). The
    /// 13B BF16 variant is ~26 GB on disk. The file-size-based preflight
    /// applies unchanged; no streaming cap override.
    LtxVideo,
    /// LTX-2 (19B / 22B) video transformer. Always loaded via
    /// the streaming block source (`new_streaming` in
    /// `crates/mold-inference/src/ltx2/model/video_transformer.rs`) — only
    /// `streaming_prefetch_count` blocks are GPU-resident at any one time,
    /// so the file size on disk (~46 GB at BF16 for the 22B preset)
    /// massively over-estimates GPU residency.
    Ltx2Video,
    /// Wan 2.1/2.2 video DiT. The currently shipped checkpoints (1.3B bf16
    /// ~2.8 GB, TI2V-5B fp16 ~10 GB) load fully GPU-resident like LtxVideo;
    /// the A14B two-expert tier arrives with quantized/streamed residency in
    /// a later layer and revisits this classification then.
    WanVideo,
}

impl ActivationFamily {
    /// Whether this family loads its transformer in a block-streaming mode
    /// (only a few blocks GPU-resident at a time, the rest mmap'd / paged).
    /// The preflight uses this to bypass the file-size-based transformer
    /// budget, which would otherwise reject 22B LTX-2 on a 24 GB card even
    /// though only ~2 GB of transformer weights are co-resident at peak.
    pub fn streaming_transformer(self) -> bool {
        matches!(self, ActivationFamily::Ltx2Video)
    }

    /// Whether this family needs the full-weight peak (transformer fully
    /// resident on GPU at inference time). Used to choose the right headroom
    /// constant in the preflight suggestion message.
    pub fn is_full_weight_video(self) -> bool {
        matches!(
            self,
            ActivationFamily::LtxVideo | ActivationFamily::WanVideo
        )
    }
}

/// Estimated activation memory (in bytes) for a single forward pass.
///
/// Mirrors ComfyUI's `model_base.memory_required(input_shape)` shape: peak
/// activation memory at fp16/bf16 scales as
/// `area × dtype_bytes × batch × per_arch_factor` where `area = h × w` in
/// image space. The factor encapsulates per-arch overhead (residuals,
/// attention KV cache, MLP intermediate, etc.). When flash /
/// memory-efficient attention is in use the factor is smaller because peak
/// attention workspace stops scaling as `B × H × N × N`.
///
/// `width`/`height` are image-space dimensions — the helper internally
/// accounts for the 8× VAE downsample via the per-family factor, so you
/// pass `req.width`/`req.height` directly, not latent dims.
/// `batch` is typically `1` for non-CFG (FLUX, Z-Image, distilled
/// schedulers) and `2` for CFG-doubled forwards (SDXL and SD3 when
/// classifier-free guidance is active).
/// `dtype_bytes` is `2` for bf16/fp16, `4` for f32, `8` for f64.
///
/// Floors at 256 MB so even tiny inputs reserve enough for kernel workspaces
/// (cuBLAS / cuDNN scratch, tokenizer / embedding buffers).
///
/// Empirical anchors:
///   * FLUX dit 1024² bf16 cfg=1   → ~273 MB
///   * FLUX dit 2048² bf16 cfg=1   → ~1.09 GB
///   * SDXL UNet 1024² bf16 cfg=2  → ~726 MB
///   * Wuerstchen 1024² bf16       → ~456 MB
pub fn activation_bytes(
    width: u32,
    height: u32,
    batch: u32,
    dtype_bytes: u32,
    family: ActivationFamily,
) -> u64 {
    let area = (width as u64).saturating_mul(height as u64);
    let bytes_per_pixel = (dtype_bytes as u64).saturating_mul(batch.max(1) as u64);
    // Per-family factor — calibrated to produce ~273 MB at 1024² bf16 cfg=1
    // for FLUX dit (just above the 256 MB floor, ~1.09 GB at 2048²), with
    // arch-specific multipliers for the heavier-attention families. Units
    // are roughly "bytes of activation per pixel per dtype-byte per batch".
    //
    // The original spec doc had factors in the 0.01 – 0.025 range, which
    // produced sub-megabyte raw budgets at 1024² (the formula's units make
    // those values meaningful only if reinterpreted as megabytes-per-
    // megapixel) — the floor would always dominate and the
    // resolution-scaling test would fail. The factors here are scaled up
    // by ~8000× to actually realize the documented ~250 MB / ~1 GB
    // targets.
    let factor: f64 = match family {
        // FLUX v1 dit: double + single block residuals dominate, flash-attn
        // collapses the B×N×N peak. 1024² → 273 MB; 2048² → 1.09 GB.
        ActivationFamily::FluxDit => 130.0,
        // Flux.2 dit: same activation shape as FLUX v1 (Klein/Pro).
        ActivationFamily::Flux2Dit => 130.0,
        // Z-Image dit: single chunk of dit blocks, similar to FLUX.
        ActivationFamily::ZImageDit => 130.0,
        // SD3 MMDiT: joint attention sits between FLUX's split and SDXL's
        // cross-attn — calibrated ~20% above FLUX.
        ActivationFamily::Sd3Mmdit => 156.0,
        // SDXL UNet: CFG runs `[uncond, cond]` and cross-attn KV is cached
        // for the prompt sequence — ~33% above FLUX. Callers pass `batch=2`
        // when CFG is active, so this factor covers per-batch overhead.
        ActivationFamily::SdxlUnet => 173.0,
        // Qwen-Image dit: similar dual-stream structure to SDXL.
        ActivationFamily::QwenImageDit => 173.0,
        // Wuerstchen v2: cascade Stage B has a chunky conv stack — ~67%
        // above FLUX.
        ActivationFamily::Wuerstchen => 217.0,
        // Never the authority — see the variant's doc. This factor exists so
        // an unexpected caller gets a FLUX-class number rather than a panic.
        ActivationFamily::Hunyuan3dShape => 130.0,
        // T5 / CLIP / Qwen3 encoders work over tokens × hidden, not pixels.
        // Image-space scaling is a soft proxy for "small workspace" — the
        // floor usually dominates for typical inputs.
        ActivationFamily::SmallTransformer => 87.0,
        // LTX-Video (0.9.6 / 0.9.8): per-frame activation is similar to FLUX
        // dit (split blocks, RMSNorm, no CFG-batched workspace). The full
        // transformer is resident on GPU during denoise, so the activation
        // budget here is an additional workspace on top of the weight peak
        // captured by the file-size estimator.
        ActivationFamily::LtxVideo => 130.0,
        // LTX-2: same per-pixel activation shape as LTX-Video. Only 1-2
        // streaming blocks are GPU-resident at a time, so the dominant
        // cost on a 24 GB card is encoder + block workspace, not the
        // full-weight sum.
        ActivationFamily::Ltx2Video => 130.0,
        // Wan video DiT: split-attention flow transformer in the same
        // shape class as the LTX families; starts at their calibration
        // until a bespoke wan admission model lands with the 14B tier.
        ActivationFamily::WanVideo => 130.0,
    };
    let raw = (area as f64 * bytes_per_pixel as f64 * factor) as u64;
    /// Sanity floor: even tiny inputs reserve ~256 MB for kernel workspaces
    /// (cuBLAS / cuDNN scratch, tokenizer / embedding buffers).
    const ACTIVATION_FLOOR_BYTES: u64 = 256_000_000;
    raw.max(ACTIVATION_FLOOR_BYTES)
}

/// Bytes per element for a given candle dtype, used to feed
/// [`activation_bytes`] from a runtime `DType`. Returns `2` for bf16/fp16 and
/// `4` for f32; integer / quantized weights still flow as bf16/fp16
/// activations during forward, so `2` is the right answer there too.
pub fn dtype_bytes(dt: candle_core::DType) -> u32 {
    use candle_core::DType;
    match dt {
        DType::BF16 | DType::F16 => 2,
        DType::F32 => 4,
        DType::F64 => 8,
        // Everything else (quantized / int storage / sub-byte floats):
        // activations during forward travel as bf16/fp16, so `2` matches
        // the runtime activation cost.
        _ => 2,
    }
}

// ── LTX-2 token-based activation budget ──────────────────────────────────────
//
// The pixel-area heuristic in `activation_bytes` is wrong for LTX-2 in both
// directions: the video transformer works over *tokens*, and the token count
// is a 3D quantity (latent frames × patch grid), not `w × h`. Charging
// `area × latent_frames` over-estimates a half-resolution stage-1 render by 4×
// and under-estimates the feed-forward peak at stage 2. The helpers below
// price the real tensors instead.

/// LTX-2 video VAE temporal compression — 8 pixel frames per latent frame,
/// plus the ungrouped first frame.
const LTX2_TEMPORAL_STRIDE: u64 = 8;
/// Pixels per token along each spatial axis: 8× VAE spatial downsample followed
/// by the transformer's 4×4 latent patchify.
const LTX2_PIXELS_PER_TOKEN_AXIS: u64 = 32;
/// Transformer inner dim (`patchify_proj` output width) for the 19B/22B
/// presets.
const LTX2_INNER_DIM: u64 = 4096;
/// Feed-forward hidden width (`ff.net.0.proj` output) — 4× the inner dim.
const LTX2_FF_HIDDEN_DIM: u64 = 16_384;
/// Per-token AdaLN width (`adaln_single.linear` output) when the checkpoint's
/// own width is not known — 6 × inner dim, the LTX-2 19B layout. Only
/// materialized per token when the run is conditioned.
///
/// This is a fallback, not the truth: LTX-2.3's 22B checkpoint ships nine
/// components (`[36864, 4096]`), so callers that have read the checkpoint
/// header pass its real width to
/// [`ltx2_activation_budget_bytes`].
pub const LTX2_ADALN_DEFAULT_DIM: u64 = 24_576;
const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;

/// Token-independent LTX-2 denoise workspace.
///
/// Covers the chunked feed-forward's two live F32 `[chunk, 16384]` staging
/// buffers, the RoPE cos/sin tables held for the whole denoise loop, the video
/// (and optional audio) latent/noise tensors, and cuBLAS/cuDNN scratch. None of
/// these grow with the token count, so they are a flat term rather than part of
/// the per-token slope.
const LTX2_FIXED_WORKSPACE_BYTES: u64 = 1_073_741_824;

/// Number of simultaneously live `[1, T, inner]` BF16 buffers in a block's
/// residual stream (input, normed, attention output, gated residual, and the
/// feed-forward's residual copy).
const LTX2_RESIDUAL_LIVE_BUFFERS: u64 = 5;

/// Video self-attention heads for the 19B/22B presets (`num_attention_heads`
/// in `ltx2/model/video_transformer.rs`; `inner_dim / head_dim` = 4096 / 128).
const LTX2_ATTENTION_HEADS: u64 = 32;

/// Score-matrix bytes one LTX-2 video self-attention holds on CUDA (#735).
///
/// The shared dispatcher (`crate::attention`) materializes the score tile in
/// the input dtype: `[heads, query_chunk, tokens]` BF16, twice over — the
/// raw scores and the softmax output both live while the second matmul runs.
/// Its `Auto` policy chunks the query axis at
/// [`crate::attention::CUDA_AUTO_QUERY_CHUNK`] rows once the sequence passes
/// [`crate::attention::CUDA_AUTO_CHUNK_MIN_QUERY_ROWS`], and otherwise takes
/// the whole matrix at once. At the 13,312-token stage-2 shape that is
/// `32 x 512 x 13,312 x 2 B x 2` = 872 MB — far above the 128 MB F32 tiles the
/// hand-rolled path budgeted, which is why this is its own term in the
/// activation budget rather than a share of the flat workspace.
/// The estimate follows the RESOLVED policy, not the default: a raised
/// `MOLD_ATTN_CHUNK`, or `off`, grows the tile the dispatcher actually
/// allocates (at 13,312 tokens `off` is a 22.7 GB full matrix that a
/// 512-row estimate would under-charge 26x), and an EFFECTIVE `flash`
/// backend holds no score matrix at all, so charging one would refuse
/// renders that fit. "Effective" is load-bearing: a requested `flash` in a
/// build without the kernels executes as math (the dispatcher warns once
/// and falls back), so the zero-tile answer is keyed on
/// `AttentionBackend::resolve_effective()`, never `resolve()` — estimating
/// from the request under-charges the math tile that actually allocates.
/// Both envs are engine-shaping fingerprint inputs, so admission and the
/// dispatcher read the same frozen decision.
pub fn ltx2_attention_tile_bytes(tokens: u64, heads: u64) -> u64 {
    if crate::attention::AttentionBackend::resolve_effective()
        == crate::attention::AttentionBackend::Flash
    {
        return 0;
    }
    let query_rows = crate::attention::cuda_query_chunk_rows(tokens as usize)
        .map(|rows| rows as u64)
        .unwrap_or(tokens);
    heads
        .saturating_mul(query_rows)
        .saturating_mul(tokens)
        .saturating_mul(BF16_BYTES)
        .saturating_mul(2)
}

/// Token count for an LTX-2 render at `width × height × frames`.
///
/// `frames` is in pixel space; the VAE groups them 8:1 after the ungrouped
/// first frame, and each latent frame contributes a `(h/32) × (w/32)` patch
/// grid.
pub fn ltx2_token_count(width: u32, height: u32, frames: u32) -> u64 {
    let latent_frames = (frames.max(1) as u64 - 1) / LTX2_TEMPORAL_STRIDE + 1;
    let grid_h = (height as u64 / LTX2_PIXELS_PER_TOKEN_AXIS).max(1);
    let grid_w = (width as u64 / LTX2_PIXELS_PER_TOKEN_AXIS).max(1);
    latent_frames.saturating_mul(grid_h).saturating_mul(grid_w)
}

/// Peak activation bytes for one LTX-2 transformer forward at this shape.
///
/// Derived from the real tensor shapes in
/// `crates/mold-inference/src/ltx2/model/video_transformer.rs`, per token of
/// the `[1, T, 4096]` hidden state:
///
/// * residual stream — 5 live `[1, T, 4096]` BF16 buffers → `5 × 4096 × 2`
/// * attention — `q`/`k`/`v` BF16 `[1, T, 4096]` plus the F32 output
///   accumulator → `3 × 4096 × 2 + 4096 × 4`. `chunked_attention` narrows and
///   upcasts one tile at a time, so its F32 staging is flat, not per-token
/// * feed-forward — the BF16 `[1, T, 16384]` projection. The F32 GELU work is
///   chunked (two `[chunk, 16384]` buffers) and therefore priced in
///   [`LTX2_FIXED_WORKSPACE_BYTES`], not per token
/// * AdaLN — the per-token `[1, T, adaln_dim]` BF16 modulation, materialized
///   only when `conditioned` (source image, keyframes, or extend carryover)
///   makes the modulation token-varying rather than a single broadcast row
///
/// `adaln_dim` is the checkpoint's own `adaln_single.linear` output width. It
/// is **not** a constant across LTX-2: the 19B ships six components
/// (`[24576, 4096]`) and LTX-2.3's 22B ships nine (`[36864, 4096]`), a 327 MB
/// difference at the stage-2 shape. Pass `None` only when the header has not
/// been read; it falls back to [`LTX2_ADALN_DEFAULT_DIM`].
///
/// * self-attention score tile — the BF16 dispatcher's chunked `[heads,
///   512, T]` scores plus their softmax, priced by
///   [`ltx2_attention_tile_bytes`]; past 1,024 tokens that is a further
///   64 KiB/token
///
/// That is 112 KiB/token unconditioned and 160 KiB/token conditioned at the
/// six-component width, plus the attention tile and the flat workspace.
/// Anchors: 1024² × 97 frames (13,312 tokens) → 3.47 GB unconditioned /
/// 4.13 GB conditioned; the 512² stage-1 render of the same job (3,328
/// tokens) → 1.68 GB.
pub fn ltx2_activation_budget_bytes(
    width: u32,
    height: u32,
    frames: u32,
    conditioned: bool,
    adaln_dim: Option<u64>,
) -> u64 {
    let residual = LTX2_RESIDUAL_LIVE_BUFFERS * LTX2_INNER_DIM * BF16_BYTES;
    // `chunked_attention` narrows q/k/v and upcasts per tile, so the only
    // per-token attention cost is the caller's three BF16 projections plus the
    // F32 output accumulator. The tile buffers are flat and priced in
    // `LTX2_FIXED_WORKSPACE_BYTES`.
    let attention = 3 * LTX2_INNER_DIM * BF16_BYTES + LTX2_INNER_DIM * F32_BYTES;
    let feed_forward = LTX2_FF_HIDDEN_DIM * BF16_BYTES;
    let adaln = if conditioned {
        adaln_dim.unwrap_or(LTX2_ADALN_DEFAULT_DIM) * BF16_BYTES
    } else {
        0
    };
    let per_token = residual + attention + feed_forward + adaln;
    let tokens = ltx2_token_count(width, height, frames);
    tokens
        .saturating_mul(per_token)
        .saturating_add(ltx2_attention_tile_bytes(tokens, LTX2_ATTENTION_HEADS))
        .saturating_add(LTX2_FIXED_WORKSPACE_BYTES)
}

// ── Wan activation model ────────────────────────────────────────────────────

/// The Wan VAE groups four pixel frames into one latent frame after the
/// ungrouped first frame, in both generations
/// (`crates/mold-inference/src/wan/model/vae.rs:1398`).
const WAN_TEMPORAL_STRIDE: u64 = 4;

/// Query-chunk width `math_attention_chunk_size` picks on CUDA
/// (`crates/mold-inference/src/attention.rs:206-218`). The chunk bounds the
/// query axis of the score matrix; the key axis stays the full token count,
/// which is what makes the score tensor a *per-token* cost.
const WAN_ATTENTION_QUERY_CHUNK: u64 = 512;

/// Live `[1, T, dim]` BF16 buffers in a block's residual stream: the block
/// input, the normalized copy, and the attention/FFN output
/// (`wan/model/transformer.rs:866-906`).
const WAN_RESIDUAL_LIVE_BUFFERS: u64 = 3;

/// Live `[1, T, dim]` buffers at a gated residual, in the *output* dtype.
///
/// This used to be three F32 buffers scaling with the clip — the upcast hidden
/// state, the upcast branch output, and their sum. `gated_residual` now
/// evaluates the add in fixed 4096-token slices (#776 item 2), so the F32
/// arithmetic no longer scales with tokens at all; what does is the BF16
/// staging around it, which is the accumulated slice list plus the `Tensor::cat`
/// that joins it — two buffers, live together at the seam.
///
/// The slice-sized F32 transients that replaced the per-token ones are a flat
/// ~335 MiB at `dim = 5120` and are absorbed by the measured slope below,
/// which is fitted as a *difference* between two frame counts and therefore
/// prices only what actually varies with tokens.
const WAN_GATED_RESIDUAL_LIVE_BUFFERS: u64 = 2;

/// Modulation components a block slices out of its timestep table —
/// shift/scale/gate for attention and for the feed-forward
/// (`transformer.rs:849-864`).
const WAN_MODULATION_COMPONENTS: u64 = 6;

/// Which VAE generation a checkpoint pairs with. Only the spatial stride
/// differs for token math: 2.1 encodes 8:1 and 2.2 16:1, and the DiT's
/// `(1, 2, 2)` patch halves each spatial axis again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WanVaeGeneration {
    /// Wan 2.1 — 16-channel latents, 8:1 spatial.
    V21,
    /// Wan 2.2 — 48-channel latents, 16:1 spatial (TI2V-5B).
    V22,
}

impl WanVaeGeneration {
    /// Spatial compression of the VAE itself, before the DiT's patch.
    pub const fn spatial_stride(self) -> u64 {
        match self {
            Self::V21 => 8,
            Self::V22 => 16,
        }
    }
}

/// Checkpoint geometry the wan activation model prices against. Read from the
/// DiT's own safetensors header, never from the model name — a repack or a
/// community fine-tune carries its shape in the file, which is the same rule
/// `wan::pipeline::detect_transformer_config` follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WanActivationGeometry {
    /// Model width (`patch_embedding.weight[0]`).
    pub dim: u64,
    /// Feed-forward width (`blocks.0.ffn.0.weight[0]`).
    pub ffn_dim: u64,
    /// Attention heads — every shipped variant uses a 128-wide head.
    pub num_heads: u64,
    /// Transformer block count, read from the highest `blocks.{i}` index.
    pub num_layers: u64,
    /// VAE generation, which sets the latent grid.
    pub vae: WanVaeGeneration,
    /// The DiT patch's spatial width (`patch_size.1`), which halves each
    /// latent axis again. Read from the checkpoint rather than assumed: every
    /// shipped variant is `(1, 2, 2)`, but a community fine-tune is exactly
    /// the case a shape-driven probe exists to cover.
    pub patch_spatial: u64,
    /// Whether the checkpoint drives per-token timesteps. TI2V's latent
    /// inpaint gives frame 0 its own timestep
    /// (`wan/conditioning.rs:331-360`), which turns every block's modulation
    /// table from one broadcast row into a full `[1, T, 6, dim]` F32 tensor.
    pub per_token_timesteps: bool,
}

impl WanActivationGeometry {
    /// Wan 2.1 T2V-1.3B.
    pub const fn t2v_1_3b() -> Self {
        Self {
            dim: 1536,
            ffn_dim: 8960,
            num_heads: 12,
            num_layers: 30,
            vae: WanVaeGeneration::V21,
            patch_spatial: 2,
            per_token_timesteps: false,
        }
    }

    /// Wan 2.1 T2V-14B and both Wan 2.2 A14B experts.
    pub const fn a14b() -> Self {
        Self {
            dim: 5120,
            ffn_dim: 13824,
            num_heads: 40,
            num_layers: 40,
            vae: WanVaeGeneration::V21,
            patch_spatial: 2,
            per_token_timesteps: false,
        }
    }

    /// Wan 2.2 TI2V-5B.
    pub const fn ti2v_5b() -> Self {
        Self {
            dim: 3072,
            ffn_dim: 14336,
            num_heads: 24,
            num_layers: 30,
            vae: WanVaeGeneration::V22,
            patch_spatial: 2,
            per_token_timesteps: true,
        }
    }
}

/// Token count for a wan render at `width × height × frames`.
///
/// `((F - 1) / 4 + 1)` latent frames, each contributing a patch grid that is
/// the frame divided by the VAE stride and again by the DiT's patch width.
pub fn wan_token_count(
    width: u32,
    height: u32,
    frames: u32,
    geometry: WanActivationGeometry,
) -> u64 {
    let latent_frames = (u64::from(frames.max(1)) - 1) / WAN_TEMPORAL_STRIDE + 1;
    let axis = geometry
        .vae
        .spatial_stride()
        .saturating_mul(geometry.patch_spatial.max(1));
    let grid_h = (u64::from(height) / axis).max(1);
    let grid_w = (u64::from(width) / axis).max(1);
    latent_frames.saturating_mul(grid_h).saturating_mul(grid_w)
}

/// Peak transformer activation bytes for one wan forward at this shape.
///
/// Derived from the tensors `WanBlock::forward` actually materializes
/// (`wan/model/transformer.rs:866-906`), priced per token of the `[1, T, dim]`
/// hidden state. A block runs two gated-residual phases; they do not overlap,
/// so the block's cost is the larger phase, not their sum:
///
/// * **self-attention phase** — the BF16 residual buffers, `q`/`k`/`v`, and the
///   F32 upcast trio at the gate.
/// * **feed-forward phase** — the same F32 trio plus the BF16 `[1, T, ffn_dim]`
///   projection, which is the wider of the two on every shipped variant.
///
/// **The score matrix is the dominant term and is per-token.** Chunking bounds
/// the *query* axis at [`WAN_ATTENTION_QUERY_CHUNK`], but the key axis is the
/// full token count, so `[heads, 512, T]` costs `heads × 512 × 4` bytes for
/// every token — 82 KiB/token at A14B's 40 heads, more than the whole residual
/// stream. The softmax result is a second tensor of the same shape.
///
/// That linear form depends on chunking being active, which
/// `math_attention_chunk_size` makes true on CUDA above 1,024 queries
/// (`crates/mold-inference/src/attention.rs:206-218`). Below that the score
/// matrix is the unchunked `[heads, T, T]`, and the two forms **coincide
/// exactly at T = 1,024** because the chunked form counts both the score and
/// the softmax copy (`2 × 512 = 1024`); under that threshold this estimate is
/// conservative, over it chunking holds. A build with `flash-attn` compiled
/// materializes no score matrix at all, so this over-estimates there — the
/// safe direction for admission. Forcing `MOLD_ATTN_CHUNK=off` at a real video
/// shape is the one case this does not model, and it OOMs on its own long
/// before admission could help (`[40, 21840, 21840]` F32 is ~76 TB).
///
/// Per-token modulation is charged only when the checkpoint drives per-token
/// timesteps: for a scalar timestep the table is a single broadcast row.
pub fn wan_activation_budget_bytes(
    width: u32,
    height: u32,
    frames: u32,
    geometry: WanActivationGeometry,
) -> u64 {
    wan_activation_budget_bytes_for(
        width,
        height,
        frames,
        geometry,
        crate::attention::AttentionBackend::Math,
    )
}

/// [`wan_activation_budget_bytes`] against an explicit attention backend.
///
/// The score matrix is the single largest per-token term on the math path and
/// **flash materializes none of it** — FA2 keeps the score tile in registers
/// and shared memory and never writes an `[heads, chunk, T]` buffer to global
/// memory. An estimate blind to the backend therefore prices a render that is
/// not the one running, which matters twice over: admission decides whether
/// the shape is feasible, and `wan::block_offload` decides how many blocks to
/// park to make it so. Both read this.
pub fn wan_activation_budget_bytes_for(
    width: u32,
    height: u32,
    frames: u32,
    geometry: WanActivationGeometry,
    backend: crate::attention::AttentionBackend,
) -> u64 {
    let dim = geometry.dim;
    let residual = WAN_RESIDUAL_LIVE_BUFFERS * dim * BF16_BYTES;
    let gated = WAN_GATED_RESIDUAL_LIVE_BUFFERS * dim * BF16_BYTES;
    let qkv = 3 * dim * BF16_BYTES;
    let attention_phase = residual + gated + qkv;
    let feed_forward_phase = residual + gated + geometry.ffn_dim * BF16_BYTES;
    let block = attention_phase.max(feed_forward_phase);

    // Score plus softmax, both `[heads, chunk, T]`. BF16, not F32:
    // `q_chunk.matmul(&k_t)` inherits q/k's dtype
    // (`crates/mold-inference/src/attention.rs:235`) and `softmax_last_dim`
    // preserves it, and the wan compute dtype on CUDA is BF16
    // (`gpu_dtype`).
    //
    // Zero under flash: the kernel's online softmax never materializes the
    // tile. This is the term the two calibrations differ over, which is why
    // they are fitted separately rather than sharing a slope.
    let scores = match backend {
        crate::attention::AttentionBackend::Math => {
            2 * geometry.num_heads * WAN_ATTENTION_QUERY_CHUNK * BF16_BYTES
        }
        crate::attention::AttentionBackend::Flash => 0,
    };
    // `math_attention_chunked_flat` accumulates every chunk's `[heads, len,
    // head_dim]` output and then `Tensor::cat` allocates the joined copy
    // (`attention.rs:240-251`), so the output is live twice at the seam.
    let attention_output = 2 * dim * BF16_BYTES;

    let modulation = if geometry.per_token_timesteps {
        // One block's expanded `[1, tokens, 6, dim]` modulation table.
        //
        // This used to be charged twice: the forward-wide `timestep_proj` was
        // dense too, so a per-token schedule held one table for the whole
        // forward *and* built another in every block. `WanTimestepEmbedding`
        // now keeps the forward-wide copy at its handful of distinct rows
        // (~150 KiB instead of 1.87 GiB at the 5B's 121-frame default), so
        // only the per-block expansion remains.
        //
        // Charging the retired term made admission refuse a 121-frame 1280x704
        // image-to-video render at ~25.9 GB against ~24.8 GB usable — a shape
        // whose recorded measurement is 18,460 MiB.
        WAN_MODULATION_COMPONENTS * dim * F32_BYTES
    } else {
        0
    };

    let per_token = block + scores + attention_output + modulation;
    wan_token_count(width, height, frames, geometry).saturating_mul(per_token)
}

/// Measured multiplier on the derived per-token tensor sum, and the flat terms
/// that go with it.
///
/// These live here, beside the derived formula, because two consumers need the
/// same number: `mold-server`'s admission decides whether a shape is feasible,
/// and the wan engine's block-offload policy decides how many blocks to park to
/// make it so. They diverged once — the engine used the RAW derived budget and
/// therefore under-estimated by 2.14x, parked nothing, and OOM'd at a shape
/// admission had just accepted. One authority, so that cannot recur.
///
/// See `mold_server::wan_admission` for the fit: RTX 4090, `wan22-t2v-a14b:q5`,
/// 832x480, CFG off, 16,074 MiB at 17f and 21,354 MiB at 53f.
pub const WAN_MEASURED_SLOPE_NUMERATOR: u64 = 214;
pub const WAN_MEASURED_SLOPE_DENOMINATOR: u64 = 100;
/// Token-independent denoise workspace, fitted as the intercept of that pair.
pub const WAN_FIXED_WORKSPACE_BYTES: u64 = 2301 * 1024 * 1024;
/// Token-shaped BF16 buffers the cached branch holds beyond the uncached one,
/// within a single trajectory's forward. See [`wan_step_cache_bytes`].
const WAN_STEP_CACHE_FORWARD_BUFFERS: u64 = 4;

/// Token-shaped BF16 buffers the *other* trajectory's cache retains across the
/// denoise when CFG runs two of them: its `previous_first_block` and its `tail`.
const WAN_STEP_CACHE_PERSISTENT_BUFFERS: u64 = 2;

/// F32 chunk buffers live at once inside the distance reduction.
///
/// Three, not two: `previous_chunk.abs()` allocates a third while both upcast
/// chunks are still live, and the `(current - previous)` / `.abs()` pair does
/// the same on the other side of the reduction.
const WAN_STEP_CACHE_REDUCTION_LIVE_BUFFERS: u64 = 3;

/// Extra device memory a CFG step holds beyond a single forward.
pub const WAN_CFG_RESIDENT_BYTES: u64 = 256 * 1024 * 1024;

/// The same fit, re-measured on the flash path.
///
/// **This pair cannot be inherited from the math one, and getting that wrong
/// is an OOM rather than a slow render.** Dropping the score term shrinks the
/// derived per-token sum by ~44% at A14B (184,320 B to 102,400 B), so keeping
/// the 2.14 slope would under-estimate the real peak by several GB, park too
/// few blocks, and hand the driver a shape it cannot hold. The residual the
/// slope covers — LayerNorm F32 copies, RoPE application, the q/k RMS norms,
/// statement-scoped temporaries — does not shrink when attention stops
/// materializing its tile, so the *ratio* has to rise even though the absolute
/// per-token cost falls.
///
/// Same protocol and hardware as the math fit — RTX 4090,
/// `wan22-t2v-a14b:q5`, 832x480, CFG off, two renders differing only in frame
/// count so every fixed cost cancels — with `MOLD_ATTN=flash`:
///
/// | Frames | Tokens | Peak |
/// | -----: | -----: | ---: |
/// | 17 | 7,800 | 15,011 MiB |
/// | 53 | 21,840 | 19,011 MiB |
///
/// 4,000 MiB over 14,040 tokens is 298,740 B/token against the derived flash
/// sum of 102,400, i.e. **2.92** — higher than math's 2.14 exactly because the
/// derived sum shrank 44% while the residual it stands in for did not. The
/// cross-check that this is a real re-fit and not an artifact: the measured
/// per-token costs are 298,740 (flash) against 394,336 (math), a ratio of
/// 0.757, and the two calibrated models predict 0.758. The slope moved because
/// the formula changed, not because the hardware did.
///
/// Both points were taken with `MOLD_WAN_OFFLOAD_BLOCKS=0`. That is not
/// optional: parking blocks lowers the peak, so a fit taken while the policy
/// was engaged measures the policy rather than the render, and the resulting
/// slope would then tell the policy to park less — which is the wrong
/// direction and OOMs.
pub const WAN_FLASH_MEASURED_SLOPE_NUMERATOR: u64 = 292;
pub const WAN_FLASH_MEASURED_SLOPE_DENOMINATOR: u64 = 100;
/// Token-independent denoise workspace on the flash path, the intercept of
/// that pair. Larger than the math intercept is expected and not a mistake:
/// FA2 allocates its own fixed workspace, and the terms the derived formula no
/// longer names have to land somewhere.
pub const WAN_FLASH_FIXED_WORKSPACE_BYTES: u64 = 1949 * 1024 * 1024;

/// The calibration pair for `backend`.
///
/// One accessor so a caller cannot pair one backend's slope with the other's
/// intercept — the two are fitted together and are meaningless apart.
fn wan_calibration_for(backend: crate::attention::AttentionBackend) -> (u64, u64, u64) {
    match backend {
        crate::attention::AttentionBackend::Math => (
            WAN_MEASURED_SLOPE_NUMERATOR,
            WAN_MEASURED_SLOPE_DENOMINATOR,
            WAN_FIXED_WORKSPACE_BYTES,
        ),
        crate::attention::AttentionBackend::Flash => (
            WAN_FLASH_MEASURED_SLOPE_NUMERATOR,
            WAN_FLASH_MEASURED_SLOPE_DENOMINATOR,
            WAN_FLASH_FIXED_WORKSPACE_BYTES,
        ),
    }
}

/// [`wan_activation_budget_bytes`] with the measured calibration applied.
///
/// The derived sum counts the tensors a block visibly materializes; this is
/// what a real forward actually holds.
pub fn wan_calibrated_activation_bytes(
    width: u32,
    height: u32,
    frames: u32,
    geometry: WanActivationGeometry,
    cfg: bool,
    steps: u32,
    distilled: bool,
) -> u64 {
    wan_calibrated_activation_bytes_for(
        width,
        height,
        frames,
        geometry,
        cfg,
        steps,
        distilled,
        wan_effective_attention_backend(),
    )
}

/// The attention backend a wan render on this process will actually execute.
///
/// Reads the *video* policy, because that is the one the Wan DiT resolves
/// (`wan::model::transformer`), and the *effective* answer, because a `Flash`
/// request in a build without the kernels runs as math and would otherwise be
/// priced with no score tile at all.
///
/// Device-blind on purpose. The estimate is computed at admission, before a
/// device is leased, and every device this family runs on is CUDA; a CPU wan
/// render is a correctness path nobody admits against a VRAM budget.
pub fn wan_effective_attention_backend() -> crate::attention::AttentionBackend {
    crate::attention::AttentionBackend::resolve_effective_for(
        crate::attention::AttentionPolicy::Video,
    )
}

/// [`wan_calibrated_activation_bytes`] against an explicit backend, so the two
/// calibrations are testable without the process-frozen `MOLD_ATTN` cache.
#[allow(clippy::too_many_arguments)]
pub fn wan_calibrated_activation_bytes_for(
    width: u32,
    height: u32,
    frames: u32,
    geometry: WanActivationGeometry,
    cfg: bool,
    steps: u32,
    distilled: bool,
    backend: crate::attention::AttentionBackend,
) -> u64 {
    let (slope_num, slope_den, workspace) = wan_calibration_for(backend);
    let derived = wan_activation_budget_bytes_for(width, height, frames, geometry, backend);
    let transformer = derived.saturating_mul(slope_num) / slope_den;
    let cfg_bytes = if cfg { WAN_CFG_RESIDENT_BYTES } else { 0 };
    // The step cache is charged HERE, after the slope, beside the other exact
    // terms — never inside `derived`. The slope is a fitted multiplier on the
    // tensors a block materializes; these are a separate, exact allocation,
    // and folding them into `derived` would inflate them 2.14x (math) or 2.92x
    // (flash). The calibration test would still pass, because it prices a
    // distilled 4-step tier where the charge is zero, so that error lands
    // silently and parks too many blocks — giving back the speedup the cache
    // exists to buy.
    let step_cache = wan_step_cache_bytes(width, height, frames, geometry, cfg, steps, distilled);
    transformer
        .saturating_add(cfg_bytes)
        .saturating_add(step_cache)
        .saturating_add(workspace)
}

/// Device bytes engaging the step cache adds, or zero when it will not engage.
///
/// Exact, not fitted. Every buffer below is a `[1, T, dim]` BF16 tensor the
/// cached branch of `wan::model::transformer::forward_with_rope_cached`
/// (`transformer.rs:1883-1904`) holds beyond what the uncached branch holds,
/// so the charge is a token count times a buffer count:
///
/// The four are the buffers live DURING the block 1..N loop, because that is
/// where the block transients the fitted slope prices are also live — it is the
/// global high-water mark, not merely a local one.
///
/// | Buffer | Site | Lifetime | Count |
/// | --- | --- | --- | --- |
/// | `entry` | `transformer.rs:1887` | pins the patchify output for the whole arm; the uncached branch drops it at the first `block.forward` | 1 |
/// | `first_block_residual` | `transformer.rs:1891` | `Arc`-shared into `previous_first_block`, so counted once | 1 |
/// | `after_first` | `transformer.rs:1897` | pins block 0's output across all remaining blocks | 1 |
/// | retained `tail` | `step_cache.rs` `record_tail` | live until the new tail overwrites it | 1 |
/// | the other trajectory's `previous_first_block` + `tail` | — | across the denoise | 2 when CFG |
///
/// `record_tail` (`transformer.rs:1902`) briefly holds a FIFTH — the new tail
/// alongside the old — but that moment is after the loop, with the block
/// transients already dropped, so it lands far below the loop peak. Do not
/// "simplify" the count to 3 by reasoning that one of these is transient: every
/// row above is simultaneously live inside the loop, and dropping one
/// re-introduces the OOM this charge exists to prevent.
///
/// `hidden` is not charged: the uncached branch holds it too, so it is already
/// in the derived budget. `entry` and `after_first` are `Arc` clones sharing
/// storage with `hidden` — they allocate nothing but pin allocations alive,
/// which is why each is counted once and not twice.
///
/// The two trajectories do not run concurrently — the conditional and
/// unconditional forwards are separate calls with separate caches
/// (`wan::pipeline`) — so CFG adds only the other trajectory's *persistent*
/// pair, not a second copy of the transients.
///
/// Engagement is delegated to [`WanStepCachePolicy::resolve`], never
/// re-derived. A copy of `steps >= 12 && !distilled` here is how the estimate
/// and the engine drift into disagreeing about whether the cache is even on.
pub fn wan_step_cache_bytes(
    width: u32,
    height: u32,
    frames: u32,
    geometry: WanActivationGeometry,
    cfg: bool,
    steps: u32,
    distilled: bool,
) -> u64 {
    // A malformed value prices as off here; the engine fails the render on it
    // (`wan::pipeline` propagates the parse error), so the two cannot disagree
    // about a render that actually runs.
    let requested = crate::wan::step_cache::requested_threshold().unwrap_or(None);
    wan_step_cache_bytes_for(
        width, height, frames, geometry, cfg, steps, distilled, requested,
    )
}

/// [`wan_step_cache_bytes`] against an explicit requested threshold, so a test
/// is not at the mercy of a `MOLD_WAN_STEP_CACHE` the developer has exported.
#[allow(clippy::too_many_arguments)]
pub fn wan_step_cache_bytes_for(
    width: u32,
    height: u32,
    frames: u32,
    geometry: WanActivationGeometry,
    cfg: bool,
    steps: u32,
    distilled: bool,
    requested: Option<f64>,
) -> u64 {
    use crate::wan::step_cache::{WanStepCachePolicy, WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS};

    let (policy, _) = WanStepCachePolicy::resolve(requested, steps, distilled);
    if !policy.is_on() {
        return 0;
    }

    let tokens = wan_token_count(width, height, frames, geometry);
    let per_token = geometry.dim.saturating_mul(BF16_BYTES);
    let buffers = WAN_STEP_CACHE_FORWARD_BUFFERS
        + if cfg {
            WAN_STEP_CACHE_PERSISTENT_BUFFERS
        } else {
            0
        };
    let retained = tokens.saturating_mul(per_token).saturating_mul(buffers);

    // The distance reduction upcasts a bounded token slice to F32, twice
    // (the previous residual and the difference against it). Bounded by the
    // chunk width rather than the clip length, which is the whole reason the
    // reduction is chunked — see `step_cache::WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS`.
    let reduction = (WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS as u64)
        .min(tokens)
        .saturating_mul(geometry.dim)
        .saturating_mul(F32_BYTES)
        .saturating_mul(WAN_STEP_CACHE_REDUCTION_LIVE_BUFFERS);

    retained.saturating_add(reduction)
}

/// Map a manifest family slug (e.g. `"flux"`, `"sdxl"`, `"qwen-image"`) to the
/// activation-budget family. Falls back to [`ActivationFamily::FluxDit`] for
/// unknown slugs — the FLUX factor is the most common diffusion default and
/// errs toward a conservative-but-not-over-budget estimate.
pub fn activation_family_for(family_slug: &str) -> ActivationFamily {
    match family_slug {
        "flux" => ActivationFamily::FluxDit,
        "flux2" => ActivationFamily::Flux2Dit,
        "sd3" => ActivationFamily::Sd3Mmdit,
        "sdxl" | "sd15" => ActivationFamily::SdxlUnet,
        "qwen-image" | "qwen-image-edit" => ActivationFamily::QwenImageDit,
        "z-image" => ActivationFamily::ZImageDit,
        "wuerstchen" => ActivationFamily::Wuerstchen,
        "hunyuan3d" | "hunyuan-3d" => ActivationFamily::Hunyuan3dShape,
        // LTX-Video (0.9.6 / 0.9.8 2B or 13B): loads the entire transformer
        // into VRAM during each generate call. The file-size-based preflight
        // applies normally — the 13B BF16 checkpoint is ~26 GB and must be
        // counted in full.
        "ltx-video" => ActivationFamily::LtxVideo,
        // LTX-2 (19B / 22B): streaming-loaded transformer — only a couple of
        // blocks are GPU-resident at peak, so the preflight skips the
        // file-size estimate and uses a fixed streaming cap instead.
        "ltx2" | "ltx-2" | "ltx-2.3" => ActivationFamily::Ltx2Video,
        // Wan 2.1/2.2: fully GPU-resident transformer at every shipped size —
        // 1.3B, 5B, and both A14B experts, which are resident one at a time
        // rather than streamed. The file-size preflight applies in full, and
        // `estimate_peak_memory` sizes an expert pair as the larger of the two.
        "wan" => ActivationFamily::WanVideo,
        // Unknown families default to FLUX dit shape — same activation
        // class, conservative against unknowns.
        _ => ActivationFamily::FluxDit,
    }
}

/// Compute VRAM threshold for an expand LLM of a given size (weights + headroom).
pub fn expand_vram_threshold(model_size_bytes: u64) -> u64 {
    model_size_bytes + EXPAND_ACTIVATION_HEADROOM
}

/// Resolved placement for the expand LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandPlacement {
    /// Place on GPU with the given ordinal.
    Gpu(usize),
    /// Place on CPU (system RAM).
    Cpu,
}

/// Pick where to run the expand LLM: main GPU first, then remaining GPUs in
/// ordinal order, then CPU.
///
/// - `gpus` must be in ordinal order (as returned by `discover_gpus()`).
/// - On Metal (unified memory) the single discovered "GPU" is always chosen
///   — memory policing happens via system-RAM preflight at the call site,
///   since GPU VRAM and system RAM are the same pool.
/// - A GPU is considered to fit when `free_vram_bytes > threshold`.
/// - Returns `ExpandPlacement::Cpu` when no GPU has room (or when `gpus` is
///   empty). The caller is responsible for running a system-RAM preflight
///   before actually allocating on CPU.
pub fn select_expand_device(
    gpus: &[DiscoveredGpu],
    threshold: u64,
    is_metal: bool,
) -> ExpandPlacement {
    select_expand_device_with_preference(gpus, threshold, is_metal, None)
}

/// Same as [`select_expand_device`], but prefers `preferred_ordinal` when it
/// is in the allowed GPU set and has enough free VRAM.
pub fn select_expand_device_with_preference(
    gpus: &[DiscoveredGpu],
    threshold: u64,
    is_metal: bool,
    preferred_ordinal: Option<usize>,
) -> ExpandPlacement {
    if is_metal {
        if let Some(ordinal) = preferred_ordinal {
            if let Some(g) = gpus.iter().find(|g| g.ordinal == ordinal) {
                return ExpandPlacement::Gpu(g.ordinal);
            }
        }
        if let Some(g) = gpus.first() {
            return ExpandPlacement::Gpu(g.ordinal);
        }
        return ExpandPlacement::Cpu;
    }
    if let Some(ordinal) = preferred_ordinal {
        if let Some(g) = gpus
            .iter()
            .find(|g| g.ordinal == ordinal && g.free_vram_bytes > threshold)
        {
            return ExpandPlacement::Gpu(g.ordinal);
        }
    }
    for g in gpus {
        if g.free_vram_bytes > threshold {
            return ExpandPlacement::Gpu(g.ordinal);
        }
    }
    ExpandPlacement::Cpu
}

// ── LTX-2 Gemma encoder placement ────────────────────────────────────────────

/// Minimum free VRAM (bytes) needed to land the LTX-2 Gemma 3 12B prompt
/// encoder on a single GPU alongside its activation workspace.
///
/// This tracks *streaming* residency, not the ~24 GB the BF16 weights occupy
/// on disk. [`crate::ltx2::text::GemmaHiddenStateEncoder::load_from_assets`]
/// builds through `new_streaming`, and `forward_hidden_states` constructs each
/// of the 48 `DecoderLayer`s inside the forward loop and drops it before the
/// next — so the layers are never co-resident. What stays live is the token
/// embedding table (262,208 × 3,840 BF16 ≈ 2.01 GB), at most two decoder
/// layers in flight (≈ 0.45 GB each), and the 49 retained hidden states
/// (≈ 0.39 GB budgeted at a 1,024-token context — deliberately conservative:
/// `ltx2::text::gemma::DEFAULT_GEMMA_MAX_LENGTH` is 256, and every LTX-2
/// call site uses it, but the budget keeps the larger figure so a future
/// longer context cannot silently outgrow it) — a peak near 3.3 GB.
///
/// 6 GB leaves roughly 2× headroom over that peak while keeping a 24 GB
/// consumer card eligible. The previous value was 24 GB, which described the
/// long-removed eager loader and made the encoder unplaceable on exactly the
/// cards that need it most: it pinned Gemma to CPU on a 4090, where prompt
/// encoding cost ~93 s of a ~180 s render.
///
/// Placement stays advisory — [`crate::ltx2`]'s OOM path retries on CPU.
pub const LTX2_GEMMA_VRAM_THRESHOLD: u64 = 6_000_000_000;

/// Resolved placement for the LTX-2 Gemma 3 12B prompt encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtxGemmaPlacement {
    /// Place the encoder on GPU with the given ordinal.
    Gpu(usize),
    /// Place the encoder on CPU (system RAM).
    Cpu,
}

impl LtxGemmaPlacement {
    /// Convert to a candle `Device`. CUDA failures fall back to CPU rather
    /// than panic — the caller already paid for the placement decision and a
    /// runtime CUDA error here is far worse than honoring the hint as CPU.
    pub fn into_device(self) -> candle_core::Device {
        match self {
            LtxGemmaPlacement::Gpu(ordinal) => match candle_core::Device::new_cuda(ordinal) {
                Ok(d) => d,
                Err(err) => {
                    tracing::warn!(
                        ordinal,
                        error = %err,
                        "failed to open CUDA device for LTX-2 Gemma encoder, falling back to CPU"
                    );
                    candle_core::Device::Cpu
                }
            },
            LtxGemmaPlacement::Cpu => candle_core::Device::Cpu,
        }
    }
}

/// Pick where to load the LTX-2 Gemma 3 12B prompt encoder.
///
/// The encoder may use only the GPU already leased to this stage or CPU.
/// Sibling GPUs are intentionally ignored: allocating there without a lease
/// would violate the one-device-per-stage invariant and could double-book
/// another worker.
///
/// - `gpus` is the output of [`discover_gpus`] — ordinals in ascending order.
/// - `active_ordinal` is the GPU the LTX-2 transformer was loaded onto.
/// - A GPU is considered to fit when `free_vram_bytes > threshold` (strict
///   greater-than, mirrors [`select_expand_device`]).
/// - Returns [`LtxGemmaPlacement::Cpu`] when no GPU has room.
pub fn select_ltx2_gemma_device(
    gpus: &[DiscoveredGpu],
    active_ordinal: usize,
    threshold: u64,
) -> LtxGemmaPlacement {
    if let Some(gpu) = gpus
        .iter()
        .find(|gpu| gpu.ordinal == active_ordinal && gpu.free_vram_bytes > threshold)
    {
        return LtxGemmaPlacement::Gpu(gpu.ordinal);
    }
    LtxGemmaPlacement::Cpu
}

/// Read [`MOLD_LTX2_GEMMA_DEVICE`] (and the deprecated
/// [`MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER`] alias) and return an explicit
/// placement override. `auto`, an unset env, or a value the parser doesn't
/// recognise return `None` so the caller falls through to the auto-resolver.
///
/// The returned `Gpu` placement always points at `gpu_ordinal` — explicit
/// `gpu` doesn't try to outsmart the user by walking siblings.
pub fn resolve_ltx2_gemma_device_override(gpu_ordinal: usize) -> Option<LtxGemmaPlacement> {
    resolve_ltx2_gemma_device_override_from_values(
        crate::runtime_env::value("MOLD_LTX2_GEMMA_DEVICE").as_deref(),
        crate::runtime_env::value("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER").as_deref(),
        gpu_ordinal,
    )
}

/// Pure parser used by immutable scheduler plans and environment-isolated
/// tests. Production callers normally use
/// [`resolve_ltx2_gemma_device_override`].
pub fn resolve_ltx2_gemma_device_override_from_values(
    primary: Option<&str>,
    legacy_force_cpu: Option<&str>,
    gpu_ordinal: usize,
) -> Option<LtxGemmaPlacement> {
    if let Some(raw) = primary {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let lower = trimmed.to_ascii_lowercase();
            match lower.as_str() {
                "cpu" => return Some(LtxGemmaPlacement::Cpu),
                "gpu" => return Some(LtxGemmaPlacement::Gpu(gpu_ordinal)),
                "auto" => return None,
                _ => {
                    tracing::warn!(
                        value = %trimmed,
                        "unrecognised MOLD_LTX2_GEMMA_DEVICE value; expected cpu/gpu/auto",
                    );
                    return None;
                }
            }
        }
    }

    if legacy_force_cpu.is_some() {
        warn_once_legacy_force_cpu_prompt_encoder();
        return Some(LtxGemmaPlacement::Cpu);
    }

    None
}

fn warn_once_legacy_force_cpu_prompt_encoder() {
    use std::sync::OnceLock;
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        tracing::warn!(
            "MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER is deprecated; \
             use MOLD_LTX2_GEMMA_DEVICE=cpu instead",
        );
    });
}

/// Resolve the LTX-2 Gemma encoder placement once, honoring the env override
/// before falling through to the assigned-GPU-or-CPU fallback. The runtime and
/// the server-side preflight both call this so they reach the same decision
/// for the same observation of free VRAM and env vars.
pub fn resolve_ltx2_gemma_placement(gpu_ordinal: usize) -> LtxGemmaPlacement {
    if let Some(p) = resolve_ltx2_gemma_device_override(gpu_ordinal) {
        return p;
    }
    let gpus = discover_gpus();
    select_ltx2_gemma_device(&gpus, gpu_ordinal, LTX2_GEMMA_VRAM_THRESHOLD)
}

/// Minimum free VRAM for BF16 Qwen3-4B on GPU with drop-and-reload.
/// 8.2GB model + 2GB activation headroom = 10.2GB.
/// With drop-and-reload, the encoder is temporary — loaded for encoding, then dropped.
pub const QWEN3_FP16_VRAM_THRESHOLD: u64 = 10_200_000_000;

/// Headroom for activation memory during inference (denoising + VAE decode workspace).
const MEMORY_BUDGET_HEADROOM: u64 = 2_000_000_000; // 2GB
/// Conservative GPU working-set cap for block-streamed transformers. Covers
/// resident top-level weights, streamed block weights, VAE residency, and
/// allocator slack; activations are budgeted separately.
pub const STREAMING_TRANSFORMER_CAP_BYTES: u64 = 6_000_000_000;

// ── Placement resolution ─────────────────────────────────────────────────────

/// Resolve a caller-supplied `DeviceRef` override into a concrete candle
/// `Device`, falling back to `auto` when the override is missing or `Auto`.
///
/// - `None`, `Some(Auto)` — call `auto()` (existing VRAM-aware logic).
/// - `Some(Cpu)`          — `Device::Cpu`, never invoke `auto()`.
/// - `Some(Gpu { ordinal })` — try CUDA first, then Metal. Each backend is
///   gated by its candle feature flag so a CPU-only build returns a clear
///   error message instead of a build failure.
/// - `Some(Device { id })` — open the ordinal already bound to the owner
///   thread. Stable-ID eligibility must have been validated by the scheduler;
///   inference never searches for or silently substitutes another device.
pub fn resolve_device<F>(
    req: Option<mold_core::types::DeviceRef>,
    auto: F,
) -> anyhow::Result<candle_core::Device>
where
    F: FnOnce() -> anyhow::Result<candle_core::Device>,
{
    use mold_core::types::DeviceRef;
    match req {
        None | Some(DeviceRef::Auto) => auto(),
        Some(DeviceRef::Cpu) => Ok(candle_core::Device::Cpu),
        Some(DeviceRef::Gpu { ordinal }) => resolve_gpu_ordinal(ordinal),
        Some(DeviceRef::Device { id }) => {
            let ordinal = thread_gpu_ordinal().ok_or_else(|| {
                anyhow::anyhow!(
                    "stable device placement '{id}' requires a scheduler-bound GPU owner thread"
                )
            })?;
            resolve_gpu_ordinal(ordinal)
        }
    }
}

#[cfg(feature = "cuda")]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    debug_assert_ordinal_matches_thread(ordinal, "resolve_device");
    candle_core::Device::new_cuda(ordinal)
        .map_err(|e| anyhow::anyhow!("failed to open CUDA device {ordinal}: {e}"))
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    debug_assert_ordinal_matches_thread(ordinal, "resolve_device");
    metal_device(ordinal)
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    Err(anyhow::anyhow!(
        "GPU ordinal {ordinal} requested but this build has neither CUDA nor Metal enabled"
    ))
}

/// Resolve a component-level `DeviceRef` from a `DevicePlacement`, honoring
/// the Tier 2 per-component override first, then the Tier 1 `text_encoders`
/// group knob when appropriate.
///
/// Precedence:
///   1. `advanced_override` (Tier 2 per-component) if `Some`.
///   2. Fall back to `placement.text_encoders` (group knob) when
///      `fallback_is_component_auto` is `true` (typically for text-encoder
///      components — T5/CLIP-L/Qwen — that follow the group knob by default).
///   3. Fall back to `DeviceRef::Auto` (non-text-encoder components like the
///      VAE, which don't inherit from the text-encoder group knob).
pub fn effective_device_ref(
    placement: Option<&mold_core::types::DevicePlacement>,
    advanced_override: impl FnOnce(
        &mold_core::types::AdvancedPlacement,
    ) -> Option<mold_core::types::DeviceRef>,
    fallback_is_component_auto: bool,
) -> mold_core::types::DeviceRef {
    use mold_core::types::DeviceRef;
    let Some(placement) = placement else {
        return DeviceRef::Auto;
    };
    if let Some(adv) = placement.advanced.as_ref() {
        if let Some(r) = advanced_override(adv) {
            return r;
        }
        if fallback_is_component_auto {
            return placement.text_encoders.clone();
        }
        DeviceRef::Auto
    } else {
        placement.text_encoders.clone()
    }
}

// ── macOS memory query ───────────────────────────────────────────────────────

/// Raw VM statistics from macOS host_statistics64.
#[cfg(target_os = "macos")]
struct MacOSMemInfo {
    free: u64,
    inactive: u64,
}

/// Query macOS VM statistics using host_statistics64 FFI.
#[cfg(target_os = "macos")]
fn macos_vm_stats() -> Option<MacOSMemInfo> {
    type MachPort = u32;
    type KernReturn = i32;
    type HostFlavor = i32;
    type MachMsgType = u32;

    const HOST_VM_INFO64: HostFlavor = 4;
    const HOST_VM_INFO64_COUNT: MachMsgType = 38;
    const KERN_SUCCESS: KernReturn = 0;

    extern "C" {
        fn mach_host_self() -> MachPort;
        fn host_statistics64(
            host: MachPort,
            flavor: HostFlavor,
            info: *mut i32,
            count: *mut MachMsgType,
        ) -> KernReturn;
        fn host_page_size(host: MachPort, page_size: *mut usize) -> KernReturn;
    }

    unsafe {
        let mut buf = [0i32; HOST_VM_INFO64_COUNT as usize];
        let mut count = HOST_VM_INFO64_COUNT;
        let ret = host_statistics64(
            mach_host_self(),
            HOST_VM_INFO64,
            buf.as_mut_ptr(),
            &mut count,
        );
        if ret != KERN_SUCCESS {
            return None;
        }
        let mut page_size: usize = 0;
        let ret = host_page_size(mach_host_self(), &mut page_size);
        if ret != KERN_SUCCESS {
            return None;
        }
        let page_size = page_size as u64;
        // Layout: [0]=free_count, [1]=active_count, [2]=inactive_count (all natural_t = u32)
        Some(MacOSMemInfo {
            free: buf[0] as u32 as u64 * page_size,
            inactive: buf[2] as u32 as u64 * page_size,
        })
    }
}

/// Immediately free system memory on macOS (free pages only).
///
/// This is the conservative metric — memory available WITHOUT reclaiming inactive pages.
/// Use this for variant selection to avoid triggering page reclamation storms that
/// make the system unresponsive.
#[cfg(target_os = "macos")]
pub fn free_system_memory_bytes() -> Option<u64> {
    macos_vm_stats().map(|s| s.free)
}

/// Total available system memory on macOS (free + inactive pages).
///
/// Inactive pages are trivially reclaimable by the OS (no I/O for anonymous pages).
/// Used for both memory budget checks and variant selection on unified-memory systems,
/// where free-only is too conservative (often ~1-2GB on a busy 16GB Mac).
#[cfg(target_os = "macos")]
pub fn available_system_memory_bytes() -> Option<u64> {
    macos_vm_stats().map(|s| s.free + s.inactive)
}

/// Currently used macOS swap bytes.
///
/// Large Metal jobs share physical memory with the host. A free-page sample
/// alone can lag the onset of destructive paging, so H3's live guard also
/// bounds swap growth relative to the start of one attempt.
#[cfg(target_os = "macos")]
pub fn used_system_swap_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::xsw_usage>::zeroed();
    let mut len = std::mem::size_of::<libc::xsw_usage>();
    let name = c"vm.swapusage";
    // SAFETY: `usage` has the exact kernel ABI type and `len` describes its
    // writable storage for the duration of this sysctl call.
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            usage.as_mut_ptr().cast::<libc::c_void>(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 || len != std::mem::size_of::<libc::xsw_usage>() {
        return None;
    }
    // SAFETY: the successful sysctl initialized the complete `xsw_usage`.
    Some(unsafe { usage.assume_init() }.xsu_used)
}

/// Installed physical RAM on macOS (`hw.memsize`).
///
/// Unified memory has no separate VRAM total, so this is the ceiling the Metal
/// device can address. It is a hardware constant: unlike
/// [`available_system_memory_bytes`] it must not move as pages are dirtied,
/// because the scheduler clamps a device's effective capacity to its total.
#[cfg(target_os = "macos")]
pub fn total_system_memory_bytes() -> Option<u64> {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = c"hw.memsize";
    // SAFETY: `name` is a NUL-terminated C string, and `size`/`len` are valid
    // writable pointers to a u64 and its length for the duration of the call.
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut size).cast::<libc::c_void>(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (ret == 0 && size > 0).then_some(size)
}

/// Stable unified-memory capacity for an explicitly selected large Metal job.
///
/// `free + inactive` is the right immediate-pressure sample for ordinary
/// scheduling, but it excludes clean active file cache and compressed or
/// swappable application pages. H3 authenticates ~44 GB of model files before
/// allocation, so that sample falls sharply from the verification I/O itself
/// and can make a 48 GB Mac report only ~20 GB even though the reviewed 33.9 GB
/// runtime fits while retaining the canonical host safety floor. For this
/// explicit large-model path, use installed unified memory minus the same
/// `max(15%, 8 GiB)` floor as the host ledger. If installed memory cannot be
/// queried, retain the live sample unchanged.
pub fn metal_unified_capacity_with_safety_floor(live_available_bytes: u64) -> u64 {
    total_system_memory_bytes().map_or(live_available_bytes, |total| {
        let safety_floor = total.saturating_mul(15).saturating_div(100).max(8 << 30);
        total.saturating_sub(safety_floor)
    })
}

/// Unified-memory bytes a streaming runtime may allocate from the current
/// live sample while preserving the host safety floor.
///
/// Unlike [`metal_unified_capacity_with_safety_floor`], this is deliberately
/// pressure-sensitive: adaptive residency uses it after prompt encoding, so a
/// busy Mac retains fewer model blocks and streams the remainder instead of
/// reclaiming the memory needed by the desktop and other applications.
pub fn metal_live_allocation_budget(live_available_bytes: u64) -> u64 {
    let total = total_system_memory_bytes().unwrap_or(live_available_bytes);
    let safety_floor = total.saturating_mul(15).saturating_div(100).max(8 << 30);
    live_available_bytes.min(total).saturating_sub(safety_floor)
}

#[cfg(not(target_os = "macos"))]
pub fn total_system_memory_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn free_system_memory_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn available_system_memory_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn used_system_swap_bytes() -> Option<u64> {
    None
}

// ── Text-encoder retention ───────────────────────────────────────────────────

/// Whether to park text-encoder weights on CPU instead of dropping them after
/// encoding finishes (opt-in via `MOLD_KEEP_TE_RAM=1`).
///
/// Default off. Two things keep it that way rather than letting mold decide
/// per host:
///
/// * A park is a multi-gigabyte **host** allocation, and the only probes
///   available here (`/proc/meminfo`, the macOS VM statistics) describe the
///   machine, not this process's cgroup. Inside a memory-capped container —
///   which is how mold ships (the GHCR matrix, Lambda/RunPod provisioning) —
///   `MemAvailable` reports the host's free RAM, so a probe-driven default
///   would engage against a limit it cannot see and get the process
///   OOM-killed. Nothing in the tree reads `memory.max`.
/// * `MOLD_KEEP_TE_RAM` is an [`crate::runtime_env::ENGINE_SHAPING_VARIABLES`]
///   member precisely because memory residency must be frozen at admission.
///   A live host probe would let two runs with byte-identical frozen
///   snapshots take opposite residency paths while serializing to one
///   execution-equivalence fingerprint and one learned-timing bucket — the
///   same trap `MOLD_WAN_OFFLOAD_BLOCKS` is documented against.
///
/// When on, encoders survive between requests on host RAM; only the
/// lightweight GPU↔CPU tensor copy happens between requests. Since #1044 this
/// covers Qwen-Image's quantized GGUF Qwen2 encoder too — its `QTensor` bytes
/// move losslessly through `wan::block_offload` rather than being re-read from
/// disk (measured at 35.1 s per cold prompt). Other families' GGUF encoders
/// (T5, Qwen3) still drop and reload.
///
/// This mirrors ComfyUI's `text_encoder_offload_device()` behavior
/// (`comfy/model_management.py:1012`).
pub fn keep_te_in_ram() -> bool {
    crate::runtime_env::value("MOLD_KEEP_TE_RAM")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn fatal_cuda_driver_message(message: &str) -> bool {
    [
        "CUDA_ERROR_ILLEGAL_ADDRESS",
        "CUDA_ERROR_ECC_UNCORRECTABLE",
        "CUDA_ERROR_LAUNCH_FAILED",
        "CUDA_ERROR_ASSERT",
        "CUDA_ERROR_MISALIGNED_ADDRESS",
        "CUDA_ERROR_HARDWARE_STACK_ERROR",
        "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        "CUDA_ERROR_INVALID_PC",
        "CUDA_ERROR_LAUNCH_TIMEOUT",
        "CUDA_ERROR_EXTERNAL_DEVICE",
        "CUDA_ERROR_MPS_CLIENT_TERMINATED",
        "CUDA_ERROR_CONTAINED",
        "CUDA_ERROR_TENSOR_MEMORY_LEAK",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

#[cfg(feature = "cuda")]
fn memory_error(operation: &'static str, error: impl std::fmt::Display) -> DeviceMemoryError {
    let message = error.to_string();
    if fatal_cuda_driver_message(&message) {
        DeviceMemoryError::FatalCuda { operation, message }
    } else {
        DeviceMemoryError::Unavailable { operation, message }
    }
}

/// Run an authoritative post-drop observation. Kept separate from the CUDA
/// bindings so fault-ordering can be tested without inducing a real illegal
/// access: a failed synchronize must prevent the memory callback from running.
#[doc(hidden)]
pub fn post_drop_free_vram_bytes_with(
    synchronize: impl FnOnce() -> Result<(), DeviceMemoryError>,
    sample_free: impl FnOnce() -> Result<u64, DeviceMemoryError>,
) -> Result<u64, DeviceMemoryError> {
    synchronize()?;
    sample_free()
}

/// Synchronize pending CUDA work.
///
/// After a `CUDA_ERROR_OUT_OF_MEMORY` the CUDA context may have in-flight work
/// that hasn't been flushed; subsequent allocations can inherit a poisoned
/// scheduler state. Calling synchronize before reporting the OOM and before
/// any retry lets CUDA drain pending work and reset internal queues so the
/// next allocation attempt starts clean.
///
/// On non-CUDA platforms this is a no-op.
#[cfg(feature = "cuda")]
pub fn try_synchronize_device(ordinal: usize) -> Result<(), DeviceMemoryError> {
    debug_assert_ordinal_matches_thread(ordinal, "try_synchronize_device");
    let context = candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal)
        .map_err(|error| memory_error("context retain", error))?;
    context
        .synchronize()
        .map_err(|error| memory_error("device synchronize", error))
}

/// No-op on non-CUDA platforms.
#[cfg(not(feature = "cuda"))]
pub fn try_synchronize_device(_ordinal: usize) -> Result<(), DeviceMemoryError> {
    Ok(())
}

/// Synchronize pending work after device-backed values have been dropped, then
/// return the actual reserve-adjusted free VRAM reported by the driver.
///
/// Dropping Mold-owned objects does not imply that nominal total VRAM is
/// available: driver workspaces, fragmentation, and external allocations
/// remain unavailable pressure and are preserved in this measurement.
#[cfg(feature = "cuda")]
pub fn post_drop_free_vram_bytes(ordinal: usize) -> Result<u64, DeviceMemoryError> {
    debug_assert_ordinal_matches_thread(ordinal, "post_drop_free_vram_bytes");
    // One retained context spans both calls. Using the global mem_get_info
    // entry point after dropping this retain can sample whichever context the
    // thread later binds and defeats the owner-thread invariant.
    let context = candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal)
        .map_err(|error| memory_error("context retain", error))?;
    post_drop_free_vram_bytes_with(
        || {
            context
                .synchronize()
                .map_err(|error| memory_error("device synchronize", error))
        },
        || {
            context
                .mem_get_info()
                .map(|(free, _)| usable_free_vram_from_raw(free as u64, reserved_vram_bytes()))
                .map_err(|error| memory_error("free VRAM query", error))
        },
    )
}

/// Metal has unified memory and no CUDA stream to drain. The server deliberately
/// does not use this as a second post-drop admission gate; this implementation
/// remains useful to inference components that want a conservative observation.
#[cfg(not(feature = "cuda"))]
pub fn post_drop_free_vram_bytes(ordinal: usize) -> Result<u64, DeviceMemoryError> {
    usable_free_vram_bytes(ordinal).ok_or_else(|| DeviceMemoryError::Unavailable {
        operation: "free GPU memory query",
        message: "this backend does not expose a memory sample".to_string(),
    })
}

// ── VRAM query ───────────────────────────────────────────────────────────────

/// Query free VRAM in bytes for the specified GPU ordinal.
///
/// On CUDA, sets the context to the specified device before querying.
/// On macOS (unified memory), returns available system memory (free + inactive).
/// On other non-CUDA platforms, no VRAM info available.
#[cfg(feature = "cuda")]
pub fn free_vram_bytes(ordinal: usize) -> Option<u64> {
    let context = candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal).ok()?;
    context
        .mem_get_info()
        .ok()
        .map(|(free, _total)| free as u64)
}

/// On macOS (unified memory), return available system memory (free + inactive).
///
/// macOS reclaims inactive pages trivially with no I/O, so free-only is too
/// conservative for variant selection — it can reject quantized encoders that
/// would actually fit, forcing a BF16 fallback that doesn't fit either.
/// On other non-CUDA platforms, no VRAM info available.
#[cfg(not(feature = "cuda"))]
pub fn free_vram_bytes(_ordinal: usize) -> Option<u64> {
    available_system_memory_bytes().or_else(free_system_memory_bytes)
}

/// Bytes reserved from VRAM for the OS / desktop / cuBLAS workspace.
///
/// Even on a "headless" GPU some VRAM is always claimed by the driver, the
/// CUDA runtime, and (on Windows) the Desktop Window Manager — querying
/// `cuMemGetInfo` returns a number that the next allocation cannot fully
/// realise. ComfyUI bakes the same constant in (`comfy/model_management.py`):
/// 400 MB on Linux, 600 MB on Windows, plus an extra 100 MB on 16 GB+ cards.
///
/// Default: 400 MB on Linux, 600 MB on Windows, 0 on macOS (Metal unified
/// memory has its own headroom and the OS swap already accounts for desktop
/// pressure). Override via `MOLD_RESERVE_VRAM_MB`.
pub fn reserved_vram_bytes() -> u64 {
    if let Some(s) = crate::runtime_env::value("MOLD_RESERVE_VRAM_MB") {
        if let Some(mb) = crate::runtime_env::parse_u64(&s) {
            return mb.saturating_mul(1_000_000);
        }
    }
    #[cfg(target_os = "linux")]
    {
        400_000_000
    }
    #[cfg(target_os = "windows")]
    {
        600_000_000
    }
    #[cfg(target_os = "macos")]
    {
        0
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        400_000_000
    }
}

/// Wraps [`free_vram_bytes`] with the OS reserve subtracted.
///
/// Use this for **budget decisions** (does the transformer fit? should we
/// offload? which T5 variant should we pick?). For **diagnostic logging** keep
/// calling [`free_vram_bytes`] so the displayed value matches what the driver
/// reports — otherwise the ComfyUI-style reserve looks like ghost VRAM in
/// `nvidia-smi`.
pub fn usable_free_vram_bytes(ordinal: usize) -> Option<u64> {
    let reserve = reserved_vram_bytes();
    free_vram_bytes(ordinal).map(|free| usable_free_vram_from_raw(free, reserve))
}

/// Authoritative reserve-adjusted current-free reading used by CUDA admission.
/// Unlike the compatibility `Option` API, failure is typed so callers cannot
/// silently substitute host RAM or nominal device capacity.
#[cfg(feature = "cuda")]
pub fn usable_free_vram_bytes_result(ordinal: usize) -> Result<u64, DeviceMemoryError> {
    debug_assert_ordinal_matches_thread(ordinal, "usable_free_vram_bytes_result");
    let context = candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal)
        .map_err(|error| memory_error("context retain", error))?;
    context
        .mem_get_info()
        .map(|(free, _)| usable_free_vram_from_raw(free as u64, reserved_vram_bytes()))
        .map_err(|error| memory_error("free VRAM query", error))
}

#[cfg(not(feature = "cuda"))]
pub fn usable_free_vram_bytes_result(ordinal: usize) -> Result<u64, DeviceMemoryError> {
    usable_free_vram_bytes(ordinal).ok_or_else(|| DeviceMemoryError::Unavailable {
        operation: "free GPU memory query",
        message: "this backend does not expose a memory sample".to_string(),
    })
}

fn usable_free_vram_from_raw(free: u64, reserve: u64) -> u64 {
    free.saturating_sub(reserve)
}

/// Total VRAM currently in use (`total - free`) for the specified GPU
/// ordinal. Returns 0 if unavailable.
///
/// This is a **global** device measurement, not a per-model footprint. To
/// estimate the VRAM consumed by loading a model, take the delta between a
/// pre-load baseline and a post-load reading via [`vram_load_delta`].
#[cfg(feature = "cuda")]
pub fn vram_in_use_bytes(ordinal: usize) -> u64 {
    candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal)
        .and_then(|context| context.mem_get_info())
        .map(|(free, total)| total as u64 - free as u64)
        .unwrap_or(0)
}

/// Non-CUDA stub — no VRAM tracking available.
#[cfg(not(feature = "cuda"))]
pub fn vram_in_use_bytes(_ordinal: usize) -> u64 {
    0
}

/// Total VRAM (bytes) physically present on the specified GPU ordinal.
///
/// This is inventory/telemetry capacity, not an allocation promise. Memory
/// admission must use the current free reading because driver workspaces,
/// fragmentation, retained handles, and external processes may consume part
/// of the total.
///
/// Returns `None` when the device cannot be queried (CUDA disabled, ordinal
/// out of range, driver error). Callers should fall back to a less generous
/// budget in that case rather than treating `None` as unlimited.
#[cfg(feature = "cuda")]
pub fn total_vram_bytes(ordinal: usize) -> Option<u64> {
    let context = candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal).ok()?;
    context
        .mem_get_info()
        .ok()
        .map(|(_free, total)| total as u64)
}

/// Non-CUDA stub — no per-device total VRAM available outside CUDA.
#[cfg(not(feature = "cuda"))]
pub fn total_vram_bytes(_ordinal: usize) -> Option<u64> {
    None
}

/// Bytes loaded onto the GPU since `baseline` was sampled.
///
/// `baseline = vram_in_use_bytes(ordinal)` taken **before** loading a model;
/// this returns `vram_in_use_bytes(ordinal).saturating_sub(baseline)` so the
/// model cache records the new load's per-model footprint, not whatever the
/// device was already using.
pub fn vram_load_delta(ordinal: usize, baseline: u64) -> u64 {
    vram_in_use_bytes(ordinal).saturating_sub(baseline)
}

// ── Phase VRAM telemetry ─────────────────────────────────────────────────────
//
// `cuMemGetInfo` alone cannot describe what one inference phase spent. Candle
// allocates through cudarc's stream-ordered pool (`cuMemAllocAsync`) and Mold
// never trims that pool, so free VRAM does not recover when tensors drop: a
// free-delta reads every transient buffer as permanent retention, and a phase
// that reused pooled memory reads as spending nothing.
//
// The memory-pool attributes are the right authority. They are process-local
// (correct in containers, MIG, and WSL2 where the global number describes
// other tenants), they carry a driver-maintained high-water mark, and writing
// `0` to a `*_HIGH` attribute rearms it — which is exactly a per-phase peak.
//
// This is diagnostics only. Per the memory-authority rule, a phase report is
// never fed back into admission: the scheduler's frozen grant stays the
// authority even when a probe measures something different.

/// Which memory authority produced a [`PhaseVramReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramAttribution {
    /// CUDA memory-pool attributes were readable: the peaks are real,
    /// process-local high-water marks for this phase.
    Pool,
    /// Only `cuMemGetInfo` was readable. Free VRAM is device-global and
    /// includes every other process, so it is reported as context and
    /// deliberately never promoted into a peak.
    GlobalOnly,
    /// No GPU memory observation was available (non-CUDA build, driver error,
    /// or a device without memory-pool support).
    Unavailable,
}

impl VramAttribution {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pool => "pool",
            Self::GlobalOnly => "global-only",
            Self::Unavailable => "unavailable",
        }
    }
}

/// One raw GPU memory observation. Every field is optional because each
/// underlying query can independently be unavailable, and a missing sample
/// must never be substituted with a zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VramSample {
    /// `CU_MEMPOOL_ATTR_USED_MEM_CURRENT` — bytes currently held by live
    /// pool allocations.
    pub pool_used_bytes: Option<u64>,
    /// `CU_MEMPOOL_ATTR_USED_MEM_HIGH` — high-water mark since the last reset.
    pub pool_used_high_bytes: Option<u64>,
    /// `CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH` — high-water mark of physical
    /// memory the pool reserved from the driver (used plus pool overhead).
    pub pool_reserved_high_bytes: Option<u64>,
    /// `cuMemGetInfo` free bytes — device-global, all processes.
    pub global_free_bytes: Option<u64>,
    /// `cuMemGetInfo` total bytes.
    pub global_total_bytes: Option<u64>,
}

/// What one inference phase actually cost on the GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseVramReport {
    /// Phase label, e.g. `distilled.stage2.denoise`.
    pub phase: String,
    /// Peak pool bytes in use during the phase. `None` unless the
    /// attribution is [`VramAttribution::Pool`].
    pub peak_pool_used: Option<u64>,
    /// Peak pool bytes reserved from the driver during the phase.
    pub peak_pool_reserved: Option<u64>,
    /// Pool bytes live when the phase probe opened. This lets qualification
    /// distinguish a phase's incremental workspace from resident weights.
    pub entry_pool_used: Option<u64>,
    /// Signed change in live pool bytes across the phase. Negative means the
    /// phase released more than it retained — a real and useful outcome that
    /// a saturating unsigned delta would hide.
    pub delta_pool_used: Option<i64>,
    pub global_free_entry: Option<u64>,
    pub global_free_exit: Option<u64>,
    pub global_total: Option<u64>,
    /// What the planner said this phase would cost, when the caller knows.
    pub predicted_bytes: Option<u64>,
    pub elapsed: Duration,
    pub attribution: VramAttribution,
}

impl PhaseVramReport {
    /// Whether this report carries a peak Mold can attribute to itself.
    ///
    /// False for [`VramAttribution::GlobalOnly`] and
    /// [`VramAttribution::Unavailable`]: a device-global free delta is not a
    /// process-local peak and must not be presented (or reasoned about) as one.
    pub fn has_authoritative_peak(&self) -> bool {
        matches!(self.attribution, VramAttribution::Pool) && self.peak_pool_used.is_some()
    }

    /// Peak pool growth above the phase-entry residency.
    pub fn incremental_peak_pool_used(&self) -> Option<u64> {
        self.peak_pool_used
            .zip(self.entry_pool_used)
            .map(|(peak, entry)| peak.saturating_sub(entry))
    }

    /// Peak temporary growth after removing allocations still live at exit.
    /// Admission accounts retained weights/outputs separately from workspace.
    pub fn incremental_transient_pool_used(&self) -> Option<u64> {
        let growth = self.incremental_peak_pool_used()?;
        let retained = self.delta_pool_used?.max(0) as u64;
        Some(growth.saturating_sub(retained))
    }

    /// How far device-global free VRAM fell across the phase, saturating at
    /// zero when the phase ended with more free memory than it started with.
    pub fn global_free_drop(&self) -> Option<u64> {
        match (self.global_free_entry, self.global_free_exit) {
            (Some(entry), Some(exit)) => Some(entry.saturating_sub(exit)),
            _ => None,
        }
    }
}

fn fmt_gb_opt(bytes: Option<u64>) -> String {
    bytes.map(fmt_gb).unwrap_or_else(|| "n/a".to_string())
}

impl std::fmt::Display for PhaseVramReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} peak={} reserved={} predicted={} free {}->{} of {} attribution={} {}ms",
            self.phase,
            fmt_gb_opt(self.peak_pool_used),
            fmt_gb_opt(self.peak_pool_reserved),
            fmt_gb_opt(self.predicted_bytes),
            fmt_gb_opt(self.global_free_entry),
            fmt_gb_opt(self.global_free_exit),
            fmt_gb_opt(self.global_total),
            self.attribution.as_str(),
            self.elapsed.as_millis(),
        )
    }
}

/// Build a phase report from two raw observations.
///
/// Kept pure and free of CUDA bindings so the attribution rules are unit
/// testable without a GPU, mirroring `post_drop_free_vram_bytes_with`.
pub fn phase_vram_report_from(
    phase: impl Into<String>,
    entry: VramSample,
    exit: VramSample,
    predicted_bytes: Option<u64>,
    elapsed: Duration,
) -> PhaseVramReport {
    let pool_authoritative = exit.pool_used_high_bytes.is_some();
    let attribution = if pool_authoritative {
        VramAttribution::Pool
    } else if entry.global_free_bytes.is_some() || exit.global_free_bytes.is_some() {
        VramAttribution::GlobalOnly
    } else {
        VramAttribution::Unavailable
    };
    let delta_pool_used = match (entry.pool_used_bytes, exit.pool_used_bytes) {
        (Some(entry_used), Some(exit_used)) if pool_authoritative => {
            Some(exit_used as i64 - entry_used as i64)
        }
        _ => None,
    };
    PhaseVramReport {
        phase: phase.into(),
        peak_pool_used: pool_authoritative
            .then_some(exit.pool_used_high_bytes)
            .flatten(),
        peak_pool_reserved: pool_authoritative
            .then_some(exit.pool_reserved_high_bytes)
            .flatten(),
        entry_pool_used: pool_authoritative
            .then_some(entry.pool_used_bytes)
            .flatten(),
        delta_pool_used,
        global_free_entry: entry.global_free_bytes,
        global_free_exit: exit.global_free_bytes,
        global_total: exit.global_total_bytes.or(entry.global_total_bytes),
        predicted_bytes,
        elapsed,
        attribution,
    }
}

/// Sample GPU memory for the calling thread's bound device.
///
/// `rearm_high_water` resets `CU_MEMPOOL_ATTR_USED_MEM_HIGH` and
/// `CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH` to the pool's current usage after
/// reading them, so the next sample's high-water marks describe only the
/// interval that just started.
#[cfg(feature = "cuda")]
fn sample_phase_vram(ordinal: usize, rearm_high_water: bool) -> VramSample {
    use candle_core::cuda_backend::cudarc::driver::{result, sys, CudaContext};

    let mut sample = VramSample::default();
    let Ok(context) = CudaContext::new(ordinal) else {
        return sample;
    };
    if let Ok((free, total)) = context.mem_get_info() {
        sample.global_free_bytes = Some(free as u64);
        sample.global_total_bytes = Some(total as u64);
    }
    // Without pool-backed allocation, cudarc allocates with `cuMemAlloc` and
    // the pool attributes describe an empty pool. Reporting those zeros as a
    // measured peak would be worse than reporting nothing.
    if !context.has_async_alloc() {
        return sample;
    }
    if context.preflight_raw_call().is_err() {
        return sample;
    }
    // SAFETY: `context.cu_device()` is a live CUdevice owned by the retained
    // cudarc context, and this is the pool `cuMemAllocAsync` allocates from.
    let Ok(pool) = (unsafe { result::device::get_mem_pool(context.cu_device()) }) else {
        return sample;
    };
    sample.pool_used_bytes = pool_attribute(
        &context,
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
    );
    sample.pool_used_high_bytes = pool_attribute(
        &context,
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
    );
    sample.pool_reserved_high_bytes = pool_attribute(
        &context,
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
    );
    if rearm_high_water {
        // Writing zero does not zero the counter: CUDA resets each high-water
        // mark to the pool's *current* value, which is precisely the baseline
        // the next interval should measure against.
        reset_pool_high_water(
            &context,
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        );
        reset_pool_high_water(
            &context,
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
        );
    }
    sample
}

#[cfg(feature = "cuda")]
fn pool_attribute(
    context: &candle_core::cuda_backend::cudarc::driver::CudaContext,
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
    attribute: candle_core::cuda_backend::cudarc::driver::sys::CUmemPool_attribute,
) -> Option<u64> {
    use candle_core::cuda_backend::cudarc::driver::result;

    let mut value: u64 = 0;
    context.preflight_raw_call().ok()?;
    // SAFETY: `pool` is a live pool handle from the driver and `value` is a
    // writable `cuuint64_t`, the documented type for every `*_MEM_*` pool
    // attribute queried here.
    unsafe {
        result::mem_pool::get_attribute(
            pool,
            attribute,
            (&mut value) as *mut u64 as *mut std::ffi::c_void,
        )
        .ok()?;
    }
    Some(value)
}

#[cfg(feature = "cuda")]
fn reset_pool_high_water(
    context: &candle_core::cuda_backend::cudarc::driver::CudaContext,
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
    attribute: candle_core::cuda_backend::cudarc::driver::sys::CUmemPool_attribute,
) {
    use candle_core::cuda_backend::cudarc::driver::result;

    let mut value: u64 = 0;
    if context.preflight_raw_call().is_err() {
        return;
    }
    // SAFETY: same contract as `pool_attribute`; the driver reads one
    // `cuuint64_t` from `value`.
    unsafe {
        let _ = result::mem_pool::set_attribute(
            pool,
            attribute,
            (&mut value) as *mut u64 as *mut std::ffi::c_void,
        );
    }
}

/// Non-CUDA builds have no per-phase GPU memory authority.
#[cfg(not(feature = "cuda"))]
fn sample_phase_vram(_ordinal: usize, _rearm_high_water: bool) -> VramSample {
    VramSample::default()
}

thread_local! {
    /// Peak floors for the probes currently open on this thread.
    ///
    /// The device high-water mark is a single per-pool counter, so a nested
    /// probe's rearm would otherwise erase whatever its parent had already
    /// peaked at (LTX-2 nests a VAE load inside the conditioning encode).
    /// Each open probe therefore carries a `(used, reserved)` floor that its
    /// children fold their observations into.
    static PROBE_PEAK_FLOORS: std::cell::RefCell<Vec<(u64, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// How many probes this thread currently holds open.
///
/// The floor stack is the only durable trace a `PhaseVramProbe` leaves, so it
/// is the one signal that says whether a phase probe was actually closed. A
/// caller that returns without finishing its probes leaves this above where it
/// started, and the next phase on the same worker thread inherits the frame.
#[cfg(test)]
// Its only caller is the runtime-bound observer's withheld-observation test,
// which compiles under the H3 features without `cuda`; every other test build
// must still see a used function under `-D warnings`.
#[cfg_attr(
    not(all(any(feature = "h3", feature = "h3-private-uat"), not(feature = "cuda"))),
    allow(dead_code)
)]
pub(crate) fn open_probe_depth() -> usize {
    PROBE_PEAK_FLOORS.with(|floors| floors.borrow().len())
}

fn raise_open_probe_floor(used: Option<u64>, reserved: Option<u64>) {
    let (Some(used), reserved) = (used, reserved.unwrap_or(0)) else {
        return;
    };
    PROBE_PEAK_FLOORS.with(|floors| {
        if let Some(parent) = floors.borrow_mut().last_mut() {
            parent.0 = parent.0.max(used);
            parent.1 = parent.1.max(reserved);
        }
    });
}

/// A live measurement of one inference phase's GPU memory cost.
///
/// `enter` samples the thread's bound GPU (see [`thread_gpu_ordinal`]) and
/// rearms the pool high-water marks; `finish` samples again and reports the
/// interval. Diagnostics only — never an admission input.
#[derive(Debug)]
pub struct PhaseVramProbe {
    phase: String,
    entry: VramSample,
    started: Instant,
    predicted_bytes: Option<u64>,
    ordinal: usize,
    depth: usize,
}

impl PhaseVramProbe {
    pub fn enter(phase: impl Into<String>) -> Self {
        Self::enter_if(phase, true)
    }

    /// Open a probe only when the phase actually runs on CUDA.
    ///
    /// Sampling retains a CUDA context, which on a CPU-only run would create
    /// one — hundreds of MB of VRAM claimed by a run that asked for none. An
    /// inert probe reports [`VramAttribution::Unavailable`] and touches no
    /// driver call.
    pub fn enter_if(phase: impl Into<String>, sample: bool) -> Self {
        let ordinal = thread_gpu_ordinal().unwrap_or(0);
        if !sample {
            return Self {
                phase: phase.into(),
                entry: VramSample::default(),
                started: Instant::now(),
                predicted_bytes: None,
                ordinal,
                depth: usize::MAX,
            };
        }
        // Sampling reads the current high-water marks and then rearms them.
        // Hand what was read to the enclosing probe first: that peak belongs
        // to its interval and the rearm is about to discard it.
        let entry = sample_phase_vram(ordinal, true);
        raise_open_probe_floor(entry.pool_used_high_bytes, entry.pool_reserved_high_bytes);
        let depth = PROBE_PEAK_FLOORS.with(|floors| {
            let mut floors = floors.borrow_mut();
            floors.push((0, 0));
            floors.len() - 1
        });
        Self {
            phase: phase.into(),
            entry,
            started: Instant::now(),
            predicted_bytes: None,
            ordinal,
            depth,
        }
    }

    /// Attach the planner's prediction for this phase so one log line carries
    /// predicted-versus-actual.
    pub fn with_predicted(mut self, predicted_bytes: Option<u64>) -> Self {
        self.predicted_bytes = predicted_bytes;
        self
    }

    pub fn phase(&self) -> &str {
        &self.phase
    }

    pub fn finish(self) -> PhaseVramReport {
        let predicted = self.predicted_bytes;
        self.finish_with_predicted(predicted)
    }

    /// Finish with a prediction that only became known while the phase ran
    /// (an adaptive residency plan, for instance).
    pub fn finish_with_predicted(self, predicted_bytes: Option<u64>) -> PhaseVramReport {
        let elapsed = self.started.elapsed();
        if self.depth == usize::MAX {
            // Inert probe: it never sampled, never rearmed a counter, and owns
            // no slot on the floor stack. Timing is still real.
            return phase_vram_report_from(
                self.phase,
                VramSample::default(),
                VramSample::default(),
                predicted_bytes,
                elapsed,
            );
        }
        let mut exit = sample_phase_vram(self.ordinal, false);
        // Fold in what nested probes observed, then repair the stack: a
        // truncate also discards any child probe that was dropped without
        // finishing, so one leak cannot desynchronize later phases.
        let floor = PROBE_PEAK_FLOORS.with(|floors| {
            let mut floors = floors.borrow_mut();
            let mine = floors.get(self.depth).copied().unwrap_or((0, 0));
            if floors.len() > self.depth {
                floors.truncate(self.depth);
            }
            mine
        });
        exit.pool_used_high_bytes = exit.pool_used_high_bytes.map(|high| high.max(floor.0));
        exit.pool_reserved_high_bytes = exit.pool_reserved_high_bytes.map(|high| high.max(floor.1));
        // The enclosing phase's counter was rearmed by this probe's `enter`,
        // so it needs this interval's peak as its own floor.
        raise_open_probe_floor(exit.pool_used_high_bytes, exit.pool_reserved_high_bytes);
        phase_vram_report_from(self.phase, self.entry, exit, predicted_bytes, elapsed)
    }
}

// ── Formatting ───────────────────────────────────────────────────────────────

// ── Device helpers ───────────────────────────────────────────────────────────

/// Check whether a device is a GPU (CUDA or Metal).
pub(crate) fn is_gpu(device: &candle_core::Device) -> bool {
    device.is_cuda() || device.is_metal()
}

/// Select the optimal compute dtype for GPU inference.
///
/// - CUDA and Metal: BF16 (well-supported by tensor cores / Apple Neural Engine)
/// - CPU: F32
///
/// Note: this is the default compute dtype for model families that support BF16.
/// Some model families (SD1.5, SDXL) prefer F16 — they handle dtype selection
/// in their own pipelines.
#[allow(dead_code)]
pub(crate) fn gpu_compute_dtype(device: &candle_core::Device) -> candle_core::DType {
    if is_gpu(device) {
        candle_core::DType::BF16
    } else {
        candle_core::DType::F32
    }
}

/// Select the optimal dtype for GPU inference (CUDA-only BF16 variant).
///
/// - CUDA: BF16 (well-supported by tensor cores, standard for diffusion)
/// - Metal/MPS: F32 (BF16 on Metal has precision issues that cause washed-out,
///   blurry images — matmul accumulation errors compound through denoising loops.
///   This matches InvokeAI/diffusers which also avoid BF16 on MPS.)
/// - CPU: F32
pub(crate) fn gpu_dtype(device: &candle_core::Device) -> candle_core::DType {
    if device.is_cuda() {
        candle_core::DType::BF16
    } else {
        candle_core::DType::F32
    }
}

/// Resolve the VAE decode dtype for the current generation.
///
/// Reads `MOLD_VAE_DTYPE` to let users force a different precision for the
/// VAE decode pass than the rest of the model. Default (`auto` or unset)
/// preserves the per-pipeline historical choice (typically BF16 on CUDA,
/// F16 on SDXL/SD1.5, F32 on CPU). Forcing `fp32` fixes occasional banding
/// artifacts on FLUX/SD3 finetuned VAEs whose conv weights round badly at
/// half precision; the trade-off is ~2× peak VRAM on the decode step,
/// which `vae_tiling::decode_with_oom_fallback` will absorb by retrying
/// with tiles when the full-tensor decode OOMs.
///
/// Accepted values: `auto` (= unset), `bf16`, `fp16` / `f16`, `fp32` / `f32`.
/// Any other value emits a one-shot warn and falls back to the default —
/// loud enough to surface typos without failing the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeDtypePolicy {
    Auto,
    Bf16,
    F16,
    F32,
}

fn parse_vae_dtype_policy(value: Option<&str>) -> VaeDtypePolicy {
    match value.map(str::trim) {
        None | Some("") | Some("auto") => VaeDtypePolicy::Auto,
        Some("bf16") | Some("BF16") => VaeDtypePolicy::Bf16,
        Some("fp16") | Some("f16") | Some("FP16") | Some("F16") => VaeDtypePolicy::F16,
        Some("fp32") | Some("f32") | Some("FP32") | Some("F32") => VaeDtypePolicy::F32,
        Some(other) => {
            tracing::warn!(
                value = other,
                "MOLD_VAE_DTYPE has unrecognised value; expected one of auto/bf16/fp16/fp32 — falling back to default"
            );
            VaeDtypePolicy::Auto
        }
    }
}

pub fn resolved_vae_dtype_policy() -> VaeDtypePolicy {
    static CACHED: std::sync::OnceLock<VaeDtypePolicy> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_vae_dtype_policy(crate::runtime_env::value("MOLD_VAE_DTYPE").as_deref())
    })
}

pub(crate) fn resolve_vae_dtype(default_dtype: candle_core::DType) -> candle_core::DType {
    use candle_core::DType;
    match resolved_vae_dtype_policy() {
        VaeDtypePolicy::Auto => default_dtype,
        VaeDtypePolicy::Bf16 => DType::BF16,
        VaeDtypePolicy::F16 => DType::F16,
        VaeDtypePolicy::F32 => DType::F32,
    }
}

/// Format bytes as a human-readable size (e.g. "11.7 GB").
pub(crate) fn fmt_gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
}

// ── Decision functions ───────────────────────────────────────────────────────

/// Determine whether a component should be placed on GPU given free VRAM.
///
/// On Metal (Apple Silicon), always returns true — unified memory means GPU
/// placement is purely a compute performance decision, not a memory one.
/// On CUDA, checks that free discrete VRAM exceeds the threshold.
pub(crate) fn should_use_gpu(
    is_cuda: bool,
    is_metal: bool,
    _free_vram: u64,
    _threshold: u64,
) -> bool {
    if is_metal {
        return true;
    }
    is_cuda && _free_vram > _threshold
}

/// Check if block-level offloading should be auto-enabled.
///
/// Returns true when the transformer + activation headroom won't fit in VRAM
/// but there's enough for a single block + activations (~4GB). This allows
/// streaming blocks one at a time between CPU and GPU.
/// Minimum VRAM needed for one block + activations during offloaded inference.
pub(crate) const MIN_OFFLOAD_VRAM: u64 = 4_000_000_000; // 4 GB
/// Extra workspace needed when a full BF16/FP transformer stays GPU-resident.
///
/// `activation_bytes` covers resolution-scaled tensor peaks. It does not cover
/// CUDA allocator slack, attention workspaces, and short-lived per-layer
/// buffers that only appear once denoising starts. Without this reserve a
/// 23.8 GB FLUX transformer can pass preflight on a 24 GB card, then OOM in
/// the first denoise step before adaptive offload ever gets a chance to run.
pub(crate) const FULL_RESIDENT_RUNTIME_HEADROOM: u64 = 2_000_000_000; // 2 GB

/// Decide whether to enable block-level offloading.
///
/// `activation_bytes` is the per-request activation budget from
/// [`activation_bytes`] — scaled with resolution and dtype. Replaces the
/// previous fixed 3 GB `INFERENCE_HEADROOM` so a 768² generation isn't
/// false-offloaded on a 16 GB card while a 2048² generation isn't
/// under-budgeted on a 24 GB card. Full-resident inference also reserves
/// [`FULL_RESIDENT_RUNTIME_HEADROOM`] because kernels and allocator workspaces
/// are not represented in the safetensors byte count.
pub(crate) fn should_offload(transformer_size: u64, free_vram: u64, activation_bytes: u64) -> bool {
    let needed = transformer_size
        .saturating_add(activation_bytes)
        .saturating_add(FULL_RESIDENT_RUNTIME_HEADROOM);
    free_vram > 0 && needed > free_vram && free_vram >= MIN_OFFLOAD_VRAM
}

/// Check whether a model component fits comfortably in memory.
///
/// On CUDA, checks discrete VRAM against threshold.
/// On Metal, checks the passed `free_vram` (which should be available system
/// memory on unified-memory systems) against threshold.
pub(crate) fn fits_in_memory(
    is_cuda: bool,
    is_metal: bool,
    free_vram: u64,
    threshold: u64,
) -> bool {
    if is_metal {
        if free_vram > 0 {
            return free_vram > threshold;
        }
        // No memory info — assume it fits
        return true;
    }
    is_cuda && free_vram > threshold
}

// ── Memory budget ────────────────────────────────────────────────────────────

/// Estimate peak memory usage for a model given its component file sizes and loading strategy.
///
/// For Eager: sum of all component files + headroom.
/// For Sequential: max(encoder_total, transformer + VAE) + headroom.
///
/// **Single-file convention.** The catalog bridge sets
/// `paths.transformer == paths.vae` to a single `.safetensors` (transformer
/// and VAE keys both extracted from the same file at runtime — see
/// `crates/mold-cli/src/catalog_bridge.rs:196`). Naive `transformer + vae`
/// double-counts that file. We detect and elide it.
pub fn estimate_peak_memory(paths: &mold_core::ModelPaths, strategy: LoadStrategy) -> u64 {
    let file_size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let same_file = |a: &std::path::Path, b: &std::path::Path| -> bool {
        a == b
            || std::fs::canonicalize(a)
                .ok()
                .zip(std::fs::canonicalize(b).ok())
                .is_some_and(|(a, b)| a == b)
    };
    let path_matches_any = |path: &std::path::Path, paths: &[std::path::PathBuf]| -> bool {
        paths.iter().any(|candidate| same_file(path, candidate))
    };

    let transformer_size = if !paths.transformer_shards.is_empty() {
        paths.transformer_shards.iter().map(|p| file_size(p)).sum()
    } else {
        file_size(&paths.transformer)
    };
    // A two-expert pair is resident one expert at a time, so its demand is the
    // **max** over the two, not their sum. Taking the max rather than the first
    // expert matters because the pair need not be symmetric: a config or a
    // `MOLD_LOW_NOISE_TRANSFORMER_PATH` override can put a Q8 low expert behind
    // a Q5 high one, and both pass the architecture check. Sizing only the high
    // expert would admit ~10.8 GB for a plan that needs ~15.4 GB after the
    // swap, which OOMs mid-generation rather than at admission.
    let transformer_size = transformer_size.max(
        paths
            .low_noise_transformer
            .as_deref()
            .map(file_size)
            .unwrap_or(0),
    );
    // Single-file: transformer & vae point at the same on-disk file. Keys are
    // extracted from one mmap, so the file's bytes are paged in once. Don't
    // double-count. Use same-file identity rather than path-string equality
    // because catalog resolution can surface equivalent paths through distinct
    // spellings, and shard-backed configs can name the primary file in
    // transformer_shards.
    let vae_is_transformer_file = if paths.transformer_shards.is_empty() {
        same_file(&paths.transformer, &paths.vae)
    } else {
        paths
            .transformer_shards
            .iter()
            .any(|shard| same_file(shard, &paths.vae))
    };
    let vae_is_separate_file = !vae_is_transformer_file;
    let vae_size = if vae_is_separate_file {
        file_size(&paths.vae)
    } else {
        0
    };

    let mut base_component_paths: Vec<std::path::PathBuf> = paths.transformer_shards.to_vec();
    if base_component_paths.is_empty() {
        base_component_paths.push(paths.transformer.clone());
    }
    if vae_is_separate_file {
        base_component_paths.push(paths.vae.clone());
    }

    let mut counted_encoder_paths: Vec<std::path::PathBuf> = Vec::new();
    let mut encoder_size = |path: &std::path::Path| -> u64 {
        if path_matches_any(path, &base_component_paths)
            || path_matches_any(path, &counted_encoder_paths)
        {
            return 0;
        }
        counted_encoder_paths.push(path.to_path_buf());
        file_size(path)
    };

    let t5_size = paths
        .t5_encoder
        .as_ref()
        .map(|p| encoder_size(p))
        .unwrap_or(0);
    let clip_size = paths
        .clip_encoder
        .as_ref()
        .map(|p| encoder_size(p))
        .unwrap_or(0);
    let clip2_size = paths
        .clip_encoder_2
        .as_ref()
        .map(|p| encoder_size(p))
        .unwrap_or(0);
    let text_encoder_size: u64 = paths
        .text_encoder_files
        .iter()
        .map(|p| encoder_size(p))
        .sum();

    let encoder_total = t5_size + clip_size + clip2_size + text_encoder_size;

    match strategy {
        LoadStrategy::Eager => transformer_size + vae_size + encoder_total + MEMORY_BUDGET_HEADROOM,
        LoadStrategy::Sequential => {
            let peak_encoder = encoder_total;
            let peak_inference = transformer_size + vae_size;
            std::cmp::max(peak_encoder, peak_inference) + MEMORY_BUDGET_HEADROOM
        }
    }
}

/// Check whether estimated peak memory fits within available system memory.
///
/// Uses the generous free+inactive metric (can this model run at all?).
/// Returns a warning message if peak memory exceeds 80% of available memory,
/// or `None` if sufficient memory is available (or if memory info is unavailable).
pub fn check_memory_budget(
    paths: &mold_core::ModelPaths,
    strategy: LoadStrategy,
) -> Option<String> {
    let available = available_system_memory_bytes()?;
    let peak = estimate_peak_memory(paths, strategy);
    let threshold = available * 80 / 100;

    if peak > threshold {
        Some(format!(
            "Model needs ~{} but only ~{} available. \
             Consider a smaller quantized variant or close other applications.",
            fmt_gb(peak),
            fmt_gb(available),
        ))
    } else {
        None
    }
}

// ── Pre-flight memory guard ──────────────────────────────────────────────────

/// Check if loading a component of `size_bytes` plus its activation workspace
/// would exceed available system memory.
///
/// `activation_bytes` is the per-request activation budget from
/// [`activation_bytes`] — when the caller has a resolution / family hint use
/// that; otherwise pass `0` and the check degrades to "does the component
/// itself fit", matching the pre-budget behavior.
///
/// Uses available memory (free + inactive/reclaimable) as the primary metric,
/// since macOS moves recently-freed pages to inactive rather than free —
/// those are trivially reclaimable with no I/O. Hard-fails if
/// `size_bytes + activation_bytes` exceeds 90% of available memory; warns
/// (but proceeds) if the same quantity exceeds 2× free memory. On CUDA or
/// when memory info is unavailable, always returns Ok.
pub(crate) fn preflight_memory_check(
    component_name: &str,
    size_bytes: u64,
    activation_bytes: u64,
) -> anyhow::Result<()> {
    // --eager or MOLD_EAGER=1 bypasses the check
    if crate::runtime_env::value("MOLD_EAGER").is_some_and(|v| v == "1") {
        return Ok(());
    }

    let available = match available_system_memory_bytes() {
        Some(a) if a > 0 => a,
        _ => return Ok(()), // No info or CUDA — can't check
    };

    let free = free_system_memory_bytes();
    let total = size_bytes.saturating_add(activation_bytes);

    preflight_check_budget(component_name, total, available, free)
}

/// Pure logic for the preflight memory check, factored out for testability.
/// `available` = free + inactive (reclaimable); `free` = free pages only.
///
/// - Hard-fails if `size_bytes > 90%` of available (truly doesn't fit).
/// - Warns if `size_bytes > 2 * free` but within available (page reclamation expected).
fn preflight_check_budget(
    component_name: &str,
    size_bytes: u64,
    available: u64,
    free: Option<u64>,
) -> anyhow::Result<()> {
    // Hard fail: component won't fit even with full page reclamation
    if size_bytes > available * 90 / 100 {
        anyhow::bail!(
            "Not enough memory to load {} ({} needed, {} available).\n\
             Close other applications or use a smaller quantized model.",
            component_name,
            fmt_gb(size_bytes),
            fmt_gb(available),
        );
    }

    // Soft warning: fits in available but may trigger page reclamation
    if let Some(f) = free {
        if size_bytes > f * 2 {
            tracing::warn!(
                "{} ({}) exceeds free memory ({}), will reclaim inactive pages",
                component_name,
                fmt_gb(size_bytes),
                fmt_gb(f),
            );
        }
    }

    Ok(())
}

// ── Memory status reporting ──────────────────────────────────────────────────

/// Return a human-readable memory status string for display.
///
/// On CUDA: "VRAM: X.X GB free"
/// On macOS: "Memory: X.X GB free / Y.Y GB available"
/// Returns None if no memory info is available.
pub fn memory_status_string() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        if let Some(free) = free_vram_bytes(0) {
            return Some(format!("VRAM: {} free", fmt_gb(free)));
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(stats) = macos_vm_stats() {
            let available = stats.free + stats.inactive;
            return Some(format!(
                "Memory: {} free, {} available",
                fmt_gb(stats.free),
                fmt_gb(available),
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two Wan calibrations must each reproduce the renders they were
    /// fitted to, on an RTX 4090 with `wan22-t2v-a14b:q5` at 832x480 and CFG
    /// off. Both pairs are two renders differing only in frame count, so every
    /// fixed cost cancels and the pair pins the slope; the intercept is then
    /// the token-independent workspace plus the UMT5 floor charged elsewhere.
    ///
    /// This test exists because the flash pair is the easy thing to get wrong:
    /// dropping the score term shrinks the derived sum 44%, and a flash
    /// estimate that inherited math's 2.14 slope would under-charge by several
    /// GB, park too few blocks, and OOM. A change to either calibration that
    /// stops predicting its own measurements fails here.
    #[test]
    fn both_wan_calibrations_reproduce_their_measured_fits() {
        use crate::attention::AttentionBackend;

        const UMT5_FLOOR_MIB: u64 = 10_840;
        const MIB: u64 = 1024 * 1024;
        let a14b = WanActivationGeometry::a14b();

        // (backend, frames, measured whole-render peak in MiB)
        let fits = [
            (AttentionBackend::Math, 17u32, 16_074u64),
            (AttentionBackend::Math, 53, 21_354),
            (AttentionBackend::Flash, 17, 15_011),
            (AttentionBackend::Flash, 53, 19_011),
        ];

        for (backend, frames, measured_peak_mib) in fits {
            // `steps: 4, distilled: true` is not a convenience for zeroing the
            // step-cache charge -- it is what these renders were: the fits were
            // measured on `wan22-t2v-a14b:q5`, the 4-step Lightning distill
            // tier, which `WanStepCachePolicy::resolve` refuses on both counts.
            // Pricing them any other way would compare the calibration against
            // a render nobody measured.
            let budget = wan_calibrated_activation_bytes_for(
                832, 480, frames, a14b, false, 4, true, backend,
            );
            let predicted_mib = budget / MIB + UMT5_FLOOR_MIB;
            let measured = measured_peak_mib as i64;
            let predicted = predicted_mib as i64;
            assert!(
                (predicted - measured).abs() <= 300,
                "{backend:?} at {frames}f: predicted {predicted} MiB against a measured \
                 {measured} MiB — the calibration no longer reproduces its own fit"
            );
        }
    }

    /// The step-cache charge is exact, and it lands AFTER the fitted slope.
    ///
    /// This is the test that catches the error the issue's own plan would have
    /// produced. Charging inside `wan_activation_budget_bytes_for` puts the
    /// term on the wrong side of the 2.14x (math) multiplier, and nothing else
    /// in the suite notices: the calibration test prices a distilled 4-step
    /// tier where the charge is zero. The delta below is the arithmetic the
    /// buffers actually are, so a misplaced term fails here loudly.
    #[test]
    fn the_step_cache_charge_is_exact_and_lands_after_the_slope() {
        use crate::wan::step_cache::WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS;
        let a14b = WanActivationGeometry::a14b();
        let (width, height, frames) = (832u32, 480u32, 53u32);
        let tokens = wan_token_count(width, height, frames, a14b);

        for (cfg, buffers) in [(false, 4u64), (true, 6)] {
            // Explicit threshold, not the env: `MOLD_WAN_STEP_CACHE=off` is the
            // documented escape hatch and a developer may well have it exported.
            let charge = |steps: u32, distilled: bool| {
                wan_step_cache_bytes_for(
                    width,
                    height,
                    frames,
                    a14b,
                    cfg,
                    steps,
                    distilled,
                    Some(crate::wan::step_cache::AUTO_THRESHOLD),
                )
            };
            let engaged = charge(20, false);
            let refused = charge(4, true);

            let retained = tokens * a14b.dim * BF16_BYTES * buffers;
            let reduction = (WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS as u64).min(tokens)
                * a14b.dim
                * F32_BYTES
                * WAN_STEP_CACHE_REDUCTION_LIVE_BUFFERS;

            assert_eq!(
                engaged.saturating_sub(refused),
                retained + reduction,
                "cfg={cfg}: the step-cache charge must be exactly the buffers it \
                 names, unscaled by the fitted slope"
            );
        }
    }

    /// A refused cache costs nothing, on both refusal grounds.
    #[test]
    fn a_refused_step_cache_charges_nothing() {
        let a14b = WanActivationGeometry::a14b();
        for (steps, distilled) in [(4u32, false), (20, true), (4, true), (11, false)] {
            assert_eq!(
                wan_step_cache_bytes_for(
                    832,
                    480,
                    53,
                    a14b,
                    true,
                    steps,
                    distilled,
                    Some(crate::wan::step_cache::AUTO_THRESHOLD),
                ),
                0,
                "steps={steps} distilled={distilled} must charge nothing"
            );
        }
    }

    /// The charge must key on the ENGINE's policy, never on a copy of it.
    ///
    /// A second `steps >= 12 && !distilled` here is how the estimate and the
    /// denoise loop drift into disagreeing about whether the cache is even on
    /// -- the same failure the raw-vs-calibrated split already produced once.
    #[test]
    fn the_charge_follows_the_policy_not_a_copy_of_it() {
        use crate::wan::step_cache::{requested_threshold, WanStepCachePolicy};
        let a14b = WanActivationGeometry::a14b();
        let requested = requested_threshold().unwrap_or(None);

        for steps in [4u32, 11, 12, 20, 50] {
            for distilled in [false, true] {
                let (policy, _) = WanStepCachePolicy::resolve(requested, steps, distilled);
                let charged = wan_step_cache_bytes(832, 480, 53, a14b, true, steps, distilled) > 0;
                assert_eq!(
                    charged,
                    policy.is_on(),
                    "steps={steps} distilled={distilled}: the charge and the policy disagree"
                );
            }
        }
    }

    /// Flash must always price BELOW math at the same shape, and the gap must
    /// grow with the clip. If a change ever inverted this, the backend-aware
    /// estimate would be charging for a tile that is not allocated.
    #[test]
    fn flash_prices_below_math_and_the_gap_grows_with_frames() {
        use crate::attention::AttentionBackend;
        let a14b = WanActivationGeometry::a14b();
        let mut previous_gap = 0u64;
        for frames in [17u32, 53, 81, 121] {
            let math = wan_calibrated_activation_bytes_for(
                832,
                480,
                frames,
                a14b,
                false,
                4,
                true,
                AttentionBackend::Math,
            );
            let flash = wan_calibrated_activation_bytes_for(
                832,
                480,
                frames,
                a14b,
                false,
                4,
                true,
                AttentionBackend::Flash,
            );
            assert!(
                flash < math,
                "{frames}f: flash ({flash}) must price below math ({math})"
            );
            let gap = math - flash;
            assert!(
                gap > previous_gap,
                "{frames}f: the flash saving must grow with the clip"
            );
            previous_gap = gap;
        }
    }

    /// The score term is the whole difference between the two derived sums,
    /// and it is exactly the math-attention tile: `2 * heads * chunk * bf16`.
    #[test]
    fn the_derived_difference_is_exactly_the_score_tile() {
        use crate::attention::AttentionBackend;
        let a14b = WanActivationGeometry::a14b();
        let tokens = wan_token_count(832, 480, 53, a14b);
        let math = wan_activation_budget_bytes_for(832, 480, 53, a14b, AttentionBackend::Math);
        let flash = wan_activation_budget_bytes_for(832, 480, 53, a14b, AttentionBackend::Flash);
        let per_token_tile = 2 * a14b.num_heads * WAN_ATTENTION_QUERY_CHUNK * BF16_BYTES;
        assert_eq!(math - flash, tokens * per_token_tile);
    }

    fn identified_gpu(ordinal: usize, uuid: [u8; 16]) -> DiscoveredGpu {
        DiscoveredGpu {
            ordinal,
            stable_id: Some(stable_cuda_id(uuid)),
            raw_cuda_uuid: Some(uuid),
            device_kind: Some(CudaDeviceKind::UnknownCuda),
            identity_error: None,
            backend: GpuBackend::Cuda,
            name: format!("gpu{ordinal}"),
            compute_capability: Some((8, 6)),
            pci_bus_id: Some(format!("00000000:{ordinal:02x}:00.0")),
            total_vram_bytes: 24 * GB,
            free_vram_bytes: 20 * GB,
        }
    }

    #[test]
    fn stable_cuda_identity_uses_lowercase_raw_uuid_bytes() {
        assert_eq!(
            stable_cuda_id([
                0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
                0x32, 0x10,
            ]),
            "cuda:0123456789abcdeffedcba9876543210"
        );
    }

    #[test]
    fn stable_metal_selector_resolves_the_advertised_device_id() {
        let metal = DiscoveredGpu {
            ordinal: 0,
            stable_id: Some("metal:default".to_string()),
            raw_cuda_uuid: None,
            device_kind: None,
            identity_error: None,
            backend: GpuBackend::Metal,
            name: "Apple Metal GPU".to_string(),
            compute_capability: None,
            pci_bus_id: None,
            total_vram_bytes: 64_000_000_000,
            free_vram_bytes: 48_000_000_000,
        };
        let selected = resolve_gpu_selection(
            std::slice::from_ref(&metal),
            &GpuSelection::parse("metal:default").unwrap(),
        )
        .unwrap();

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].stable_id.as_deref(), Some("metal:default"));
        assert_eq!(selected[0].backend, GpuBackend::Metal);
    }

    #[test]
    fn cuda_visible_tokens_prove_only_explicit_full_or_mig_kind() {
        assert_eq!(
            cuda_device_kind_from_visible_token(Some("GPU-01234567-89ab-cdef-0123-456789abcdef")),
            CudaDeviceKind::FullGpu
        );
        assert_eq!(
            cuda_device_kind_from_visible_token(Some("MIG-01234567-89ab-cdef-0123-456789abcdef")),
            CudaDeviceKind::Mig
        );
        assert_eq!(
            cuda_device_kind_from_visible_token(Some("1")),
            CudaDeviceKind::UnknownCuda
        );
        assert_eq!(
            cuda_device_kind_from_visible_token(None),
            CudaDeviceKind::UnknownCuda
        );
    }

    #[test]
    fn startup_selection_supports_none_all_and_legacy_ordinals() {
        let gpus = vec![identified_gpu(0, [0x11; 16]), identified_gpu(1, [0x22; 16])];

        assert!(resolve_gpu_selection(&gpus, &GpuSelection::None)
            .unwrap()
            .is_empty());
        assert_eq!(
            resolve_gpu_selection(&gpus, &GpuSelection::All)
                .unwrap()
                .iter()
                .map(|gpu| gpu.ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            resolve_gpu_selection(
                &gpus,
                &GpuSelection::Specific(vec![GpuSelector::Ordinal(1)])
            )
            .unwrap()[0]
                .stable_id
                .as_deref(),
            Some("cuda:22222222222222222222222222222222")
        );
    }

    #[test]
    fn startup_selection_tracks_uuid_across_visible_ordinal_reordering() {
        let first = vec![identified_gpu(0, [0xaa; 16]), identified_gpu(1, [0xbb; 16])];
        let reordered = vec![identified_gpu(0, [0xbb; 16]), identified_gpu(1, [0xaa; 16])];
        let selection = GpuSelection::Specific(vec![GpuSelector::Identifier(
            "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        )]);

        assert_eq!(
            resolve_gpu_selection(&first, &selection).unwrap()[0].ordinal,
            0
        );
        assert_eq!(
            resolve_gpu_selection(&reordered, &selection).unwrap()[0].ordinal,
            1
        );
    }

    #[test]
    fn startup_selection_accepts_nvidia_gpu_and_mig_uuid_forms() {
        let mut full = identified_gpu(
            0,
            [
                0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ],
        );
        full.device_kind = Some(CudaDeviceKind::FullGpu);
        let mut mig = identified_gpu(
            1,
            [
                0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
                0x32, 0x10,
            ],
        );
        mig.device_kind = Some(CudaDeviceKind::Mig);
        let gpus = vec![full, mig];

        let full_selection = GpuSelection::Specific(vec![GpuSelector::Identifier(
            "GPU-01234567-89ab-cdef-0123-456789abcdef".to_string(),
        )]);
        assert_eq!(
            resolve_gpu_selection(&gpus, &full_selection).unwrap()[0].ordinal,
            0
        );

        let mig_selection = GpuSelection::Specific(vec![GpuSelector::Identifier(
            "MIG-fedcba98-7654-3210-fedc-ba9876543210".to_string(),
        )]);
        assert_eq!(
            resolve_gpu_selection(&gpus, &mig_selection).unwrap()[0].ordinal,
            1
        );
    }

    #[test]
    fn startup_selection_rejects_ambiguous_or_missing_uuid_prefixes() {
        let gpus = vec![
            identified_gpu(0, [0xaa; 16]),
            identified_gpu(1, [0xaa, 0xab, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ];

        let ambiguous =
            GpuSelection::Specific(vec![GpuSelector::Identifier("cuda:aa".to_string())]);
        let error = resolve_gpu_selection(&gpus, &ambiguous)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous"));
        assert!(error.contains("cuda:aaaaaaaa"));
        assert!(error.contains("cuda:aaab0000"));

        let missing = GpuSelection::Specific(vec![GpuSelector::Identifier("GPU-ff".to_string())]);
        let error = resolve_gpu_selection(&gpus, &missing)
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not match"));
        assert!(error.contains("visible choices"));
    }

    #[test]
    fn uuid_lookup_failures_never_fall_back_to_ordinal_identity() {
        let unavailable = DiscoveredGpu {
            ordinal: 0,
            stable_id: None,
            raw_cuda_uuid: None,
            device_kind: Some(CudaDeviceKind::UnknownCuda),
            identity_error: Some("UUID unavailable".to_string()),
            backend: GpuBackend::Cuda,
            name: "visible but unavailable".to_string(),
            compute_capability: Some((8, 6)),
            pci_bus_id: None,
            total_vram_bytes: 24 * GB,
            free_vram_bytes: 20 * GB,
        };

        assert!(
            resolve_gpu_selection(std::slice::from_ref(&unavailable), &GpuSelection::All)
                .unwrap()
                .is_empty()
        );
        let error = resolve_gpu_selection(
            &[unavailable],
            &GpuSelection::Specific(vec![GpuSelector::Ordinal(0)]),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("stable CUDA identity"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn live_cuda_discovery_uses_driver_uuid_identity_when_devices_exist() {
        let gpus = discover_gpus();
        for gpu in gpus {
            assert_eq!(gpu.backend, GpuBackend::Cuda);
            let raw = gpu
                .raw_cuda_uuid
                .unwrap_or_else(|| panic!("visible GPU {} has no raw CUDA UUID", gpu.ordinal));
            let expected_id = stable_cuda_id(raw);
            assert_eq!(gpu.stable_id.as_deref(), Some(expected_id.as_str()));
            assert!(gpu.identity_error.is_none());
            assert!(gpu.compute_capability.is_some());
            assert!(gpu.pci_bus_id.as_deref().is_some_and(|id| id.contains(':')));
        }
    }

    // --- fmt_gb tests ---

    #[test]
    fn fmt_gb_zero() {
        assert_eq!(fmt_gb(0), "0.0 GB");
    }

    #[test]
    fn fmt_gb_one_gb() {
        assert_eq!(fmt_gb(1_000_000_000), "1.0 GB");
    }

    #[test]
    fn fmt_gb_fractional() {
        assert_eq!(fmt_gb(14_600_000_000), "14.6 GB");
    }

    #[test]
    fn fmt_gb_small() {
        assert_eq!(fmt_gb(800_000_000), "0.8 GB");
    }

    // --- macOS memory query ---

    #[cfg(target_os = "macos")]
    #[test]
    fn free_system_memory_returns_positive() {
        let mem = free_system_memory_bytes();
        assert!(mem.is_some());
        assert!(mem.unwrap() > 0, "free system memory should be positive");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_includes_inactive() {
        let free = free_system_memory_bytes().unwrap();
        let available = available_system_memory_bytes().unwrap();
        assert!(
            available >= free,
            "available (free+inactive) should be >= free alone"
        );
    }

    // --- free_vram_bytes ---

    #[test]
    fn free_vram_returns_some_on_macos_or_none_on_other() {
        let _result = free_vram_bytes(0);
        #[cfg(target_os = "macos")]
        assert!(_result.is_some(), "macOS should return system memory info");
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        assert_eq!(_result, None);
    }

    /// On macOS (unified memory), free_vram_bytes should return available memory
    /// (free + inactive), not just free pages. This ensures variant selection
    /// doesn't reject quantized encoders that would actually fit.
    #[cfg(target_os = "macos")]
    #[test]
    fn free_vram_returns_available_not_just_free_on_macos() {
        let vram = free_vram_bytes(0).unwrap();
        let available = available_system_memory_bytes().unwrap();
        let free = free_system_memory_bytes().unwrap();
        // free_vram_bytes should return available (>= free), not just free
        assert!(
            vram >= free,
            "free_vram_bytes ({vram}) should be >= free_system_memory ({free})"
        );
        // Allow small delta between separate syscalls (TOCTOU: inactive pages may
        // change between the two macos_vm_stats() calls on a busy system)
        let max_drift = 256 * 4096; // 256 pages (~1MB)
        assert!(
            vram.abs_diff(available) < max_drift,
            "free_vram_bytes ({vram}) should approximately equal available_system_memory ({available})"
        );
    }

    /// Metal discovery must budget on the same reclaimable-memory metric
    /// `free_vram_bytes` already uses. It called `free_system_memory_bytes()`
    /// instead, so a 48 GB M4 Max with 23 GB reclaimable advertised ~7 GB free.
    /// The scheduler clamps `free + reclaimable_cache` to `total_vram_bytes`,
    /// which discovery also set to a transient availability reading rather than
    /// installed RAM, so every LTX-2 plan was rejected before a weight was read
    /// ("metal:default needs ~13.9 GB of ~10.1 GB usable").
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    #[test]
    fn metal_discovery_budgets_on_reclaimable_memory() {
        let gpus = discover_gpus();
        let metal = gpus
            .iter()
            .find(|gpu| gpu.backend == GpuBackend::Metal)
            .expect("macOS exposes a Metal device");
        let available = available_system_memory_bytes().expect("macOS VM statistics");
        let installed = total_system_memory_bytes().expect("macOS reports installed RAM");

        // Separate syscalls sample live VM state, so compare with drift room.
        let max_drift = 512 * 1024 * 1024;
        assert!(
            metal.free_vram_bytes.abs_diff(available) < max_drift,
            "discovered free {} should track reclaimable {available}, not free pages alone",
            metal.free_vram_bytes,
        );
        assert_eq!(
            metal.total_vram_bytes, installed,
            "total capacity is installed RAM, not a transient availability reading",
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn installed_memory_is_reported_and_bounds_available() {
        let installed = total_system_memory_bytes().expect("macOS reports installed RAM");
        let available = available_system_memory_bytes().expect("macOS VM statistics");
        assert!(installed > 0, "installed RAM should be positive");
        assert!(
            installed >= available,
            "installed RAM ({installed}) must bound available ({available})"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn used_swap_is_reported_by_macos() {
        used_system_swap_bytes().expect("macOS reports vm.swapusage");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn large_metal_capacity_retains_the_canonical_host_safety_floor() {
        let installed = total_system_memory_bytes().expect("macOS reports installed RAM");
        let floor = (installed.saturating_mul(15) / 100).max(8 << 30);
        assert_eq!(
            metal_unified_capacity_with_safety_floor(1),
            installed.saturating_sub(floor)
        );
        assert!(floor >= 8 << 30);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn live_metal_budget_preserves_the_floor_under_current_pressure() {
        let installed = total_system_memory_bytes().expect("macOS reports installed RAM");
        let floor = (installed.saturating_mul(15) / 100).max(8 << 30);
        let live = installed.saturating_sub(4 << 30);
        assert_eq!(metal_live_allocation_budget(live), live - floor);
        assert_eq!(metal_live_allocation_budget(floor), 0);
        assert_eq!(metal_live_allocation_budget(floor / 2), 0);
    }

    // --- should_use_gpu: Metal always GPU ---

    #[test]
    fn metal_always_uses_gpu() {
        assert!(should_use_gpu(false, true, 0, T5_VRAM_THRESHOLD));
        assert!(should_use_gpu(false, true, 1_000, T5_VRAM_THRESHOLD));
        assert!(should_use_gpu(
            false,
            true,
            100_000_000_000,
            T5_VRAM_THRESHOLD
        ));
    }

    // --- fits_in_memory: Metal threshold-based ---

    #[test]
    fn metal_fits_when_enough_free() {
        assert!(fits_in_memory(
            false,
            true,
            20_000_000_000,
            T5_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn metal_does_not_fit_when_free_low() {
        assert!(!fits_in_memory(
            false,
            true,
            2_000_000_000,
            T5_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn metal_fits_fallback_when_no_memory_info() {
        assert!(fits_in_memory(false, true, 0, T5_VRAM_THRESHOLD));
    }

    // --- CUDA threshold tests ---

    #[test]
    fn t5_on_gpu_when_plenty_of_vram() {
        assert!(should_use_gpu(
            true,
            false,
            16_700_000_000,
            T5_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn t5_on_cpu_when_q6_on_24gb() {
        assert!(!should_use_gpu(
            true,
            false,
            14_600_000_000,
            T5_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn t5_on_cpu_when_q8_on_24gb() {
        assert!(!should_use_gpu(
            true,
            false,
            11_700_000_000,
            T5_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn t5_on_cpu_when_bf16_fills_vram() {
        assert!(!should_use_gpu(true, false, 700_000_000, T5_VRAM_THRESHOLD));
    }

    #[test]
    fn t5_on_cpu_when_exactly_at_threshold() {
        assert!(!should_use_gpu(
            true,
            false,
            T5_VRAM_THRESHOLD,
            T5_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn t5_on_cpu_when_no_gpu() {
        assert!(!should_use_gpu(
            false,
            false,
            100_000_000_000,
            T5_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn t5_on_gpu_on_48gb_card() {
        assert!(should_use_gpu(
            true,
            false,
            35_700_000_000,
            T5_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn clip_on_gpu_when_vram_available() {
        assert!(should_use_gpu(
            true,
            false,
            7_500_000_000,
            CLIP_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn clip_on_gpu_with_minimal_vram() {
        assert!(should_use_gpu(
            true,
            false,
            900_000_000,
            CLIP_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn clip_on_cpu_when_vram_tight() {
        assert!(!should_use_gpu(
            true,
            false,
            500_000_000,
            CLIP_VRAM_THRESHOLD
        ));
    }

    // --- Threshold constant sanity checks ---

    #[test]
    fn t5_threshold_accounts_for_headroom() {
        let threshold = std::hint::black_box(T5_VRAM_THRESHOLD);
        assert!(threshold > 9_200_000_000);
        assert!(threshold < 25_000_000_000);
    }

    #[test]
    fn clip_threshold_accounts_for_headroom() {
        let threshold = std::hint::black_box(CLIP_VRAM_THRESHOLD);
        assert!(threshold > 246_000_000);
        assert!(threshold < 2_000_000_000);
    }

    // --- Dynamic T5 threshold tests ---

    #[test]
    fn t5_threshold_for_fp16() {
        let threshold = t5_vram_threshold(9_200_000_000);
        assert!(threshold > 9_200_000_000);
        assert!(threshold <= 16_000_000_000);
    }

    #[test]
    fn t5_threshold_for_q8() {
        let threshold = t5_vram_threshold(5_060_000_000);
        assert_eq!(threshold, 7_060_000_000);
        assert!(should_use_gpu(true, false, 17_000_000_000, threshold));
        assert!(should_use_gpu(true, false, 12_000_000_000, threshold));
    }

    #[test]
    fn t5_threshold_for_q5() {
        let threshold = t5_vram_threshold(3_390_000_000);
        assert_eq!(threshold, 5_390_000_000);
        assert!(should_use_gpu(true, false, 12_000_000_000, threshold));
    }

    #[test]
    fn t5_threshold_for_q3() {
        let threshold = t5_vram_threshold(2_100_000_000);
        assert_eq!(threshold, 4_100_000_000);
    }

    // --- Qwen3 VRAM threshold tests ---

    #[test]
    fn qwen3_fp16_threshold_with_drop_and_reload() {
        assert_eq!(QWEN3_FP16_VRAM_THRESHOLD, 10_200_000_000);
        assert!(should_use_gpu(
            true,
            false,
            17_000_000_000,
            QWEN3_FP16_VRAM_THRESHOLD
        ));
        assert!(should_use_gpu(
            true,
            false,
            19_000_000_000,
            QWEN3_FP16_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn qwen3_threshold_for_q8() {
        let threshold = qwen3_vram_threshold(4_280_000_000);
        assert_eq!(threshold, 6_280_000_000);
        assert!(should_use_gpu(true, false, 17_000_000_000, threshold));
    }

    #[test]
    fn qwen3_threshold_for_q3() {
        let threshold = qwen3_vram_threshold(2_080_000_000);
        assert_eq!(threshold, 4_080_000_000);
        assert!(should_use_gpu(true, false, 5_000_000_000, threshold));
    }

    #[test]
    fn qwen2_threshold_for_q6() {
        let threshold = qwen2_vram_threshold(6_250_000_000);
        assert_eq!(threshold, 8_250_000_000);
        assert!(should_use_gpu(true, false, 12_000_000_000, threshold));
    }

    #[test]
    fn qwen3_fp16_does_not_fit_with_bf16_transformer() {
        assert!(!should_use_gpu(
            true,
            false,
            400_000_000,
            QWEN3_FP16_VRAM_THRESHOLD
        ));
    }

    // --- preflight_check_budget (preflight memory check logic) ---

    const GB: u64 = 1_000_000_000;

    #[test]
    fn budget_ok_when_plenty_of_memory() {
        // 5 GB component, 20 GB available, 10 GB free — no issue
        let result = preflight_check_budget("UNet", 5 * GB, 20 * GB, Some(10 * GB));
        assert!(result.is_ok());
    }

    #[test]
    fn budget_hard_fail_when_exceeds_90pct_available() {
        // 19 GB component, 20 GB available → 19 > 18 (90% of 20) → fail
        let result = preflight_check_budget("UNet", 19 * GB, 20 * GB, Some(GB));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Not enough memory"), "got: {msg}");
    }

    #[test]
    fn budget_ok_at_exactly_90pct_available() {
        // 18 GB component, 20 GB available → 18 == 18 (90% of 20) → pass (not >)
        let result = preflight_check_budget("UNet", 18 * GB, 20 * GB, Some(GB));
        assert!(result.is_ok());
    }

    #[test]
    fn budget_hard_fail_just_over_90pct() {
        // Component barely over 90% of available
        let available = 10 * GB;
        let size = available * 90 / 100 + 1; // one byte over
        let result = preflight_check_budget("Transformer", size, available, Some(0));
        assert!(result.is_err());
    }

    #[test]
    fn budget_ok_when_low_free_but_high_available() {
        // The key scenario: 5 GB UNet, only 0.4 GB free, but 18 GB available
        // Old code would bail here; new code proceeds with a warning
        let result = preflight_check_budget("UNet", 5 * GB, 18 * GB, Some(400_000_000));
        assert!(result.is_ok());
    }

    #[test]
    fn budget_ok_with_no_free_info() {
        // free = None (e.g. CUDA), available is sufficient → ok
        let result = preflight_check_budget("UNet", 5 * GB, 20 * GB, None);
        assert!(result.is_ok());
    }

    #[test]
    fn budget_hard_fail_with_no_free_info() {
        // free = None but available too low
        let result = preflight_check_budget("UNet", 19 * GB, 20 * GB, None);
        assert!(result.is_err());
    }

    #[test]
    fn budget_ok_small_component() {
        // Tiny component always fits
        let result = preflight_check_budget("CLIP-L", 250_000_000, 16 * GB, Some(8 * GB));
        assert!(result.is_ok());
    }

    #[test]
    fn budget_error_message_includes_component_name() {
        let result = preflight_check_budget("MyModel", 19 * GB, 20 * GB, Some(GB));
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("MyModel"),
            "error should mention component name"
        );
    }

    #[test]
    fn budget_error_message_includes_sizes() {
        let result = preflight_check_budget("UNet", 19 * GB, 20 * GB, Some(GB));
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("19.0 GB"), "should show needed size");
        assert!(msg.contains("20.0 GB"), "should show available size");
    }

    // ── should_offload tests ─────────────────────────────────────────────

    /// 1024² FLUX bf16 cfg=1 activation budget — used as a default in
    /// existing offload tests so resolution scaling is exercised in the
    /// dedicated `should_offload_uses_resolution_scaled_activation` test.
    fn flux_1024_activation() -> u64 {
        activation_bytes(1024, 1024, 1, 2, ActivationFamily::FluxDit)
    }

    #[test]
    fn offload_when_transformer_exceeds_vram() {
        // 24GB transformer, 16GB free → needs offloading
        assert!(should_offload(24 * GB, 16 * GB, flux_1024_activation()));
    }

    #[test]
    fn offload_when_transformer_fits_but_no_headroom() {
        // 23.8GB transformer on 24.5GB usable free: the file plus activations
        // appears to fit, but runtime workspace does not. This is the 24GB
        // CUDA-card FLUX BF16 regression: without resident-runtime headroom,
        // full load succeeds and denoising OOMs before adaptive offload starts.
        let xformer = 23_800_000_000;
        let free = 24_500_000_000;
        assert!(should_offload(xformer, free, flux_1024_activation()));
    }

    #[test]
    fn no_offload_when_plenty_of_vram() {
        // 12GB transformer on 24GB free → plenty of room
        assert!(!should_offload(12 * GB, 24 * GB, flux_1024_activation()));
    }

    #[test]
    fn no_offload_when_vram_unknown() {
        // free = 0 means we couldn't query VRAM
        assert!(!should_offload(24 * GB, 0, flux_1024_activation()));
    }

    #[test]
    fn no_offload_when_vram_too_small_for_single_block() {
        // 24GB transformer but only 2GB free — not enough for even one block
        assert!(!should_offload(24 * GB, 2 * GB, flux_1024_activation()));
    }

    // ── activation_bytes tests ─────────────────────────────────────────

    /// Doubling each axis quadruples the area, so the activation budget
    /// should also quadruple (modulo the floor) — the core scaling property
    /// that fixed-headroom missed.
    #[test]
    fn activation_bytes_scales_with_area() {
        let small = activation_bytes(1024, 1024, 1, 2, ActivationFamily::FluxDit);
        let big = activation_bytes(2048, 2048, 1, 2, ActivationFamily::FluxDit);
        // Both must be above the floor for the scaling to be observable.
        assert!(
            small > 256_000_000,
            "1024² FLUX bf16 should clear the floor, got {small}"
        );
        let ratio = big as f64 / small as f64;
        assert!(
            (ratio - 4.0).abs() < 0.04,
            "expected 4× scaling, got {ratio:.4} (small={small}, big={big})"
        );
    }

    /// bf16 (dtype_bytes=2) should produce exactly half the budget of f32
    /// (dtype_bytes=4) at the same resolution — both above the floor.
    #[test]
    fn activation_bytes_scales_with_dtype() {
        // Pick a resolution large enough that both dtype variants clear the floor.
        let bf16 = activation_bytes(2048, 2048, 1, 2, ActivationFamily::FluxDit);
        let f32 = activation_bytes(2048, 2048, 1, 4, ActivationFamily::FluxDit);
        assert!(bf16 > 256_000_000, "2048² FLUX bf16 should clear floor");
        let ratio = f32 as f64 / bf16 as f64;
        assert!(
            (ratio - 2.0).abs() < 0.02,
            "expected f32 = 2× bf16, got {ratio:.4} (bf16={bf16}, f32={f32})"
        );
    }

    /// CFG-style batch=2 should double the budget vs non-CFG batch=1.
    #[test]
    fn activation_bytes_scales_with_batch() {
        let b1 = activation_bytes(2048, 2048, 1, 2, ActivationFamily::SdxlUnet);
        let b2 = activation_bytes(2048, 2048, 2, 2, ActivationFamily::SdxlUnet);
        assert!(b1 > 256_000_000, "2048² SDXL bf16 b=1 should clear floor");
        let ratio = b2 as f64 / b1 as f64;
        assert!(
            (ratio - 2.0).abs() < 0.02,
            "expected b=2 → 2× b=1, got {ratio:.4} (b1={b1}, b2={b2})"
        );
    }

    // ── LTX-2 token-based activation budget ────────────────────────────

    /// The 1024² × 97-frame stage-2 render is 13,312 tokens; the 512² stage-1
    /// render of the same job is 3,328. Token count is what the transformer
    /// actually sees — not `w × h`.
    #[test]
    fn ltx2_token_count_matches_latent_patch_grid() {
        assert_eq!(ltx2_token_count(1024, 1024, 97), 13_312);
        assert_eq!(ltx2_token_count(512, 512, 97), 3_328);
        // Single frame still yields one latent frame.
        assert_eq!(ltx2_token_count(1024, 1024, 1), 1_024);
    }

    /// Golden per-token slopes: 112 KiB/token unconditioned, 160 KiB/token
    /// conditioned (the extra 48 KiB is the per-token AdaLN row), plus the
    /// BF16 attention tile (64 KiB/token past 1,024 tokens) and the flat
    /// 1 GiB workspace.
    #[test]
    fn ltx2_activation_budget_models_ff_and_adaln() {
        let uncond = ltx2_activation_budget_bytes(1024, 1024, 97, false, None);
        let cond = ltx2_activation_budget_bytes(1024, 1024, 97, true, None);
        let tile = ltx2_attention_tile_bytes(13_312, LTX2_ATTENTION_HEADS);
        assert_eq!(tile, 32 * 512 * 13_312 * 2 * 2);
        assert_eq!(uncond, 13_312 * 114_688 + tile + 1_073_741_824);
        assert_eq!(cond, 13_312 * 163_840 + tile + 1_073_741_824);
        assert_eq!(uncond, 3_472_883_712);
        assert_eq!(cond, 4_127_195_136);
        assert_eq!(cond - uncond, 13_312 * LTX2_ADALN_DEFAULT_DIM * BF16_BYTES);
    }

    /// The tile follows the dispatcher's own CUDA chunk policy (#735): the
    /// full `[heads, T, T]` matrix up to 1,024 tokens, then a 512-row query
    /// chunk against the whole key axis, in BF16, twice (scores + softmax).
    #[test]
    fn ltx2_attention_tile_mirrors_the_dispatcher_chunk_policy() {
        // Below the chunk threshold the whole matrix is one tile.
        assert_eq!(
            ltx2_attention_tile_bytes(1_024, 32),
            32 * 1_024 * 1_024 * 2 * 2
        );
        assert_eq!(ltx2_attention_tile_bytes(100, 32), 32 * 100 * 100 * 2 * 2);
        // Past it the query axis is chunked at 512 rows, so the tile grows
        // linearly with the key axis: 64 KiB per token at 32 heads.
        assert_eq!(
            ltx2_attention_tile_bytes(1_025, 32),
            32 * 512 * 1_025 * 2 * 2
        );
        assert_eq!(ltx2_attention_tile_bytes(13_376, 32), 13_376 * 65_536);
        assert_eq!(ltx2_attention_tile_bytes(3_344, 32), 3_344 * 65_536);
        // Stage-2 1216x704x121 (the LTX-2.5 default): 876 MB.
        assert_eq!(ltx2_attention_tile_bytes(13_376, 32), 876_609_536);
        assert_eq!(ltx2_attention_tile_bytes(0, 32), 0);
        assert_eq!(ltx2_attention_tile_bytes(u64::MAX, 32), u64::MAX);
    }

    /// LTX-2.3's 22B checkpoint ships `adaln_single.linear.weight` at
    /// `[36864, 4096]` — **nine** AdaLN components, not the 19B's six. Measured
    /// from the two checkpoints' safetensors headers. Assuming the 19B width
    /// under-reserved a conditioned 22B render by 327 MB at the stage-2 shape,
    /// in the exact budget this estimator exists to keep honest.
    #[test]
    fn ltx2_activation_budget_follows_the_checkpoints_adaln_width() {
        const NINE_COMPONENT_ADALN: u64 = 36_864;
        let six = ltx2_activation_budget_bytes(1024, 1024, 97, true, Some(LTX2_ADALN_DEFAULT_DIM));
        let nine = ltx2_activation_budget_bytes(1024, 1024, 97, true, Some(NINE_COMPONENT_ADALN));

        assert_eq!(
            six,
            ltx2_activation_budget_bytes(1024, 1024, 97, true, None),
            "an unknown width must fall back to the 19B/6-component default"
        );
        assert_eq!(
            nine - six,
            13_312 * (NINE_COMPONENT_ADALN - LTX2_ADALN_DEFAULT_DIM) * BF16_BYTES,
            "the extra three components must be priced per token"
        );
        assert_eq!(nine - six, 327_155_712);

        // An unconditioned render never materializes the per-token modulation,
        // so the width cannot matter there.
        assert_eq!(
            ltx2_activation_budget_bytes(1024, 1024, 97, false, Some(NINE_COMPONENT_ADALN)),
            ltx2_activation_budget_bytes(1024, 1024, 97, false, None),
        );
    }

    /// A half-resolution stage-1 render costs a quarter of the tokens, so it
    /// must cost a quarter of the *token* term — the old pixel-area heuristic
    /// charged it the full-resolution budget.
    #[test]
    fn ltx2_activation_budget_scales_with_tokens_not_pixels() {
        let stage1 = ltx2_activation_budget_bytes(512, 512, 97, false, None);
        let stage2 = ltx2_activation_budget_bytes(1024, 1024, 97, false, None);
        assert_eq!(
            stage1,
            3_328 * 114_688 + ltx2_attention_tile_bytes(3_328, 32) + 1_073_741_824
        );
        // Both shapes sit past the 1,024-token chunk threshold, so the
        // attention tile is linear in tokens there too.
        let stage1_tokens = stage1 - 1_073_741_824;
        let stage2_tokens = stage2 - 1_073_741_824;
        assert_eq!(stage2_tokens, stage1_tokens * 4);
    }

    /// Even a degenerate shape reserves the flat workspace.
    #[test]
    fn ltx2_activation_budget_floors_at_fixed_workspace() {
        let tiny = ltx2_activation_budget_bytes(0, 0, 0, false, None);
        assert!(tiny >= 1_073_741_824, "got {tiny}");
    }

    /// Tiny inputs return at least 256 MB (kernel-workspace floor).
    #[test]
    fn activation_bytes_floors_at_256mb() {
        let tiny = activation_bytes(64, 64, 1, 2, ActivationFamily::FluxDit);
        assert_eq!(
            tiny, 256_000_000,
            "tiny input must hit the 256 MB floor exactly, got {tiny}"
        );
        // SmallTransformer family with same inputs should also floor.
        let tiny_te = activation_bytes(64, 64, 1, 2, ActivationFamily::SmallTransformer);
        assert_eq!(tiny_te, 256_000_000);
    }

    /// 1024² FLUX bf16 cfg=1 must land in the [200 MB, 1 GB] sanity band — if
    /// this drifts, recalibrate the FLUX dit factor *and* update the comment
    /// in `activation_bytes` so the empirical anchor stays trustworthy.
    #[test]
    fn activation_bytes_flux_dit_at_1024_is_in_expected_range() {
        let budget = activation_bytes(1024, 1024, 1, 2, ActivationFamily::FluxDit);
        assert!(
            (200_000_000..=1_000_000_000).contains(&budget),
            "FLUX 1024² bf16 cfg=1 budget {budget} bytes outside [200 MB, 1 GB]"
        );
    }

    /// At 2048² the activation budget should be substantially higher than at
    /// 768², so the same `(transformer, free_vram)` pair flips into "offload"
    /// at the larger resolution. This is the regression that motivated the
    /// switch from the fixed 3 GB headroom: under the old constant the
    /// transformer alone would have already triggered offload (or not) the
    /// same way at both resolutions.
    #[test]
    fn should_offload_uses_resolution_scaled_activation() {
        // Pick a transformer + VRAM pair where the answer still depends on
        // resolution after the fixed resident-runtime reserve is included.
        // 22 GB transformer on 24.5 GB usable free: 768² fits; 2048² does not.
        // The old fixed 3 GB inference headroom would have triggered offload
        // at both resolutions even though only the larger one needs it.
        let xformer = 22_000_000_000;
        let free = 24_500_000_000;
        let act_768 = activation_bytes(768, 768, 1, 2, ActivationFamily::FluxDit);
        let act_2048 = activation_bytes(2048, 2048, 1, 2, ActivationFamily::FluxDit);
        assert!(act_2048 > act_768, "2048² must exceed 768²");
        assert!(
            !should_offload(xformer, free, act_768),
            "small budget must NOT trigger offload at this VRAM (act_768={act_768})"
        );
        assert!(
            should_offload(xformer, free, act_2048),
            "large budget MUST trigger offload at this VRAM (act_2048={act_2048})"
        );
    }

    /// `activation_family_for` maps manifest slugs to the right enum and
    /// defaults unknown slugs to the FLUX-dit factor.
    #[test]
    fn activation_family_for_maps_known_and_falls_back() {
        assert_eq!(activation_family_for("flux"), ActivationFamily::FluxDit);
        assert_eq!(activation_family_for("sdxl"), ActivationFamily::SdxlUnet);
        assert_eq!(
            activation_family_for("qwen-image"),
            ActivationFamily::QwenImageDit
        );
        assert_eq!(
            activation_family_for("wuerstchen"),
            ActivationFamily::Wuerstchen
        );
        assert_eq!(activation_family_for("wan"), ActivationFamily::WanVideo);
        assert!(ActivationFamily::WanVideo.is_full_weight_video());
        assert!(!ActivationFamily::WanVideo.streaming_transformer());
        // Unknown slug — falls back to FluxDit.
        assert_eq!(
            activation_family_for("bogus-family"),
            ActivationFamily::FluxDit
        );
    }

    #[test]
    fn wan_token_count_follows_the_vae_generation_and_frame_grid() {
        // 2.1 groups 4:1 temporally and 8:1 spatially, then the (1,2,2) patch
        // halves each spatial axis: 832x480 -> 52 x 30.
        let v21 = WanActivationGeometry::a14b();
        let v22 = WanActivationGeometry::ti2v_5b();
        assert_eq!(wan_token_count(832, 480, 53, v21), 14 * 30 * 52);
        // The ungrouped first frame: 1 pixel frame is still 1 latent frame.
        assert_eq!(wan_token_count(832, 480, 1, v21), 30 * 52);
        // 2.2 is 16:1 spatially, so the same pixels are a quarter of the grid.
        assert_eq!(wan_token_count(1280, 704, 121, v22), 31 * 22 * 40);
        // Frames are the axis the old pixel-area estimate was blind to.
        let short = wan_token_count(832, 480, 17, v21);
        let long = wan_token_count(832, 480, 81, v21);
        assert_eq!(long, short * 21 / 5);
    }

    #[test]
    fn wan_activation_scales_with_frames_and_never_collapses_to_area() {
        let geometry = WanActivationGeometry::a14b();
        let short = wan_activation_budget_bytes(832, 480, 17, geometry);
        let long = wan_activation_budget_bytes(832, 480, 81, geometry);
        // 5 latent frames -> 21: the estimate must move by the token ratio,
        // which is the whole point of #774. The generic pixel-area estimate
        // returns the same number for both.
        assert_eq!(long, short * 21 / 5);
        assert!(
            long > short,
            "frames must move the wan activation estimate: {short} -> {long}"
        );
    }

    #[test]
    fn wan_attention_scores_dominate_the_per_token_slope() {
        // Chunking bounds the query axis, not the key axis, so the score
        // matrix is per-token and larger than the residual stream. If a future
        // change prices it as flat workspace the estimate silently loses its
        // dominant term.
        let geometry = WanActivationGeometry::a14b();
        let tokens = wan_token_count(832, 480, 53, geometry);
        let per_token = wan_activation_budget_bytes(832, 480, 53, geometry) / tokens;
        let scores = 2 * geometry.num_heads * WAN_ATTENTION_QUERY_CHUNK * F32_BYTES;
        assert!(
            scores > per_token / 2,
            "score matrix {scores} should dominate the {per_token} B/token slope"
        );
    }

    #[test]
    fn wan_score_term_meets_the_unchunked_form_exactly_at_the_chunk_threshold() {
        // The linear score term assumes `math_attention_chunk_size` is active,
        // which it is on CUDA above 1,024 queries. At exactly the threshold the
        // chunked estimate (score + softmax, each `[heads, 512, T]`) and the
        // unchunked reality (`[heads, T, T]`) are the same number, because
        // 2 x 512 = 1024. Below it this over-estimates, which is the safe
        // direction; above it chunking holds. If either constant moves without
        // the other, that continuity breaks and the model silently starts
        // under-counting small shapes.
        const THRESHOLD_TOKENS: u64 = 1024;
        let heads = WanActivationGeometry::a14b().num_heads;
        let chunked = 2 * heads * WAN_ATTENTION_QUERY_CHUNK * F32_BYTES * THRESHOLD_TOKENS;
        let unchunked = heads * THRESHOLD_TOKENS * THRESHOLD_TOKENS * F32_BYTES;
        assert_eq!(
            chunked, unchunked,
            "the chunked and unchunked score forms must meet at the threshold"
        );
    }

    #[test]
    fn wan_per_token_timesteps_only_charge_ti2v() {
        // A scalar timestep broadcasts one modulation row; TI2V's latent
        // inpaint makes it a full [1, T, 6, dim] table in every block.
        let scalar = WanActivationGeometry {
            per_token_timesteps: false,
            ..WanActivationGeometry::ti2v_5b()
        };
        let per_token = WanActivationGeometry::ti2v_5b();
        assert!(
            wan_activation_budget_bytes(1280, 704, 121, per_token)
                > wan_activation_budget_bytes(1280, 704, 121, scalar),
        );
        // The A14B pair drives a scalar timestep, so it must not be charged.
        assert!(!WanActivationGeometry::a14b().per_token_timesteps);
        assert!(!WanActivationGeometry::t2v_1_3b().per_token_timesteps);
    }

    /// `dtype_bytes` returns the right element width for activation budget math.
    #[test]
    fn dtype_bytes_matches_runtime() {
        use candle_core::DType;
        assert_eq!(dtype_bytes(DType::BF16), 2);
        assert_eq!(dtype_bytes(DType::F16), 2);
        assert_eq!(dtype_bytes(DType::F32), 4);
        assert_eq!(dtype_bytes(DType::F64), 8);
        // Quantized / int storage flows as bf16 activations.
        assert_eq!(dtype_bytes(DType::U8), 2);
    }

    // ── select_expand_device tests ─────────────────────────────────────────

    fn gpu(ordinal: usize, free_gb: u64) -> DiscoveredGpu {
        DiscoveredGpu {
            ordinal,
            stable_id: Some(format!("cuda:{ordinal:032x}")),
            raw_cuda_uuid: Some((ordinal as u128).to_be_bytes()),
            device_kind: Some(CudaDeviceKind::UnknownCuda),
            identity_error: None,
            backend: GpuBackend::Cuda,
            name: format!("gpu{ordinal}"),
            compute_capability: Some((8, 6)),
            pci_bus_id: None,
            total_vram_bytes: 24 * GB,
            free_vram_bytes: free_gb * GB,
        }
    }

    #[test]
    fn expand_picks_main_gpu_when_it_fits() {
        let gpus = vec![gpu(0, 20), gpu(1, 20)];
        assert_eq!(
            select_expand_device(&gpus, 3 * GB, false),
            ExpandPlacement::Gpu(0),
        );
    }

    #[test]
    fn expand_falls_through_to_second_gpu_when_main_full() {
        let gpus = vec![gpu(0, 1), gpu(1, 10)];
        assert_eq!(
            select_expand_device(&gpus, 3 * GB, false),
            ExpandPlacement::Gpu(1),
        );
    }

    #[test]
    fn expand_walks_all_gpus_in_ordinal_order() {
        // GPU 1 also full, GPU 2 fits — should reach ordinal 2
        let gpus = vec![gpu(0, 1), gpu(1, 2), gpu(2, 10)];
        assert_eq!(
            select_expand_device(&gpus, 3 * GB, false),
            ExpandPlacement::Gpu(2),
        );
    }

    #[test]
    fn expand_falls_back_to_cpu_when_no_gpu_fits() {
        let gpus = vec![gpu(0, 1), gpu(1, 2)];
        assert_eq!(
            select_expand_device(&gpus, 3 * GB, false),
            ExpandPlacement::Cpu,
        );
    }

    #[test]
    fn expand_falls_back_to_cpu_when_no_gpus_discovered() {
        let gpus: Vec<DiscoveredGpu> = vec![];
        assert_eq!(
            select_expand_device(&gpus, 3 * GB, false),
            ExpandPlacement::Cpu,
        );
    }

    #[test]
    fn expand_metal_always_picks_gpu_0_when_present() {
        // Metal: unified memory, VRAM threshold doesn't gate — RAM preflight does.
        let gpus = vec![gpu(0, 0)];
        assert_eq!(
            select_expand_device(&gpus, 100 * GB, true),
            ExpandPlacement::Gpu(0),
        );
    }

    #[test]
    fn expand_metal_with_no_gpus_goes_to_cpu() {
        let gpus: Vec<DiscoveredGpu> = vec![];
        assert_eq!(
            select_expand_device(&gpus, 3 * GB, true),
            ExpandPlacement::Cpu,
        );
    }

    #[test]
    fn expand_threshold_sums_weights_and_headroom() {
        // 4 GB q8 model → 4 + 2 = 6 GB threshold
        assert_eq!(expand_vram_threshold(4 * GB), 6 * GB);
        // 1.3 GB q4 model → 1.3 + 2 = 3.3 GB
        assert_eq!(
            expand_vram_threshold(1_300_000_000),
            1_300_000_000 + EXPAND_ACTIVATION_HEADROOM,
        );
    }

    #[test]
    fn expand_strictly_greater_than_threshold() {
        // free_vram must exceed threshold (strict >), not just equal it —
        // matches should_use_gpu's convention so an exactly-fitting model
        // still leaves no room for OS overhead.
        let gpus = vec![gpu(0, 3)]; // exactly 3 GB free
        assert_eq!(
            select_expand_device(&gpus, 3 * GB, false),
            ExpandPlacement::Cpu,
        );
    }

    #[test]
    fn expand_prefers_requested_gpu_when_it_fits() {
        let gpus = vec![gpu(0, 20), gpu(1, 20)];
        assert_eq!(
            select_expand_device_with_preference(&gpus, 3 * GB, false, Some(1)),
            ExpandPlacement::Gpu(1),
        );
    }

    #[test]
    fn expand_preference_falls_back_when_requested_gpu_cannot_fit() {
        let gpus = vec![gpu(0, 20), gpu(1, 1)];
        assert_eq!(
            select_expand_device_with_preference(&gpus, 3 * GB, false, Some(1)),
            ExpandPlacement::Gpu(0),
        );
    }

    // ── select_ltx2_gemma_device ─────────────────────────────────────────

    /// Single GPU with room: encoder lands on the active GPU. Mirrors a
    /// 2× 3090 host where one card is busy with another model and the
    /// other has 24+ GB free for Gemma.
    #[test]
    fn select_ltx2_gemma_device_picks_active_gpu_when_room() {
        let gpus = vec![gpu(0, 25)];
        assert_eq!(
            select_ltx2_gemma_device(&gpus, 0, 24 * GB),
            LtxGemmaPlacement::Gpu(0),
        );
    }

    /// Single 24 GB card already streaming a 22B LTX-2 transformer: only
    /// ~17 GB free, doesn't clear the 24 GB Gemma threshold, so the encoder
    /// must land on CPU instead of OOMing.
    #[test]
    fn select_ltx2_gemma_device_falls_to_cpu_when_no_gpu_fits() {
        let gpus = vec![gpu(0, 17)];
        assert_eq!(
            select_ltx2_gemma_device(&gpus, 0, 24 * GB),
            LtxGemmaPlacement::Cpu,
        );
    }

    /// Multi-GPU host: the assigned GPU is full even though a sibling has
    /// room. The sibling is not leased to this stage, so auto placement must
    /// fall back to CPU instead of allocating there behind the scheduler.
    #[test]
    fn select_ltx2_gemma_device_never_uses_unleased_sibling() {
        let gpus = vec![gpu(0, 4), gpu(1, 25)];
        assert_eq!(
            select_ltx2_gemma_device(&gpus, 0, 24 * GB),
            LtxGemmaPlacement::Cpu,
        );
    }

    /// No amount of sibling capacity changes the one-device-per-stage
    /// contract when the assigned device cannot fit Gemma.
    #[test]
    fn select_ltx2_gemma_device_ignores_all_sibling_capacity() {
        let gpus = vec![gpu(0, 25), gpu(1, 4), gpu(2, 25)];
        assert_eq!(
            select_ltx2_gemma_device(&gpus, 1, 24 * GB),
            LtxGemmaPlacement::Cpu,
        );
    }

    #[test]
    fn select_ltx2_gemma_device_returns_cpu_when_no_gpus_discovered() {
        let gpus: Vec<DiscoveredGpu> = vec![];
        assert_eq!(
            select_ltx2_gemma_device(&gpus, 0, 24 * GB),
            LtxGemmaPlacement::Cpu,
        );
    }

    /// `LTX2_GEMMA_VRAM_THRESHOLD` is the headline knob; pin its bytes so
    /// edits go through the constant rather than scattering literal figures
    /// across call sites.
    ///
    /// The value tracks *streaming* residency, not the 24 GB of BF16 weights:
    /// [`GemmaHiddenStateEncoder::load_from_assets`] builds via `new_streaming`,
    /// whose `forward_hidden_states` constructs and drops one `DecoderLayer`
    /// per iteration, so the 48 layers are never co-resident.
    #[test]
    fn ltx2_gemma_vram_threshold_covers_streaming_residency() {
        // Derived from `ltx_gemma_config()` in ltx2/text/encoder.rs.
        const HIDDEN: u64 = 3_840;
        const VOCAB: u64 = 262_208;
        const INTERMEDIATE: u64 = 15_360;
        const LAYERS: u64 = 48;
        const SEQ: u64 = 1_024;
        const BF16: u64 = 2;

        // Permanently resident: the token embedding table and final norm.
        let embed = VOCAB * HIDDEN * BF16;
        // One decoder layer: q/k/v/o attention projections + the gated MLP.
        let attn = HIDDEN * (16 * 256) * 2 + HIDDEN * (8 * 256) * 2;
        let mlp = 3 * HIDDEN * INTERMEDIATE;
        let layer = (attn + mlp) * BF16;
        // Every layer's output is retained to feed the connectors.
        let hidden_states = (LAYERS + 1) * SEQ * HIDDEN * BF16;

        // Two layers in flight bounds the construct-then-drop overlap.
        let peak = embed + 2 * layer + hidden_states;

        assert!(
            peak < 4 * GB,
            "streaming Gemma should peak under 4 GB, computed {peak} bytes"
        );
        assert!(
            LTX2_GEMMA_VRAM_THRESHOLD > peak,
            "threshold {LTX2_GEMMA_VRAM_THRESHOLD} must clear the {peak}-byte streaming peak"
        );
        // ...with headroom for the encode workspace, but well under a 24 GB
        // card so the most common consumer GPU can host the encoder.
        const {
            assert!(
                LTX2_GEMMA_VRAM_THRESHOLD < 12 * GB,
                "threshold must leave a 24 GB card eligible for GPU Gemma"
            )
        };
    }

    // ── resolve_ltx2_gemma_device_override ───────────────────────────────

    /// All `MOLD_LTX2_GEMMA_DEVICE` / `MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER`
    /// env-var behaviors live under one `#[test]` to serialize access to the
    /// shared process-global env vars (cargo's parallel runner can't race
    /// between `set_var`/`remove_var` of two adjacent tests).
    #[test]
    fn resolve_ltx2_gemma_device_override_env_behaviors() {
        // Snapshot then clear both vars so we start from a known state.
        let prior_main = std::env::var_os("MOLD_LTX2_GEMMA_DEVICE");
        let prior_legacy = std::env::var_os("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
        unsafe {
            std::env::remove_var("MOLD_LTX2_GEMMA_DEVICE");
            std::env::remove_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
        }

        // Unset → None (auto path).
        assert_eq!(resolve_ltx2_gemma_device_override(0), None);

        // Explicit cpu → Cpu.
        unsafe { std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", "cpu") };
        assert_eq!(
            resolve_ltx2_gemma_device_override(0),
            Some(LtxGemmaPlacement::Cpu),
        );

        // Explicit gpu → Gpu(active_ordinal).
        unsafe { std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", "gpu") };
        assert_eq!(
            resolve_ltx2_gemma_device_override(1),
            Some(LtxGemmaPlacement::Gpu(1)),
        );

        // Case-insensitive parse.
        unsafe { std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", "CPU") };
        assert_eq!(
            resolve_ltx2_gemma_device_override(0),
            Some(LtxGemmaPlacement::Cpu),
        );

        // Explicit auto → None (lets the auto-resolver run).
        unsafe { std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", "auto") };
        assert_eq!(resolve_ltx2_gemma_device_override(0), None);

        // Garbage → None + warn (warn isn't asserted; we just confirm None).
        unsafe { std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", "wat") };
        assert_eq!(resolve_ltx2_gemma_device_override(0), None);

        // Legacy alias still pins to CPU when the new var is unset.
        unsafe {
            std::env::remove_var("MOLD_LTX2_GEMMA_DEVICE");
            std::env::set_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", "1");
        }
        assert_eq!(
            resolve_ltx2_gemma_device_override(0),
            Some(LtxGemmaPlacement::Cpu),
        );

        // New var beats legacy alias (legacy `=1` would say cpu, new var
        // pins to gpu — new wins).
        unsafe { std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", "gpu") };
        assert_eq!(
            resolve_ltx2_gemma_device_override(2),
            Some(LtxGemmaPlacement::Gpu(2)),
        );

        // Restore.
        unsafe {
            std::env::remove_var("MOLD_LTX2_GEMMA_DEVICE");
            std::env::remove_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER");
            if let Some(v) = prior_main {
                std::env::set_var("MOLD_LTX2_GEMMA_DEVICE", v);
            }
            if let Some(v) = prior_legacy {
                std::env::set_var("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER", v);
            }
        }
    }

    // ── keep_te_in_ram ───────────────────────────────────────────────────

    /// Every `MOLD_KEEP_TE_RAM` behavior lives under one `#[test]` so cargo's
    /// parallel runner cannot race two `set_var`/`remove_var` sequences on the
    /// same process-global variable — the same reason
    /// `test_reserved_vram_and_usable_free_vram` below is a single test.
    ///
    /// One claim is pinned here: parking is opt-in and the decision is a pure
    /// function of the frozen environment. There is deliberately no host-memory
    /// probe in this path — a `/proc/meminfo` read cannot see a container's
    /// cgroup limit, and an unfrozen input would let two runs with identical
    /// frozen snapshots take opposite residency paths under one
    /// execution-equivalence fingerprint.
    #[test]
    fn test_keep_te_in_ram_env_behaviors() {
        unsafe { std::env::remove_var("MOLD_KEEP_TE_RAM") };
        assert!(!keep_te_in_ram(), "missing var must be off");

        unsafe { std::env::set_var("MOLD_KEEP_TE_RAM", "1") };
        assert!(keep_te_in_ram(), "\"1\" must enable park");

        for v in ["", "0", "true", "yes", "TRUE"] {
            unsafe { std::env::set_var("MOLD_KEEP_TE_RAM", v) };
            assert!(
                !keep_te_in_ram(),
                "value {v:?} must not enable park (helper is strict ==\"1\")"
            );
        }
        unsafe { std::env::remove_var("MOLD_KEEP_TE_RAM") };
    }

    // ── reserved_vram_bytes / usable_free_vram_bytes ─────────────────────

    /// All `MOLD_RESERVE_VRAM_MB` env-var behaviors live under one `#[test]`
    /// to serialize access to the shared process-global env var.
    /// Combined into a single `#[test]` so cargo's parallel runner can't race
    /// between the `set_var`/`remove_var` of `test_reserved_vram_env_behaviors`
    /// and the `remove_var` of `test_usable_free_vram_bytes_subtracts_reserve`.
    /// (Two parallel tests touching the same env var caused intermittent
    /// failures where one test's set_var was clobbered before its assertion.)
    #[test]
    fn test_reserved_vram_and_usable_free_vram() {
        // ── Part 1: reserved_vram_bytes env behavior ────────────────────
        unsafe { std::env::remove_var("MOLD_RESERVE_VRAM_MB") };

        let default = reserved_vram_bytes();
        #[cfg(target_os = "linux")]
        assert_eq!(default, 400_000_000, "Linux default reserve = 400 MB");
        #[cfg(target_os = "macos")]
        assert_eq!(default, 0, "macOS default = 0 (Metal unified memory)");

        unsafe { std::env::set_var("MOLD_RESERVE_VRAM_MB", "1024") };
        assert_eq!(reserved_vram_bytes(), 1_024_000_000);

        unsafe { std::env::set_var("MOLD_RESERVE_VRAM_MB", "0") };
        assert_eq!(reserved_vram_bytes(), 0);

        for v in ["", "abc", "-1"] {
            unsafe { std::env::set_var("MOLD_RESERVE_VRAM_MB", v) };
            assert_eq!(
                reserved_vram_bytes(),
                default,
                "unparseable {v:?} must fall back to default"
            );
        }

        unsafe { std::env::remove_var("MOLD_RESERVE_VRAM_MB") };

        // ── Part 2: usable_free_vram_bytes wrapper ──────────────────────
        assert_eq!(usable_free_vram_from_raw(1_500, 500), 1_000);
        assert_eq!(usable_free_vram_from_raw(500, 1_500), 0);

        unsafe { std::env::set_var("MOLD_RESERVE_VRAM_MB", u64::MAX.to_string()) };
        let has_raw_reading = free_vram_bytes(0).is_some();
        let usable = usable_free_vram_bytes(0);
        assert_eq!(
            usable.is_some(),
            has_raw_reading,
            "usable_free_vram_bytes must mirror free_vram_bytes presence"
        );
        if has_raw_reading {
            assert_eq!(usable, Some(0));
        }

        unsafe { std::env::remove_var("MOLD_RESERVE_VRAM_MB") };
    }

    // ── vram_load_delta ──────────────────────────────────────────────────

    /// `vram_load_delta` must be a pure `saturating_sub` against the
    /// post-load reading. When the device has no CUDA (the test environment),
    /// `vram_in_use_bytes` returns 0, so the delta is always 0 — but the
    /// function shape (saturating_sub, not panic on underflow) must hold
    /// regardless of the live reading.
    #[test]
    fn vram_load_delta_is_saturating_sub() {
        // Without CUDA, vram_in_use_bytes(0) == 0, and 0.saturating_sub(N) == 0
        // for any N. This locks in the saturating semantic — a flaky reading
        // (post < pre) must never panic or wrap.
        assert_eq!(vram_load_delta(0, 0), 0);
        assert_eq!(vram_load_delta(0, 1_000_000_000), 0);
        assert_eq!(vram_load_delta(0, u64::MAX), 0);
    }

    // --- estimate_peak_memory: single-file convention must not double-count ---

    fn write_dummy_file(dir: &std::path::Path, name: &str, size: u64) -> std::path::PathBuf {
        let p = dir.join(name);
        let f = std::fs::File::create(&p).expect("create dummy");
        f.set_len(size).expect("set_len");
        p
    }

    #[test]
    fn estimate_peak_memory_single_file_does_not_double_count_vae() {
        // Civitai single-file convention: ModelPaths::transformer == ModelPaths::vae,
        // both pointing at the primary .safetensors. Naive math would compute
        // transformer_size + vae_size twice for the same on-disk bytes — exactly
        // the bug behind the misleading "94 GB needed" preflight error.
        let dir = tempfile::tempdir().expect("tempdir");
        let single = write_dummy_file(dir.path(), "single.safetensors", 44_000_000_000);
        let te = write_dummy_file(dir.path(), "te.safetensors", 24_000_000_000);
        let paths = mold_core::ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: single.clone(),
            transformer_shards: vec![],
            vae: single, // Same file as transformer — single-file path
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![te],
            text_tokenizer: None,
            decoder: None,
        };
        let peak = estimate_peak_memory(&paths, LoadStrategy::Sequential);
        // Sequential = max(encoders, transformer + vae[==transformer dedup'd]) + headroom
        //            = max(24 GB, 44 GB) + 2 GB = 46 GB
        // NOT 24 + 44 + 44 + 2 = 114 GB (the double-count bug).
        let peak_gb = peak as f64 / 1e9;
        assert!(
            peak_gb < 50.0,
            "single-file peak should be ~46 GB, got {peak_gb:.1} GB (double-count bug returned)"
        );
        assert!(
            peak_gb > 45.0,
            "single-file peak should be ~46 GB, got {peak_gb:.1} GB"
        );
    }

    /// A two-expert pair is resident one expert at a time, so admission sizes
    /// the **larger** of the two — not the first, and not their sum.
    ///
    /// Sizing the first expert is the dangerous direction: an asymmetric pair
    /// (a Q8 low expert behind a Q5 high one, which the architecture check
    /// accepts and a `MOLD_LOW_NOISE_TRANSFORMER_PATH` override can produce)
    /// would admit against the smaller number and then OOM at the swap,
    /// halfway through a generation.
    #[test]
    fn estimate_peak_memory_sizes_the_larger_expert_of_a_pair() {
        let dir = tempfile::tempdir().expect("tempdir");
        let high = write_dummy_file(dir.path(), "high.gguf", 10_800_000_000);
        let low = write_dummy_file(dir.path(), "low.gguf", 15_400_000_000);
        let vae = write_dummy_file(dir.path(), "vae.safetensors", 250_000_000);
        let te = write_dummy_file(dir.path(), "umt5.safetensors", 11_400_000_000);

        let paths = |low_noise: Option<std::path::PathBuf>| mold_core::ModelPaths {
            low_noise_transformer: low_noise,
            low_noise_distilled_lora: None,
            transformer: high.clone(),
            transformer_shards: vec![],
            vae: vae.clone(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![te.clone()],
            text_tokenizer: None,
            decoder: None,
        };

        let high_alone = estimate_peak_memory(&paths(None), LoadStrategy::Sequential);
        let paired = estimate_peak_memory(&paths(Some(low.clone())), LoadStrategy::Sequential);

        // The invariant: a pair costs exactly what its larger expert would cost
        // on its own. Stated as an equality against that model rather than as a
        // byte delta, because the sequential peak is a max over phases and the
        // 11.4 GB encoder can dominate the smaller expert.
        let mut larger_alone = paths(None);
        larger_alone.transformer = low.clone();
        assert_eq!(
            paired,
            estimate_peak_memory(&larger_alone, LoadStrategy::Sequential),
            "a pair must be sized as its larger expert alone"
        );
        assert!(
            paired > high_alone,
            "the larger expert must raise the estimate above the smaller one's"
        );
        // And emphatically not their sum: one expert is resident at a time.
        assert!(
            paired < high_alone + 15_400_000_000,
            "the pair must not be sized as high + low"
        );

        // Order-independent: the max does not care which slot the big one is in.
        let mut swapped = paths(Some(high.clone()));
        swapped.transformer = low;
        assert_eq!(
            estimate_peak_memory(&swapped, LoadStrategy::Sequential),
            paired
        );
    }

    #[test]
    fn estimate_peak_memory_separate_vae_file_still_sums() {
        // Sanity: when transformer and vae are distinct files (e.g. FLUX with
        // separate vae companion), both file sizes contribute as before.
        let dir = tempfile::tempdir().expect("tempdir");
        let transformer = write_dummy_file(dir.path(), "tx.safetensors", 4_000_000_000);
        let vae = write_dummy_file(dir.path(), "vae.safetensors", 1_000_000_000);
        let te = write_dummy_file(dir.path(), "te.safetensors", 9_000_000_000);
        let paths = mold_core::ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: vec![],
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(te),
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        };
        let peak = estimate_peak_memory(&paths, LoadStrategy::Sequential);
        // max(9, 4 + 1) + 2 = 11 GB
        let peak_gb = peak as f64 / 1e9;
        assert!(
            (10.5..11.5).contains(&peak_gb),
            "expected ~11 GB, got {peak_gb:.1} GB"
        );
    }

    #[test]
    fn estimate_peak_memory_sharded_transformer_with_separate_vae_sums() {
        // Multi-shard transformer (e.g. FLUX2 diffusers): shards are listed
        // explicitly and vae is a separate file. The single-file dedup must
        // not fire here.
        let dir = tempfile::tempdir().expect("tempdir");
        let s1 = write_dummy_file(dir.path(), "tx-1.safetensors", 4_000_000_000);
        let s2 = write_dummy_file(dir.path(), "tx-2.safetensors", 4_000_000_000);
        let vae = write_dummy_file(dir.path(), "vae.safetensors", 1_000_000_000);
        let paths = mold_core::ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: s1.clone(), // primary shard
            transformer_shards: vec![s1, s2],
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        };
        let peak = estimate_peak_memory(&paths, LoadStrategy::Sequential);
        // max(0, 4+4 + 1) + 2 = 11 GB
        let peak_gb = peak as f64 / 1e9;
        assert!(
            (10.5..11.5).contains(&peak_gb),
            "sharded peak should be ~11 GB, got {peak_gb:.1} GB"
        );
    }

    #[test]
    fn estimate_peak_memory_sharded_single_file_vae_does_not_double_count() {
        // Catalog single-file checkpoints can reach the estimator with the
        // primary checkpoint listed as a transformer shard and as the bundled
        // VAE. That is still one mmap-backed safetensors file, not two GPU
        // resident weight sets.
        let dir = tempfile::tempdir().expect("tempdir");
        let single = write_dummy_file(dir.path(), "single.safetensors", 14_000_000_000);
        let paths = mold_core::ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: single.clone(),
            transformer_shards: vec![single.clone()],
            vae: single,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        };
        let peak = estimate_peak_memory(&paths, LoadStrategy::Sequential);
        // max(0, 14 GB + deduped VAE) + 2 GB headroom = 16 GB.
        let peak_gb = peak as f64 / 1e9;
        assert!(
            (15.5..16.5).contains(&peak_gb),
            "sharded single-file peak should be ~16 GB, got {peak_gb:.1} GB"
        );
    }

    #[test]
    fn estimate_peak_memory_single_file_sdxl_does_not_count_clip_views_as_full_checkpoints() {
        // Cached Civitai SDXL single-file engines expose a diffusers-shaped
        // ModelPaths view where transformer, VAE, CLIP-L, and CLIP-G all point
        // at the same safetensors file. These are component views into one
        // checkpoint, not four full checkpoint-sized GPU phases.
        let dir = tempfile::tempdir().expect("tempdir");
        let single = write_dummy_file(dir.path(), "single.safetensors", 14_000_000_000);
        let paths = mold_core::ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: single.clone(),
            transformer_shards: vec![],
            vae: single.clone(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: Some(single.clone()),
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: Some(single),
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        };
        let peak = estimate_peak_memory(&paths, LoadStrategy::Sequential);
        // max(deduped encoders=0, transformer + deduped VAE=14 GB) + 2 GB.
        let peak_gb = peak as f64 / 1e9;
        assert!(
            (15.5..16.5).contains(&peak_gb),
            "single-file SDXL peak should be ~16 GB, got {peak_gb:.1} GB"
        );
    }

    #[test]
    fn parse_vae_dtype_unset_and_auto_preserve_default_policy() {
        assert_eq!(parse_vae_dtype_policy(None), VaeDtypePolicy::Auto);
        assert_eq!(parse_vae_dtype_policy(Some("auto")), VaeDtypePolicy::Auto);
    }

    #[test]
    fn parse_vae_dtype_fp32_forces_f32() {
        assert_eq!(parse_vae_dtype_policy(Some("fp32")), VaeDtypePolicy::F32);
    }

    #[test]
    fn parse_vae_dtype_bf16_forces_bf16() {
        assert_eq!(parse_vae_dtype_policy(Some("bf16")), VaeDtypePolicy::Bf16);
    }

    #[test]
    fn parse_vae_dtype_fp16_alias_recognised() {
        for value in ["fp16", "f16", "FP16", "F16"] {
            assert_eq!(
                parse_vae_dtype_policy(Some(value)),
                VaeDtypePolicy::F16,
                "value `{value}` should resolve to F16"
            );
        }
    }

    #[test]
    fn parse_vae_dtype_invalid_value_falls_back_to_auto() {
        assert_eq!(
            parse_vae_dtype_policy(Some("fp64")),
            VaeDtypePolicy::Auto,
            "invalid value must fall back, not error"
        );
    }

    /// Pin the `MEMORY_BUDGET_HEADROOM` constant so a future change forces a
    /// matching update to the preflight rejection error message in
    /// `mold-server::model_manager::check_model_memory_budget`, which prints
    /// "with 2 GB activation headroom" in its formatted output.
    #[test]
    fn memory_budget_headroom_is_2gb() {
        assert_eq!(
            MEMORY_BUDGET_HEADROOM, 2_000_000_000,
            "MEMORY_BUDGET_HEADROOM changed — update the rejection error message \
             in mold-server::model_manager::check_model_memory_budget to match"
        );
    }

    #[test]
    fn fatal_post_drop_synchronize_prevents_memory_query() {
        let sample_calls = std::cell::Cell::new(0);
        let result = post_drop_free_vram_bytes_with(
            || {
                Err(DeviceMemoryError::FatalCuda {
                    operation: "device synchronize",
                    message: "CUDA_ERROR_ILLEGAL_ADDRESS".to_string(),
                })
            },
            || {
                sample_calls.set(sample_calls.get() + 1);
                Ok(24_000_000_000)
            },
        );

        assert!(result.unwrap_err().is_fatal_cuda());
        assert_eq!(
            sample_calls.get(),
            0,
            "fatal synchronize must fence every later CUDA callback"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn every_contained_driver_status_is_a_typed_fatal_memory_error() {
        for status in [
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            "CUDA_ERROR_ECC_UNCORRECTABLE",
            "CUDA_ERROR_LAUNCH_FAILED",
            "CUDA_ERROR_ASSERT",
            "CUDA_ERROR_MISALIGNED_ADDRESS",
            "CUDA_ERROR_HARDWARE_STACK_ERROR",
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "CUDA_ERROR_INVALID_ADDRESS_SPACE",
            "CUDA_ERROR_INVALID_PC",
            "CUDA_ERROR_LAUNCH_TIMEOUT",
            "CUDA_ERROR_EXTERNAL_DEVICE",
            "CUDA_ERROR_MPS_CLIENT_TERMINATED",
            "CUDA_ERROR_CONTAINED",
            "CUDA_ERROR_TENSOR_MEMORY_LEAK",
        ] {
            let error = memory_error("synthetic memory sample", status);
            assert!(error.is_fatal_cuda(), "not classified as fatal: {status}");
        }
        assert!(matches!(
            memory_error("synthetic memory sample", "CUDA_ERROR_OUT_OF_MEMORY"),
            DeviceMemoryError::Unavailable { .. }
        ));
    }

    #[test]
    fn unavailable_post_drop_sample_stays_typed() {
        let result = post_drop_free_vram_bytes_with(
            || Ok(()),
            || {
                Err(DeviceMemoryError::Unavailable {
                    operation: "free VRAM query",
                    message: "driver unavailable".to_string(),
                })
            },
        );

        assert!(matches!(
            result,
            Err(DeviceMemoryError::Unavailable {
                operation: "free VRAM query",
                ..
            })
        ));
    }

    // ── Phase VRAM telemetry ────────────────────────────────────────────

    fn pool_sample(used: u64, used_high: u64, reserved_high: u64, free: u64) -> VramSample {
        VramSample {
            pool_used_bytes: Some(used),
            pool_used_high_bytes: Some(used_high),
            pool_reserved_high_bytes: Some(reserved_high),
            global_free_bytes: Some(free),
            global_total_bytes: Some(24 * GB),
        }
    }

    #[test]
    fn pool_samples_attribute_the_phase_peak_to_the_pool_high_water_mark() {
        let report = phase_vram_report_from(
            "transformer_load",
            pool_sample(2 * GB, 2 * GB, 3 * GB, 20 * GB),
            pool_sample(9 * GB, 11 * GB, 12 * GB, 12 * GB),
            Some(10 * GB),
            Duration::from_millis(41_821),
        );

        assert_eq!(report.attribution, VramAttribution::Pool);
        // The peak is the exit high-water mark, not the exit current usage and
        // not a free-VRAM delta.
        assert_eq!(report.peak_pool_used, Some(11 * GB));
        assert_eq!(report.peak_pool_reserved, Some(12 * GB));
        assert_eq!(report.entry_pool_used, Some(2 * GB));
        assert_eq!(report.incremental_peak_pool_used(), Some(9 * GB));
        assert_eq!(report.incremental_transient_pool_used(), Some(2 * GB));
        assert_eq!(report.delta_pool_used, Some(7 * GB as i64));
        assert_eq!(report.global_free_entry, Some(20 * GB));
        assert_eq!(report.global_free_exit, Some(12 * GB));
        assert_eq!(report.global_total, Some(24 * GB));
        assert_eq!(report.predicted_bytes, Some(10 * GB));
        assert_eq!(report.elapsed, Duration::from_millis(41_821));
        assert!(report.has_authoritative_peak());
    }

    #[test]
    fn global_only_samples_never_promote_a_free_delta_into_a_peak() {
        let entry = VramSample {
            global_free_bytes: Some(20 * GB),
            global_total_bytes: Some(24 * GB),
            ..VramSample::default()
        };
        let exit = VramSample {
            global_free_bytes: Some(12 * GB),
            global_total_bytes: Some(24 * GB),
            ..VramSample::default()
        };

        let report =
            phase_vram_report_from("vae_load", entry, exit, None, Duration::from_millis(10));

        assert_eq!(report.attribution, VramAttribution::GlobalOnly);
        assert_eq!(report.peak_pool_used, None);
        assert_eq!(report.peak_pool_reserved, None);
        assert_eq!(report.entry_pool_used, None);
        assert_eq!(report.incremental_peak_pool_used(), None);
        assert_eq!(report.incremental_transient_pool_used(), None);
        assert_eq!(report.delta_pool_used, None);
        assert!(
            !report.has_authoritative_peak(),
            "a global free delta is not a Mold-attributable peak"
        );
        assert_eq!(report.global_free_drop(), Some(8 * GB));
    }

    #[test]
    fn absent_samples_report_unavailable_without_a_zero_valued_peak() {
        let report = phase_vram_report_from(
            "denoise",
            VramSample::default(),
            VramSample::default(),
            Some(5 * GB),
            Duration::from_millis(1),
        );

        assert_eq!(report.attribution, VramAttribution::Unavailable);
        assert_eq!(report.peak_pool_used, None);
        assert_eq!(report.peak_pool_reserved, None);
        assert_eq!(report.delta_pool_used, None);
        assert_eq!(report.global_free_entry, None);
        assert_eq!(report.global_free_exit, None);
        assert_eq!(report.global_total, None);
        assert_eq!(report.global_free_drop(), None);
        assert!(!report.has_authoritative_peak());
        // Still renders, and never claims a measured zero.
        assert!(report.to_string().contains("attribution=unavailable"));
        assert!(!report.to_string().contains("peak=0.0 GB"));
    }

    #[test]
    fn a_phase_that_releases_memory_reports_a_negative_delta_and_no_free_underflow() {
        let report = phase_vram_report_from(
            "prompt_encode",
            pool_sample(9 * GB, 9 * GB, 10 * GB, 12 * GB),
            pool_sample(2 * GB, 9 * GB, 10 * GB, 20 * GB),
            None,
            Duration::from_millis(5),
        );

        assert_eq!(report.delta_pool_used, Some(-(7 * GB as i64)));
        // Exit free above entry free must saturate, never underflow.
        assert_eq!(report.global_free_drop(), Some(0));
    }

    #[test]
    fn a_probe_on_a_gpuless_build_finishes_as_unavailable() {
        let probe = PhaseVramProbe::enter("vae_decode");
        let report = probe.finish();

        assert_eq!(report.phase, "vae_decode");
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(report.attribution, VramAttribution::Unavailable);
            assert_eq!(report.peak_pool_used, None);
        }
    }

    /// Live check that the pool attributes really describe Mold's own
    /// allocations. Skipped (not failed) when no CUDA device is present, the
    /// same pattern the LTX-2 CUDA handoff test uses.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_probe_measures_the_allocation_made_inside_the_phase() {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return;
        };
        let probe = PhaseVramProbe::enter("test_alloc");
        let tensor =
            candle_core::Tensor::zeros((256, 1024, 1024), candle_core::DType::F32, &device)
                .expect("1 GB allocation");
        device.synchronize().ok();
        let report = probe.finish_with_predicted(Some(1_073_741_824));
        drop(tensor);

        assert_eq!(
            report.attribution,
            VramAttribution::Pool,
            "cudarc allocates through the stream-ordered pool on CUDA: {report}"
        );
        let peak = report.peak_pool_used.expect("pool peak");
        assert!(
            peak >= 1_073_741_824,
            "the phase peak must cover the 1 GB allocated inside it: {report}"
        );
        assert!(
            report.delta_pool_used.unwrap_or(0) >= 1_073_741_824,
            "the tensor is still live at finish, so the delta must show it: {report}"
        );
    }

    /// The reason this is pool-attributed rather than `cuMemGetInfo`-attributed:
    /// a later phase must not inherit an earlier phase's peak, even though the
    /// pool keeps the freed memory reserved and device-global free VRAM never
    /// recovers.
    #[cfg(feature = "cuda")]
    #[test]
    fn each_cuda_phase_measures_only_its_own_peak() {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return;
        };
        let big = PhaseVramProbe::enter("big");
        {
            let _tensor =
                candle_core::Tensor::zeros((512, 1024, 1024), candle_core::DType::F32, &device)
                    .expect("2 GB allocation");
            device.synchronize().ok();
        }
        let big_report = big.finish();
        device.synchronize().ok();

        let small = PhaseVramProbe::enter("small");
        let _small_tensor =
            candle_core::Tensor::zeros((64, 1024, 1024), candle_core::DType::F32, &device)
                .expect("256 MB allocation");
        device.synchronize().ok();
        let small_report = small.finish();

        assert!(
            big_report.peak_pool_used.expect("big peak") >= 2_147_483_648,
            "the first phase owns its own 2 GB: {big_report}"
        );
        assert!(
            small_report.peak_pool_used.expect("small peak") < 2_147_483_648,
            "the high-water mark must be rearmed per phase, so the second phase \
             never inherits the first's peak: {small_report}"
        );
    }

    /// A CPU-only run must not pay for telemetry it cannot use: sampling
    /// retains a CUDA context, which on a GPU box would claim VRAM for a run
    /// that asked for none.
    #[test]
    fn an_inert_probe_reports_timing_without_touching_the_driver() {
        let probe = PhaseVramProbe::enter_if("cpu_denoise", false);
        let report = probe.finish();

        assert_eq!(report.phase, "cpu_denoise");
        assert_eq!(report.attribution, VramAttribution::Unavailable);
        assert_eq!(report.peak_pool_used, None);
        assert_eq!(report.global_free_entry, None);
        // A probe opened after it must still be well-formed: the inert probe
        // owns no slot on the nesting stack.
        let next = PhaseVramProbe::enter_if("next", false).finish();
        assert_eq!(next.attribution, VramAttribution::Unavailable);
    }

    /// LTX-2 nests a VAE load inside the conditioning encode, and the device
    /// high-water mark is one counter: without folding, the inner probe's
    /// rearm would silently erase the outer phase's peak.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_nested_cuda_probe_never_erases_its_parents_peak() {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return;
        };
        let outer = PhaseVramProbe::enter("outer");
        {
            let _tensor =
                candle_core::Tensor::zeros((512, 1024, 1024), candle_core::DType::F32, &device)
                    .expect("2 GB allocation");
            device.synchronize().ok();
        }
        let inner = PhaseVramProbe::enter("inner");
        let _small = candle_core::Tensor::zeros((16, 1024, 1024), candle_core::DType::F32, &device)
            .expect("64 MB allocation");
        device.synchronize().ok();
        let inner_report = inner.finish();
        let outer_report = outer.finish();

        assert!(
            inner_report.peak_pool_used.expect("inner peak") < 2_147_483_648,
            "the nested phase reports only its own work: {inner_report}"
        );
        assert!(
            outer_report.peak_pool_used.expect("outer peak") >= 2_147_483_648,
            "the enclosing phase keeps the peak it reached before the nested \
             probe rearmed the counter: {outer_report}"
        );
    }

    // ── Metal device identity ──────────────────────────────────────────────

    #[test]
    fn memoized_creates_at_most_one_value_per_key() {
        let cache = std::sync::Mutex::new(std::collections::BTreeMap::new());
        let calls = std::cell::Cell::new(0_usize);
        let create = |ordinal: usize| {
            calls.set(calls.get() + 1);
            Ok(format!("device-{ordinal}"))
        };

        assert_eq!(memoized(&cache, 0, || create(0)).unwrap(), "device-0");
        assert_eq!(memoized(&cache, 0, || create(0)).unwrap(), "device-0");
        assert_eq!(calls.get(), 1, "a repeated ordinal must reuse its device");

        assert_eq!(memoized(&cache, 1, || create(1)).unwrap(), "device-1");
        assert_eq!(calls.get(), 2, "a distinct ordinal gets its own device");
    }

    #[test]
    fn memoized_does_not_cache_a_failed_creation() {
        let cache = std::sync::Mutex::new(std::collections::BTreeMap::<usize, String>::new());
        assert!(memoized(&cache, 0, || anyhow::bail!("no such GPU")).is_err());
        assert_eq!(
            memoized(&cache, 0, || Ok("recovered".to_string())).unwrap(),
            "recovered"
        );
    }

    #[test]
    fn repeated_metal_resolution_yields_one_shared_device() {
        if !candle_core::utils::metal_is_available() {
            eprintln!("skipping: no Metal device");
            return;
        }
        // candle identifies a Metal device by a process-global counter, not by
        // the physical GPU, and `Tensor::to_device` has no Metal->Metal arm. Two
        // devices for one GPU therefore make every cross-component tensor move
        // fail with "not implemented yet, self.device: Metal(..), device:
        // Metal(..)" — which is exactly what Z-Image hit once the scheduler
        // started handing each component an explicit placement.
        let transformer = metal_device(0).expect("metal device");
        let text_encoder = metal_device(0).expect("metal device");

        assert!(
            transformer.same_device(&text_encoder),
            "every component on GPU 0 must share one MetalDevice"
        );

        let embeddings = candle_core::Tensor::zeros((2, 4), candle_core::DType::F32, &text_encoder)
            .expect("allocate on the text-encoder device");
        embeddings
            .to_device(&transformer)
            .expect("moving text embeddings onto the transformer device must not fail");
    }

    #[test]
    fn production_code_never_constructs_a_metal_device_directly() {
        // `metal_device` is only load-bearing if it is the sole constructor.
        // A stray `Device::new_metal` reintroduces the split-identity bug, and
        // the failure is a runtime error in one family rather than a build
        // break, so pin it here. Tests may construct freely.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates dir");
        let mut offenders = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("readable crate dir") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().is_none_or(|ext| ext != "rs") {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("readable source");
                // Everything from the first `#[cfg(test)]` on is test code.
                let production = source
                    .split_once("#[cfg(test)]")
                    .map_or(source.as_str(), |(before, _)| before);
                for (index, line) in production.lines().enumerate() {
                    if line.contains("new_metal(")
                        && !line.trim_start().starts_with("//")
                        && !line.contains("candle_core::Device::new_metal(ordinal)")
                    {
                        offenders.push(format!("{}:{}", path.display(), index + 1));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "production code must open Metal through device::metal_device: {offenders:?}"
        );
    }

    #[test]
    fn releasing_pooled_metal_memory_is_a_noop_for_an_unopened_ordinal() {
        // The sweep must never construct a device: an unload on a box with no
        // Metal GPU, or for an ordinal this process never touched, has nothing
        // to return and must not allocate one to find that out.
        release_pooled_metal_memory(usize::MAX);
    }

    #[test]
    fn releasing_pooled_metal_memory_keeps_the_device_usable() {
        if !candle_core::utils::metal_is_available() {
            eprintln!("skipping: no Metal device");
            return;
        }
        let device = metal_device(0).expect("metal device");
        let before =
            candle_core::Tensor::ones((64, 64), candle_core::DType::F32, &device).expect("alloc");
        drop(before);

        release_pooled_metal_memory(0);

        // Sweeping returns pooled buffers to the OS; the memoized device must
        // survive it, keep its identity, and still allocate.
        let after = metal_device(0).expect("metal device after sweep");
        assert!(
            after.same_device(&device),
            "the sweep must not re-mint the device"
        );
        candle_core::Tensor::ones((64, 64), candle_core::DType::F32, &after).expect("alloc after");
    }

    #[test]
    fn every_metal_entry_point_shares_the_memoized_device() {
        if !candle_core::utils::metal_is_available() || candle_core::utils::cuda_is_available() {
            eprintln!("skipping: not a Metal-only host");
            return;
        }
        let progress = ProgressReporter::default();
        let auto = create_device(0, &progress).expect("auto device");
        let exact = create_exact_gpu_device(GpuBackend::Metal, 0, &progress).expect("exact device");
        let resolved = resolve_gpu_ordinal(0).expect("resolved device");

        assert!(
            auto.same_device(&exact),
            "create_device vs create_exact_gpu_device"
        );
        assert!(
            auto.same_device(&resolved),
            "create_device vs resolve_gpu_ordinal"
        );
    }
}

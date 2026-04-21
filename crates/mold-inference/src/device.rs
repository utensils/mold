use crate::engine::LoadStrategy;
use crate::progress::ProgressReporter;
use mold_core::types::GpuSelection;
use std::cell::Cell;

// ── Thread-local GPU ordinal guard ─────────────────────────────────────────
//
// Each GPU worker thread is pinned to a single ordinal. We stash that ordinal
// in a thread-local so cross-engine hotpaths (`create_device`, `reclaim_gpu_memory`)
// can debug-assert the caller isn't drifting onto a sibling GPU's context —
// the exact footgun that took the process down on killswitch when LTX-2 had
// `reclaim_gpu_memory(0)` hardcoded and nuked GPU 0's context while SD3.5
// was still denoising there.
//
// Threads without a bound ordinal (tokio blocking pool, tests) see `None`
// and the assert is skipped.

thread_local! {
    static THREAD_GPU_ORDINAL: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Bind the current thread to a GPU ordinal. Call once from each GPU worker
/// thread's entry point. Any subsequent `create_device` / `reclaim_gpu_memory`
/// call on this thread must match `ordinal` (debug builds only).
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

/// Panic in debug builds if `ordinal` doesn't match the thread's bound GPU.
/// A mismatch means a call site is ignoring its engine's `gpu_ordinal` and
/// reaching for another GPU's context — the SD3.5/LTX-2 crash pattern.
#[inline]
fn debug_assert_ordinal_matches_thread(ordinal: usize, context: &'static str) {
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
    pub name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
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
                                // `CudaContext::new` binds the calling thread to
                                // this ordinal, so `mem_get_info` returns this GPU's
                                // VRAM.
                                let (free, total) =
                                    driver::result::mem_get_info().unwrap_or((0, 0));
                                gpus.push(DiscoveredGpu {
                                    ordinal,
                                    name,
                                    total_vram_bytes: total as u64,
                                    free_vram_bytes: free as u64,
                                });
                            }
                            Err(e) => tracing::warn!("failed to open CUDA device {ordinal}: {e}"),
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
            let total = available_system_memory_bytes().unwrap_or(0);
            let free = free_system_memory_bytes().unwrap_or(0);
            gpus.push(DiscoveredGpu {
                ordinal: 0,
                name: "Apple Metal GPU".to_string(),
                total_vram_bytes: total,
                free_vram_bytes: free,
            });
        }
    }

    gpus
}

/// Filter discovered GPUs by user selection.
pub fn filter_gpus(gpus: &[DiscoveredGpu], selection: &GpuSelection) -> Vec<DiscoveredGpu> {
    match selection {
        GpuSelection::All => gpus.to_vec(),
        GpuSelection::Specific(ordinals) => gpus
            .iter()
            .filter(|g| ordinals.contains(&g.ordinal))
            .cloned()
            .collect(),
    }
}

/// Select the single best GPU (most free VRAM) for local CLI use.
pub fn select_best_gpu(gpus: &[DiscoveredGpu]) -> Option<&DiscoveredGpu> {
    gpus.iter().max_by_key(|g| g.free_vram_bytes)
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
    let force_cpu = std::env::var("MOLD_DEVICE")
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
        Ok(Device::new_metal(ordinal)?)
    } else {
        progress.info("No GPU detected, using CPU");
        tracing::warn!("No GPU detected, falling back to CPU");
        Ok(Device::Cpu)
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
    if is_metal {
        if let Some(g) = gpus.first() {
            return ExpandPlacement::Gpu(g.ordinal);
        }
        return ExpandPlacement::Cpu;
    }
    for g in gpus {
        if g.free_vram_bytes > threshold {
            return ExpandPlacement::Gpu(g.ordinal);
        }
    }
    ExpandPlacement::Cpu
}

/// Minimum free VRAM for BF16 Qwen3-4B on GPU with drop-and-reload.
/// 8.2GB model + 2GB activation headroom = 10.2GB.
/// With drop-and-reload, the encoder is temporary — loaded for encoding, then dropped.
pub const QWEN3_FP16_VRAM_THRESHOLD: u64 = 10_200_000_000;

/// Headroom for activation memory during inference (denoising + VAE decode workspace).
const MEMORY_BUDGET_HEADROOM: u64 = 2_000_000_000; // 2GB

// ── Placement resolution ─────────────────────────────────────────────────────

/// Resolve a caller-supplied `DeviceRef` override into a concrete candle
/// `Device`, falling back to `auto` when the override is missing or `Auto`.
///
/// - `None`, `Some(Auto)` — call `auto()` (existing VRAM-aware logic).
/// - `Some(Cpu)`          — `Device::Cpu`, never invoke `auto()`.
/// - `Some(Gpu { ordinal })` — try CUDA first, then Metal. Each backend is
///   gated by its candle feature flag so a CPU-only build returns a clear
///   error message instead of a build failure.
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
    }
}

#[cfg(feature = "cuda")]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    candle_core::Device::new_cuda(ordinal)
        .map_err(|e| anyhow::anyhow!("failed to open CUDA device {ordinal}: {e}"))
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    candle_core::Device::new_metal(ordinal)
        .map_err(|e| anyhow::anyhow!("failed to open Metal device {ordinal}: {e}"))
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
            return placement.text_encoders;
        }
        DeviceRef::Auto
    } else {
        placement.text_encoders
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

#[cfg(not(target_os = "macos"))]
pub fn free_system_memory_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn available_system_memory_bytes() -> Option<u64> {
    None
}

// ── GPU memory reclamation ───────────────────────────────────────────────────

/// Reclaim GPU memory by resetting the CUDA primary context for the specified device.
///
/// **Must only be called when no CUDA objects (tensors, devices, engines) exist on this device.**
/// This resets CUDA state on the specified GPU: driver context, cuBLAS workspace caches,
/// compiled kernel modules, and memory pools. After calling this, the next
/// `Device::new_cuda(ordinal)` will create a fresh context.
///
/// On non-CUDA platforms, this is a no-op.
#[cfg(feature = "cuda")]
pub fn reclaim_gpu_memory(ordinal: usize) {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};

    debug_assert_ordinal_matches_thread(ordinal, "reclaim_gpu_memory");

    // Synchronize to ensure all async GPU work completes before reset.
    let _ = result::ctx::synchronize();

    // Get the CUdevice handle for the specified GPU ordinal.
    let cu_device = match result::device::get(ordinal as i32) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("reclaim_gpu_memory: failed to get device {ordinal}: {e}");
            return;
        }
    };

    // Reset the primary context — frees all allocations, destroys cuBLAS/cuDNN
    // workspace caches, and releases compiled kernel modules.
    let result = unsafe { sys::cuDevicePrimaryCtxReset_v2(cu_device) };
    if result != sys::CUresult::CUDA_SUCCESS {
        tracing::warn!(
            "reclaim_gpu_memory: cuDevicePrimaryCtxReset for device {ordinal} returned {result:?}"
        );
    } else {
        tracing::info!("CUDA primary context reset for device {ordinal}, GPU memory reclaimed");
    }
}

/// No-op on non-CUDA platforms.
#[cfg(not(feature = "cuda"))]
pub fn reclaim_gpu_memory(_ordinal: usize) {}

// ── VRAM query ───────────────────────────────────────────────────────────────

/// Query free VRAM in bytes for the specified GPU ordinal.
///
/// On CUDA, sets the context to the specified device before querying.
/// On macOS (unified memory), returns available system memory (free + inactive).
/// On other non-CUDA platforms, no VRAM info available.
#[cfg(feature = "cuda")]
pub fn free_vram_bytes(ordinal: usize) -> Option<u64> {
    // Create/bind the device context for the specified ordinal before querying.
    if candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal).is_ok() {
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            .ok()
            .map(|(free, _total)| free as u64)
    } else {
        None
    }
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

/// Estimate current VRAM usage (total - free) for the specified GPU ordinal.
/// Returns 0 if unavailable. Used by the model cache to track per-model VRAM footprint.
#[cfg(feature = "cuda")]
pub fn vram_used_estimate(ordinal: usize) -> u64 {
    if candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal).is_ok() {
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            .ok()
            .map(|(_free, total)| total as u64 - _free as u64)
            .unwrap_or(0)
    } else {
        0
    }
}

/// Non-CUDA stub — no VRAM tracking available.
#[cfg(not(feature = "cuda"))]
pub fn vram_used_estimate(_ordinal: usize) -> u64 {
    0
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

pub(crate) fn should_offload(transformer_size: u64, free_vram: u64) -> bool {
    /// Headroom needed beyond the transformer for activations, noise, VAE workspace.
    const INFERENCE_HEADROOM: u64 = 3_000_000_000; // 3 GB
    let needed = transformer_size.saturating_add(INFERENCE_HEADROOM);
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
pub fn estimate_peak_memory(paths: &mold_core::ModelPaths, strategy: LoadStrategy) -> u64 {
    let file_size = |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);

    let transformer_size = if !paths.transformer_shards.is_empty() {
        paths.transformer_shards.iter().map(|p| file_size(p)).sum()
    } else {
        file_size(&paths.transformer)
    };
    let vae_size = file_size(&paths.vae);

    let t5_size = paths.t5_encoder.as_ref().map(|p| file_size(p)).unwrap_or(0);
    let clip_size = paths
        .clip_encoder
        .as_ref()
        .map(|p| file_size(p))
        .unwrap_or(0);
    let clip2_size = paths
        .clip_encoder_2
        .as_ref()
        .map(|p| file_size(p))
        .unwrap_or(0);
    let text_encoder_size: u64 = paths.text_encoder_files.iter().map(|p| file_size(p)).sum();

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

/// Check if loading a component of `size_bytes` would exceed available system memory.
/// Uses available memory (free + inactive/reclaimable) as the primary metric, since
/// macOS moves recently-freed pages to inactive rather than free — these are trivially
/// reclaimable with no I/O. Hard-fails if component exceeds 90% of available memory;
/// warns (but proceeds) if component exceeds 2x free memory. On CUDA or when memory
/// info is unavailable, always returns Ok.
pub(crate) fn preflight_memory_check(component_name: &str, size_bytes: u64) -> anyhow::Result<()> {
    // --eager or MOLD_EAGER=1 bypasses the check
    if std::env::var("MOLD_EAGER").is_ok_and(|v| v == "1") {
        return Ok(());
    }

    let available = match available_system_memory_bytes() {
        Some(a) if a > 0 => a,
        _ => return Ok(()), // No info or CUDA — can't check
    };

    let free = free_system_memory_bytes();

    preflight_check_budget(component_name, size_bytes, available, free)
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

    #[test]
    fn offload_when_transformer_exceeds_vram() {
        // 24GB transformer, 16GB free → needs offloading
        assert!(should_offload(24 * GB, 16 * GB));
    }

    #[test]
    fn offload_when_transformer_fits_but_no_headroom() {
        // 23.8GB transformer on 24.7GB free: file fits but 23.8+3.0 headroom > 24.7
        let xformer = 23_800_000_000;
        let free = 24_700_000_000;
        assert!(should_offload(xformer, free));
    }

    #[test]
    fn no_offload_when_plenty_of_vram() {
        // 12GB transformer on 24GB free → plenty of room
        assert!(!should_offload(12 * GB, 24 * GB));
    }

    #[test]
    fn no_offload_when_vram_unknown() {
        // free = 0 means we couldn't query VRAM
        assert!(!should_offload(24 * GB, 0));
    }

    #[test]
    fn no_offload_when_vram_too_small_for_single_block() {
        // 24GB transformer but only 2GB free — not enough for even one block
        assert!(!should_offload(24 * GB, 2 * GB));
    }

    // ── select_expand_device tests ─────────────────────────────────────────

    fn gpu(ordinal: usize, free_gb: u64) -> DiscoveredGpu {
        DiscoveredGpu {
            ordinal,
            name: format!("gpu{ordinal}"),
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
}

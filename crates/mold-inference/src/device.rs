use crate::engine::LoadStrategy;
use crate::progress::ProgressReporter;
use mold_core::types::{GpuBackend, GpuSelection, GpuSelector};
use std::cell::Cell;
use std::collections::BTreeSet;

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
            let total = available_system_memory_bytes().unwrap_or(0);
            let free = free_system_memory_bytes().unwrap_or(0);
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
                        let normalized = normalize_uuid_selector(identifier)?;
                        gpus.iter()
                            .filter(|gpu| uuid_selector_matches(gpu, &normalized))
                            .collect::<Vec<_>>()
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
        matches!(self, ActivationFamily::LtxVideo)
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
/// schedulers) and `2` for CFG-doubled forwards (SDXL, SD3 with
/// `guidance != 1.0`).
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
        // LTX-Video (0.9.6 / 0.9.8 2B or 13B): loads the entire transformer
        // into VRAM during each generate call. The file-size-based preflight
        // applies normally — the 13B BF16 checkpoint is ~26 GB and must be
        // counted in full.
        "ltx-video" => ActivationFamily::LtxVideo,
        // LTX-2 (19B / 22B): streaming-loaded transformer — only a couple of
        // blocks are GPU-resident at peak, so the preflight skips the
        // file-size estimate and uses a fixed streaming cap instead.
        "ltx2" | "ltx-2" | "ltx-2.3" => ActivationFamily::Ltx2Video,
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

/// Minimum free VRAM (bytes) needed to land Gemma 3 12B BF16 on a single GPU
/// alongside its activation workspace. ~23 GB resident weights + ~1 GB
/// activation overhead. Encoder isn't streamed — picking GPU means the whole
/// thing is co-resident with the LTX-2 transformer phase.
pub const LTX2_GEMMA_VRAM_THRESHOLD: u64 = 24_000_000_000;

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

/// Pick where to load the LTX-2 Gemma 3 12B prompt encoder: active GPU first,
/// then sibling GPUs in ordinal order, then CPU.
///
/// - `gpus` is the output of [`discover_gpus`] — ordinals in ascending order.
/// - `active_ordinal` is the GPU the LTX-2 transformer was loaded onto. We
///   prefer co-residency (no cross-device tensor copy at encode time) but
///   fall through to siblings when the active GPU is full.
/// - A GPU is considered to fit when `free_vram_bytes > threshold` (strict
///   greater-than, mirrors [`select_expand_device`]).
/// - Returns [`LtxGemmaPlacement::Cpu`] when no GPU has room.
pub fn select_ltx2_gemma_device(
    gpus: &[DiscoveredGpu],
    active_ordinal: usize,
    threshold: u64,
) -> LtxGemmaPlacement {
    if let Some(g) = gpus
        .iter()
        .find(|g| g.ordinal == active_ordinal && g.free_vram_bytes > threshold)
    {
        return LtxGemmaPlacement::Gpu(g.ordinal);
    }
    for g in gpus {
        if g.ordinal == active_ordinal {
            continue;
        }
        if g.free_vram_bytes > threshold {
            return LtxGemmaPlacement::Gpu(g.ordinal);
        }
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
    if let Ok(raw) = std::env::var("MOLD_LTX2_GEMMA_DEVICE") {
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

    if std::env::var_os("MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER").is_some() {
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
/// before falling through to the GPU-walk + CPU fallback. The runtime and
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
    debug_assert_ordinal_matches_thread(ordinal, "resolve_device");
    candle_core::Device::new_cuda(ordinal)
        .map_err(|e| anyhow::anyhow!("failed to open CUDA device {ordinal}: {e}"))
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
fn resolve_gpu_ordinal(ordinal: usize) -> anyhow::Result<candle_core::Device> {
    debug_assert_ordinal_matches_thread(ordinal, "resolve_device");
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

// ── Text-encoder retention ───────────────────────────────────────────────────

/// Whether to park text-encoder weights on CPU instead of dropping them after
/// encoding finishes (opt-in via `MOLD_KEEP_TE_RAM=1`).
///
/// Default off for backward compatibility — the existing drop-and-reload path
/// re-mmaps safetensors / re-dequantizes GGUF on every request, costing
/// ~2-4 s per FLUX generation (~1 s on SD3).
///
/// When on, FP16/BF16 encoders survive between requests on host RAM (~9 GB
/// for T5-XXL fp16); only the lightweight GPU↔CPU tensor copy happens between
/// requests. Quantized GGUF encoders fall back to drop-and-reload regardless,
/// because their `QTensor` storage is device-tied and not trivially walkable.
///
/// This mirrors ComfyUI's `text_encoder_offload_device()` behavior
/// (`comfy/model_management.py:1012`).
pub fn keep_te_in_ram() -> bool {
    std::env::var("MOLD_KEEP_TE_RAM")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Best-effort CUDA device synchronize, ignoring errors.
///
/// After a `CUDA_ERROR_OUT_OF_MEMORY` the CUDA context may have in-flight work
/// that hasn't been flushed; subsequent allocations can inherit a poisoned
/// scheduler state. Calling synchronize before reporting the OOM and before
/// any retry lets CUDA drain pending work and reset internal queues so the
/// next allocation attempt starts clean.
///
/// Errors are silently swallowed — this is a "best effort" hygiene step, not a
/// hard requirement. The caller has already decided to surface an OOM error;
/// a secondary synchronize failure shouldn't shadow the primary message.
///
/// On non-CUDA platforms this is a no-op.
#[cfg(feature = "cuda")]
pub fn try_synchronize_device(ordinal: usize) {
    debug_assert_ordinal_matches_thread(ordinal, "try_synchronize_device");
    if let Ok(context) = candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal) {
        let _ = context.synchronize();
    }
}

/// No-op on non-CUDA platforms.
#[cfg(not(feature = "cuda"))]
pub fn try_synchronize_device(_ordinal: usize) {}

/// Synchronize pending work after device-backed values have been dropped, then
/// return the actual reserve-adjusted free VRAM reported by the driver.
///
/// Dropping Mold-owned objects does not imply that nominal total VRAM is
/// available: driver workspaces, fragmentation, and external allocations
/// remain unavailable pressure and are preserved in this measurement.
pub fn post_drop_free_vram_bytes(ordinal: usize) -> Option<u64> {
    try_synchronize_device(ordinal);
    usable_free_vram_bytes(ordinal)
}

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
    if let Ok(s) = std::env::var("MOLD_RESERVE_VRAM_MB") {
        if let Ok(mb) = s.parse::<u64>() {
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
    if candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal).is_ok() {
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            .ok()
            .map(|(free, total)| total as u64 - free as u64)
            .unwrap_or(0)
    } else {
        0
    }
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
    if candle_core::cuda_backend::cudarc::driver::CudaContext::new(ordinal).is_ok() {
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            .ok()
            .map(|(_free, total)| total as u64)
    } else {
        None
    }
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
pub(crate) fn resolve_vae_dtype(default_dtype: candle_core::DType) -> candle_core::DType {
    use candle_core::DType;
    match std::env::var("MOLD_VAE_DTYPE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        None | Some("") | Some("auto") => default_dtype,
        Some("bf16") | Some("BF16") => DType::BF16,
        Some("fp16") | Some("f16") | Some("FP16") | Some("F16") => DType::F16,
        Some("fp32") | Some("f32") | Some("FP32") | Some("F32") => DType::F32,
        Some(other) => {
            tracing::warn!(
                value = other,
                "MOLD_VAE_DTYPE has unrecognised value; expected one of auto/bf16/fp16/fp32 — falling back to default"
            );
            default_dtype
        }
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
    if std::env::var("MOLD_EAGER").is_ok_and(|v| v == "1") {
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
        // Unknown slug — falls back to FluxDit.
        assert_eq!(
            activation_family_for("bogus-family"),
            ActivationFamily::FluxDit
        );
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

    /// Multi-GPU host: the active GPU is full (4 GB free), but a sibling
    /// GPU has plenty of room. Encoder runs there and pays a single
    /// cross-device copy at encode time.
    #[test]
    fn select_ltx2_gemma_device_picks_sibling_gpu_when_active_full() {
        let gpus = vec![gpu(0, 4), gpu(1, 25)];
        assert_eq!(
            select_ltx2_gemma_device(&gpus, 0, 24 * GB),
            LtxGemmaPlacement::Gpu(1),
        );
    }

    /// Three-GPU walk: the active GPU is GPU 1; both GPU 0 and GPU 2 have
    /// room. The walk picks the first sibling in ordinal order (GPU 0)
    /// rather than starting from `active_ordinal`.
    #[test]
    fn select_ltx2_gemma_device_walks_remaining_in_ordinal_order() {
        let gpus = vec![gpu(0, 25), gpu(1, 4), gpu(2, 25)];
        assert_eq!(
            select_ltx2_gemma_device(&gpus, 1, 24 * GB),
            LtxGemmaPlacement::Gpu(0),
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
    /// edits go through the constant rather than scattering literal 24-GB
    /// figures across call sites.
    #[test]
    fn ltx2_gemma_vram_threshold_is_24gb() {
        assert_eq!(LTX2_GEMMA_VRAM_THRESHOLD, 24_000_000_000);
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

    /// `MOLD_KEEP_TE_RAM` defaults to off so the existing drop-and-reload
    /// behavior is preserved when the env var is absent.
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

    #[test]
    fn estimate_peak_memory_separate_vae_file_still_sums() {
        // Sanity: when transformer and vae are distinct files (e.g. FLUX with
        // separate vae companion), both file sizes contribute as before.
        let dir = tempfile::tempdir().expect("tempdir");
        let transformer = write_dummy_file(dir.path(), "tx.safetensors", 4_000_000_000);
        let vae = write_dummy_file(dir.path(), "vae.safetensors", 1_000_000_000);
        let te = write_dummy_file(dir.path(), "te.safetensors", 9_000_000_000);
        let paths = mold_core::ModelPaths {
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

    // --- resolve_vae_dtype tests ---
    //
    // MOLD_VAE_DTYPE is process-global; tests serialize via a static mutex
    // (mirrors the MOLD_LONG_PROMPTS / MOLD_CFG_PLUS test pattern elsewhere
    // in the crate).

    fn vae_env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn resolve_vae_dtype_unset_returns_default() {
        let _g = vae_env_lock();
        // SAFETY: serialized via vae_env_lock to avoid racing parallel tests.
        unsafe { std::env::remove_var("MOLD_VAE_DTYPE") };
        assert_eq!(
            resolve_vae_dtype(candle_core::DType::BF16),
            candle_core::DType::BF16
        );
        assert_eq!(
            resolve_vae_dtype(candle_core::DType::F16),
            candle_core::DType::F16
        );
    }

    #[test]
    fn resolve_vae_dtype_auto_returns_default() {
        let _g = vae_env_lock();
        unsafe { std::env::set_var("MOLD_VAE_DTYPE", "auto") };
        let resolved = resolve_vae_dtype(candle_core::DType::BF16);
        unsafe { std::env::remove_var("MOLD_VAE_DTYPE") };
        assert_eq!(resolved, candle_core::DType::BF16);
    }

    #[test]
    fn resolve_vae_dtype_fp32_forces_f32_regardless_of_default() {
        let _g = vae_env_lock();
        unsafe { std::env::set_var("MOLD_VAE_DTYPE", "fp32") };
        let resolved = resolve_vae_dtype(candle_core::DType::BF16);
        unsafe { std::env::remove_var("MOLD_VAE_DTYPE") };
        assert_eq!(resolved, candle_core::DType::F32);
    }

    #[test]
    fn resolve_vae_dtype_bf16_forces_bf16_even_when_default_is_f32() {
        // CPU default is F32; user opts back into BF16 explicitly. Pins the
        // contract that the env knob can both raise *and* lower precision.
        let _g = vae_env_lock();
        unsafe { std::env::set_var("MOLD_VAE_DTYPE", "bf16") };
        let resolved = resolve_vae_dtype(candle_core::DType::F32);
        unsafe { std::env::remove_var("MOLD_VAE_DTYPE") };
        assert_eq!(resolved, candle_core::DType::BF16);
    }

    #[test]
    fn resolve_vae_dtype_fp16_alias_recognised() {
        // f16 / fp16 / F16 / FP16 must all resolve identically — different
        // tools and shells normalise case differently.
        let _g = vae_env_lock();
        for value in ["fp16", "f16", "FP16", "F16"] {
            unsafe { std::env::set_var("MOLD_VAE_DTYPE", value) };
            let resolved = resolve_vae_dtype(candle_core::DType::BF16);
            assert_eq!(
                resolved,
                candle_core::DType::F16,
                "value `{value}` should resolve to F16"
            );
        }
        unsafe { std::env::remove_var("MOLD_VAE_DTYPE") };
    }

    #[test]
    fn resolve_vae_dtype_invalid_value_falls_back_to_default() {
        let _g = vae_env_lock();
        unsafe { std::env::set_var("MOLD_VAE_DTYPE", "fp64") };
        let resolved = resolve_vae_dtype(candle_core::DType::BF16);
        unsafe { std::env::remove_var("MOLD_VAE_DTYPE") };
        assert_eq!(
            resolved,
            candle_core::DType::BF16,
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
}

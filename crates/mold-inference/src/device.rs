use crate::engine::LoadStrategy;
use crate::progress::ProgressReporter;

/// Create a GPU device, falling back to CPU if no accelerator is available.
/// Reports device selection via the progress reporter.
pub fn create_device(progress: &ProgressReporter) -> anyhow::Result<candle_core::Device> {
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
    if candle_core::utils::cuda_is_available() {
        progress.info("CUDA detected, using GPU");
        tracing::info!("CUDA detected, using GPU");
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        progress.info("Metal detected, using GPU");
        tracing::info!("Metal detected, using MPS");
        Ok(Device::new_metal(0)?)
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

/// Minimum free VRAM for BF16 Qwen3-4B on GPU with drop-and-reload.
/// 8.2GB model + 2GB activation headroom = 10.2GB.
/// With drop-and-reload, the encoder is temporary — loaded for encoding, then dropped.
pub const QWEN3_FP16_VRAM_THRESHOLD: u64 = 10_200_000_000;

/// Headroom for activation memory during inference (denoising + VAE decode workspace).
const MEMORY_BUDGET_HEADROOM: u64 = 2_000_000_000; // 2GB

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
pub(crate) fn free_system_memory_bytes() -> Option<u64> {
    macos_vm_stats().map(|s| s.free)
}

/// Total available system memory on macOS (free + inactive pages).
///
/// Inactive pages CAN be reclaimed by the OS, but doing so involves I/O for dirty pages.
/// Use this for memory budget checks (can this model run at all?) not for real-time decisions.
#[cfg(target_os = "macos")]
pub(crate) fn available_system_memory_bytes() -> Option<u64> {
    macos_vm_stats().map(|s| s.free + s.inactive)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn free_system_memory_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn available_system_memory_bytes() -> Option<u64> {
    None
}

// ── GPU memory reclamation ───────────────────────────────────────────────────

/// Reclaim GPU memory by resetting the CUDA primary context.
///
/// **Must only be called when no CUDA objects (tensors, devices, engines) exist.**
/// This resets all CUDA state on GPU 0: driver context, cuBLAS workspace caches,
/// compiled kernel modules, and memory pools. After calling this, the next
/// `Device::new_cuda(0)` will create a fresh context.
///
/// On non-CUDA platforms, this is a no-op.
#[cfg(feature = "cuda")]
pub fn reclaim_gpu_memory() {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};

    // Synchronize to ensure all async GPU work completes before reset.
    let _ = result::ctx::synchronize();

    // Get the CUdevice handle for GPU 0.
    let cu_device = match result::device::get(0) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("reclaim_gpu_memory: failed to get device: {e}");
            return;
        }
    };

    // Reset the primary context — frees all allocations, destroys cuBLAS/cuDNN
    // workspace caches, and releases compiled kernel modules.
    let result = unsafe { sys::cuDevicePrimaryCtxReset_v2(cu_device) };
    if result != sys::CUresult::CUDA_SUCCESS {
        tracing::warn!("reclaim_gpu_memory: cuDevicePrimaryCtxReset returned {result:?}");
    } else {
        tracing::info!("CUDA primary context reset, GPU memory reclaimed");
    }
}

/// No-op on non-CUDA platforms.
#[cfg(not(feature = "cuda"))]
pub fn reclaim_gpu_memory() {}

// ── VRAM query ───────────────────────────────────────────────────────────────

/// Query free VRAM in bytes from the current CUDA context.
#[cfg(feature = "cuda")]
pub(crate) fn free_vram_bytes() -> Option<u64> {
    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _total)| free as u64)
}

/// On macOS, return immediately free system memory (conservative estimate).
/// On other non-CUDA platforms, no VRAM info available.
#[cfg(not(feature = "cuda"))]
pub(crate) fn free_vram_bytes() -> Option<u64> {
    free_system_memory_bytes()
}

// ── Formatting ───────────────────────────────────────────────────────────────

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

/// Check whether a model component fits comfortably in memory.
///
/// On CUDA, checks discrete VRAM (same as should_use_gpu).
/// On Metal, checks immediately free system memory against threshold.
/// Returns false if loading this component would require heavy page reclamation.
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
pub(crate) fn memory_status_string() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        if let Some(free) = free_vram_bytes() {
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
        let result = free_vram_bytes();
        #[cfg(target_os = "macos")]
        assert!(result.is_some(), "macOS should return system memory info");
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        assert_eq!(result, None);
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
}

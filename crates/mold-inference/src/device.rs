/// Minimum free VRAM (bytes) required to place T5-XXL on GPU.
/// This accounts for T5 weights (~9.2GB) plus headroom for denoising activations
/// and VAE decode (~6.5GB at 1024x1024). Conservative to avoid CUDA OOM.
pub const T5_VRAM_THRESHOLD: u64 = 16_000_000_000;
/// Minimum free VRAM (bytes) required to place CLIP-L on GPU: ~246MB model + 500MB headroom.
pub const CLIP_VRAM_THRESHOLD: u64 = 800_000_000;

/// Query free VRAM in bytes from the current CUDA context.
#[cfg(feature = "cuda")]
pub(crate) fn free_vram_bytes() -> Option<u64> {
    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _total)| free as u64)
}

/// No VRAM info available without CUDA.
#[cfg(not(feature = "cuda"))]
pub(crate) fn free_vram_bytes() -> Option<u64> {
    None
}

/// Format bytes as a human-readable size (e.g. "11.7 GB").
pub(crate) fn fmt_gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
}

/// Determine whether a component should be placed on GPU given free VRAM.
pub(crate) fn should_use_gpu(is_cuda: bool, free_vram: u64, threshold: u64) -> bool {
    is_cuda && free_vram > threshold
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

    // --- free_vram_bytes (non-CUDA stub) ---

    #[test]
    fn free_vram_returns_none_without_cuda() {
        #[cfg(not(feature = "cuda"))]
        assert_eq!(free_vram_bytes(), None);
    }

    // --- VRAM threshold decision tests ---

    #[test]
    fn t5_on_gpu_when_plenty_of_vram() {
        // Q4 transformer (7GB) + VAE (0.3GB) on 24GB card → ~16.7GB free
        assert!(should_use_gpu(true, 16_700_000_000, T5_VRAM_THRESHOLD));
    }

    #[test]
    fn t5_on_cpu_when_q6_on_24gb() {
        // Q6 transformer (9.9GB) + VAE (0.3GB) on 24GB card → ~14.6GB free
        assert!(!should_use_gpu(true, 14_600_000_000, T5_VRAM_THRESHOLD));
    }

    #[test]
    fn t5_on_cpu_when_q8_on_24gb() {
        // Q8 transformer (12GB) + VAE (0.3GB) on 24GB card → ~11.7GB free
        assert!(!should_use_gpu(true, 11_700_000_000, T5_VRAM_THRESHOLD));
    }

    #[test]
    fn t5_on_cpu_when_bf16_fills_vram() {
        // BF16 dev (23GB) + VAE (0.3GB) on 24GB card → ~0.7GB free
        assert!(!should_use_gpu(true, 700_000_000, T5_VRAM_THRESHOLD));
    }

    #[test]
    fn t5_on_cpu_when_exactly_at_threshold() {
        // Exactly at threshold should NOT place on GPU (we need strictly more)
        assert!(!should_use_gpu(true, T5_VRAM_THRESHOLD, T5_VRAM_THRESHOLD));
    }

    #[test]
    fn t5_on_cpu_when_no_cuda() {
        // Even with plenty of "free memory", non-CUDA devices always use CPU
        assert!(!should_use_gpu(false, 100_000_000_000, T5_VRAM_THRESHOLD));
    }

    #[test]
    fn t5_on_gpu_on_48gb_card() {
        // Q8 transformer (12GB) + VAE (0.3GB) on 48GB card → ~35.7GB free
        assert!(should_use_gpu(true, 35_700_000_000, T5_VRAM_THRESHOLD));
    }

    #[test]
    fn clip_on_gpu_when_vram_available() {
        // After Q4 transformer + T5 on GPU: ~7.5GB free → CLIP easily fits
        assert!(should_use_gpu(true, 7_500_000_000, CLIP_VRAM_THRESHOLD));
    }

    #[test]
    fn clip_on_gpu_with_minimal_vram() {
        // Just above 800MB threshold
        assert!(should_use_gpu(true, 900_000_000, CLIP_VRAM_THRESHOLD));
    }

    #[test]
    fn clip_on_cpu_when_vram_tight() {
        // Only 500MB free — below CLIP threshold
        assert!(!should_use_gpu(true, 500_000_000, CLIP_VRAM_THRESHOLD));
    }

    // --- Threshold constant sanity checks ---

    #[test]
    fn t5_threshold_accounts_for_headroom() {
        // T5 is ~9.2GB, threshold should be higher to account for
        // denoising activation memory and VAE decode headroom
        assert!(T5_VRAM_THRESHOLD > 9_200_000_000);
        // But not unreasonably high (should still work on 48GB cards)
        assert!(T5_VRAM_THRESHOLD < 25_000_000_000);
    }

    #[test]
    fn clip_threshold_accounts_for_headroom() {
        // CLIP is ~246MB, threshold should be higher
        assert!(CLIP_VRAM_THRESHOLD > 246_000_000);
        // But not unreasonably high
        assert!(CLIP_VRAM_THRESHOLD < 2_000_000_000);
    }
}

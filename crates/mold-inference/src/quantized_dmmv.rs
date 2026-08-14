//! Mold's readable mirror of candle's process-global `FORCE_DMMV` switch.
//!
//! `candle_core::quantized::cuda::set_force_dmmv` writes a process-global
//! `AtomicBool` (`candle-core/src/quantized/cuda.rs`) and exposes no reader, so
//! an engine that dispatches on the quantized fast path cannot ask candle
//! whether that path is still reachable. Mold is the only thing in this process
//! that flips it — Wan's `MOLD_WAN_FORCE_DMMV=1` diagnostic, inside its denoise
//! loop, once and never cleared — so mold keeps the answer here and routes
//! every write through [`set_force_dmmv`].
//!
//! This became load-bearing when Qwen-Image started running `QMatMul` on CUDA.
//! With the fallback forced, `QCudaStorage::fwd` skips both `fast_mmvq` and
//! `fast_mmq` and lands in `dequantize_matmul_vec` / `dequantize_matmul`, which
//! read the activation as `f32` and therefore reject the BF16 activations that
//! engine runs — the exact failure its dtype gate exists to prevent. Because
//! the flag can flip after an engine is built and cached, readers must consult
//! it per forward, not only at load.

use std::sync::atomic::{AtomicBool, Ordering};

static FORCE_DMMV: AtomicBool = AtomicBool::new(false);

/// Force (or release) candle's quantized matmuls onto the
/// dequantize-per-forward fallback, recording the choice for mold's own
/// readers.
pub(crate) fn set_force_dmmv(enabled: bool) {
    FORCE_DMMV.store(enabled, Ordering::Relaxed);
    #[cfg(feature = "cuda")]
    candle_core::quantized::cuda::set_force_dmmv(enabled);
}

/// Whether candle's quantized fast paths are disabled for this process.
pub(crate) fn force_dmmv_enabled() -> bool {
    FORCE_DMMV.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mirror has to answer for candle, whose flag is write-only. A reader
    /// that cannot see the flip picks a kernel path that no longer exists.
    #[test]
    fn the_mirror_reports_what_was_set() {
        assert!(!force_dmmv_enabled(), "the process starts on the fast path");
        set_force_dmmv(true);
        assert!(force_dmmv_enabled());
        set_force_dmmv(false);
        assert!(!force_dmmv_enabled());
    }
}

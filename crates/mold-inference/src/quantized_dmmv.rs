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
use std::sync::OnceLock;

static FORCE_DMMV: AtomicBool = AtomicBool::new(false);

/// Process-constant initialization from the frozen environment.
///
/// `MOLD_WAN_FORCE_DMMV` is an env-driven diagnostic, so its value is fixed
/// for the process lifetime — but the flip used to happen at Wan's first
/// denoise loop, which raced a concurrent Qwen forward: Qwen could read the
/// mirror as `false` and then candle's global could flip to `true` before the
/// kernel dispatch read it, landing BF16 activations in the f32-only dequant
/// fallback. Initializing on FIRST READ (from any engine) makes the value
/// constant before any dispatch that consults it.
fn env_initialized() -> bool {
    static INIT: OnceLock<bool> = OnceLock::new();
    *INIT.get_or_init(|| {
        let force = crate::runtime_env::value("MOLD_WAN_FORCE_DMMV").as_deref() == Some("1");
        if force {
            FORCE_DMMV.store(true, Ordering::Relaxed);
            #[cfg(feature = "cuda")]
            candle_core::quantized::cuda::set_force_dmmv(true);
        }
        force
    })
}

/// Force (or release) candle's quantized matmuls onto the
/// dequantize-per-forward fallback, recording the choice for mold's own
/// readers.
///
/// No production call site remains — the env diagnostic initializes through
/// [`env_initialized`] — but the setter stays as the ONLY sanctioned way to
/// flip candle's write-only global, so a future scoped user cannot bypass the
/// mirror. The tests drive it.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn set_force_dmmv(enabled: bool) {
    env_initialized();
    FORCE_DMMV.store(enabled, Ordering::Relaxed);
    #[cfg(feature = "cuda")]
    candle_core::quantized::cuda::set_force_dmmv(enabled);
}

/// Whether candle's quantized fast paths are disabled for this process.
pub(crate) fn force_dmmv_enabled() -> bool {
    env_initialized() || FORCE_DMMV.load(Ordering::Relaxed)
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

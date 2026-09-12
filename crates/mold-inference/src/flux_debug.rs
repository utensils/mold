//! Opt-in per-step non-finite check for the FLUX families.
//!
//! The FLUX.2 GGUF transformer used to wrap all eighteen of its linear sites
//! in a `linear_nan_safe` helper — a full-tensor `ne` compare, a zeros
//! allocation and a `where_cond` per linear — copied from SD3's quantized
//! MMDiT in #166 without a FLUX.2 NaN ever having been observed. That is a
//! measured ~0.5 s/step spent masking a fault it would be a bug to hide: a
//! transformer emitting NaN has something wrong with it, and zeroing the
//! element turns a loud failure into a quietly wrong picture.
//!
//! This is the replacement, and it is deliberately the other shape: OFF by
//! default, ONE reduction per denoise step rather than eighteen per block, and
//! it BAILS naming the step instead of continuing. Enable it with
//! `MOLD_FLUX_DEBUG_NONFINITE=1` when a render comes out black or blank; the
//! step it names is where to look.
//!
//! It is a diagnostic, not an engine input: it adds a device synchronization
//! and a reduction, so it changes wall clock, but it cannot change a pixel —
//! which is why it stays out of `runtime_env::ENGINE_SHAPING_VARIABLES` and
//! out of the execution fingerprint.

use candle_core::{DType, Tensor};
use std::sync::OnceLock;

/// Whether `MOLD_FLUX_DEBUG_NONFINITE` asked for the check, read once.
///
/// Read straight from the process environment rather than through
/// `runtime_env::value`, which debug-asserts its argument is engine-shaping.
/// Every other diagnostic switch (`MOLD_STEP_PREVIEW`, `MOLD_FLUX2_DUMP_LATENT`)
/// does the same.
pub(crate) fn nonfinite_check_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled =
            parse_nonfinite_flag(std::env::var("MOLD_FLUX_DEBUG_NONFINITE").ok().as_deref());
        if enabled {
            tracing::info!(
                "MOLD_FLUX_DEBUG_NONFINITE is on: every denoise step is checked for \
                 non-finite values and the render will fail at the first one"
            );
        }
        enabled
    })
}

/// Pure parser, so the flag's semantics are testable without the env.
pub(crate) fn parse_nonfinite_flag(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

/// Fail the render if `tensor` holds a NaN or an infinity.
///
/// A single `sqr().sum_all()` is enough and is one kernel: NaN propagates
/// through both, and an infinity squares to an infinity, so a finite total is
/// proof every element was finite. Summing the raw values would not do — a
/// `+inf` and a `-inf` cancel.
///
/// `Ok(())` without the flag, before any work at all, so the default path
/// costs a boolean.
pub(crate) fn check_step_is_finite(tensor: &Tensor, what: &str, step: usize) -> anyhow::Result<()> {
    if !nonfinite_check_enabled() {
        return Ok(());
    }
    let total = tensor
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    if total.is_finite() {
        return Ok(());
    }
    Err(nonfinite_error(what, step))
}

/// The bail, as a value, so its CLASS is testable without the flag.
///
/// It is marked model-specific: a NaN is a fact about this checkpoint at this
/// shape, and three of them used to degrade the whole DEVICE for a minute and
/// refuse every other model on a single-GPU host.
pub(crate) fn nonfinite_error(what: &str, step: usize) -> anyhow::Error {
    crate::failure_class::model_specific_error(format!(
        "non-finite {what} at denoise step {step} (MOLD_FLUX_DEBUG_NONFINITE); \
         re-run with MOLD_FLUX2_QMATMUL=0 to take the per-forward dequant arm"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn the_flag_is_off_unless_it_is_explicitly_truthy() {
        for value in ["1", "true", "on", "yes", " TRUE "] {
            assert!(parse_nonfinite_flag(Some(value)), "{value}");
        }
        for value in ["0", "false", "off", "no", "", "garbage"] {
            assert!(!parse_nonfinite_flag(Some(value)), "{value}");
        }
        assert!(!parse_nonfinite_flag(None));
    }

    /// The sum of squares is the right reduction: it catches a NaN, a `+inf`
    /// and — the case a plain sum misses — a `+inf` paired with a `-inf`.
    #[test]
    fn the_sum_of_squares_catches_every_non_finite_shape() {
        let device = Device::Cpu;
        let finite = |v: Vec<f32>| {
            Tensor::from_vec(v, 4, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .is_finite()
        };
        assert!(finite(vec![1.0, -2.0, 3.5, 0.0]));
        assert!(!finite(vec![1.0, f32::NAN, 3.5, 0.0]));
        assert!(!finite(vec![1.0, f32::INFINITY, 3.5, 0.0]));
        assert!(!finite(vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0]));
    }

    /// A non-finite prediction is the MODEL's fault, never the card's.
    ///
    /// Three of these in a row used to trip the server's device breaker,
    /// which on a single-GPU host refused every other model for sixty
    /// seconds with "no enabled, healthy GPU device is available".
    #[test]
    fn the_bail_is_marked_model_specific() {
        let error = nonfinite_error("prediction", 3);
        assert!(crate::failure_class::is_model_specific_failure(&error));
        assert!(
            error
                .to_string()
                .contains("non-finite prediction at denoise step 3"),
            "the diagnostic still leads with the step it found: {error:#}"
        );
    }

    /// With the flag off — the default in this test process — the check is a
    /// boolean and never touches the tensor, so even a NaN passes.
    #[test]
    fn the_check_is_inert_when_the_flag_is_off() {
        let nan = Tensor::from_vec(vec![f32::NAN; 4], 4, &Device::Cpu).unwrap();
        assert!(!nonfinite_check_enabled(), "the flag must default off");
        assert!(check_step_is_finite(&nan, "prediction", 3).is_ok());
    }
}

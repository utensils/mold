//! ARRI LogC3 (EI 800) decode for the LTX-2 HDR IC-LoRA pipeline.
//!
//! The HDR adapter is trained to emit LogC3-encoded video, so the VAE's
//! ordinary `[0, 1]` output is a *log* signal, not a display signal. Applying
//! the inverse transfer function recovers scene-referred linear light suitable
//! for EXR export and compositing.
//!
//! Ported from `packages/ltx-core/src/ltx_core/hdr.py:14-71` in
//! <https://github.com/Lightricks/LTX-2>.

/// ARRI LogC3 EI 800 curve parameters (`hdr.py:14-29`).
const A: f32 = 5.555_556;
const B: f32 = 0.052_272;
const C: f32 = 0.247_190;
const D: f32 = 0.385_537;
const E: f32 = 5.367_655;
const F: f32 = 0.092_809;
const CUT: f32 = 0.010_591;

/// Log value at which the curve switches from its linear toe to its log
/// segment: `E * CUT + F` ≈ 0.1496648.
fn cut_log() -> f32 {
    E * CUT + F
}

/// Decode one LogC3 sample to scene-referred linear.
///
/// The input is clamped to `[0, 1]` but **the output is not clamped at zero**
/// — upstream's `decompress` leaves the toe unclamped, so pure black decodes
/// to `(0 - F) / E` ≈ `-0.0173`. That negative floor is faithful and is
/// written into the EXR verbatim; only the tonemapped preview clamps it.
pub(crate) fn logc3_to_linear(value: f32) -> f32 {
    let logc = value.clamp(0.0, 1.0);
    if logc >= cut_log() {
        (10f32.powf((logc - D) / C) - B) / A
    } else {
        (logc - F) / E
    }
}

/// Encode scene-referred linear back to LogC3. Upstream only needs this for
/// its own tests; we keep it for the round-trip test below.
#[cfg(test)]
pub(crate) fn linear_to_logc3(value: f32) -> f32 {
    if value >= CUT {
        C * (A * value + B).log10() + D
    } else {
        E * value + F
    }
}

/// sRGB opto-electronic transfer function (IEC 61966-2-1), used to tonemap
/// linear HDR into the SDR preview at EV 0 (`media_io.py:769-772`).
pub(crate) fn linear_to_srgb(value: f32) -> f32 {
    let v = value.max(0.0).min(1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Map a decoded VAE frame in `[-1, 1]` to scene-referred linear HDR.
///
/// The ordinary SDR path collapses this same tensor to 8-bit; HDR forks here
/// instead, keeping full float precision.
pub(crate) fn vae_output_to_linear_hdr(sample: f32) -> f32 {
    logc3_to_linear((sample.clamp(-1.0, 1.0) + 1.0) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values checked against upstream's own constants. The negative floor is
    /// the load-bearing one: "fixing" it to 0 would silently change every
    /// shadow in an exported EXR.
    #[test]
    fn logc3_decode_matches_upstream_endpoints() {
        let black = logc3_to_linear(0.0);
        assert!(
            (black - (-0.017_289_9)).abs() < 1e-5,
            "pure black decodes below zero, got {black}"
        );
        assert!(black < 0.0, "the toe is deliberately unclamped");

        let white = logc3_to_linear(1.0);
        // ~55.10 with f64 constants; f32 lands at 55.0796. The tolerance is
        // accumulated precision in `10^((1 - D) / C)`, not a curve difference.
        assert!(
            (white - 55.099_5).abs() < 0.05,
            "peak white decodes to ~55.1, got {white}"
        );
    }

    /// The two segments must meet, or the curve has a visible step in the
    /// shadows exactly where detail matters.
    #[test]
    fn logc3_segments_are_continuous_at_the_cut() {
        let cut = cut_log();
        let below = logc3_to_linear(cut - 1e-6);
        let above = logc3_to_linear(cut + 1e-6);
        assert!(
            (below - above).abs() < 1e-4,
            "discontinuity at the cut: {below} vs {above}"
        );
        // At the cut the linear value is CUT itself.
        assert!((logc3_to_linear(cut) - CUT).abs() < 1e-4);
    }

    #[test]
    fn logc3_decode_is_monotonic() {
        let mut previous = f32::NEG_INFINITY;
        for step in 0..=1000 {
            let value = logc3_to_linear(step as f32 / 1000.0);
            assert!(value > previous, "not monotonic at {step}");
            previous = value;
        }
    }

    #[test]
    fn logc3_clamps_inputs_outside_the_unit_range() {
        assert_eq!(logc3_to_linear(-5.0), logc3_to_linear(0.0));
        assert_eq!(logc3_to_linear(5.0), logc3_to_linear(1.0));
    }

    #[test]
    fn logc3_round_trips() {
        for step in 0..=200 {
            let logc = step as f32 / 200.0;
            let back = linear_to_logc3(logc3_to_linear(logc));
            assert!((back - logc).abs() < 1e-4, "round trip failed at {logc}");
        }
    }

    #[test]
    fn vae_output_maps_the_full_signed_range() {
        // -1 is pure black, +1 is peak white.
        assert!((vae_output_to_linear_hdr(-1.0) - logc3_to_linear(0.0)).abs() < 1e-6);
        assert!((vae_output_to_linear_hdr(1.0) - logc3_to_linear(1.0)).abs() < 1e-6);
        assert!(vae_output_to_linear_hdr(-2.0) == vae_output_to_linear_hdr(-1.0));
    }

    #[test]
    fn srgb_encode_matches_the_standard() {
        assert!((linear_to_srgb(0.0) - 0.0).abs() < 1e-6);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
        // The canonical mid-grey check: linear 0.18 -> ~0.4620 sRGB.
        assert!((linear_to_srgb(0.18) - 0.4620).abs() < 1e-3);
        // HDR values above 1.0 clip, and the negative toe floors at black.
        assert!((linear_to_srgb(55.0) - 1.0).abs() < 1e-6);
        assert_eq!(linear_to_srgb(-0.017), 0.0);
    }
}

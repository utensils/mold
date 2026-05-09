//! SD3 Euler flow-matching sampler with SLG (Skip Layer Guidance) support.
//!
//! Port from the candle SD3 example. SD3 uses the same flow-matching framework as FLUX
//! but with a different timestep schedule (resolution-dependent SNR shifting).

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use std::time::Instant;

use super::transformer::SD3Transformer;
use crate::engine::cfg_active;
use crate::img_utils;
use crate::progress::{ProgressEvent, ProgressReporter};

/// Configuration for Skip Layer Guidance (SLG).
/// Only supported for SD3.5 Medium (depth=24).
pub struct SkipLayerGuidanceConfig {
    pub scale: f64,
    pub start: f64,
    pub end: f64,
    pub layers: Vec<usize>,
}

fn debug_tensor_stats(name: &str, tensor: &Tensor) {
    if std::env::var_os("MOLD_SD3_DEBUG").is_none() {
        return;
    }
    let stats = || -> Result<(f32, f32)> {
        let min = tensor.min_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let max = tensor.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        Ok((min, max))
    };
    match stats() {
        Ok((min, max)) => eprintln!("[sd3-debug] {name}: min={min:.4} max={max:.4}"),
        Err(err) => eprintln!("[sd3-debug] {name}: <failed: {err}>"),
    }
}

/// Run the Euler flow-matching sampling loop for SD3.
///
/// - `y`: Concatenated [y_cond, y_uncond] vector conditioning (batch=2)
/// - `context`: Concatenated [context_cond, context_uncond] text embeddings (batch=2)
/// - `cfg_scale`: Classifier-free guidance scale (1.0 = no guidance, e.g. turbo)
/// - `cfg_plus`: When true, take the CFG++ step (x_0 from guided velocity,
///   renoise with the unconditional velocity). Falls back to the standard
///   Euler step when CFG is inactive (cfg ≈ 1.0) since there is no uncond
///   row to read from. See `cfg_plus_step` for the math derivation.
/// - `time_shift`: Alpha for resolution-dependent timestep shifting (typically 3.0)
/// - `is_quantized`: If true, use F32 dtype for noise (GGUF dequantizes to F32)
/// - `progress`: Progress reporter for per-step denoising updates
#[allow(clippy::too_many_arguments)]
pub fn euler_sample(
    mmdit: &SD3Transformer,
    y: &Tensor,
    context: &Tensor,
    num_inference_steps: usize,
    cfg_scale: f64,
    cfg_plus: bool,
    time_shift: f64,
    height: usize,
    width: usize,
    slg_config: Option<&SkipLayerGuidanceConfig>,
    is_quantized: bool,
    seed: u64,
    progress: &ProgressReporter,
    initial_latents: Option<&Tensor>,
    sigmas_override: Option<Vec<f64>>,
    inpaint_ctx: Option<&img_utils::InpaintContext>,
) -> Result<Tensor> {
    // SD3 uses the same 16-channel latent noise as FLUX
    // Quantized models (GGUF) dequantize to F32, so noise must also be F32
    let noise_dtype = if is_quantized { DType::F32 } else { DType::F16 };
    let latent_h = height / 16 * 2;
    let latent_w = width / 16 * 2;

    let mut x = if let Some(latents) = initial_latents {
        latents.clone()
    } else {
        crate::engine::seeded_randn(seed, &[1, 16, latent_h, latent_w], y.device(), noise_dtype)?
    };

    let sigmas = sigmas_override.unwrap_or_else(|| {
        (0..=num_inference_steps)
            .map(|s| s as f64 / num_inference_steps as f64)
            .rev()
            .map(|t| time_snr_shift(time_shift, t))
            .collect()
    });

    let total_steps = sigmas.len().saturating_sub(1);

    // Fast path: at cfg ≈ 1.0 the uncond pred contributes 0 to the mix
    // (`apply_cfg(1, .) = cond`), so skip the doubled forward entirely.
    // Saves ~2× denoise time for distilled-CFG (Turbo) workflows.
    // The encoder still produces `[cond, uncond]` in y/context; we slice
    // the cond row here.
    let use_cfg = cfg_active(cfg_scale);
    let (y_cond_only, context_cond_only) = if use_cfg {
        (None, None)
    } else {
        (Some(y.i(..1)?), Some(context.i(..1)?))
    };

    // CFG++ requires the doubled `[cond, uncond]` forward so we can read the
    // unconditional row at integration time. When CFG is disabled (cfg ≈ 1.0)
    // the loop runs a single conditional forward and there's no uncond row to
    // use — degrade to the standard step and warn once. Loud enough to catch
    // misconfiguration but doesn't fail the request.
    let cfg_plus_active = cfg_plus && use_cfg;
    if cfg_plus && !use_cfg {
        tracing::warn!(
            cfg_scale,
            "cfg_plus requested but cfg_scale ≈ 1.0 — falling back to standard step (no uncond available)"
        );
    }

    for (step, window) in sigmas.windows(2).enumerate() {
        let step_start = Instant::now();
        let (s_curr, s_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };

        let timestep = (*s_curr) * 1000.0;
        // `noise_pred_full` holds the raw transformer output (batched
        // `[cond, uncond]` under CFG, single `[cond]` otherwise). We keep
        // it around so SLG can recover the conditional row without rerunning
        // the transformer.
        let (mut guidance, noise_pred_full) = if use_cfg {
            let noise_pred = mmdit.forward(
                &Tensor::cat(&[&x, &x], 0)?,
                &Tensor::full(timestep as f32, (2,), x.device())?.contiguous()?,
                y,
                context,
                None,
            )?;
            if step == 0 {
                debug_tensor_stats("noise_pred", &noise_pred);
            }
            let g = apply_cfg(cfg_scale, &noise_pred)?;
            if step == 0 {
                debug_tensor_stats("guidance", &g);
            }
            (g, noise_pred)
        } else {
            // Single conditional forward — `cond` IS the guided prediction.
            let noise_pred = mmdit.forward(
                &x,
                &Tensor::full(timestep as f32, (1,), x.device())?.contiguous()?,
                y_cond_only.as_ref().expect("cfg-disabled cond slice"),
                context_cond_only.as_ref().expect("cfg-disabled cond slice"),
                None,
            )?;
            if step == 0 {
                debug_tensor_stats("noise_pred (cfg=1)", &noise_pred);
            }
            (noise_pred.clone(), noise_pred)
        };

        if let Some(slg_config) = slg_config {
            if (total_steps as f64) * slg_config.start < (step as f64)
                && (step as f64) < (total_steps as f64) * slg_config.end
            {
                let slg_noise_pred = mmdit.forward(
                    &x,
                    &Tensor::full(timestep as f32, (1,), x.device())?.contiguous()?,
                    &y.i(..1)?,
                    &context.i(..1)?,
                    Some(&slg_config.layers),
                )?;
                guidance = (guidance
                    + (slg_config.scale * (noise_pred_full.i(..1)? - slg_noise_pred.i(..1))?)?)?;
            }
        }

        x = if cfg_plus_active {
            // CFG++ Euler step: x_0 estimate from CFG-guided velocity, but
            // re-noise using the unconditional velocity (keeps the trajectory
            // on the data manifold for high-CFG runs and unlocks lower CFG
            // scales — see Chung et al. 2024 §3 and the rectified-flow
            // extension noted in the diffusers cfg_pp PR).
            //
            //   x_{i+1} = x_i - σ_i · v_guided + σ_{i+1} · v_uncond
            //          = (x_0_estimate)        + (re-noise w/ uncond)
            cfg_plus_step(&x, &guidance, &noise_pred_full, *s_curr, *s_prev)?
        } else {
            (x + (&guidance * (*s_prev - *s_curr))?)?
        };

        // Inpainting: blend preserved regions back at current noise level
        if let Some(ctx) = inpaint_ctx {
            x = crate::img2img::apply_flow_match_inpaint(&x, ctx, *s_prev)?;
        }

        if step + 1 == total_steps {
            debug_tensor_stats("latents_final", &x);
        }

        progress.emit(ProgressEvent::DenoiseStep {
            step: step + 1,
            total: total_steps,
            elapsed: step_start.elapsed(),
        });
    }
    Ok(x)
}

/// Resolution-dependent shifting of timestep schedules.
///
/// From the SD3 tech report: <https://arxiv.org/pdf/2403.03206>
/// Following ComfyUI implementation.
pub fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    alpha * t / (1.0 + (alpha - 1.0) * t)
}

/// Apply classifier-free guidance: cfg * pred_cond - (cfg - 1) * pred_uncond.
fn apply_cfg(cfg_scale: f64, noise_pred: &Tensor) -> Result<Tensor> {
    Ok(((cfg_scale * noise_pred.narrow(0, 0, 1)?)?
        - ((cfg_scale - 1.0) * noise_pred.narrow(0, 1, 1)?)?)?)
}

/// CFG++ Euler step for rectified-flow models.
///
/// Replaces the standard Euler integration `x_{i+1} = x_i + (s_{i+1}-s_i)·v_guided`
/// with the manifold-projection form `x_{i+1} = x_i - s_i·v_guided + s_{i+1}·v_uncond`.
/// Equivalent to: estimate x_0 from the CFG-guided velocity, then re-noise back
/// to sigma=s_{i+1} using the *unconditional* velocity. The two forms collapse
/// to the same value when v_uncond = v_guided (i.e. cfg=1.0); the CFG++ form
/// only matters when guidance is active.
///
/// `noise_pred_full` carries the doubled `[cond, uncond]` transformer output;
/// the uncond row is at index 1.
fn cfg_plus_step(
    x: &Tensor,
    guidance: &Tensor,
    noise_pred_full: &Tensor,
    s_curr: f64,
    s_prev: f64,
) -> Result<Tensor> {
    let v_uncond = noise_pred_full.narrow(0, 1, 1)?;
    let x0_estimate = (x - (guidance * s_curr)?)?;
    Ok((x0_estimate + (v_uncond * s_prev)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_time_snr_shift_alpha_1() {
        // alpha=1 means no shift: output should equal input for all t
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let shifted = time_snr_shift(1.0, t);
            assert!(
                (shifted - t).abs() < 1e-12,
                "alpha=1 should be identity: time_snr_shift(1.0, {t}) = {shifted}, expected {t}"
            );
        }
    }

    #[test]
    fn test_time_snr_shift_boundaries() {
        // t=0 -> 0 and t=1 -> 1 for any positive alpha
        for alpha in [0.5, 1.0, 3.0, 10.0, 100.0] {
            let at_zero = time_snr_shift(alpha, 0.0);
            let at_one = time_snr_shift(alpha, 1.0);
            assert!(
                at_zero.abs() < 1e-12,
                "time_snr_shift({alpha}, 0.0) = {at_zero}, expected 0.0"
            );
            assert!(
                (at_one - 1.0).abs() < 1e-12,
                "time_snr_shift({alpha}, 1.0) = {at_one}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_time_snr_shift_midpoint() {
        // alpha=3, t=0.5: 3*0.5 / (1 + 2*0.5) = 1.5/2.0 = 0.75
        let result = time_snr_shift(3.0, 0.5);
        assert!(
            (result - 0.75).abs() < 1e-12,
            "time_snr_shift(3.0, 0.5) = {result}, expected 0.75"
        );
    }

    #[test]
    fn test_time_snr_shift_monotonic() {
        // For alpha=3, the function should be non-decreasing over [0, 1]
        let alpha = 3.0;
        let mut prev = time_snr_shift(alpha, 0.0);
        for i in 1..=100 {
            let t = i as f64 / 100.0;
            let curr = time_snr_shift(alpha, t);
            assert!(
                curr >= prev - 1e-12,
                "non-monotonic at t={t}: {curr} < {prev}"
            );
            prev = curr;
        }
    }

    #[test]
    fn test_apply_cfg_scale_1() {
        // cfg=1: 1*cond - 0*uncond = cond
        let dev = Device::Cpu;
        let cond = Tensor::new(&[[1.0f32, 2.0, 3.0]], &dev).unwrap();
        let uncond = Tensor::new(&[[10.0f32, 20.0, 30.0]], &dev).unwrap();
        let noise_pred = Tensor::cat(&[&cond, &uncond], 0).unwrap();

        let result = apply_cfg(1.0, &noise_pred).unwrap();
        let result_vec: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let cond_vec: Vec<f32> = cond.flatten_all().unwrap().to_vec1().unwrap();
        for (r, c) in result_vec.iter().zip(cond_vec.iter()) {
            assert!(
                (r - c).abs() < 1e-6,
                "cfg=1 should return cond: got {r}, expected {c}"
            );
        }
    }

    #[test]
    fn test_apply_cfg_scale_7_5() {
        // cfg=7.5: 7.5*cond - 6.5*uncond
        let dev = Device::Cpu;
        let cond = Tensor::new(&[[2.0f32, 4.0]], &dev).unwrap();
        let uncond = Tensor::new(&[[1.0f32, 1.0]], &dev).unwrap();
        let noise_pred = Tensor::cat(&[&cond, &uncond], 0).unwrap();

        let result = apply_cfg(7.5, &noise_pred).unwrap();
        let result_vec: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // expected: 7.5*[2,4] - 6.5*[1,1] = [15-6.5, 30-6.5] = [8.5, 23.5]
        let expected = [8.5f32, 23.5];
        for (r, e) in result_vec.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-4,
                "cfg=7.5 mismatch: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_sigma_schedule_endpoints() {
        // Reproduce the sigma schedule from euler_sample:
        // sigmas = (0..=steps).map(|s| s/steps).rev().map(time_snr_shift(alpha, .))
        let num_steps = 28;
        let alpha = 3.0;
        let sigmas: Vec<f64> = (0..=num_steps)
            .map(|s| s as f64 / num_steps as f64)
            .rev()
            .map(|t| time_snr_shift(alpha, t))
            .collect();

        assert_eq!(
            sigmas.len(),
            num_steps + 1,
            "schedule length should be steps+1"
        );
        assert!(
            (sigmas[0] - 1.0).abs() < 1e-12,
            "first sigma should be 1.0, got {}",
            sigmas[0]
        );
        assert!(
            sigmas[sigmas.len() - 1].abs() < 1e-12,
            "last sigma should be 0.0, got {}",
            sigmas[sigmas.len() - 1]
        );
    }

    // `euler_sample` gates the doubled `[cond, uncond]` forward on
    // `cfg_active(cfg_scale)`. These tests pin the predicate so a regression
    // to `cfg_scale > 1.0` (which would silently keep doubling the
    // transformer at cfg=1.0 — the SD3 Turbo case) is caught here.

    #[test]
    fn test_cfg_disabled_at_guidance_1_0() {
        assert!(!cfg_active(1.0));
    }

    #[test]
    fn test_cfg_disabled_just_below_1_0() {
        assert!(!cfg_active(1.0 - 1e-5));
    }

    #[test]
    fn test_cfg_enabled_at_guidance_1_5() {
        assert!(cfg_active(1.5));
    }

    #[test]
    fn test_cfg_enabled_at_guidance_7_5() {
        assert!(cfg_active(7.5));
    }

    // CFG++ tests pin the step math against analytic ground truth on toy
    // tensors. The transformer / SLG path stays GPU-only, but the integration
    // arithmetic runs on CPU so we can verify it without GPU resources.

    /// Build a `[cond, uncond]` noise tensor of shape (2, n) and the matching
    /// CFG-guided velocity for the given scale. Returns (noise_pred_full, guidance).
    fn toy_noise_pair(cond: &[f32], uncond: &[f32], cfg_scale: f64) -> (Tensor, Tensor) {
        assert_eq!(cond.len(), uncond.len(), "cond/uncond shapes must match");
        let n = cond.len();
        let dev = Device::Cpu;
        let cond_t = Tensor::from_slice(cond, (1, n), &dev).unwrap();
        let uncond_t = Tensor::from_slice(uncond, (1, n), &dev).unwrap();
        let noise_pred = Tensor::cat(&[&cond_t, &uncond_t], 0).unwrap();
        let guidance = apply_cfg(cfg_scale, &noise_pred).unwrap();
        (noise_pred, guidance)
    }

    #[test]
    fn cfg_plus_step_matches_manifold_formula() {
        // Verify x_{i+1} = x_i - σ_i·v_guided + σ_{i+1}·v_uncond against an
        // analytic computation with concrete numbers.
        let dev = Device::Cpu;
        let x = Tensor::new(&[[10.0f32, 20.0, 30.0]], &dev).unwrap();
        let (noise_pred, guidance) = toy_noise_pair(&[2.0, 4.0, 6.0], &[1.0, 1.0, 1.0], 7.5);

        let s_curr = 0.8;
        let s_prev = 0.6;
        let result = cfg_plus_step(&x, &guidance, &noise_pred, s_curr, s_prev).unwrap();
        let result_vec: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // guidance = 7.5*[2,4,6] - 6.5*[1,1,1] = [8.5, 23.5, 38.5]
        // expected[i] = x[i] - σ_curr·guidance[i] + σ_prev·v_uncond[i]
        //            = x[i] - 0.8·guidance[i] + 0.6·1.0
        let expected = [
            10.0 - 0.8 * 8.5 + 0.6 * 1.0,
            20.0 - 0.8 * 23.5 + 0.6 * 1.0,
            30.0 - 0.8 * 38.5 + 0.6 * 1.0,
        ];
        for (got, exp) in result_vec.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-4,
                "cfg++ step mismatch: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn cfg_plus_step_collapses_to_standard_when_cond_eq_uncond() {
        // When v_cond == v_uncond, guidance == v_uncond regardless of scale,
        // and the CFG++ step must equal the standard Euler step (no manifold
        // correction needed because guidance contributes nothing extra).
        let dev = Device::Cpu;
        let x = Tensor::new(&[[5.0f32, 7.0]], &dev).unwrap();
        let (noise_pred, guidance) = toy_noise_pair(&[3.0, 4.0], &[3.0, 4.0], 7.5);

        let s_curr = 0.5;
        let s_prev = 0.25;
        let cfg_pp = cfg_plus_step(&x, &guidance, &noise_pred, s_curr, s_prev).unwrap();
        let standard = (&x + (&guidance * (s_prev - s_curr)).unwrap()).unwrap();

        let cfg_pp_vec: Vec<f32> = cfg_pp.flatten_all().unwrap().to_vec1().unwrap();
        let std_vec: Vec<f32> = standard.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in cfg_pp_vec.iter().zip(std_vec.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "cfg++ ≠ standard when v_cond=v_uncond: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn cfg_plus_step_diverges_from_standard_under_high_cfg() {
        // Sanity: at cfg=7.5 with v_cond ≠ v_uncond, the two formulas must
        // produce *different* outputs. Catches accidental no-op
        // implementations (e.g. forgetting to swap in v_uncond).
        let dev = Device::Cpu;
        let x = Tensor::new(&[[0.0f32, 0.0, 0.0]], &dev).unwrap();
        let (noise_pred, guidance) = toy_noise_pair(&[2.0, 4.0, 6.0], &[1.0, 1.0, 1.0], 7.5);

        let s_curr = 0.9;
        let s_prev = 0.7;
        let cfg_pp = cfg_plus_step(&x, &guidance, &noise_pred, s_curr, s_prev).unwrap();
        let standard = (&x + (&guidance * (s_prev - s_curr)).unwrap()).unwrap();

        let cfg_pp_vec: Vec<f32> = cfg_pp.flatten_all().unwrap().to_vec1().unwrap();
        let std_vec: Vec<f32> = standard.flatten_all().unwrap().to_vec1().unwrap();
        // At least one element must differ noticeably — guards against any
        // future refactor that silently makes CFG++ a no-op.
        let max_diff = cfg_pp_vec
            .iter()
            .zip(std_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.1,
            "cfg++ should differ from standard at cfg=7.5 with v_cond≠v_uncond, max_diff={max_diff}"
        );
    }
}

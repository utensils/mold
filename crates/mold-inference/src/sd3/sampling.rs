//! SD3 Euler flow-matching sampler with SLG (Skip Layer Guidance) support.
//!
//! Port from the candle SD3 example. SD3 uses the same flow-matching framework as FLUX
//! but with a different timestep schedule (resolution-dependent SNR shifting).

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use std::time::Instant;

use super::transformer::SD3Transformer;
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
    time_shift: f64,
    height: usize,
    width: usize,
    slg_config: Option<&SkipLayerGuidanceConfig>,
    is_quantized: bool,
    seed: u64,
    progress: &ProgressReporter,
) -> Result<Tensor> {
    // SD3 uses the same 16-channel latent noise as FLUX
    // Quantized models (GGUF) dequantize to F32, so noise must also be F32
    let noise_dtype = if is_quantized { DType::F32 } else { DType::F16 };
    let latent_h = height / 16 * 2;
    let latent_w = width / 16 * 2;
    let mut x =
        crate::engine::seeded_randn(seed, &[1, 16, latent_h, latent_w], y.device(), noise_dtype)?;

    let sigmas: Vec<f64> = (0..=num_inference_steps)
        .map(|s| s as f64 / num_inference_steps as f64)
        .rev()
        .map(|t| time_snr_shift(time_shift, t))
        .collect();

    for (step, window) in sigmas.windows(2).enumerate() {
        let step_start = Instant::now();
        let (s_curr, s_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };

        let timestep = (*s_curr) * 1000.0;
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

        let mut guidance = apply_cfg(cfg_scale, &noise_pred)?;
        if step == 0 {
            debug_tensor_stats("guidance", &guidance);
        }

        if let Some(slg_config) = slg_config {
            if (num_inference_steps as f64) * slg_config.start < (step as f64)
                && (step as f64) < (num_inference_steps as f64) * slg_config.end
            {
                let slg_noise_pred = mmdit.forward(
                    &x,
                    &Tensor::full(timestep as f32, (1,), x.device())?.contiguous()?,
                    &y.i(..1)?,
                    &context.i(..1)?,
                    Some(&slg_config.layers),
                )?;
                guidance = (guidance
                    + (slg_config.scale * (noise_pred.i(..1)? - slg_noise_pred.i(..1))?)?)?;
            }
        }

        x = (x + (guidance * (*s_prev - *s_curr))?)?;
        if step + 1 == num_inference_steps {
            debug_tensor_stats("latents_final", &x);
        }

        progress.emit(ProgressEvent::DenoiseStep {
            step: step + 1,
            total: num_inference_steps,
            elapsed: step_start.elapsed(),
        });
    }
    Ok(x)
}

/// Resolution-dependent shifting of timestep schedules.
///
/// From the SD3 tech report: <https://arxiv.org/pdf/2403.03206>
/// Following ComfyUI implementation.
fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    alpha * t / (1.0 + (alpha - 1.0) * t)
}

/// Apply classifier-free guidance: cfg * pred_cond - (cfg - 1) * pred_uncond.
fn apply_cfg(cfg_scale: f64, noise_pred: &Tensor) -> Result<Tensor> {
    Ok(((cfg_scale * noise_pred.narrow(0, 0, 1)?)?
        - ((cfg_scale - 1.0) * noise_pred.narrow(0, 1, 1)?)?)?)
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
}

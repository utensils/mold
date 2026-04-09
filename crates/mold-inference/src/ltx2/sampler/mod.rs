#![allow(dead_code)]

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};

pub const DISTILLED_SIGMA_VALUES: &[f32] = &[
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];

pub const STAGE_2_DISTILLED_SIGMA_VALUES: &[f32] = &[0.909375, 0.725, 0.421875, 0.0];

pub fn to_velocity(sample: &Tensor, sigma: f64, denoised_sample: &Tensor) -> Result<Tensor> {
    if sigma == 0.0 {
        bail!("sigma cannot be zero when converting to velocity");
    }
    sample
        .to_dtype(DType::F32)?
        .broadcast_sub(&denoised_sample.to_dtype(DType::F32)?)?
        .affine(1.0 / sigma, 0.0)
        .map_err(Into::into)
}

pub fn to_denoised(sample: &Tensor, velocity: &Tensor, sigma: f64) -> Result<Tensor> {
    Ok(sample
        .to_dtype(DType::F32)?
        .broadcast_sub(&(velocity.to_dtype(DType::F32)? * sigma)?)?)
}

pub fn euler_step(
    sample: &Tensor,
    denoised_sample: &Tensor,
    sigmas: &[f32],
    step_index: usize,
) -> Result<Tensor> {
    if step_index + 1 >= sigmas.len() {
        bail!("euler step requires a sigma and next sigma");
    }
    let sigma = sigmas[step_index] as f64;
    let sigma_next = sigmas[step_index + 1] as f64;
    let dt = sigma_next - sigma;
    let velocity = to_velocity(sample, sigma, denoised_sample)?;
    Ok(sample
        .to_dtype(DType::F32)?
        .broadcast_add(&(velocity * dt)?)?)
}

pub fn apply_denoise_mask(
    denoised: &Tensor,
    denoise_mask: Option<&Tensor>,
    clean_latent: Option<&Tensor>,
) -> Result<Tensor> {
    match (denoise_mask, clean_latent) {
        (Some(mask), Some(clean)) => denoised
            .broadcast_mul(mask)?
            .broadcast_add(&clean.broadcast_mul(&mask.affine(-1.0, 1.0)?)?)
            .map_err(Into::into),
        _ => Ok(denoised.clone()),
    }
}

pub fn euler_denoising_loop<F>(
    initial_sample: &Tensor,
    sigmas: &[f32],
    denoise_mask: Option<&Tensor>,
    clean_latent: Option<&Tensor>,
    mut denoiser: F,
) -> Result<Tensor>
where
    F: FnMut(&Tensor, usize) -> Result<Tensor>,
{
    if sigmas.len() < 2 {
        bail!("euler denoising loop requires at least two sigma values");
    }

    let mut sample = initial_sample.clone();
    for step_index in 0..(sigmas.len() - 1) {
        let denoised = denoiser(&sample, step_index)?;
        let denoised = apply_denoise_mask(&denoised, denoise_mask, clean_latent)?;
        sample = euler_step(&sample, &denoised, sigmas, step_index)?;
    }
    Ok(sample)
}

pub fn phi(j: usize, neg_h: f64) -> f64 {
    if neg_h.abs() < 1e-10 {
        return 1.0 / factorial(j) as f64;
    }
    let remainder = (0..j)
        .map(|k| neg_h.powi(k as i32) / factorial(k) as f64)
        .sum::<f64>();
    (neg_h.exp() - remainder) / neg_h.powi(j as i32)
}

pub fn res2s_coefficients(h: f64, c2: f64) -> (f64, f64, f64) {
    let a21 = c2 * phi(1, -h * c2);
    let b2 = phi(2, -h) / c2;
    let b1 = phi(1, -h) - b2;
    (a21, b1, b2)
}

pub fn res2s_sde_coefficients(sigma_next: f64, eta: f64) -> (f64, f64, f64) {
    let sigma_up = (sigma_next * eta).min(sigma_next * 0.9999);
    let sigma_signal = 1.0 - sigma_next;
    let sigma_residual = (sigma_next.powi(2) - sigma_up.powi(2)).max(0.0).sqrt();
    let alpha_ratio = sigma_signal + sigma_residual;
    let sigma_down = if alpha_ratio.abs() < f64::EPSILON {
        sigma_next
    } else {
        sigma_residual / alpha_ratio
    };
    (alpha_ratio, sigma_down, sigma_up)
}

pub fn res2s_step(
    sample: &Tensor,
    denoised_sample: &Tensor,
    sigma: f64,
    sigma_next: f64,
    noise: &Tensor,
    eta: f64,
) -> Result<Tensor> {
    let (alpha_ratio, sigma_down, sigma_up) = res2s_sde_coefficients(sigma_next, eta);
    if sigma_up == 0.0 || sigma_next == 0.0 {
        return Ok(denoised_sample.clone());
    }

    let eps_next = sample
        .to_dtype(DType::F32)?
        .broadcast_sub(&denoised_sample.to_dtype(DType::F32)?)?
        .affine(1.0 / (sigma - sigma_next), 0.0)?;
    let denoised_next = sample
        .to_dtype(DType::F32)?
        .broadcast_sub(&eps_next.affine(sigma, 0.0)?)?;
    let drift = denoised_next.broadcast_add(&eps_next.affine(sigma_down, 0.0)?)?;
    let drift = drift.affine(alpha_ratio, 0.0)?;
    let noise_term = noise.to_dtype(DType::F32)?.affine(sigma_up, 0.0)?;
    drift.broadcast_add(&noise_term).map_err(Into::into)
}

pub fn res2s_denoising_loop<F>(
    initial_sample: &Tensor,
    sigmas: &[f32],
    denoise_mask: Option<&Tensor>,
    clean_latent: Option<&Tensor>,
    noise: &Tensor,
    mut denoiser: F,
) -> Result<Tensor>
where
    F: FnMut(&Tensor, usize) -> Result<Tensor>,
{
    if sigmas.len() < 2 {
        bail!("res2s denoising loop requires at least two sigma values");
    }

    let mut sample = initial_sample.clone();
    for step_index in 0..(sigmas.len() - 1) {
        let denoised = denoiser(&sample, step_index)?;
        let denoised = apply_denoise_mask(&denoised, denoise_mask, clean_latent)?;
        sample = res2s_step(
            &sample,
            &denoised,
            sigmas[step_index] as f64,
            sigmas[step_index + 1] as f64,
            noise,
            0.5,
        )?;
    }
    Ok(sample)
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{
        euler_denoising_loop, euler_step, phi, res2s_coefficients, res2s_denoising_loop,
        res2s_sde_coefficients, DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES,
    };

    #[test]
    fn distilled_sigma_schedules_match_published_values() {
        assert_eq!(DISTILLED_SIGMA_VALUES.len(), 9);
        assert_eq!(DISTILLED_SIGMA_VALUES[0], 1.0);
        assert_eq!(DISTILLED_SIGMA_VALUES[7], 0.421875);
        assert_eq!(DISTILLED_SIGMA_VALUES[8], 0.0);

        assert_eq!(
            STAGE_2_DISTILLED_SIGMA_VALUES,
            &[0.909375, 0.725, 0.421875, 0.0]
        );
    }

    #[test]
    fn euler_step_advances_sample_by_velocity_dt() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[2f32], &device).unwrap();
        let denoised = Tensor::new(&[1f32], &device).unwrap();
        let out = euler_step(&sample, &denoised, &[1.0, 0.5], 0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, vec![1.5]);
    }

    #[test]
    fn euler_denoising_loop_supports_skip_step_like_identity_denoiser() {
        let device = Device::Cpu;
        let initial = Tensor::new(&[2f32], &device).unwrap();
        let out = euler_denoising_loop(&initial, &[1.0, 0.5, 0.0], None, None, |sample, _| {
            Ok(sample.clone())
        })
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        assert_eq!(out, vec![2.0]);
    }

    #[test]
    fn phi_matches_taylor_limit_near_zero() {
        let value = phi(2, -1e-12);
        assert!((value - 0.5).abs() < 1e-6);
    }

    #[test]
    fn res2s_coefficients_are_finite_for_midpoint_scheme() {
        let (a21, b1, b2) = res2s_coefficients(0.5, 0.5);
        assert!(a21.is_finite());
        assert!(b1.is_finite());
        assert!(b2.is_finite());
    }

    #[test]
    fn res2s_sde_coefficients_are_bounded() {
        let (alpha_ratio, sigma_down, sigma_up) = res2s_sde_coefficients(0.5, 0.5);
        assert!(alpha_ratio.is_finite());
        assert!(sigma_down >= 0.0);
        assert!(sigma_up >= 0.0);
        assert!(sigma_up < 0.5);
    }

    #[test]
    fn res2s_loop_returns_denoised_output_at_terminal_sigma() {
        let device = Device::Cpu;
        let initial = Tensor::new(&[2f32], &device).unwrap();
        let noise = Tensor::zeros((1,), candle_core::DType::F32, &device).unwrap();
        let out = res2s_denoising_loop(&initial, &[0.5, 0.0], None, None, &noise, |_sample, _| {
            Ok(Tensor::new(&[1f32], &device).unwrap())
        })
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        assert_eq!(out, vec![1.0]);
    }
}

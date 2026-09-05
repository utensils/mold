use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};

use crate::ltx2::execution::SamplerMode;

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

pub(crate) fn sampler_step(
    sampler_mode: SamplerMode,
    sample: &Tensor,
    denoised_sample: &Tensor,
    sigmas: &[f32],
    step_index: usize,
    noise: Option<&Tensor>,
    missing_noise_context: &'static str,
) -> Result<Tensor> {
    match sampler_mode {
        SamplerMode::Euler => euler_step(sample, denoised_sample, sigmas, step_index),
        SamplerMode::Res2S => {
            if step_index + 1 >= sigmas.len() {
                bail!("Res2S sampler step requires a sigma and next sigma");
            }
            res2s_step(
                sample,
                denoised_sample,
                sigmas[step_index] as f64,
                sigmas[step_index + 1] as f64,
                noise.context(missing_noise_context)?,
                0.5,
            )
        }
    }
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

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use crate::ltx2::execution::SamplerMode;

    use super::{euler_step, res2s_sde_coefficients, sampler_step};

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
    fn sampler_step_matches_euler_step_for_schedule_index() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[2f32], &device).unwrap();
        let denoised = Tensor::new(&[1f32], &device).unwrap();

        let direct = euler_step(&sample, &denoised, &[1.0, 0.5], 0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let via_helper = sampler_step(
            SamplerMode::Euler,
            &sample,
            &denoised,
            &[1.0, 0.5],
            0,
            None,
            "test sampler noise missing",
        )
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        assert_eq!(via_helper, direct);
    }

    #[test]
    fn sampler_step_matches_res2s_step_for_schedule_index() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[2f32], &device).unwrap();
        let denoised = Tensor::new(&[1f32], &device).unwrap();
        let noise = Tensor::zeros((1,), candle_core::DType::F32, &device).unwrap();

        let direct = super::res2s_step(&sample, &denoised, 0.5, 0.0, &noise, 0.5)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let via_helper = sampler_step(
            SamplerMode::Res2S,
            &sample,
            &denoised,
            &[0.5, 0.0],
            0,
            Some(&noise),
            "test sampler noise missing",
        )
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        assert_eq!(via_helper, direct);
    }

    #[test]
    fn sampler_step_requires_noise_for_res2s() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[2f32], &device).unwrap();
        let denoised = Tensor::new(&[1f32], &device).unwrap();

        let err = sampler_step(
            SamplerMode::Res2S,
            &sample,
            &denoised,
            &[0.5, 0.0],
            0,
            None,
            "test sampler noise missing",
        )
        .unwrap_err();

        assert!(err.to_string().contains("test sampler noise missing"));
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
    fn res2s_step_returns_denoised_output_at_terminal_sigma() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[2f32], &device).unwrap();
        let denoised = Tensor::new(&[1f32], &device).unwrap();
        let noise = Tensor::zeros((1,), candle_core::DType::F32, &device).unwrap();
        let out = super::res2s_step(&sample, &denoised, 0.5, 0.0, &noise, 0.5)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, vec![1.0]);
    }
}

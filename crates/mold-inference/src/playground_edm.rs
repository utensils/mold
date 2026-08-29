//! Playground v2.5's continuous EDM DPM++ 2M sampler.
//!
//! This is a direct, deliberately narrow port of Diffusers'
//! `EDMDPMSolverMultistepScheduler` configuration shipped by
//! `playgroundai/playground-v2.5-1024px-aesthetic`: Karras sigmas 80→0.002,
//! rho 7, sigma_data 0.5, epsilon preconditioning, order 2 / midpoint, and a
//! terminal zero sigma. Candle's stable-diffusion scheduler trait exposes only
//! integer timesteps, while this model's UNet requires `0.25 * ln(sigma)`;
//! keeping the sampler here avoids rounding a continuous training contract.

use candle_core::{Result, Tensor};

const SIGMA_MIN: f64 = 0.002;
const SIGMA_MAX: f64 = 80.0;
const SIGMA_DATA: f64 = 0.5;
const RHO: f64 = 7.0;

pub(crate) struct PlaygroundEdmScheduler {
    sigmas: Vec<f64>,
    timesteps: Vec<f64>,
    previous_denoised: Option<Tensor>,
    step_index: usize,
}

impl PlaygroundEdmScheduler {
    pub(crate) fn new(inference_steps: usize, begin_index: usize) -> Result<Self> {
        if inference_steps == 0 {
            return Err(candle_core::Error::Msg(
                "Playground EDM requires at least one inference step".to_string(),
            ));
        }
        if begin_index >= inference_steps {
            return Err(candle_core::Error::Msg(format!(
                "Playground EDM begin index {begin_index} is outside {inference_steps} steps"
            )));
        }

        let min_inv_rho = SIGMA_MIN.powf(1.0 / RHO);
        let max_inv_rho = SIGMA_MAX.powf(1.0 / RHO);
        let denominator = inference_steps.saturating_sub(1).max(1) as f64;
        let mut sigmas = (0..inference_steps)
            .map(|i| {
                let ramp = i as f64 / denominator;
                (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).powf(RHO)
            })
            .collect::<Vec<_>>();
        let timesteps = sigmas.iter().map(|sigma| 0.25 * sigma.ln()).collect();
        sigmas.push(0.0);

        Ok(Self {
            sigmas,
            timesteps,
            previous_denoised: None,
            step_index: begin_index,
        })
    }

    pub(crate) fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    pub(crate) fn init_noise_sigma() -> f64 {
        (SIGMA_MAX * SIGMA_MAX + 1.0).sqrt()
    }

    pub(crate) fn scale_model_input(&self, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        sample / (sigma * sigma + SIGMA_DATA * SIGMA_DATA).sqrt()
    }

    pub(crate) fn add_noise_at(
        &self,
        original: &Tensor,
        noise: &Tensor,
        index: usize,
    ) -> Result<Tensor> {
        original + (noise * self.sigmas[index])?
    }

    /// Re-noise preserved inpaint pixels at the scheduler's current sigma.
    /// Call after `step`: `step_index` has advanced to the destination sigma,
    /// including the appended terminal zero.
    pub(crate) fn add_noise_at_current_sigma(
        &self,
        original: &Tensor,
        noise: &Tensor,
    ) -> Result<Tensor> {
        original + (noise * self.sigmas[self.step_index])?
    }

    pub(crate) fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma_s0 = self.sigmas[self.step_index];
        let sigma_t = self.sigmas[self.step_index + 1];

        // Diffusers `precondition_outputs(..., prediction_type="epsilon")`.
        // Playground's UNet output is not ordinary DDPM epsilon despite that
        // config spelling: EDM combines it with the current sample first.
        let denominator = sigma_s0 * sigma_s0 + SIGMA_DATA * SIGMA_DATA;
        let c_skip = SIGMA_DATA * SIGMA_DATA / denominator;
        let c_out = sigma_s0 * SIGMA_DATA / denominator.sqrt();
        let denoised = ((sample * c_skip)? + (model_output * c_out)?)?;

        let terminal = self.step_index + 1 == self.timesteps.len();
        let prev_sample = match (&self.previous_denoised, terminal) {
            (None, _) | (_, true) => {
                // DPM-Solver++ first order. At terminal sigma=0 this reduces
                // exactly to the current denoised x0 prediction.
                if sigma_t == 0.0 {
                    denoised.clone()
                } else {
                    let h = sigma_s0.ln() - sigma_t.ln();
                    let coeff = (-h).exp_m1();
                    ((sample * (sigma_t / sigma_s0))? - (&denoised * coeff)?)?
                }
            }
            (Some(previous), false) => {
                // DPM++ 2M midpoint update, matching Diffusers lines 566–592.
                let sigma_s1 = self.sigmas[self.step_index - 1];
                let h = sigma_s0.ln() - sigma_t.ln();
                let h_0 = sigma_s1.ln() - sigma_s0.ln();
                let r0 = h_0 / h;
                let d1 = ((&denoised - previous)? / r0)?;
                let coeff = (-h).exp_m1();
                let first = (sample * (sigma_t / sigma_s0))?;
                let first = (first - (&denoised * coeff)?)?;
                (first - (d1 * (0.5 * coeff))?)?
            }
        };

        self.previous_denoised = Some(denoised);
        self.step_index += 1;
        Ok(prev_sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn upstream_schedule_endpoints_and_noise_scale_are_exact() {
        let scheduler = PlaygroundEdmScheduler::new(50, 0).unwrap();
        assert!((scheduler.sigmas[0] - 80.0).abs() < 1e-12);
        assert!((scheduler.sigmas[49] - 0.002).abs() < 1e-12);
        assert_eq!(scheduler.sigmas[50], 0.0);
        assert!((scheduler.timesteps[0] - 0.25 * 80.0f64.ln()).abs() < 1e-12);
        assert!((PlaygroundEdmScheduler::init_noise_sigma() - 80.00624975587844).abs() < 1e-12);
    }

    #[test]
    fn terminal_step_returns_edm_preconditioned_x0() {
        let device = Device::Cpu;
        let mut scheduler = PlaygroundEdmScheduler::new(1, 0).unwrap();
        let sample = Tensor::new(&[2.0f32], &device).unwrap();
        let output = Tensor::new(&[3.0f32], &device).unwrap();
        let actual = scheduler
            .step(&output, &sample)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let denominator = 80.0f64.powi(2) + 0.5f64.powi(2);
        let expected = 2.0 * (0.25 / denominator) + 3.0 * (40.0 / denominator.sqrt());
        assert!((actual as f64 - expected).abs() < 1e-6);
    }

    #[test]
    fn inpaint_blend_uses_destination_sigma_and_clean_terminal_latents() {
        let device = Device::Cpu;
        let original = Tensor::new(&[2.0f32], &device).unwrap();
        let noise = Tensor::new(&[3.0f32], &device).unwrap();
        let model_output = Tensor::zeros(1, candle_core::DType::F32, &device).unwrap();
        let mut sample = Tensor::zeros(1, candle_core::DType::F32, &device).unwrap();
        let mut scheduler = PlaygroundEdmScheduler::new(2, 0).unwrap();

        sample = scheduler.step(&model_output, &sample).unwrap();
        let intermediate = scheduler
            .add_noise_at_current_sigma(&original, &noise)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((intermediate - (2.0 + 3.0 * SIGMA_MIN as f32)).abs() < 1e-6);

        scheduler.step(&model_output, &sample).unwrap();
        let terminal = scheduler
            .add_noise_at_current_sigma(&original, &noise)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert_eq!(terminal, 2.0);
    }
}

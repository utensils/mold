//! Flow-matching Euler discrete scheduler for Qwen-Image.
//!
//! Matches the official Qwen diffusers scheduler configuration:
//! - `FlowMatchEulerDiscreteScheduler`
//! - dynamic exponential shifting based on image sequence length
//! - `base_shift=0.5`, `max_shift=0.9`
//! - `base_image_seq_len=256`, `max_image_seq_len=8192`
//! - `shift_terminal=0.02`

use candle_core::{DType, Result, Tensor};

pub(crate) const NUM_TRAIN_TIMESTEPS: usize = 1000;
pub(crate) const BASE_IMAGE_SEQ_LEN: usize = 256;
pub(crate) const MAX_IMAGE_SEQ_LEN: usize = 8192;
pub(crate) const BASE_SHIFT: f64 = 0.5;
pub(crate) const MAX_SHIFT: f64 = 0.9;
pub(crate) const SHIFT_TERMINAL: f64 = 0.02;

fn calculate_shift(image_seq_len: usize) -> f64 {
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN) as f64;
    let b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN as f64;
    image_seq_len as f64 * m + b
}

fn time_shift_exponential(mu: f64, sigma: f64, t: f64) -> f64 {
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

fn stretch_shift_to_terminal(sigmas: &mut [f64]) {
    let one_minus_terminal = 1.0 - SHIFT_TERMINAL;
    let one_minus_z = 1.0 - sigmas[sigmas.len() - 1];
    let scale_factor = one_minus_z / one_minus_terminal;
    for sigma in sigmas.iter_mut() {
        *sigma = 1.0 - ((1.0 - *sigma) / scale_factor);
    }
}

/// Sequence length after Qwen patchification.
pub(crate) fn image_seq_len(latent_h: usize, latent_w: usize, patch_size: usize) -> usize {
    (latent_h / patch_size) * (latent_w / patch_size)
}

/// Flow-matching Euler scheduler matching official Qwen diffusers behavior.
#[derive(Debug, Clone)]
pub(crate) struct QwenImageScheduler {
    pub sigmas: Vec<f64>,
    step_index: usize,
}

impl QwenImageScheduler {
    pub fn new(num_inference_steps: usize, image_seq_len: usize) -> Self {
        // diffusers:
        // sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        let mut sigmas: Vec<f64> = if num_inference_steps == 1 {
            vec![1.0]
        } else {
            let start = 1.0;
            let end = 1.0 / num_inference_steps as f64;
            let step = (end - start) / (num_inference_steps - 1) as f64;
            (0..num_inference_steps)
                .map(|i| start + step * i as f64)
                .collect()
        };

        let mu = calculate_shift(image_seq_len);
        for sigma in &mut sigmas {
            *sigma = time_shift_exponential(mu, 1.0, *sigma);
        }
        stretch_shift_to_terminal(&mut sigmas);
        sigmas.push(0.0);

        Self {
            sigmas,
            step_index: 0,
        }
    }

    pub fn current_timestep(&self) -> f64 {
        self.sigmas[self.step_index] * NUM_TRAIN_TIMESTEPS as f64
    }

    pub fn initial_sigma(&self) -> f64 {
        self.sigmas[0]
    }

    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;

        let out_dtype = model_output.dtype();
        let sample = sample.to_dtype(DType::F32)?;
        let model_output = model_output.to_dtype(DType::F32)?;
        let prev_sample = (sample + (model_output * dt)?)?;

        self.step_index += 1;
        prev_sample.to_dtype(out_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_shift_matches_qwen_defaults() {
        let mu = calculate_shift(4096);
        assert!((mu - 0.6935483870967742).abs() < 1e-9);
    }

    #[test]
    fn exponential_time_shift_is_monotonic() {
        let low = time_shift_exponential(0.69, 1.0, 0.1);
        let high = time_shift_exponential(0.69, 1.0, 0.9);
        assert!(low < high);
    }

    #[test]
    fn image_seq_len_matches_patchified_latents() {
        assert_eq!(image_seq_len(128, 128, 2), 4096);
        assert_eq!(image_seq_len(116, 208, 2), 6032);
    }

    #[test]
    fn scheduler_creates_descending_sigmas_and_terminal_zero() {
        let scheduler = QwenImageScheduler::new(50, 4096);
        assert_eq!(scheduler.sigmas.len(), 51);
        assert_eq!(*scheduler.sigmas.last().unwrap(), 0.0);
        for pair in scheduler.sigmas.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "sigmas should be monotonically decreasing"
            );
        }
    }

    #[test]
    fn current_timestep_is_sigma_times_train_steps() {
        let scheduler = QwenImageScheduler::new(50, 4096);
        assert!(
            (scheduler.current_timestep() - scheduler.sigmas[0] * NUM_TRAIN_TIMESTEPS as f64).abs()
                < 1e-10
        );
    }

    #[test]
    fn stretched_schedule_ends_at_shift_terminal_before_zero_append() {
        let mut sigmas: Vec<f64> = vec![1.0, 0.5];
        stretch_shift_to_terminal(&mut sigmas);
        assert!((sigmas[1] - SHIFT_TERMINAL).abs() < 1e-10);
    }
}

//! Flow-matching Euler discrete scheduler for Qwen-Image-2512.
//!
//! Implements `FlowMatchEulerDiscreteScheduler` with dynamic exponential time shifting,
//! matching the HuggingFace diffusers scheduler configuration for Qwen-Image.
//!
//! Key parameters from `scheduler_config.json`:
//! - base_shift=0.5, max_shift=0.9
//! - shift_terminal=0.02
//! - exponential time shift type
//! - base_image_seq_len=256, num_train_timesteps=1000

use candle_core::{Result, Tensor};

/// Scheduler shift constants for Qwen-Image (from `scheduler_config.json`).
pub(crate) const BASE_IMAGE_SEQ_LEN: usize = 256;
pub(crate) const MAX_IMAGE_SEQ_LEN: usize = 8192;
pub(crate) const BASE_SHIFT: f64 = 0.5;
pub(crate) const MAX_SHIFT: f64 = 0.9;
pub(crate) const SHIFT_TERMINAL: f64 = 0.02;
pub(crate) const NUM_TRAIN_TIMESTEPS: usize = 1000;

/// Calculate the dynamic shift parameter `mu` based on image sequence length.
///
/// Linear interpolation between base_shift and max_shift based on image_seq_len
/// position between base_seq_len and max_seq_len. This is then used as the
/// `mu` parameter for exponential time shifting.
pub(crate) fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

/// Apply exponential time shift: sigma' = exp(mu) / (exp(mu) + (1/sigma - 1))
///
/// This warps the time schedule so more steps are spent at higher noise levels,
/// improving quality for flow-matching models.
fn time_shift(mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    if sigma >= 1.0 {
        return 1.0;
    }
    let e_mu = mu.exp();
    e_mu / (e_mu + (1.0 / sigma - 1.0))
}

/// Flow-matching Euler discrete scheduler for Qwen-Image.
///
/// Uses dynamic exponential time shifting based on image resolution.
#[derive(Debug, Clone)]
pub(crate) struct QwenImageScheduler {
    /// Sigma values for each step (from 1.0 to 0.0, with terminal sigma).
    pub sigmas: Vec<f64>,
    /// Current step index.
    step_index: usize,
}

impl QwenImageScheduler {
    /// Create a new scheduler with the given number of inference steps and shift parameter.
    ///
    /// `mu` is the dynamic shift parameter from `calculate_shift()`.
    pub fn new(num_inference_steps: usize, mu: f64) -> Self {
        // Generate linearly spaced sigma values from 1.0 to shift_terminal
        let mut sigmas: Vec<f64> = (0..num_inference_steps)
            .map(|i| {
                let t = i as f64 / num_inference_steps as f64;
                1.0 * (1.0 - t) + SHIFT_TERMINAL * t
            })
            .collect();

        // Apply exponential time shift
        sigmas = sigmas.iter().map(|&s| time_shift(mu, s)).collect();

        // Add terminal sigma = 0.0
        sigmas.push(0.0);

        Self {
            sigmas,
            step_index: 0,
        }
    }

    /// Get the current timestep normalized for model input.
    ///
    /// Returns `sigma * num_train_timesteps` which the transformer expects.
    pub fn current_timestep(&self) -> f64 {
        self.sigmas[self.step_index] * NUM_TRAIN_TIMESTEPS as f64
    }

    /// Euler step: advance the latent sample from x_t to x_{t-1}.
    ///
    /// Uses the flow-matching Euler update:
    ///   x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * model_output
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;

        // prev_sample = sample + dt * model_output
        let prev_sample = (sample + (model_output * dt)?)?;

        self.step_index += 1;
        Ok(prev_sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_shift_at_base_seq_len() {
        let mu = calculate_shift(
            BASE_IMAGE_SEQ_LEN,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        assert!((mu - BASE_SHIFT).abs() < 1e-10);
    }

    #[test]
    fn calculate_shift_at_max_seq_len() {
        let mu = calculate_shift(
            MAX_IMAGE_SEQ_LEN,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        assert!((mu - MAX_SHIFT).abs() < 1e-10);
    }

    #[test]
    fn calculate_shift_1024x1024() {
        // 1024x1024 -> latent 128x128 -> patches 64x64 = 4096 seq_len
        let mu = calculate_shift(
            4096,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        assert!(mu > BASE_SHIFT);
        assert!(mu < MAX_SHIFT);
    }

    #[test]
    fn time_shift_boundaries() {
        assert_eq!(time_shift(0.7, 0.0), 0.0);
        assert_eq!(time_shift(0.7, 1.0), 1.0);
    }

    #[test]
    fn time_shift_midpoint() {
        let mu = 0.7;
        let shifted = time_shift(mu, 0.5);
        // With positive mu, shift should push midpoint higher
        assert!(shifted > 0.5);
        assert!(shifted < 1.0);
    }

    #[test]
    fn scheduler_creates_correct_sigmas() {
        let scheduler = QwenImageScheduler::new(20, 0.7);
        // num_inference_steps + 1 sigmas (including terminal 0.0)
        assert_eq!(scheduler.sigmas.len(), 21);
        // First sigma should be close to 1.0 (after time shift)
        assert!(scheduler.sigmas[0] > 0.5);
        // Last sigma should be 0.0
        assert_eq!(*scheduler.sigmas.last().unwrap(), 0.0);
        // Monotonically decreasing
        for w in scheduler.sigmas.windows(2) {
            assert!(w[0] >= w[1], "sigmas should be monotonically decreasing");
        }
    }
}

//! Flow-matching Euler discrete scheduler for Qwen-Image-2512.
//!
//! Implements ComfyUI-style scheduling with SNR time shift and simple sigma spacing.
//! The ComfyUI reference workflow for Qwen-Image uses:
//! - `ModelSamplingAuraFlow` with shift=3.1
//! - `simple` scheduler
//! - `euler` sampler
//! - CONST noise scaling: initial_noise = sigma[0] * randn()

use candle_core::{Result, Tensor};

/// Default shift for Qwen-Image, matching ComfyUI's `ModelSamplingAuraFlow` node.
pub(crate) const DEFAULT_SHIFT: f64 = 3.1;

/// Number of precomputed sigma values (matches ComfyUI's default timesteps=1000).
const NUM_SIGMAS: usize = 1000;

/// SNR-based time shift used by ComfyUI's `ModelSamplingDiscreteFlow`.
///
/// `shifted = alpha * t / (1 + (alpha - 1) * t)`
///
/// This is different from diffusers' exponential shift. At alpha=3.1 it allocates
/// more denoising budget to higher noise levels, which empirically reduces grain.
fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    if alpha == 1.0 {
        return t;
    }
    alpha * t / (1.0 + (alpha - 1.0) * t)
}

/// Flow-matching Euler discrete scheduler for Qwen-Image.
///
/// Uses ComfyUI-style SNR time shift with simple sigma spacing.
#[derive(Debug, Clone)]
pub(crate) struct QwenImageScheduler {
    /// Sigma values for each step (from ~1.0 down to 0.0).
    pub sigmas: Vec<f64>,
    /// Current step index.
    step_index: usize,
}

impl QwenImageScheduler {
    /// Create a new scheduler with the given number of inference steps.
    ///
    /// Matches ComfyUI's `simple` scheduler with `ModelSamplingAuraFlow(shift)`:
    /// 1. Precompute 1000 sigmas: `time_snr_shift(shift, i/1000)` for i=1..1000
    /// 2. Sample backwards at equal intervals for num_inference_steps
    /// 3. Append terminal sigma = 0.0
    pub fn new(num_inference_steps: usize, shift: f64) -> Self {
        // Step 1: Precompute 1000 sigmas (matching ComfyUI's set_parameters)
        let all_sigmas: Vec<f64> = (1..=NUM_SIGMAS)
            .map(|i| time_snr_shift(shift, i as f64 / NUM_SIGMAS as f64))
            .collect();

        // Step 2: Simple scheduler — sample backwards at equal intervals
        let step_size = NUM_SIGMAS as f64 / num_inference_steps as f64;
        let mut sigmas: Vec<f64> = (0..num_inference_steps)
            .map(|x| {
                let idx = NUM_SIGMAS - 1 - (x as f64 * step_size) as usize;
                all_sigmas[idx.min(NUM_SIGMAS - 1)]
            })
            .collect();

        // Step 3: Append terminal sigma
        sigmas.push(0.0);

        Self {
            sigmas,
            step_index: 0,
        }
    }

    /// Get the current timestep for model input.
    ///
    /// ComfyUI with `ModelSamplingAuraFlow(multiplier=1.0)` feeds raw sigma
    /// as the timestep, not sigma*1000.
    pub fn current_timestep(&self) -> f64 {
        self.sigmas[self.step_index]
    }

    /// Get sigma[0] for initial noise scaling.
    ///
    /// ComfyUI's CONST noise scaling: `initial_noise = sigma[0] * randn()`.
    /// For txt2img with denoise=1.0, the latent starts as zeros, so the
    /// initial noisy sample is just `sigma[0] * noise`.
    pub fn initial_sigma(&self) -> f64 {
        self.sigmas[0]
    }

    /// Euler step: advance the latent sample from x_t to x_{t-1}.
    ///
    /// Uses the flow-matching Euler update:
    ///   x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * model_output
    ///
    /// Upcasts to F32 for the step to prevent BF16 precision loss.
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;

        let out_dtype = model_output.dtype();
        let sample = sample.to_dtype(candle_core::DType::F32)?;
        let model_output = model_output.to_dtype(candle_core::DType::F32)?;

        let prev_sample = (sample + (model_output * dt)?)?;

        self.step_index += 1;
        prev_sample.to_dtype(out_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_snr_shift_identity_at_alpha_1() {
        assert!((time_snr_shift(1.0, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn time_snr_shift_boundaries() {
        // At t=0, shift should be 0 regardless of alpha
        assert!((time_snr_shift(3.1, 0.0)).abs() < 1e-10);
        // At t=1, shift should be 1.0 regardless of alpha
        assert!((time_snr_shift(3.1, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn time_snr_shift_pushes_midpoint_up() {
        // With alpha > 1, the midpoint should be pushed higher
        let shifted = time_snr_shift(3.1, 0.5);
        assert!(shifted > 0.5);
        assert!(shifted < 1.0);
        // Expected: 3.1 * 0.5 / (1 + 2.1 * 0.5) = 1.55 / 2.05 ≈ 0.7561
        assert!((shifted - 0.7561).abs() < 0.001);
    }

    #[test]
    fn scheduler_creates_correct_sigmas() {
        let scheduler = QwenImageScheduler::new(50, DEFAULT_SHIFT);
        // num_inference_steps + 1 sigmas (including terminal 0.0)
        assert_eq!(scheduler.sigmas.len(), 51);
        // First sigma should be close to 1.0
        assert!(scheduler.sigmas[0] > 0.9, "sigma[0]={}", scheduler.sigmas[0]);
        // Last sigma should be 0.0
        assert_eq!(*scheduler.sigmas.last().unwrap(), 0.0);
        // Monotonically decreasing
        for w in scheduler.sigmas.windows(2) {
            assert!(w[0] >= w[1], "sigmas should be monotonically decreasing");
        }
    }

    #[test]
    fn current_timestep_is_raw_sigma() {
        let scheduler = QwenImageScheduler::new(50, DEFAULT_SHIFT);
        // ComfyUI feeds raw sigma, not sigma*1000
        assert!(
            (scheduler.current_timestep() - scheduler.sigmas[0]).abs() < 1e-10,
            "current_timestep should be raw sigma"
        );
        assert!(
            scheduler.current_timestep() <= 1.0,
            "raw sigma should be <= 1.0"
        );
    }

    #[test]
    fn initial_sigma_matches_first_sigma() {
        let scheduler = QwenImageScheduler::new(50, DEFAULT_SHIFT);
        assert!((scheduler.initial_sigma() - scheduler.sigmas[0]).abs() < 1e-10);
    }

    #[test]
    fn scheduler_20_steps() {
        let scheduler = QwenImageScheduler::new(20, DEFAULT_SHIFT);
        assert_eq!(scheduler.sigmas.len(), 21);
        assert!(scheduler.sigmas[0] > 0.9);
        assert_eq!(*scheduler.sigmas.last().unwrap(), 0.0);
    }
}

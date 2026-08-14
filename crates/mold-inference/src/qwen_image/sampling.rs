//! Flow-matching Euler discrete scheduler for Qwen-Image.
//!
//! Matches the official Qwen diffusers scheduler configuration
//! (`Qwen/Qwen-Image` and `Qwen/Qwen-Image-Edit-2511`,
//! `scheduler/scheduler_config.json`):
//! - `FlowMatchEulerDiscreteScheduler`
//! - dynamic exponential shifting based on image sequence length
//! - `base_shift=0.5`, `max_shift=0.9`
//! - `base_image_seq_len=256`, `max_image_seq_len=8192`
//! - `shift_terminal=0.02`
//!
//! That is not the only contract in the family. A checkpoint packages its own
//! scheduler config, and a distill may pin a *fixed* shift with no terminal
//! stretch — `nvidia/Qwen-Image-Flash` ships `use_dynamic_shifting=false`,
//! `shift=3.0`, `shift_terminal=null` for its four-step trajectory. Both
//! branches of diffusers' `set_timesteps` are implemented here and selected
//! per checkpoint by [`shift_policy_for_model`], because running the base
//! model's resolution-dependent schedule on a fixed-shift distill produces a
//! materially different trajectory (at 1328x1328 over four steps:
//! `[1000, 781, 476, 20]` instead of `[1000, 900, 750, 500]`).

use candle_core::{DType, Result, Tensor};

pub(crate) const NUM_TRAIN_TIMESTEPS: usize = 1000;
pub(crate) const BASE_IMAGE_SEQ_LEN: usize = 256;
pub(crate) const MAX_IMAGE_SEQ_LEN: usize = 8192;
pub(crate) const BASE_SHIFT: f64 = 0.5;
pub(crate) const MAX_SHIFT: f64 = 0.9;
pub(crate) const SHIFT_TERMINAL: f64 = 0.02;
/// `shift` from `nvidia/Qwen-Image-Flash`'s packaged
/// `scheduler/scheduler_config.json`.
pub(crate) const QWEN_FLASH_SHIFT: f64 = 3.0;

/// The timestep-shift contract a Qwen-Image checkpoint's packaged scheduler
/// declares. Mirrors step 2 (and the step-3 guard) of diffusers'
/// `FlowMatchEulerDiscreteScheduler::set_timesteps`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum QwenShiftPolicy {
    /// `use_dynamic_shifting=true` with `shift_terminal=0.02`: mu is derived
    /// from the image sequence length and the schedule is stretched to
    /// terminate at `SHIFT_TERMINAL`.
    DynamicResolution,
    /// `use_dynamic_shifting=false` with `shift_terminal=null`: one fixed
    /// shift, no resolution dependence, and no terminal stretch.
    Fixed { shift: f64 },
}

/// Resolve the shift policy from the resolved model name.
///
/// Only checkpoints that ship a scheduler config disagreeing with the base
/// Qwen pipeline get their own arm. Transformer-only exports (DiffSynth's
/// Distill-Full, Phr00t's Rapid AIO edit merge, the Lightning FP8 tiers)
/// publish no scheduler and are run through the base pipeline's, so they keep
/// [`QwenShiftPolicy::DynamicResolution`].
pub(crate) fn shift_policy_for_model(model_name: &str) -> QwenShiftPolicy {
    if model_name.starts_with("qwen-image-flash") {
        return QwenShiftPolicy::Fixed {
            shift: QWEN_FLASH_SHIFT,
        };
    }
    QwenShiftPolicy::DynamicResolution
}

fn calculate_shift(image_seq_len: usize) -> f64 {
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN) as f64;
    let b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN as f64;
    image_seq_len as f64 * m + b
}

fn time_shift_exponential(mu: f64, sigma: f64, t: f64) -> f64 {
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

/// diffusers, `use_dynamic_shifting=false`:
/// `sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)`.
fn time_shift_fixed(shift: f64, t: f64) -> f64 {
    shift * t / (1.0 + (shift - 1.0) * t)
}

fn stretch_shift_to_terminal(sigmas: &mut [f64]) {
    let one_minus_terminal = 1.0 - SHIFT_TERMINAL;
    let one_minus_z = 1.0 - sigmas[sigmas.len() - 1];
    if one_minus_z.abs() < 1e-12 {
        return;
    }
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
    /// The policy is not defaulted on purpose: a checkpoint that packages its
    /// own scheduler config must not silently inherit the base model's.
    pub fn new(num_inference_steps: usize, image_seq_len: usize, policy: QwenShiftPolicy) -> Self {
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

        match policy {
            QwenShiftPolicy::DynamicResolution => {
                let mu = calculate_shift(image_seq_len);
                for sigma in &mut sigmas {
                    *sigma = time_shift_exponential(mu, 1.0, *sigma);
                }
                stretch_shift_to_terminal(&mut sigmas);
            }
            QwenShiftPolicy::Fixed { shift } => {
                for sigma in &mut sigmas {
                    *sigma = time_shift_fixed(shift, *sigma);
                }
                // `shift_terminal` is null on a fixed-shift checkpoint, so
                // diffusers skips `stretch_shift_to_terminal` entirely.
            }
        }
        sigmas.push(0.0);

        Self {
            sigmas,
            step_index: 0,
        }
    }

    pub fn current_timestep(&self) -> f64 {
        self.sigmas[self.step_index] * NUM_TRAIN_TIMESTEPS as f64
    }

    /// Create an img2img scheduler by slicing the full schedule at the
    /// strength-derived start index.
    ///
    /// Returns `(scheduler, num_effective_steps)`. The caller should loop
    /// `num_effective_steps` times (same as `sigmas.len() - 1`).
    pub fn new_img2img(
        num_inference_steps: usize,
        image_seq_len: usize,
        strength: f64,
        policy: QwenShiftPolicy,
    ) -> (Self, usize) {
        let full = Self::new(num_inference_steps, image_seq_len, policy);
        let start_index = crate::img2img::img2img_start_index(full.num_steps(), strength);
        let sigmas = full.sigmas[start_index..].to_vec();
        let num_steps = sigmas.len().saturating_sub(1);
        (
            Self {
                sigmas,
                step_index: 0,
            },
            num_steps,
        )
    }

    pub fn initial_sigma(&self) -> f64 {
        self.sigmas[0]
    }

    /// Number of denoising steps (sigmas.len() - 1).
    pub fn num_steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
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
        let scheduler = QwenImageScheduler::new(50, 4096, QwenShiftPolicy::DynamicResolution);
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
        let scheduler = QwenImageScheduler::new(50, 4096, QwenShiftPolicy::DynamicResolution);
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

    #[test]
    fn stretch_shift_handles_single_step_schedule() {
        let mut sigmas = vec![1.0];
        stretch_shift_to_terminal(&mut sigmas);
        assert_eq!(sigmas, vec![1.0]);
    }

    #[test]
    fn img2img_schedule_starts_at_strength_and_is_shorter() {
        let strength = 0.75;
        let (scheduler, num_steps) =
            QwenImageScheduler::new_img2img(50, 4096, strength, QwenShiftPolicy::DynamicResolution);
        let full = QwenImageScheduler::new(50, 4096, QwenShiftPolicy::DynamicResolution);
        let start_index = crate::img2img::img2img_start_index(full.num_steps(), strength);
        // First sigma should match the scheduler sigma selected by strength.
        assert!(
            (scheduler.sigmas[0] - full.sigmas[start_index]).abs() < 1e-10,
            "first sigma should equal the truncated full-schedule sigma: got {}",
            scheduler.sigmas[0]
        );
        // Last sigma should be 0.0
        assert_eq!(*scheduler.sigmas.last().unwrap(), 0.0);
        // Schedule should be shorter than full schedule
        assert!(
            num_steps < full.num_steps(),
            "img2img should have fewer steps ({}) than full ({})",
            num_steps,
            full.num_steps()
        );
        assert_eq!(
            num_steps,
            crate::img2img::img2img_effective_steps(full.num_steps(), strength)
        );
        // Sigmas should be monotonically decreasing
        for pair in scheduler.sigmas.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "sigmas should be monotonically decreasing: {} < {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn img2img_full_strength_matches_txt2img() {
        let (_scheduler, num_steps) =
            QwenImageScheduler::new_img2img(50, 4096, 1.0, QwenShiftPolicy::DynamicResolution);
        let full = QwenImageScheduler::new(50, 4096, QwenShiftPolicy::DynamicResolution);
        // At strength=1.0, img2img should produce the full schedule
        assert_eq!(num_steps, full.num_steps());
    }

    #[test]
    fn num_steps_matches_sigmas_minus_one() {
        let scheduler = QwenImageScheduler::new(30, 4096, QwenShiftPolicy::DynamicResolution);
        assert_eq!(scheduler.num_steps(), scheduler.sigmas.len() - 1);
    }

    #[test]
    fn shift_policy_follows_the_checkpoints_packaged_scheduler() {
        // nvidia/Qwen-Image-Flash ships `use_dynamic_shifting=false`,
        // `shift=3.0`, `shift_terminal=null`.
        assert_eq!(
            shift_policy_for_model("qwen-image-flash:q8"),
            QwenShiftPolicy::Fixed {
                shift: QWEN_FLASH_SHIFT
            }
        );
        assert_eq!(
            shift_policy_for_model("qwen-image-flash:q4"),
            QwenShiftPolicy::Fixed {
                shift: QWEN_FLASH_SHIFT
            }
        );
        // Everything else inherits the base Qwen pipeline's scheduler:
        // `use_dynamic_shifting=true`, `shift_terminal=0.02`. The Distill-Full
        // and Rapid AIO merges are transformer-only exports with no scheduler
        // of their own, so they take the base contract.
        for name in [
            "qwen-image:bf16",
            "qwen-image-2512:q8",
            "qwen-image-lightning:fp8",
            "qwen-image-distill:q8",
            "qwen-image-distill:q4",
            "qwen-image-edit-2511:q8",
            "qwen-image-edit-rapid:q4",
        ] {
            assert_eq!(
                shift_policy_for_model(name),
                QwenShiftPolicy::DynamicResolution,
                "{name} must keep the base dynamic-shift schedule"
            );
        }
    }

    #[test]
    fn every_shipped_flash_tier_is_covered_by_the_fixed_policy() {
        // A new Flash tier in the manifest must not quietly inherit the base
        // model's resolution-dependent schedule.
        let flash: Vec<&str> = mold_core::manifest::known_manifests()
            .iter()
            .map(|m| m.name.as_str())
            .filter(|n| n.starts_with("qwen-image-flash"))
            .collect();
        assert!(!flash.is_empty(), "no qwen-image-flash manifests found");
        for name in flash {
            assert_eq!(
                shift_policy_for_model(name),
                QwenShiftPolicy::Fixed {
                    shift: QWEN_FLASH_SHIFT
                },
                "{name} must use its packaged fixed-shift scheduler"
            );
        }
    }

    #[test]
    fn fixed_shift_reproduces_the_flash_four_step_trajectory() {
        // diffusers `FlowMatchEulerDiscreteScheduler.set_timesteps` with
        // `use_dynamic_shifting=false`: sigma = shift*s / (1 + (shift-1)*s),
        // and `shift_terminal=null` skips `stretch_shift_to_terminal`.
        // At shift=3.0 over linspace(1.0, 0.25, 4) that is exactly
        // [1000, 900, 750, 500].
        let scheduler = QwenImageScheduler::new(
            4,
            6889,
            QwenShiftPolicy::Fixed {
                shift: QWEN_FLASH_SHIFT,
            },
        );
        let timesteps: Vec<f64> = scheduler.sigmas[..4]
            .iter()
            .map(|s| s * NUM_TRAIN_TIMESTEPS as f64)
            .collect();
        for (got, want) in timesteps.iter().zip([1000.0, 900.0, 750.0, 500.0]) {
            assert!(
                (got - want).abs() < 1e-9,
                "fixed shift-3 trajectory mismatch: got {timesteps:?}"
            );
        }
        assert_eq!(scheduler.sigmas.len(), 5);
        assert_eq!(*scheduler.sigmas.last().unwrap(), 0.0);
    }

    #[test]
    fn fixed_shift_is_resolution_independent() {
        // `use_dynamic_shifting=false` ignores `mu`, so the same request at a
        // different image_seq_len must produce the same schedule.
        let policy = QwenShiftPolicy::Fixed {
            shift: QWEN_FLASH_SHIFT,
        };
        let small = QwenImageScheduler::new(4, 1024, policy);
        let large = QwenImageScheduler::new(4, 6889, policy);
        assert_eq!(small.sigmas, large.sigmas);
    }

    #[test]
    fn dynamic_policy_trajectory_is_unchanged() {
        // Same request the Flash test uses (1328x1328, four steps) under the
        // base contract. This is what Flash was getting before the policy
        // existed, and what every non-Flash checkpoint must keep getting.
        let base = QwenImageScheduler::new(4, 6889, QwenShiftPolicy::DynamicResolution);
        let timesteps: Vec<f64> = base.sigmas[..4]
            .iter()
            .map(|s| s * NUM_TRAIN_TIMESTEPS as f64)
            .collect();
        for (got, want) in timesteps
            .iter()
            .zip([1000.0, 780.980_133, 475.548_796, 20.0])
        {
            assert!(
                (got - want).abs() < 1e-6,
                "base trajectory changed: got {timesteps:?}"
            );
        }
        // The base contract still terminates at `shift_terminal` before the
        // appended zero — which is exactly what Flash must NOT do.
        assert!((base.sigmas[3] - SHIFT_TERMINAL).abs() < 1e-10);
    }

    #[test]
    fn img2img_honours_the_shift_policy() {
        let policy = QwenShiftPolicy::Fixed {
            shift: QWEN_FLASH_SHIFT,
        };
        let full = QwenImageScheduler::new(8, 6889, policy);
        let (sliced, num_steps) = QwenImageScheduler::new_img2img(8, 6889, 1.0, policy);
        assert_eq!(sliced.sigmas, full.sigmas);
        assert_eq!(num_steps, full.num_steps());
    }
}

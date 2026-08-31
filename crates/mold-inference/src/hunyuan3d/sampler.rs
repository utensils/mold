//! The Hunyuan3D flow-matching sampler.
//!
//! # Which reference this follows
//!
//! ComfyUI, not Tencent's `hy3dgen`. Per `CLAUDE.md`, where an executable
//! oracle exists it is the documented primary reference, and ComfyUI is the
//! one that renders these exact checkpoints on this exact hardware. The
//! canonical recipe is its shipped blueprint (`blueprints/Image to Model
//! (Hunyuan3d 2.1).json`): `ModelSamplingAuraFlow(shift = 1.0)` +
//! `KSampler(euler, normal)`.
//!
//! # The whole schedule, derived
//!
//! `ModelSamplingAuraFlow` is `ModelSamplingDiscreteFlow` with
//! `multiplier = 1.0` (`comfy_extras/nodes_model_advanced.py:158-159`), so:
//!
//! ```text
//! sigma(t)     = time_snr_shift(shift, t / multiplier)     model_sampling.py:318
//! time_snr_shift(a, t) = t                       if a == 1 model_sampling.py:279
//!                      = a*t / (1 + (a-1)*t)     otherwise
//! timestep(sigma) = sigma * multiplier = sigma              model_sampling.py:315
//! ```
//!
//! `set_parameters` fills a 1000-entry sigma table from `arange(1, 1001)/1000`
//! (`model_sampling.py:301`), so `sigma_min = sigma(0.001)` and
//! `sigma_max = sigma(1.0)`. `normal_scheduler` (`comfy/samplers.py:671-693`)
//! then walks `linspace(timestep(sigma_max), timestep(sigma_min), steps)` and
//! appends a terminal zero, because `sigma(0.001)` is not within `1e-5` of
//! zero.
//!
//! The update rule collapses to one line. `CONST.calculate_denoised` is
//! `x - v*sigma` (`model_sampling.py:90-92`), and `to_d` is
//! `(x - denoised)/sigma`, so `d == v` — the model output *is* the Euler
//! derivative, and `sample_euler` (`k_diffusion/sampling.py:190-212`) reduces
//! to `x += v * (sigma_next - sigma)`.
//!
//! Classifier-free guidance commutes with that: mixing the two `denoised`
//! values and re-deriving `d` gives exactly
//! `v_uncond + (v_cond - v_uncond) * scale`, so [`apply_cfg`] mixes the
//! velocities directly and skips two tensor ops per step.
//!
//! # Guided vs distilled
//!
//! The two tiers spend the same `guidance = 5.0` differently. The undistilled
//! 2.0 checkpoint has no `guidance_in`, so it runs a real guided branch: two
//! forward passes per step, conditioned and unconditioned, mixed here. The
//! Turbo and mini-Turbo checkpoints carry the embedding, so the value goes
//! into `guidance_in` and one pass per step is the whole step. That is why
//! [`SamplingPlan::passes_per_step`] exists — the memory estimate and the
//! progress accounting both need to know, before any weight is read.

use candle_core::{Result, Tensor};

/// The sigma ladder plus how each step must be evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingPlan {
    /// `steps + 1` entries, descending, terminating at exactly `0.0`.
    pub sigmas: Vec<f64>,
    /// The CFG scale for a guided checkpoint, or `None` for a distilled one.
    pub cfg_scale: Option<f64>,
    /// The value fed to `guidance_in` on a distilled checkpoint.
    pub guidance_embed: Option<f64>,
}

impl SamplingPlan {
    /// `1` for a distilled checkpoint, `2` for a guided one.
    pub fn passes_per_step(&self) -> usize {
        match self.cfg_scale {
            Some(_) => 2,
            None => 1,
        }
    }

    /// Number of Euler steps, i.e. `sigmas.len() - 1`.
    pub fn steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }
}

/// `time_snr_shift` — `comfy/model_sampling.py:279-282`.
///
/// The `alpha == 1.0` early return is not an optimization: it is upstream's
/// own identity case, and the blueprint pins `shift = 1.0`, so in practice
/// the schedule is a plain descending linspace.
pub fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    if alpha == 1.0 {
        t
    } else {
        alpha * t / (1.0 + (alpha - 1.0) * t)
    }
}

/// Default flow shift for the family. `ModelSamplingAuraFlow` in the shipped
/// blueprint is set to `1.0`, NOT the node's own `1.73` default.
pub const DEFAULT_SHIFT: f64 = 1.0;

/// The smallest non-zero timestep `set_parameters` tabulates: `1 / 1000`.
const MIN_TIMESTEP: f64 = 1.0 / 1000.0;

/// Threshold `normal_scheduler` uses to decide whether the ladder already
/// reaches zero (`comfy/samplers.py:680`).
const ZERO_TOLERANCE: f64 = 0.00001;

/// Build the sigma ladder.
///
/// Faithful to `normal_scheduler` including the branch that adds a step when
/// the schedule already bottoms out at zero — that branch never fires for
/// `AuraFlow` sigmas, but reproducing it is what keeps this function correct
/// if a future tier ships a different `shift`.
pub fn normal_schedule(steps: usize, shift: f64) -> Vec<f64> {
    if steps == 0 {
        return vec![0.0];
    }
    let sigma_max = time_snr_shift(shift, 1.0);
    let sigma_min = time_snr_shift(shift, MIN_TIMESTEP);
    // `timestep(sigma) == sigma` at multiplier 1.0.
    let (start, end) = (sigma_max, sigma_min);

    let mut steps = steps;
    let mut append_zero = true;
    if time_snr_shift(shift, end).abs() < ZERO_TOLERANCE {
        steps += 1;
        append_zero = false;
    }

    let mut sigmas = Vec::with_capacity(steps + 1);
    for index in 0..steps {
        let t = if steps == 1 {
            start
        } else {
            start + (end - start) * (index as f64 / (steps - 1) as f64)
        };
        sigmas.push(time_snr_shift(shift, t));
    }
    if append_zero {
        sigmas.push(0.0);
    }
    sigmas
}

/// Assemble the plan for one request.
///
/// `guidance` carries the manifest's single 5.0 and is routed by
/// `checkpoint_has_guidance_embed`, which is a property of the weights, never
/// of the request — a caller cannot ask a distilled checkpoint for a guided
/// branch it was trained not to need, nor ask a guided one to read an
/// embedding it does not have.
pub fn plan(
    steps: usize,
    shift: f64,
    guidance: f64,
    checkpoint_has_guidance_embed: bool,
) -> SamplingPlan {
    let sigmas = normal_schedule(steps, shift);
    if checkpoint_has_guidance_embed {
        SamplingPlan {
            sigmas,
            cfg_scale: None,
            guidance_embed: Some(guidance),
        }
    } else {
        SamplingPlan {
            sigmas,
            cfg_scale: Some(guidance),
            guidance_embed: None,
        }
    }
}

/// `uncond + (cond - uncond) * scale`, on velocities.
///
/// See the module doc for why mixing velocities is identical to mixing
/// `denoised` values and re-deriving the Euler derivative.
pub fn apply_cfg(cond: &Tensor, uncond: &Tensor, scale: f64) -> Result<Tensor> {
    uncond + ((cond - uncond)? * scale)?
}

/// One Euler step: `x + v * (sigma_next - sigma)`.
pub fn euler_step(
    latents: &Tensor,
    velocity: &Tensor,
    sigma: f64,
    sigma_next: f64,
) -> Result<Tensor> {
    latents + (velocity * (sigma_next - sigma))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn shift_of_one_is_the_identity() {
        for t in [0.0, 0.001, 0.25, 0.5, 1.0] {
            assert_eq!(time_snr_shift(1.0, t), t);
        }
    }

    #[test]
    fn shift_above_one_pushes_mass_toward_high_noise() {
        // The defining property of a flow shift: it is monotone, fixes the
        // endpoints, and raises every interior point.
        assert_eq!(time_snr_shift(3.0, 0.0), 0.0);
        assert!((time_snr_shift(3.0, 1.0) - 1.0).abs() < 1e-12);
        assert!(time_snr_shift(3.0, 0.5) > 0.5);
    }

    #[test]
    fn schedule_descends_from_one_to_zero() {
        let sigmas = normal_schedule(30, DEFAULT_SHIFT);
        assert_eq!(sigmas.len(), 31, "steps + 1 entries");
        assert!((sigmas[0] - 1.0).abs() < 1e-12, "starts at sigma_max");
        assert_eq!(*sigmas.last().unwrap(), 0.0, "terminates at exactly zero");
        for pair in sigmas.windows(2) {
            assert!(pair[0] > pair[1], "must be strictly descending: {pair:?}");
        }
        // The penultimate entry is sigma_min, not zero — the terminal zero is
        // appended, which is what makes the last step land on a clean latent.
        assert!((sigmas[29] - MIN_TIMESTEP).abs() < 1e-12);
    }

    #[test]
    fn schedule_handles_the_distilled_step_counts() {
        for steps in [1usize, 4, 5, 8] {
            let sigmas = normal_schedule(steps, DEFAULT_SHIFT);
            assert_eq!(sigmas.len(), steps + 1, "{steps} steps");
            assert_eq!(*sigmas.last().unwrap(), 0.0);
            assert!((sigmas[0] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_steps_is_a_no_op_ladder_not_a_panic() {
        assert_eq!(normal_schedule(0, DEFAULT_SHIFT), vec![0.0]);
    }

    #[test]
    fn the_checkpoint_decides_how_guidance_is_spent() {
        let guided = plan(30, DEFAULT_SHIFT, 5.0, false);
        assert_eq!(guided.cfg_scale, Some(5.0));
        assert_eq!(guided.guidance_embed, None);
        assert_eq!(guided.passes_per_step(), 2);
        assert_eq!(guided.steps(), 30);

        let distilled = plan(5, DEFAULT_SHIFT, 5.0, true);
        assert_eq!(distilled.cfg_scale, None);
        assert_eq!(distilled.guidance_embed, Some(5.0));
        assert_eq!(distilled.passes_per_step(), 1);
        assert_eq!(distilled.steps(), 5);
    }

    #[test]
    fn cfg_interpolates_between_the_two_branches() {
        let device = Device::Cpu;
        let cond = Tensor::new(&[2.0f32, 4.0], &device).unwrap();
        let uncond = Tensor::new(&[1.0f32, 1.0], &device).unwrap();

        let at_one = apply_cfg(&cond, &uncond, 1.0).unwrap();
        assert_eq!(
            at_one.to_vec1::<f32>().unwrap(),
            vec![2.0, 4.0],
            "scale 1 is cond"
        );

        let at_zero = apply_cfg(&cond, &uncond, 0.0).unwrap();
        assert_eq!(
            at_zero.to_vec1::<f32>().unwrap(),
            vec![1.0, 1.0],
            "scale 0 is uncond"
        );

        let at_five = apply_cfg(&cond, &uncond, 5.0).unwrap();
        assert_eq!(at_five.to_vec1::<f32>().unwrap(), vec![6.0, 16.0]);
    }

    #[test]
    fn a_full_euler_walk_lands_on_the_clean_latent() {
        // With a constant velocity the exact solution is x0 = x1 - v, because
        // the sigmas sum telescopes to -(sigma_0 - sigma_n) = -1.
        let device = Device::Cpu;
        let mut x = Tensor::zeros((1, 4, 3), DType::F32, &device).unwrap();
        let v = Tensor::ones((1, 4, 3), DType::F32, &device).unwrap();
        let sigmas = normal_schedule(8, DEFAULT_SHIFT);
        for window in sigmas.windows(2) {
            x = euler_step(&x, &v, window[0], window[1]).unwrap();
        }
        let values = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for value in values {
            assert!((value + 1.0).abs() < 1e-5, "expected -1, got {value}");
        }
    }
}

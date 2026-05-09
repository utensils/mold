//! CFG++ step for the DDIM scheduler used by SDXL / SD1.5.
//!
//! Standard DDIM (eta=0, Epsilon prediction) integrates
//! `x_{t-1} = sqrt(α_{t-1}) · x_0(ε̂_guided) + sqrt(1-α_{t-1}) · ε̂_guided`.
//! CFG++ keeps `ε̂_guided` for the x_0 estimate but replaces the renoise
//! direction with the unconditional row, yielding
//! `x_{t-1} = sqrt(α_{t-1}) · x_0(ε̂_guided) + sqrt(1-α_{t-1}) · ε_uncond`
//! (Chung et al. 2024, "CFG++: Manifold-constrained Classifier Free Guidance",
//! arXiv:2406.08070; same formula HF diffusers ships for DDIM).
//!
//! The candle scheduler trait is opaque — `step()` is the only entry point
//! and `alphas_cumprod` is private — so we cannot ask the scheduler for α
//! values per step. Instead we mirror the DDIM alpha schedule on our side
//! (the math is deterministic from the config, so the two stay in lockstep
//! by construction). This module is the only place that computation lives;
//! pipelines build a `DdimAlphaSchedule` alongside the candle scheduler and
//! dispatch to `cfg_plus_step` in place of `scheduler.step()` when CFG++ is
//! active.
//!
//! Scope: Epsilon prediction + DDIM only. mold's SDXL and SD1.5 pipelines
//! both use Epsilon (the pipelines' only call site for `build_scheduler`),
//! and the candle DDIM defaults are eta=0 / ScaledLinear which matches what
//! the pipelines hand the scheduler. Other schedulers (EulerAncestral,
//! UniPC) and other prediction types fall back to standard CFG with a warn
//! at the call site.

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::stable_diffusion::ddim::DDIMSchedulerConfig;
use candle_transformers::models::stable_diffusion::schedulers::{BetaSchedule, TimestepSpacing};

/// Default candle `DDIMSchedulerConfig::default()` `train_timesteps`. Pinned
/// here so the mirrored schedule can drift-test if the upstream default
/// ever changes — `upstream_ddim_defaults_match_baked_in_constants` compares
/// both sides at test-time. Not referenced from non-test code (the actual
/// schedule pulls `train_timesteps` from the live config), so the constant
/// itself is test-only.
#[cfg(test)]
const DEFAULT_TRAIN_TIMESTEPS: usize = 1000;

/// CPU-side mirror of DDIM's alpha schedule and step-ratio metadata.
///
/// Built once per generation alongside `build_scheduler`; the alphas_cumprod
/// computation matches `candle_transformers::models::stable_diffusion::ddim::
/// DDIMScheduler::new` (lines 100-117 in candle-transformers-mold-0.9.12).
/// Independent of the candle scheduler instance — exists purely to expose
/// `(α_t, α_{t-1})` per timestep so the CFG++ step can be computed without
/// reaching into private fields.
pub(crate) struct DdimAlphaSchedule {
    alphas_cumprod: Vec<f64>,
    step_ratio: usize,
}

impl DdimAlphaSchedule {
    /// Build the schedule from mold's default DDIM configuration.
    ///
    /// `crates/mold-inference/src/scheduler.rs::build_scheduler` constructs
    /// `DDIMSchedulerConfig { prediction_type, ..Default::default() }` —
    /// every other field comes from candle's default. We mirror that here.
    pub(crate) fn from_default(inference_steps: usize) -> Self {
        let cfg = DDIMSchedulerConfig::default();
        Self::from_config(inference_steps, &cfg)
    }

    /// Build from an explicit config. Public for tests that want to vary
    /// `beta_schedule` without going through pipeline plumbing.
    pub(crate) fn from_config(inference_steps: usize, cfg: &DDIMSchedulerConfig) -> Self {
        let train_timesteps = cfg.train_timesteps;
        let step_ratio = train_timesteps / inference_steps.max(1);

        // Mirror candle DDIM constructor's beta arrays. TimestepSpacing only
        // affects which timesteps the scheduler emits — alphas_cumprod is
        // computed identically regardless.
        let _ = TimestepSpacing::Leading; // pin import
        let betas: Vec<f64> = match cfg.beta_schedule {
            BetaSchedule::ScaledLinear => {
                // linspace(sqrt(beta_start), sqrt(beta_end), N).square()
                linspace(cfg.beta_start.sqrt(), cfg.beta_end.sqrt(), train_timesteps)
                    .into_iter()
                    .map(|x| x * x)
                    .collect()
            }
            BetaSchedule::Linear => linspace(cfg.beta_start, cfg.beta_end, train_timesteps),
            BetaSchedule::SquaredcosCapV2 => betas_for_alpha_bar(train_timesteps, 0.999),
        };

        let mut alphas_cumprod = Vec::with_capacity(betas.len());
        for &beta in &betas {
            let alpha = 1.0 - beta;
            let last = *alphas_cumprod.last().unwrap_or(&1.0);
            alphas_cumprod.push(alpha * last);
        }

        Self {
            alphas_cumprod,
            step_ratio,
        }
    }

    /// Returns `(α_t, α_{t-1})` for the given step's timestep.
    ///
    /// `timestep` follows the same convention as candle DDIM's `step()`: it
    /// is one of the values in `Scheduler::timesteps()`, which originate
    /// from `train_timesteps` indexing (so values can be ≥ alphas_cumprod's
    /// length when offsets push them off the end — same wrap-down candle
    /// performs at `step` line 131-135).
    pub(crate) fn alphas_for_step(&self, timestep: usize) -> (f64, f64) {
        let t = if timestep >= self.alphas_cumprod.len() {
            timestep - 1
        } else {
            timestep
        };
        let prev_t = t.saturating_sub(self.step_ratio);
        (self.alphas_cumprod[t], self.alphas_cumprod[prev_t])
    }

    /// CFG++ DDIM step (Epsilon prediction, eta=0).
    ///
    /// Inputs are the *raw* uncond and the CFG-blended guided prediction:
    /// `eps_guided = eps_uncond + s · (eps_cond - eps_uncond)`. The caller
    /// is responsible for the CFG mix; this function only handles the
    /// integration. `eps_guided` and `eps_uncond` must have the same shape
    /// as `x_t` (one batch row each).
    pub(crate) fn cfg_plus_step(
        &self,
        x_t: &Tensor,
        eps_guided: &Tensor,
        eps_uncond: &Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        let (alpha_t, alpha_t_prev) = self.alphas_for_step(timestep);
        let beta_t = 1.0 - alpha_t;
        let beta_t_prev = 1.0 - alpha_t_prev;

        // x_0 estimate uses the CFG-guided prediction (so guidance still
        // shapes the high-level content):
        //   x_0 = (x_t - sqrt(1-α_t) · ε̂_guided) / sqrt(α_t)
        let x0 = ((x_t - (eps_guided * beta_t.sqrt())?)? * (1.0 / alpha_t.sqrt()))?;

        // Re-noise direction uses the unconditional prediction (so the
        // trajectory stays on the data manifold):
        //   x_{t-1} = sqrt(α_{t-1}) · x_0 + sqrt(1-α_{t-1}) · ε_uncond
        let prev = ((x0 * alpha_t_prev.sqrt())? + (eps_uncond * beta_t_prev.sqrt())?)?;
        Ok(prev)
    }
}

/// CPU-side `linspace` (mirrors `candle::utils::linspace` for f64). Inclusive
/// of both endpoints. `n == 0` returns empty; `n == 1` returns `[start]`.
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Mirror of candle's private `betas_for_alpha_bar` (cosine schedule). Pure
/// f64 — candle's version returns a Tensor, which we don't need here.
fn betas_for_alpha_bar(num_diffusion_timesteps: usize, max_beta: f64) -> Vec<f64> {
    let alpha_bar =
        |t: usize| f64::cos((t as f64 + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).powi(2);
    let mut betas = Vec::with_capacity(num_diffusion_timesteps);
    for i in 0..num_diffusion_timesteps {
        let t1 = i / num_diffusion_timesteps;
        let t2 = (i + 1) / num_diffusion_timesteps;
        betas.push((1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta));
    }
    betas
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Pin upstream defaults — if candle bumps the default `train_timesteps`
    /// or beta endpoints, mold's mirrored schedule drifts and CFG++ output
    /// diverges from what the standard scheduler would produce on the same
    /// trajectory. Catch the drift here.
    #[test]
    fn upstream_ddim_defaults_match_baked_in_constants() {
        let cfg = DDIMSchedulerConfig::default();
        assert_eq!(cfg.train_timesteps, DEFAULT_TRAIN_TIMESTEPS);
        assert_eq!(cfg.beta_start, 0.00085);
        assert_eq!(cfg.beta_end, 0.012);
        assert!(matches!(cfg.beta_schedule, BetaSchedule::ScaledLinear));
        assert_eq!(cfg.eta, 0.0);
    }

    /// alphas_cumprod must be strictly decreasing for any valid beta
    /// schedule, monotonically approach 0, start at (1 - beta[0]).
    #[test]
    fn alphas_cumprod_monotone_decreasing() {
        let sched = DdimAlphaSchedule::from_default(50);
        assert_eq!(sched.alphas_cumprod.len(), DEFAULT_TRAIN_TIMESTEPS);
        let mut prev = sched.alphas_cumprod[0];
        assert!(
            prev < 1.0 && prev > 0.999,
            "alphas[0] should be ~ 1 - beta_start, got {prev}"
        );
        for &a in &sched.alphas_cumprod[1..] {
            assert!(a < prev, "alphas_cumprod must be strictly decreasing");
            assert!(a > 0.0, "alphas_cumprod must stay positive");
            prev = a;
        }
        // SD-style schedule (ScaledLinear, beta_end=0.012) lands α_final ≈ 5e-3
        // — close to but not exactly zero, since beta never saturates at 1.
        assert!(
            prev < 0.01,
            "alphas_cumprod[final] should be ≈ 0, got {prev}"
        );
    }

    /// Step ratio matches candle: train_timesteps / inference_steps,
    /// floor-division.
    #[test]
    fn step_ratio_floor_div() {
        assert_eq!(DdimAlphaSchedule::from_default(50).step_ratio, 20);
        assert_eq!(DdimAlphaSchedule::from_default(28).step_ratio, 35); // 1000/28 = 35
        assert_eq!(DdimAlphaSchedule::from_default(1).step_ratio, 1000);
    }

    /// alphas_for_step: clamps timestep ≥ alphas_cumprod.len() to len-1
    /// (mirrors candle DDIM step lines 131-135) and clamps prev_timestep
    /// to 0 via saturating_sub.
    #[test]
    fn alphas_for_step_clamps_at_boundaries() {
        let sched = DdimAlphaSchedule::from_default(50);
        // First inference step's timestep is 999 (Leading spacing offsets by 1).
        // We index 999 directly. prev = 999 - 20 = 979.
        let (a, ap) = sched.alphas_for_step(999);
        assert_eq!(a, sched.alphas_cumprod[999]);
        assert_eq!(ap, sched.alphas_cumprod[979]);

        // Last inference step's timestep is 19 (after 49 step-down steps).
        // prev_timestep saturating_sub(20) = 0 (not negative).
        let (a, ap) = sched.alphas_for_step(19);
        assert_eq!(a, sched.alphas_cumprod[19]);
        assert_eq!(ap, sched.alphas_cumprod[0]);

        // Beyond-array timestep wraps down by one (candle's defensive guard).
        let (a, _) = sched.alphas_for_step(DEFAULT_TRAIN_TIMESTEPS);
        assert_eq!(a, sched.alphas_cumprod[DEFAULT_TRAIN_TIMESTEPS - 1]);
    }

    /// When ε_uncond == ε_guided the CFG++ step must equal the standard
    /// DDIM step — there's no guidance signal to reroute, so the manifold
    /// projection is a no-op.
    #[test]
    fn cfg_plus_step_collapses_to_standard_when_eps_uncond_eq_eps_guided() {
        let sched = DdimAlphaSchedule::from_default(50);
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev).unwrap();
        let eps = Tensor::from_slice(&[0.5f32, -0.3, 0.1, 0.7], (1, 4), &dev).unwrap();
        let timestep = 999;

        let cfg_plus = sched.cfg_plus_step(&x, &eps, &eps, timestep).unwrap();

        // Standard DDIM step inline: x_{t-1} = sqrt(α')·x_0 + sqrt(1-α')·ε
        let (alpha_t, alpha_t_prev) = sched.alphas_for_step(timestep);
        let beta_t = 1.0 - alpha_t;
        let beta_t_prev = 1.0 - alpha_t_prev;
        let x0 =
            ((&x - (&eps * beta_t.sqrt()).unwrap()).unwrap() * (1.0 / alpha_t.sqrt())).unwrap();
        let standard =
            ((x0 * alpha_t_prev.sqrt()).unwrap() + (&eps * beta_t_prev.sqrt()).unwrap()).unwrap();

        let cfg_vec: Vec<f32> = cfg_plus.flatten_all().unwrap().to_vec1().unwrap();
        let std_vec: Vec<f32> = standard.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in cfg_vec.iter().zip(std_vec.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "cfg++ ≠ standard at degenerate eps_uncond=eps_guided"
            );
        }
    }

    /// Sanity: at high CFG with eps_uncond ≠ eps_guided, the CFG++ output
    /// must differ from the standard DDIM output. Catches accidental no-op
    /// implementations (e.g. forgetting to swap eps_uncond into the renoise
    /// term, mirroring the SD3 sampling test).
    #[test]
    fn cfg_plus_step_diverges_from_standard_under_high_cfg() {
        let sched = DdimAlphaSchedule::from_default(28);
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[0.5f32; 8], (1, 8), &dev).unwrap();
        let eps_uncond = Tensor::from_slice(&[0.1f32; 8], (1, 8), &dev).unwrap();
        let eps_cond = Tensor::from_slice(&[0.4f32; 8], (1, 8), &dev).unwrap();
        let s = 7.5_f64;
        // eps_guided = uncond + s·(cond - uncond) = 0.1 + 7.5·0.3 = 2.35
        let eps_guided =
            (&eps_uncond + (((&eps_cond - &eps_uncond).unwrap() * s).unwrap())).unwrap();
        let timestep = 999;

        let cfg_plus = sched
            .cfg_plus_step(&x, &eps_guided, &eps_uncond, timestep)
            .unwrap();
        // Standard step uses eps_guided in BOTH the x_0 estimate AND the renoise term.
        let standard = sched
            .cfg_plus_step(&x, &eps_guided, &eps_guided, timestep)
            .unwrap();

        let cfg_vec: Vec<f32> = cfg_plus.flatten_all().unwrap().to_vec1().unwrap();
        let std_vec: Vec<f32> = standard.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = cfg_vec
            .iter()
            .zip(std_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.05,
            "cfg++ must diverge from standard at cfg=7.5, max_diff={max_diff}"
        );
    }

    /// First step (timestep=999) and last step (timestep=19 with step_ratio=20)
    /// must both produce finite output — guards against integer underflow on
    /// the prev_timestep calculation.
    #[test]
    fn cfg_plus_step_finite_at_boundary_timesteps() {
        let sched = DdimAlphaSchedule::from_default(50);
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[0.5f32; 4], (1, 4), &dev).unwrap();
        let eps_g = Tensor::from_slice(&[0.3f32; 4], (1, 4), &dev).unwrap();
        let eps_u = Tensor::from_slice(&[0.1f32; 4], (1, 4), &dev).unwrap();

        for &ts in &[999_usize, 19] {
            let out = sched.cfg_plus_step(&x, &eps_g, &eps_u, ts).unwrap();
            let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            for x in &v {
                assert!(
                    x.is_finite(),
                    "cfg_plus_step produced non-finite output at timestep {ts}"
                );
            }
        }
    }
}

//! Qwen Image 2.1's packaged FlowMatch Euler schedule.
//!
//! The 2.1 checkpoint declares the same dynamic exponential-shift scheduler
//! contract as the original Qwen-Image base release: 1000 train timesteps,
//! base/max image sequence lengths 256/8192, shifts 0.5/0.9, and a terminal
//! sigma of 0.02 (`scheduler/scheduler_config.json` at `b3179ad`, captured in
//! `testdata/qwen_image21/schedules.json`).  Reuse the one implementation
//! rather than fork identical floating-point schedule arithmetic.

use candle_core::DType;

pub(crate) use crate::qwen_image::sampling::{
    QwenImageScheduler as QwenImage21Scheduler, QwenShiftPolicy,
};

/// Viggle's 6-step turbo trajectory (`Viggle/Qwen-Image-2.1-viggle-turbo`
/// model card): these are the pipeline's raw `sigmas=` — the mu shift is
/// still applied — with `shift_terminal: null`.
pub(crate) const TURBO_BASE_SIGMAS: [f64; 6] = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25];

/// Which trajectory a checkpoint samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduleKind {
    /// The shipped scheduler: linspace sigmas, dynamic shift, terminal 0.02.
    Base,
    /// The distilled turbo recipe: fixed six base sigmas, dynamic shift, no
    /// terminal stretch. Selected by the turbo tier together with its
    /// distilled LoRA.
    #[cfg_attr(not(test), allow(dead_code))]
    Turbo,
}

/// Build the scheduler for `steps` over `target_tokens` latent tokens.
///
/// `mu` always comes from the TARGET tokens alone
/// (`pipeline_qwenimage21.py:724`, `latents.shape[1]`): condition-image
/// tokens never move the schedule. A turbo render at a step count other than
/// its six keeps the no-terminal policy over the default linspace and returns
/// a warning, because there is no published trajectory to follow.
pub(crate) fn scheduler_for(
    kind: ScheduleKind,
    steps: usize,
    target_tokens: usize,
) -> (QwenImage21Scheduler, Option<String>) {
    match kind {
        ScheduleKind::Base => (
            QwenImage21Scheduler::new(steps, target_tokens, QwenShiftPolicy::DynamicResolution),
            None,
        ),
        ScheduleKind::Turbo if steps == TURBO_BASE_SIGMAS.len() => (
            QwenImage21Scheduler::with_base_sigmas(
                &TURBO_BASE_SIGMAS,
                target_tokens,
                QwenShiftPolicy::DynamicNoTerminal,
            ),
            None,
        ),
        ScheduleKind::Turbo => (
            QwenImage21Scheduler::new(steps, target_tokens, QwenShiftPolicy::DynamicNoTerminal),
            Some(format!(
                "The Qwen Image 2.1 turbo distill was trained for {} steps; {steps} steps use an evenly spaced trajectory instead",
                TURBO_BASE_SIGMAS.len()
            )),
        ),
    }
}

/// The normalized timestep the transformer receives for `sigma`.
///
/// Upstream holds sigmas as float32 and forms `t = sigma * 1000` in float32
/// (`scheduling_flow_match_euler_discrete.py:366-367`), then casts `t` to the
/// LATENT dtype before dividing by 1000 (`pipeline_qwenimage21.py:770, 775`),
/// so in BF16 the division itself rounds: 900 -> 0.8984375, 600 ->
/// 0.6015625. Reproduce that rounding for the working dtype.
pub(crate) fn transformer_timestep(sigma: f64, dtype: DType) -> f64 {
    let t = (sigma as f32) * 1000.0f32;
    match dtype {
        DType::BF16 => {
            let t = half::bf16::from_f32(t).to_f32();
            half::bf16::from_f32(t / 1000.0).to_f64()
        }
        DType::F16 => {
            let t = half::f16::from_f32(t).to_f32();
            half::f16::from_f32(t / 1000.0).to_f64()
        }
        _ => f64::from(t / 1000.0f32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_image::sampling::calculate_shift;

    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }

    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        target_tokens: usize,
        steps: usize,
        mu: f64,
        sigmas: Vec<f64>,
        timesteps: Vec<f64>,
    }

    fn fixture() -> Fixture {
        serde_json::from_str(include_str!("../../testdata/qwen_image21/schedules.json")).unwrap()
    }

    /// U11: every captured base and turbo trajectory (float32 upstream),
    /// with `mu` from target tokens only and unclamped at 2K.
    #[test]
    fn schedules_match_the_captured_upstream_trajectories() {
        let fixture = fixture();
        assert!(fixture
            .cases
            .iter()
            .any(|case| case.name.starts_with("turbo")));
        for case in &fixture.cases {
            assert_eq!(
                calculate_shift(case.target_tokens),
                case.mu,
                "{}",
                case.name
            );
            let kind = if case.name.starts_with("turbo") {
                ScheduleKind::Turbo
            } else {
                ScheduleKind::Base
            };
            let (scheduler, warning) = scheduler_for(kind, case.steps, case.target_tokens);
            assert!(warning.is_none());
            assert_eq!(scheduler.sigmas.len(), case.sigmas.len(), "{}", case.name);
            for (index, (actual, expected)) in scheduler.sigmas.iter().zip(&case.sigmas).enumerate()
            {
                // Upstream shifts in float32; mold keeps its f64 arithmetic
                // (v0.32's schedule bytes), which agrees to float32 precision.
                assert!(
                    (actual - expected).abs() <= 2e-7,
                    "{} sigma {index}: {actual} vs {expected}",
                    case.name
                );
            }
            for (index, expected) in case.timesteps.iter().enumerate() {
                let actual = (scheduler.sigmas[index] as f32) * 1000.0f32;
                assert!(
                    (f64::from(actual) - expected).abs() <= 2.7e-4,
                    "{} timestep {index}: {actual} vs {expected}",
                    case.name
                );
            }
        }
        // 2752x1536 = 16,512 target tokens extrapolates past max_shift.
        assert!(calculate_shift(16_512) > 0.9);
    }

    #[test]
    fn default_schedule_is_unchanged_and_turbo_skips_the_terminal_stretch() {
        let base = QwenImage21Scheduler::new(40, 4096, QwenShiftPolicy::DynamicResolution);
        let (via, _) = scheduler_for(ScheduleKind::Base, 40, 4096);
        assert_eq!(base.sigmas, via.sigmas);
        assert!((base.sigmas[39] - 0.02).abs() < 1e-12);
        let (turbo, _) = scheduler_for(ScheduleKind::Turbo, 6, 4096);
        assert_eq!(turbo.sigmas.len(), 7);
        assert!(turbo.sigmas[5] > 0.25 && turbo.sigmas[5] < 0.5);
        assert_eq!(*turbo.sigmas.last().unwrap(), 0.0);
        let (other, warning) = scheduler_for(ScheduleKind::Turbo, 8, 4096);
        assert_eq!(other.sigmas.len(), 9);
        assert!(warning.unwrap().contains("6 steps"));
    }

    #[test]
    fn bf16_rounds_the_timestep_like_upstream() {
        assert_eq!(transformer_timestep(0.9, DType::BF16), 0.8984375);
        assert_eq!(transformer_timestep(0.6, DType::BF16), 0.6015625);
        assert_eq!(transformer_timestep(1.0, DType::BF16), 1.0);
        assert_eq!(
            transformer_timestep(0.9, DType::F32),
            f64::from(900.0f32 / 1000.0)
        );
    }
}

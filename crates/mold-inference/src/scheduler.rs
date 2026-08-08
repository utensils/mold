use anyhow::Result;
use candle_transformers::models::stable_diffusion::{
    ddim::DDIMSchedulerConfig,
    euler_ancestral_discrete::EulerAncestralDiscreteSchedulerConfig,
    schedulers::{PredictionType, Scheduler, SchedulerConfig, TimestepSpacing},
    uni_pc::UniPCSchedulerConfig,
};
use mold_core::Scheduler as MoldScheduler;

/// Build a candle scheduler from a mold `Scheduler` enum value.
///
/// `prediction_type` and `is_turbo` come from the model config (e.g. Epsilon
/// for SD1.5/SDXL, Trailing timesteps for SDXL Turbo).
pub fn build_scheduler(
    scheduler: MoldScheduler,
    inference_steps: usize,
    prediction_type: PredictionType,
    is_turbo: bool,
) -> Result<Box<dyn Scheduler>> {
    let result = match scheduler {
        MoldScheduler::Ddim => {
            let config = DDIMSchedulerConfig {
                prediction_type,
                ..Default::default()
            };
            config.build(inference_steps)
        }
        MoldScheduler::EulerAncestral => {
            let timestep_spacing = if is_turbo {
                TimestepSpacing::Trailing
            } else {
                TimestepSpacing::Leading
            };
            let config = EulerAncestralDiscreteSchedulerConfig {
                prediction_type,
                timestep_spacing,
                ..Default::default()
            };
            config.build(inference_steps)
        }
        MoldScheduler::UniPc => {
            let config = UniPCSchedulerConfig {
                prediction_type,
                ..Default::default()
            };
            config.build(inference_steps)
        }
        // Wan's flow solvers (#795) never reach the UNet scheduler factory:
        // admission rejects them for every non-wan family, and the wan engine
        // consumes the selection itself. Fail loudly rather than silently
        // sampling with the wrong algorithm if a path ever slips through.
        MoldScheduler::Euler | MoldScheduler::DpmPp => {
            anyhow::bail!(
                "scheduler '{scheduler}' is a Wan flow solver and has no UNet implementation"
            )
        }
    };
    result.map_err(|e| anyhow::anyhow!("{e}"))
}

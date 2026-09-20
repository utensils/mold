//! Qwen Image 2.1's packaged FlowMatch Euler schedule.
//!
//! The 2.1 checkpoint declares the same dynamic exponential-shift scheduler
//! contract as the original Qwen-Image base release: 1000 train timesteps,
//! base/max image sequence lengths 256/8192, shifts 0.5/0.9, and a terminal
//! sigma of 0.02.  Reuse the one implementation rather than fork identical
//! floating-point schedule arithmetic.

pub(crate) use crate::qwen_image::sampling::{
    QwenImageScheduler as QwenImage21Scheduler, QwenShiftPolicy,
};

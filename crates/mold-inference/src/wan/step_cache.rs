//! First-block residual caching for Wan's non-distilled tiers (#801).
//!
//! The A14B `:q8` quality tier runs 20 steps of two forwards each at ~30 s a
//! step, and the 5B runs 20 steps over 121 frames. The ecosystem's standard
//! mitigation is to notice that consecutive denoise steps produce nearly the
//! same residual and reuse it: TeaCache, MagCache, and Comfy-WaveSpeed's
//! FBCache all trade a distance check for a skipped forward.
//!
//! This implements the FBCache formulation, which is the smallest surface and
//! needs no offline calibration. Per step:
//!
//! 1. run block 0 and take its residual, `delta = out - in`;
//! 2. if the relative L1 distance between `delta` and the previous step's
//!    `delta` is below the threshold, add the *cached* tail residual — what
//!    blocks 1..N contributed last time — and skip them entirely;
//! 3. otherwise run the remaining blocks and record the new tail residual.
//!
//! Block 0 always runs, so the check costs one fortieth of a forward.
//!
//! **Distilled tiers get nothing from this and must never enable it.** A
//! 4-step Lightning schedule has no near-duplicate steps to skip — the field
//! reports this consistently (kijai/ComfyUI-WanVideoWrapper#811) — and a
//! threshold that fires on a 4-step schedule is skipping a step that carried
//! real signal. [`WanStepCachePolicy::resolve`] refuses those cases by
//! construction rather than trusting the operator to know.
//!
//! Two caches are needed, not one: the conditional and unconditional forwards
//! are different trajectories and their residuals must never be crossed. The
//! cache is also reset at the A14B expert swap, where the network itself
//! changes and no previous residual describes it.

use anyhow::Result;
use candle_core::Tensor;

/// Threshold used by `MOLD_WAN_STEP_CACHE=auto`.
///
/// Measured, not chosen. The first value tried here was 0.05, on the reasoning
/// that `auto` should be conservative — and on an RTX 4090 running
/// `wan22-t2v-a14b:q8` at 33f/832x480 it never fired once in 20 steps: 620.5 s
/// against the uncached 605.6 s, i.e. it bought nothing and cost the distance
/// check. At 0.10 the same render takes 327.4 s, a 1.85x speedup with no
/// visible artifacting. Comfy-WaveSpeed independently ships 0.09 as its Wan
/// `residual_diff_threshold` default, which is the same neighbourhood.
///
/// A conservative default that never engages is not conservative, it is a knob
/// that lies about being on.
pub(crate) const AUTO_THRESHOLD: f64 = 0.10;

/// Below this step count there is nothing to reuse.
///
/// The distilled tiers run 4; upstream's non-distilled recipes run 20-50. A
/// schedule this short spends every step on structure that the next step does
/// not repeat.
pub(crate) const MIN_CACHEABLE_STEPS: u32 = 12;

/// Whether a render may reuse residuals, and how eagerly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum WanStepCachePolicy {
    /// Every block runs on every step. Bit-identical to the pre-cache engine.
    Off,
    /// Skip blocks 1..N when the first-block residual moved less than this
    /// relative L1 distance.
    Threshold(f64),
}

/// Why a requested cache was refused, for disclosure in the progress output.
///
/// A silently ignored knob is worse than no knob: the user concludes the
/// feature does not help when in fact it never ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WanStepCacheRefusal {
    /// Fewer than [`MIN_CACHEABLE_STEPS`] steps.
    TooFewSteps,
    /// A distill adapter is active; there is no redundancy to skip.
    Distilled,
}

impl WanStepCacheRefusal {
    pub fn message(self) -> &'static str {
        match self {
            Self::TooFewSteps => {
                "step cache ignored: schedules under 12 steps have no redundant steps to skip"
            }
            Self::Distilled => {
                "step cache ignored: a distilled adapter is active, which leaves no \
                 near-duplicate steps"
            }
        }
    }
}

impl WanStepCachePolicy {
    /// Resolve the policy for one render.
    ///
    /// `requested` is the parsed knob; the two guards are not overridable,
    /// because both describe configurations where the cache cannot help and
    /// can only cost quality.
    pub fn resolve(
        requested: Option<f64>,
        steps: u32,
        distilled: bool,
    ) -> (Self, Option<WanStepCacheRefusal>) {
        let Some(threshold) = requested else {
            return (Self::Off, None);
        };
        if distilled {
            return (Self::Off, Some(WanStepCacheRefusal::Distilled));
        }
        if steps < MIN_CACHEABLE_STEPS {
            return (Self::Off, Some(WanStepCacheRefusal::TooFewSteps));
        }
        (Self::Threshold(threshold), None)
    }

    pub fn is_on(self) -> bool {
        matches!(self, Self::Threshold(_))
    }
}

/// Parse `MOLD_WAN_STEP_CACHE`.
///
/// `off` / unset disables. `auto` selects [`AUTO_THRESHOLD`]. A positive
/// finite number is that threshold. Anything else is an error rather than a
/// silent fallback — a typo that quietly disabled the cache would look like
/// the feature not working.
pub(crate) fn parse_threshold(raw: &str) -> Result<Option<f64>> {
    let value = raw.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("off") || value == "0" {
        return Ok(None);
    }
    if value.eq_ignore_ascii_case("auto") {
        return Ok(Some(AUTO_THRESHOLD));
    }
    let parsed: f64 = value.parse().map_err(|_| {
        anyhow::anyhow!("MOLD_WAN_STEP_CACHE must be off, auto, or a positive number, got '{raw}'")
    })?;
    if !parsed.is_finite() || parsed <= 0.0 {
        anyhow::bail!("MOLD_WAN_STEP_CACHE threshold must be finite and positive, got {parsed}");
    }
    Ok(Some(parsed))
}

/// Resolve the policy from the process environment.
pub(crate) fn requested_threshold() -> Result<Option<f64>> {
    match crate::runtime_env::value("MOLD_WAN_STEP_CACHE") {
        Some(raw) => parse_threshold(&raw),
        None => Ok(None),
    }
}

/// One denoise trajectory's residual cache.
///
/// Holds two tensors at most: the previous step's first-block residual (for
/// the distance check) and the tail residual it is allowed to replay. Both are
/// token-shaped, so this costs two hidden states — the same order as one
/// block's transients, against skipping thirty-nine blocks.
pub(crate) struct WanStepCache {
    threshold: f64,
    previous_first_block: Option<Tensor>,
    tail: Option<Tensor>,
    skipped: usize,
    considered: usize,
}

impl WanStepCache {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            previous_first_block: None,
            tail: None,
            skipped: 0,
            considered: 0,
        }
    }

    /// Forget everything. Called at the expert swap: the network changed, so
    /// no residual recorded against the previous one describes it.
    pub fn reset(&mut self) {
        self.previous_first_block = None;
        self.tail = None;
    }

    pub fn skipped(&self) -> usize {
        self.skipped
    }

    pub fn considered(&self) -> usize {
        self.considered
    }

    /// Relative L1 distance between two residuals, the FBCache criterion.
    ///
    /// Normalized by the previous residual's own magnitude so the threshold
    /// means the same thing at every noise level; an all-zero previous
    /// residual yields no reuse rather than a division by zero.
    fn relative_distance(previous: &Tensor, current: &Tensor) -> candle_core::Result<f64> {
        let previous = previous.to_dtype(candle_core::DType::F32)?;
        let current = current.to_dtype(candle_core::DType::F32)?;
        let scale = previous.abs()?.mean_all()?.to_scalar::<f32>()? as f64;
        // NaN and zero both mean "no usable reference", and both must yield
        // no reuse rather than a division that produces one.
        if !scale.is_finite() || scale <= 0.0 {
            return Ok(f64::INFINITY);
        }
        let delta = (current - previous)?
            .abs()?
            .mean_all()?
            .to_scalar::<f32>()? as f64;
        Ok(delta / scale)
    }

    /// Decide what this step should do, given block 0's residual.
    ///
    /// Returns the cached tail to add when the step may be skipped. The
    /// first-block residual is recorded either way, so the *next* step
    /// compares against this step rather than against the last one that ran
    /// in full — otherwise a long skip run drifts without ever noticing.
    pub fn decide(&mut self, first_block_residual: &Tensor) -> candle_core::Result<Option<Tensor>> {
        self.considered += 1;
        let reuse = match (&self.previous_first_block, &self.tail) {
            (Some(previous), Some(tail)) => {
                let distance = Self::relative_distance(previous, first_block_residual)?;
                (distance < self.threshold).then(|| tail.clone())
            }
            _ => None,
        };
        self.previous_first_block = Some(first_block_residual.clone());
        if reuse.is_some() {
            self.skipped += 1;
        }
        Ok(reuse)
    }

    /// Record what blocks 1..N contributed on a step that ran in full.
    pub fn record_tail(&mut self, tail: Tensor) {
        self.tail = Some(tail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn the_knob_parses_off_auto_and_explicit_thresholds() {
        assert_eq!(parse_threshold("").unwrap(), None);
        assert_eq!(parse_threshold("off").unwrap(), None);
        assert_eq!(parse_threshold("OFF").unwrap(), None);
        assert_eq!(parse_threshold("0").unwrap(), None);
        assert_eq!(parse_threshold("auto").unwrap(), Some(AUTO_THRESHOLD));
        // The measured value, not a placeholder: 0.05 never fired.
        assert_eq!(AUTO_THRESHOLD, 0.10);
        assert_eq!(parse_threshold("0.12").unwrap(), Some(0.12));
        // A typo must be an error, not a silent disable that looks like the
        // feature failing to help.
        assert!(parse_threshold("0.1x").is_err());
        assert!(parse_threshold("-1").is_err());
        assert!(parse_threshold("nan").is_err());
    }

    /// The two guards are the reason this is safe to ship.
    ///
    /// A distilled 4-step schedule has no redundancy; a threshold that fires
    /// there is deleting signal. Neither guard is overridable, and both
    /// disclose rather than silently doing nothing.
    #[test]
    fn distilled_and_short_schedules_refuse_the_cache_and_say_why() {
        let (policy, refusal) = WanStepCachePolicy::resolve(Some(0.05), 4, true);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, Some(WanStepCacheRefusal::Distilled));

        // Distilled wins even at a long schedule.
        let (policy, refusal) = WanStepCachePolicy::resolve(Some(0.05), 40, true);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, Some(WanStepCacheRefusal::Distilled));

        let (policy, refusal) = WanStepCachePolicy::resolve(Some(0.05), 4, false);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, Some(WanStepCacheRefusal::TooFewSteps));

        // The quality tiers qualify.
        let (policy, refusal) = WanStepCachePolicy::resolve(Some(0.05), 20, false);
        assert_eq!(policy, WanStepCachePolicy::Threshold(0.05));
        assert_eq!(refusal, None);

        // Unset is off with nothing to disclose.
        let (policy, refusal) = WanStepCachePolicy::resolve(None, 20, false);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, None);
    }

    fn constant(value: f32, device: &Device) -> Tensor {
        Tensor::full(value, (1, 4, 8), device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    }

    /// The first step can never reuse — there is nothing recorded yet — and a
    /// steady trajectory reuses from the step after that.
    #[test]
    fn the_first_step_always_runs_and_a_steady_trajectory_then_reuses() {
        let device = Device::Cpu;
        let mut cache = WanStepCache::new(0.05);

        assert!(cache.decide(&constant(1.0, &device)).unwrap().is_none());
        // Still nothing to replay until a full step records its tail.
        assert!(cache.decide(&constant(1.0, &device)).unwrap().is_none());
        cache.record_tail(constant(7.0, &device));

        // An identical residual is distance 0, comfortably under threshold.
        let reused = cache.decide(&constant(1.0, &device)).unwrap();
        assert!(reused.is_some());
        assert_eq!(
            reused
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0],
            7.0
        );
        assert_eq!(cache.skipped(), 1);

        // A residual that moved 100% is far past the threshold.
        assert!(cache.decide(&constant(2.0, &device)).unwrap().is_none());
        assert_eq!(cache.skipped(), 1);
        assert_eq!(cache.considered(), 4);
    }

    /// The expert swap must invalidate the cache: the network changed, so a
    /// residual recorded against the high-noise expert says nothing about the
    /// low-noise one.
    #[test]
    fn the_expert_swap_invalidates_everything() {
        let device = Device::Cpu;
        let mut cache = WanStepCache::new(0.05);
        cache.decide(&constant(1.0, &device)).unwrap();
        cache.record_tail(constant(7.0, &device));
        assert!(cache.decide(&constant(1.0, &device)).unwrap().is_some());

        cache.reset();
        // Identical residual, but nothing to replay against.
        assert!(cache.decide(&constant(1.0, &device)).unwrap().is_none());
    }

    /// An all-zero previous residual must not divide by zero into a reuse.
    #[test]
    fn a_degenerate_previous_residual_never_reuses() {
        let device = Device::Cpu;
        let mut cache = WanStepCache::new(0.05);
        cache.decide(&constant(0.0, &device)).unwrap();
        cache.record_tail(constant(7.0, &device));
        assert!(cache.decide(&constant(0.0, &device)).unwrap().is_none());
    }
}

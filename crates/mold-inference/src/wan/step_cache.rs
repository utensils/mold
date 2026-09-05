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

/// Token slice width for the distance reduction.
///
/// The FBCache criterion is two scalars — `mean|previous|` and
/// `mean|current - previous|` — but computing them over whole tensors
/// materializes `[1, T, dim]` F32 temporaries, which at A14B 53f/832x480 is
/// ~447 MB apiece against a ~224 MB BF16 residual. Three of them are live at
/// once inside the reduction, so the check cost more device memory than
/// everything the cache retains.
///
/// Reducing a slice at a time bounds that to this many tokens regardless of
/// the clip length, which is what makes the charge in
/// `device::wan_step_cache_bytes` a small constant rather than a term that
/// grows with the shape. `device.rs` reads this constant so the estimate and
/// the engine cannot disagree about the bound.
pub(crate) const WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS: usize = 1024;

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
    /// The 1.3B geometry collapses to noise with residual reuse (#1559).
    UnqualifiedGeometry,
}

impl WanStepCacheRefusal {
    pub fn message(self) -> &'static str {
        match self {
            Self::UnqualifiedGeometry => {
                "step cache ignored: Wan 1.3B residual caching is not quality-qualified"
            }
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
        hidden_dim: u64,
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
        if hidden_dim == 1536 {
            return (Self::Off, Some(WanStepCacheRefusal::UnqualifiedGeometry));
        }
        (Self::Threshold(threshold), None)
    }

    pub fn is_on(self) -> bool {
        matches!(self, Self::Threshold(_))
    }
}

/// Parse `MOLD_WAN_STEP_CACHE`.
///
/// `off` / an empty value disables. `auto` selects [`AUTO_THRESHOLD`]. A
/// positive finite number is that threshold. Anything else is an error rather than a
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
///
/// Unset means `off`: full denoising is the correctness-preserving default.
/// `auto` remains an explicit opt-in to the measured threshold for workloads
/// where its output has been inspected. Wan 1.3B (hidden width 1536) refuses
/// residual reuse even with an explicit threshold because its cached render
/// collapses to noise (#1559). Distilled adapters and schedules shorter than
/// twelve steps also refuse it. The uncached route runs every block, matching
/// ComfyUI's unpatched Wan forward (`comfy/ldm/wan/model.py:940-959`, revision
/// 8a43c6b).
pub fn requested_threshold() -> Result<Option<f64>> {
    threshold_for_env(crate::runtime_env::value("MOLD_WAN_STEP_CACHE").as_deref())
}

/// Pure half of [`requested_threshold`], so the default is testable without
/// touching the process environment.
pub(crate) fn threshold_for_env(raw: Option<&str>) -> Result<Option<f64>> {
    match raw {
        Some(raw) => parse_threshold(raw),
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
        // Reduced a token slice at a time. Upcasting both residuals whole cost
        // three `[1, T, dim]` F32 buffers live at once — more device memory
        // than the cache retains between steps, and unaccounted by anything.
        // See [`WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS`].
        //
        // The sums accumulate in f64 rather than in the tensor's F32, so this
        // is strictly more accurate than the whole-tensor `mean_all` it
        // replaces, not an approximation of it.
        let tokens = previous.dim(1)?;
        let mut previous_sum = 0.0f64;
        let mut delta_sum = 0.0f64;
        let mut counted = 0usize;

        let mut offset = 0usize;
        while offset < tokens {
            let width = WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS.min(tokens - offset);
            let previous_chunk = previous
                .narrow(1, offset, width)?
                .to_dtype(candle_core::DType::F32)?;
            let current_chunk = current
                .narrow(1, offset, width)?
                .to_dtype(candle_core::DType::F32)?;

            previous_sum += previous_chunk.abs()?.sum_all()?.to_scalar::<f32>()? as f64;
            delta_sum += (current_chunk - &previous_chunk)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()? as f64;

            counted += previous_chunk.elem_count();
            offset += width;
        }

        if counted == 0 {
            return Ok(f64::INFINITY);
        }
        let scale = previous_sum / counted as f64;
        // NaN and zero both mean "no usable reference", and both must yield
        // no reuse rather than a division that produces one.
        if !scale.is_finite() || scale <= 0.0 {
            return Ok(f64::INFINITY);
        }
        Ok((delta_sum / counted as f64) / scale)
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

    /// An unset `MOLD_WAN_STEP_CACHE` preserves full denoising. A real Metal
    /// 1.3B A/B produced a coherent scene with `off` and saturated fields with
    /// the former implicit `auto`, so approximate reuse must be explicit.
    #[test]
    fn an_unset_env_preserves_full_denoising_after_correctness_failure() {
        assert_eq!(threshold_for_env(None).unwrap(), None);
    }

    /// `off` is still the escape hatch, and still means off.
    #[test]
    fn off_still_disables() {
        for raw in ["off", "OFF", "0", "", "  "] {
            assert_eq!(
                threshold_for_env(Some(raw)).expect("valid"),
                None,
                "{raw:?} must disable the cache"
            );
        }
    }

    /// An explicit threshold still wins over the new default.
    #[test]
    fn an_explicit_threshold_outranks_the_default() {
        assert_eq!(threshold_for_env(Some("0.25")).expect("valid"), Some(0.25));
        assert_eq!(
            threshold_for_env(Some("auto")).expect("valid"),
            Some(AUTO_THRESHOLD)
        );
    }

    /// `auto` must never engage where the measurement says it cannot help: a
    /// distilled adapter, or a schedule under `MIN_CACHEABLE_STEPS`. That
    /// refusal keeps an explicit opt-in from silently adding work where reuse
    /// cannot help.
    #[test]
    fn auto_refuses_itself_on_distilled_and_short_schedules() {
        let requested = threshold_for_env(Some("auto")).expect("auto is valid");

        let (policy, refusal) = WanStepCachePolicy::resolve(requested, 20, true, 5120);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, Some(WanStepCacheRefusal::Distilled));

        let (policy, refusal) = WanStepCachePolicy::resolve(requested, 4, false, 5120);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, Some(WanStepCacheRefusal::TooFewSteps));

        // ...and does engage on the shape it was measured on.
        let (policy, refusal) = WanStepCachePolicy::resolve(requested, 20, false, 5120);
        assert_eq!(policy, WanStepCachePolicy::Threshold(AUTO_THRESHOLD));
        assert_eq!(refusal, None);
    }

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
        let (policy, refusal) = WanStepCachePolicy::resolve(Some(0.05), 4, true, 5120);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, Some(WanStepCacheRefusal::Distilled));

        // Distilled wins even at a long schedule.
        let (policy, refusal) = WanStepCachePolicy::resolve(Some(0.05), 40, true, 5120);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, Some(WanStepCacheRefusal::Distilled));

        let (policy, refusal) = WanStepCachePolicy::resolve(Some(0.05), 4, false, 5120);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, Some(WanStepCacheRefusal::TooFewSteps));

        // The quality tiers qualify.
        let (policy, refusal) = WanStepCachePolicy::resolve(Some(0.05), 20, false, 5120);
        assert_eq!(policy, WanStepCachePolicy::Threshold(0.05));
        assert_eq!(refusal, None);

        // An explicitly disabled cache has nothing to disclose.
        let (policy, refusal) = WanStepCachePolicy::resolve(None, 20, false, 5120);
        assert_eq!(policy, WanStepCachePolicy::Off);
        assert_eq!(refusal, None);
    }

    /// The chunked reduction must compute the same criterion the whole-tensor
    /// form did, including across a token count that is not a chunk multiple.
    ///
    /// This is the test that makes the memory saving safe to take: the
    /// reduction is the only thing standing between a residual pair and a
    /// skip/no-skip decision, so an error here silently changes which steps a
    /// render skips rather than failing loudly.
    #[test]
    fn chunked_relative_distance_matches_the_whole_tensor_form() {
        let device = Device::Cpu;
        // Deliberately not a multiple of the chunk width, and wide enough to
        // need three slices — the boundary cases are the tail chunk and the
        // accumulation across chunks.
        let tokens = WAN_STEP_CACHE_REDUCTION_CHUNK_TOKENS * 2 + 377;
        let dim = 8;

        let previous = Tensor::rand(-1.0f32, 1.0f32, (1, tokens, dim), &device).unwrap();
        let current = Tensor::rand(-1.0f32, 1.0f32, (1, tokens, dim), &device).unwrap();

        // The pre-#1482 whole-tensor form, kept here as the oracle.
        let reference = {
            let scale = previous.abs().unwrap().mean_all().unwrap();
            let scale = scale.to_scalar::<f32>().unwrap() as f64;
            let delta = (&current - &previous)
                .unwrap()
                .abs()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
            delta / scale
        };

        let chunked = WanStepCache::relative_distance(&previous, &current).unwrap();
        assert!(
            (chunked - reference).abs() < 1e-4,
            "chunked {chunked} against whole-tensor {reference} — the reduction \
             changed the criterion, not just its memory profile"
        );
    }

    /// A single chunk's worth of tokens must take the same path as many.
    #[test]
    fn a_residual_shorter_than_one_chunk_still_reduces() {
        let device = Device::Cpu;
        let previous = Tensor::full(2.0f32, (1, 4, 8), &device).unwrap();
        let current = Tensor::full(3.0f32, (1, 4, 8), &device).unwrap();
        // mean|prev| = 2, mean|cur - prev| = 1, so the distance is 0.5.
        let distance = WanStepCache::relative_distance(&previous, &current).unwrap();
        assert!((distance - 0.5).abs() < 1e-6, "got {distance}");
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
